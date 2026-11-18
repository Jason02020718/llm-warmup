# -*- coding: utf-8 -*-
"""
kv_cache_bench.py  --  KV Cache 到底省了多少

两件事：
    1. 等价性：贪心解码下，带缓存和不带缓存的输出必须**逐 token 完全一致**
    2. 速度：不同上下文长度下，tok/s 差多少；理论上不带缓存是 O(T²)，带缓存是 O(T)

用法：
    python kv_cache_bench.py --ckpt ./out_gpt/best.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

import torch

from tiny_gpt import GPT, GPTConfig, CharTokenizer


def load_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = CharTokenizer.from_state(ckpt["tokenizer"])
    return model, tok, cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./out_gpt/best.pt")
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--prompts", default="32,96,160,240")
    ap.add_argument("--out-dir", default="./out_gpt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    model, tok, cfg = load_model(args.ckpt, device)
    print(f"模型：{cfg.n_layer} 层 / {cfg.n_head} 头 / d_model {cfg.n_embd} / block {cfg.block_size}")

    # 预热：第一次调用要编译 kernel、分配显存，不预热会把第一个数据点测低
    print("预热中 ...")
    warm = torch.tensor([tok.encode("ROMEO: ")], device=device)
    for uc in (False, True):
        with torch.no_grad():
            model.generate(warm, 8, temperature=0.0, use_cache=uc)
    if device == "cuda":
        torch.cuda.synchronize()

    rows = []
    print()
    print(f"{'上下文':>8}{'新生成':>8}{'无缓存 tok/s':>16}{'有缓存 tok/s':>16}{'加速比':>10}{'输出一致':>10}")
    print("-" * 70)
    for plen in [int(x) for x in args.prompts.split(",")]:
        if plen >= cfg.block_size:
            print(f"{plen:>8}  跳过（>= block_size {cfg.block_size}）")
            continue
        base = tok.encode("ROMEO: " * 20)[:plen]
        if len(base) < plen:
            base = (base * (plen // len(base) + 1))[:plen]
        idx = torch.tensor([base], device=device)

        outs = {}
        speeds = {}
        for use_cache in (False, True):
            best_dt, gen = float("inf"), None
            for _ in range(args.repeat):          # 重复取最快的，压掉调度抖动
                torch.cuda.synchronize() if device == "cuda" else None
                t0 = time.perf_counter()
                with torch.no_grad():
                    gen = model.generate(idx, args.new_tokens, temperature=0.0,
                                         use_cache=use_cache)   # 贪心 → 结果确定
                torch.cuda.synchronize() if device == "cuda" else None
                best_dt = min(best_dt, time.perf_counter() - t0)
            n_new = gen.size(1) - idx.size(1)
            outs[use_cache] = gen[0].tolist()
            speeds[use_cache] = n_new / best_dt

        same = outs[False] == outs[True]
        ratio = speeds[True] / max(speeds[False], 1e-9)
        print(f"{plen:>8}{args.new_tokens:>8}{speeds[False]:>16.1f}{speeds[True]:>16.1f}"
              f"{ratio:>9.2f}x{('是' if same else '否'):>10}")
        rows.append({"prompt_len": plen, "new_tokens": args.new_tokens,
                     "tok_s_no_cache": speeds[False], "tok_s_cache": speeds[True],
                     "speedup": ratio, "identical": same})

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "kv_cache_bench.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 画图
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4))
        xs = [r["prompt_len"] for r in rows]
        ax.plot(xs, [r["tok_s_no_cache"] for r in rows], "o-", color="tab:red", label="no cache")
        ax.plot(xs, [r["tok_s_cache"] for r in rows], "s-", color="tab:blue", label="KV cache")
        ax.set_xlabel("prompt length (tokens)")
        ax.set_ylabel("generation speed (tok/s)")
        ax.set_title(f"KV cache speedup (greedy, {args.new_tokens} new tokens)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "kv_cache_bench.png", dpi=130)
        print(f"\n[ok] {out / 'kv_cache_bench.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[!] 画图失败：{type(exc).__name__}: {exc}")

    all_same = all(r["identical"] for r in rows)
    print(f"\n等价性：{'全部一致（缓存没改变结果）' if all_same else '有不一致，缓存实现有问题'}")
    return 0 if all_same else 1


if __name__ == "__main__":
    raise SystemExit(main())
