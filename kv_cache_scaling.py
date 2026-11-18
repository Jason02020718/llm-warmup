# -*- coding: utf-8 -*-
"""
kv_cache_scaling.py  --  KV Cache 的收益由什么决定

结论先行（实测得出，见第 4 周报告）：
    · 模型从 10M 涨到 300M，加速比只有 1.04 → 1.06
    · 上下文从 64 拉到 1024，单条生成时也只有 1.2x 左右
    · 原因是这个规模下 GPU 根本喂不饱：每步几十个 kernel 的调度开销（毫秒级）
      远大于注意力本身的计算量（0.1 毫秒级）

这个脚本就是用来扫这两个维度的：--ctx-list 控制上下文，--batch 控制并发。
batch 大了之后 GPU 才进入算力受限区间，缓存的 FLOPs 优势才可能显现。

用法：
    python kv_cache_scaling.py --ctx-list 64,256,1024
    python kv_cache_scaling.py --ctx-list 128,512,1024 --batch 16
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

from tiny_gpt import GPT, GPTConfig

N_NEW = 32
BLOCK = 2048


def bench(model, idx, n_new, use_cache, device, repeat=2):
    best = float("inf")
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(idx, n_new, temperature=0.0, use_cache=use_cache)
        if device.type == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return n_new / best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ctx-list", default="64,256,512,1024")
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--n-layer", type=int, default=6)
    args = ap.parse_args()

    ctx_list = [int(x) for x in args.ctx_list.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    cfg = GPTConfig(vocab_size=65, block_size=BLOCK, n_layer=args.n_layer,
                    n_head=max(4, args.n_embd // 64), n_embd=args.n_embd, dropout=0.0)
    model = GPT(cfg).to(device).eval()
    print(f"设备 {device.type.upper()} | 模型 {cfg.n_layer}L/{cfg.n_embd}d "
          f"({model.num_params()/1e6:.1f}M) | block {BLOCK} | batch {args.batch} | 新生成 {N_NEW}")
    print()
    hdr = f"{'上下文':>7}{'无缓存 tok/s':>15}{'有缓存 tok/s':>15}{'加速比':>9}{'无缓存 毫秒/token':>20}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for ctx in ctx_list:
        idx = torch.randint(0, 65, (args.batch, ctx), device=device)
        s_no = bench(model, idx, N_NEW, False, device)
        s_ca = bench(model, idx, N_NEW, True, device)
        print(f"{ctx:>7}{s_no:>15.1f}{s_ca:>15.1f}{s_ca/max(s_no,1e-9):>8.2f}x{1000/s_no:>20.2f}")
        rows.append({"batch": args.batch, "ctx": ctx, "n_new": N_NEW,
                     "tok_s_no_cache": round(s_no, 1), "tok_s_cache": round(s_ca, 1),
                     "speedup": round(s_ca / max(s_no, 1e-9), 3),
                     "ms_per_token_no_cache": round(1000 / s_no, 2)})

    out = Path("./out_gpt")
    out.mkdir(exist_ok=True)
    csv_p = out / f"kv_cache_ctx_b{args.batch}.csv"
    with csv_p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        xs = [r["ctx"] for r in rows]
        ax.plot(xs, [r["tok_s_no_cache"] for r in rows], "o-", color="tab:red", label="recompute all")
        ax.plot(xs, [r["tok_s_cache"] for r in rows], "s-", color="tab:blue", label="KV cache")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel("generation speed (tok/s)")
        ax.set_title(f"KV cache vs context length (batch={args.batch})")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        png = out / f"kv_cache_ctx_b{args.batch}.png"
        fig.savefig(png, dpi=130)
        print(f"\n[ok] {png}")
    except Exception as exc:  # noqa: BLE001
        print(f"[!] 画图失败：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
