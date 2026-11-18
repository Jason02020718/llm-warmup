# -*- coding: utf-8 -*-
"""
bench_attention.py  --  手写注意力 vs Flash Attention(SDPA) vs torch.compile

三件事：
    1. 数值等价：同一份权重、同一份输入，两种实现的 logits 必须一致
    2. 训练速度：每步耗时、token/s、显存峰值
    3. 生成速度：同样生成 128 token 的耗时

用法：
    python bench_attention.py
    python bench_attention.py --compile
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

from tiny_gpt import GPT, GPTConfig, get_batch

N_LAYER, N_HEAD, N_EMBD, BLOCK, BATCH = 6, 6, 384, 256, 32


def build(device, use_sdpa: bool, seed: int = 1234):
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=65, block_size=BLOCK, n_layer=N_LAYER,
                    n_head=N_HEAD, n_embd=N_EMBD, dropout=0.0, use_sdpa=use_sdpa)
    return GPT(cfg).to(device)


@torch.no_grad()
def logits_of(model, x):
    model.eval()
    out, _, _ = model(x, return_all=True)
    return out


def time_train_steps(model, x, y, n=40, warmup=5):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(warmup):
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


@torch.no_grad()
def time_generate(model, idx, n_new=128, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(idx, n_new, temperature=0.0, use_cache=True)
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return n_new / best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true", help="额外测 torch.compile")
    ap.add_argument("--out", default="out_gpt/attention_bench.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = torch.randint(0, 65, (500_000,), device=device)
    x, y = get_batch(ids, BLOCK, BATCH, device)

    print(f"设备 {device.type.upper()} | {N_LAYER}L/{N_EMBD}d | block {BLOCK} | batch {BATCH}")
    print()

    # ---------- 1. 数值等价 ----------
    m_manual = build(device, use_sdpa=False)
    m_sdpa = build(device, use_sdpa=True)
    m_sdpa.load_state_dict(m_manual.state_dict())
    torch.manual_seed(0)
    lg_manual = logits_of(m_manual, x)
    lg_sdpa = logits_of(m_sdpa, x)
    diff = (lg_manual - lg_sdpa).abs().max().item()
    rel = diff / lg_manual.abs().max().item()
    print("=" * 70)
    print("1. 数值等价性（同一权重，同一输入）")
    print("=" * 70)
    print(f"  logits 最大绝对差：{diff:.2e}")
    print(f"  相对误差        ：{rel:.2e}")
    print(f"  {'通过（两种实现数学等价）' if diff < 1e-3 else '不通过'}")
    del m_manual, m_sdpa
    torch.cuda.empty_cache()

    # ---------- 2. 训练速度 ----------
    print()
    print("=" * 70)
    print("2. 训练速度（40 步，含前向+反向+优化器）")
    print("=" * 70)
    rows = []
    # (名字, 用不用 SDPA, 用不用 compile, 用不用 TF32)
    variants = [
        ("manual", False, False, False),
        ("sdpa", True, False, False),
        ("sdpa+tf32", True, False, True),
    ]
    if args.compile:
        variants.append(("sdpa+compile", True, True, False))

    hdr = f"{'实现':16}{'每步 ms':>10}{'token/s':>12}{'显存 MB':>10}"
    print(hdr)
    print("-" * len(hdr))
    for name, sdpa, use_compile, use_tf32 in variants:
        torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
        m = build(device, use_sdpa=sdpa)
        if use_compile:
            m = torch.compile(m)
        torch.cuda.reset_peak_memory_stats(device)
        dt = time_train_steps(m, x, y)
        mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        tps = BATCH * BLOCK / dt / 1000
        print(f"{name:16}{dt*1000:>10.1f}{tps:>11.0f}k{mem:>10.0f}")
        rows.append({"variant": name, "ms_per_step": round(dt * 1000, 1),
                     "k_token_per_s": round(tps, 1), "peak_mem_MB": round(mem)})
        del m
        torch.cuda.empty_cache()

    # ---------- 3. 生成速度 ----------
    print()
    print("=" * 70)
    print("3. 生成速度（贪心，128 token，带 KV Cache）")
    print("=" * 70)
    idx = torch.randint(0, 65, (1, 32), device=device)
    for name, sdpa, use_compile, use_tf32 in variants:
        torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
        m = build(device, use_sdpa=sdpa)
        if use_compile:
            m = torch.compile(m)
        tps = time_generate(m, idx)
        print(f"{name:16}{tps:>10.1f} tok/s")
        for r in rows:
            if r["variant"] == name:
                r["gen_tok_s"] = round(tps, 1)
        del m
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[ok] {out}")

    if len(rows) >= 2:
        base, new = rows[0], rows[1]
        print(f"\nSDPA 相对手写：训练 {base['ms_per_step']/new['ms_per_step']:.2f}x，"
              f"生成 {new.get('gen_tok_s',0)/max(base.get('gen_tok_s',1),1e-9):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
