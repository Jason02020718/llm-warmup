# -*- coding: utf-8 -*-
"""
tiny_gpt_anatomy.py  --  把模型拆开看：每个模块多少参数、每 token 多少计算量

输出三张表：
    1. 逐模块参数量与占比
    2. 每 token 的前向 FLOPs 分解（注意力 vs MLP）
    3. loss 与困惑度的换算表
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_gpt import GPT, GPTConfig  # noqa: E402


def main() -> int:
    cfg = GPTConfig(vocab_size=65, block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.1)
    model = GPT(cfg)
    d = cfg.n_embd
    T = cfg.block_size

    # ---------- 1. 参数分解 ----------
    print("=" * 72)
    print(f"参数分解（6L / {cfg.n_head}H / d={d} / 词表 {cfg.vocab_size} / 上下文 {T}）")
    print("=" * 72)
    groups: dict[str, int] = {}
    for name, p in model.named_parameters():
        if name.startswith("wte"):
            key = "wte 词嵌入（与 lm_head 共享）"
        elif name.startswith("wpe"):
            key = "wpe 位置嵌入"
        elif name.startswith("ln_f"):
            key = "ln_f 最终 LayerNorm"
        elif ".attn." in name:
            key = "每层 attention 小计"
        elif ".mlp." in name:
            key = "每层 MLP 小计"
        elif ".ln1." in name or ".ln2." in name:
            key = "每层 LayerNorm 小计"
        else:
            key = name
        groups[key] = groups.get(key, 0) + p.numel()

    total = sum(p.numel() for p in model.parameters())
    print(f"{'模块':38}{'参数量':>12}{'占比':>9}")
    print("-" * 60)
    for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        if k.startswith("每层") and k != "每层 attention 小计":
            pass
        print(f"{k:38}{v:>12,}{v/total*100:>8.1f}%")
    print("-" * 60)
    print(f"{'合计':38}{total:>12,}{100.0:>8.1f}%")

    # 单层拆得更细
    print()
    print("单层细化（一层有多少参数）")
    blk = model.h[0]
    for nm, mod in [("attn.c_attn (qkv 合并)", blk.attn.c_attn),
                    ("attn.c_proj", blk.attn.c_proj),
                    ("mlp.c_fc", blk.mlp.c_fc),
                    ("mlp.c_proj", blk.mlp.c_proj),
                    ("ln1 + ln2", None)]:
        if mod is not None:
            n = sum(p.numel() for p in mod.parameters())
        else:
            n = sum(p.numel() for p in blk.ln1.parameters()) + sum(p.numel() for p in blk.ln2.parameters())
        print(f"  {nm:28}{n:>10,}")

    # ---------- 2. FLOPs ----------
    print()
    print("=" * 72)
    print("每 token 的前向 FLOPs（乘加各算 2 次）")
    print("=" * 72)
    attn_proj = 2 * 3 * d * d + 2 * d * d          # qkv + out
    attn_score = 2 * T * d + 2 * T * d             # QK^T 与 AV，与 T 有关
    mlp = 2 * d * 4 * d + 2 * 4 * d * d
    ln = 4 * 2 * d                                  # 两次 LayerNorm，粗略
    per_layer = attn_proj + attn_score + mlp + ln
    print(f"  attention 投影 (qkv+out)   {attn_proj:>14,}  = 8d²")
    print(f"  attention 打分 (随 T 变)    {attn_score:>14,}  = 4·T·d   （T={T}）")
    print(f"  MLP                        {mlp:>14,}  = 16d²")
    print(f"  LayerNorm                  {ln:>14,}")
    print(f"  {'单层合计':<26}{per_layer:>14,}")
    total_per_token = per_layer * cfg.n_layer + 2 * d * cfg.vocab_size * 2
    print(f"  {'全模型（含输出层）':<24}{total_per_token:>14,}")
    print()
    print(f"  注意力的占比：{attn_score / per_layer * 100:.1f}%")
    cross = 1.5 * d
    print(f"  当 T ≈ 1.5·d = {cross:.0f} 时，注意力打分 ≈ MLP 的开销；")
    print(f"  现在 T={T}，远小于 {cross:.0f}，所以这个规模下 MLP 才是计算大头。")
    print("  ——这正好解释了第四周 KV Cache 实验里「省了 FLOPs 却不省时间」的现象。")

    # ---------- 3. loss 与困惑度 ----------
    print()
    print("=" * 72)
    print("loss 换算成「困惑度」（perplexity = e^loss，即模型平均在几个候选里犹豫）")
    print("=" * 72)
    print(f"{'loss':>8}{'困惑度':>12}{'含义':>40}")
    for L in [4.174, 3.5, 3.0, 2.5, 2.0, 1.75, 1.47, 1.2, 1.0]:
        ppl = pow(2.718281828, L)
        note = ""
        if abs(L - 4.174) < 0.01:
            note = "均匀猜测（词表 65）"
        elif abs(L - 1.47) < 0.01:
            note = "本周实际达到"
        elif L <= 1.0:
            note = "接近记忆训练集"
        print(f"{L:>8.3f}{ppl:>12.2f}{note:>40}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
