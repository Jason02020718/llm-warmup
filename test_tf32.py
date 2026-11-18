# -*- coding: utf-8 -*-
"""测 TF32（fp32 矩阵乘走 Tensor Core）对训练速度的影响。"""
import sys
import time

import torch

sys.path.insert(0, r"C:\dev\llm-warmup")
from tiny_gpt import GPT, GPTConfig, get_batch

device = torch.device("cuda")
ids = torch.randint(0, 65, (300000,), device=device)
x, y = get_batch(ids, 256, 32, device)
cfg = GPTConfig(vocab_size=65, block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.1)


def bench(model, n=30, warmup=5):
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
    return (time.perf_counter() - t0) / n * 1000


for precision in ["highest", "high", "medium"]:
    torch.set_float32_matmul_precision(precision)
    torch.manual_seed(0)
    m = GPT(cfg).to(device)
    ms = bench(m)
    print(f"matmul_precision={precision:8} {ms:6.1f} ms/step  ({32*256/ms*1000/1000:.0f}k tok/s)")
    del m
    torch.cuda.empty_cache()

# 精度代价：同一份权重下 logits 差多少
torch.set_float32_matmul_precision("highest")
torch.manual_seed(0)
m1 = GPT(cfg).to(device)
lg1, _, _ = m1(x, return_all=True)
torch.set_float32_matmul_precision("high")
lg2, _, _ = m1(x, return_all=True)
d = (lg1 - lg2).abs().max().item()
print(f"\nTF32 与 fp32 的 logits 最大差：{d:.2e}")
