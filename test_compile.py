# -*- coding: utf-8 -*-
"""快速试 torch.compile 在 Windows 上能否用（不同 backend）。"""
import sys
import time

import torch

sys.path.insert(0, r"C:\dev\llm-warmup")
from tiny_gpt import GPT, GPTConfig, get_batch

device = torch.device("cuda")
ids = torch.randint(0, 65, (300000,), device=device)
x, y = get_batch(ids, 256, 32, device)


def bench(model, n=20, warmup=5):
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


torch.manual_seed(0)
cfg = GPTConfig(vocab_size=65, block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.1)
m = GPT(cfg).to(device)
print(f"eager                {bench(m):.1f} ms/step")

for backend in ["aot_eager", "inductor"]:
    try:
        torch.manual_seed(0)
        m2 = GPT(cfg).to(device)
        mc = torch.compile(m2, backend=backend)
        ms = bench(mc)
        print(f"compile[{backend:11}] {ms:.1f} ms/step")
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0][:90]
        print(f"compile[{backend:11}] 失败：{type(e).__name__}: {msg}")
