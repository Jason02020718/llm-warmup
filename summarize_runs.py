# -*- coding: utf-8 -*-
"""汇总所有训练运行：参数量、步数、见过的 token 数、等价 epoch、最好/最终 loss。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_gpt import GPT, GPTConfig  # noqa: E402

ROOT = Path(r"C:\dev\llm-warmup\out_gpt")
RUNS = ["", "long_6L384", "big_12L512", "small_4L256"]
BATCH, BLOCK = 32, 256
CORPUS_TOKENS = 1_003_854          # 训练集字符数（约等于 token 数，字符级）

rows = []
for name in RUNS:
    d = ROOT / name if name else ROOT
    ck = d / "best.pt"
    hist = d / "history.json"
    if not ck.exists():
        print(f"[缺] {d}")
        continue
    c = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**c["config"])
    n_params = GPT(cfg).num_params()
    h = json.loads(hist.read_text(encoding="utf-8")) if hist.exists() else []
    last = h[-1] if h else {"step": c["step"], "train": float("nan"), "val": float("nan")}
    steps_done = last["step"]
    tokens = steps_done * BATCH * BLOCK
    rows.append({
        "run": name or "base(3000步)",
        "n_layer": cfg.n_layer, "n_embd": cfg.n_embd,
        "params_M": round(n_params / 1e6, 2),
        "steps_done": steps_done,
        "best_val": round(c["val"], 4),
        "best_step": c["step"],
        "final_train": round(last["train"], 4),
        "final_val": round(last["val"], 4),
        "tokens_seen_M": round(tokens / 1e6, 1),
        "epochs": round(tokens / CORPUS_TOKENS, 1),
    })

hdr = f"{'运行':16}{'层':>3}{'d':>5}{'参数M':>8}{'步数':>7}{'见过tokenM':>10}{'等价epoch':>10}{'best_val':>10}{'@step':>7}{'末val':>8}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['run']:16}{r['n_layer']:>3}{r['n_embd']:>5}{r['params_M']:>8.2f}{r['steps_done']:>7}"
          f"{r['tokens_seen_M']:>10.1f}{r['epochs']:>10.0f}{r['best_val']:>10.4f}"
          f"{r['best_step']:>7}{r['final_val']:>8.4f}")

out = ROOT / "training_runs_summary.csv"
import csv
with out.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\n[ok] {out}")
