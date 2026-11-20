# -*- coding: utf-8 -*-
"""
compute_bpb.py  --  跨 tokenizer 的正确比较指标：bits per byte

为什么不能直接比 loss：
    BPE 的一个 token 平均代表 2.58 个字节，字符级 token 只代表 1 个字符。
    同样文本 BPE 的 token 数少 2.57 倍，所以"每 token 的 loss"天然更高。
    换算成"每字节需要多少比特"才是在比同一件事：

        BPB = loss(nats/token) / ln(2) / (bytes per token)

自动扫描 out_gpt 下所有实验，按 checkpoint 里的 tokenizer 类型决定 token 总数。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"C:\dev\llm-warmup")
TEXT = ROOT / "data" / "wikitext" / "input.txt"
OUT = ROOT / "out_gpt"

text = TEXT.read_text(encoding="utf-8")
N_BYTES = len(text.encode("utf-8"))
N_CHARS = len(text)
BPE_TOKENS = len(np.load(ROOT / "data" / "wikitext" / "bpe_ids.npy"))

runs = []
for d in sorted(OUT.iterdir()):
    ck = d / "best.pt"
    hist = d / "history.json"
    if not ck.exists():
        continue
    c = torch.load(ck, map_location="cpu", weights_only=False)
    tok_type = (c.get("tokenizer") or {}).get("type", "char")
    cfg = c["config"]
    n_tokens = BPE_TOKENS if tok_type == "bpe" else N_CHARS
    bpt = N_BYTES / n_tokens
    bpb = c["val"] / math.log(2) / bpt
    last = float("nan")
    if hist.exists():
        h = json.loads(hist.read_text(encoding="utf-8"))
        last = h[-1]["val"] if h else float("nan")
    runs.append({
        "run": d.name, "tok": tok_type, "vocab": cfg["vocab_size"],
        "ctx": cfg["block_size"], "layers": cfg["n_layer"], "d": cfg["n_embd"],
        "steps": c["step"], "val": round(c["val"], 4),
        "bytes_per_token": round(bpt, 2), "bpb": round(bpb, 4),
        "last_val": round(last, 4) if last == last else "",
    })

runs.sort(key=lambda r: r["bpb"])
print(f"语料：{N_CHARS:,} 字符 / {N_BYTES:,} 字节 | BPE token 数 {BPE_TOKENS:,}")
print()
hdr = f"{'运行':16}{'分词':>6}{'词表':>6}{'ctx':>5}{'层':>3}{'d':>5}{'步数':>7}{'val':>8}{'B/token':>9}{'bits/byte':>10}"
print(hdr)
print("-" * len(hdr))
for r in runs:
    print(f"{r['run']:16}{r['tok']:>6}{r['vocab']:>6}{r['ctx']:>5}{r['layers']:>3}{r['d']:>5}"
          f"{r['steps']:>7}{r['val']:>8.4f}{r['bytes_per_token']:>9.2f}{r['bpb']:>10.4f}")

out = OUT / "bpb_compare.csv"
with out.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
    w.writeheader()
    w.writerows(runs)
print(f"\n[ok] {out}")

char = [r for r in runs if r["tok"] == "char"]
bpe = [r for r in runs if r["tok"] == "bpe"]
if char and bpe:
    best_c = min(char, key=lambda r: r["bpb"])
    best_b = min(bpe, key=lambda r: r["bpb"])
    print()
    print(f"字符级最好：{best_c['run']} → {best_c['bpb']:.4f} bits/byte")
    print(f"BPE  最好：{best_b['run']} → {best_b['bpb']:.4f} bits/byte")
    print(f"**BPE 好 {(1 - best_b['bpb'] / best_c['bpb']) * 100:.1f}%**")
