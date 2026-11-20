# -*- coding: utf-8 -*-
"""汇总 wikitext 上的全部实验。"""
import json
from pathlib import Path

import torch

ROOT = Path(r"C:\dev\llm-warmup\out_gpt")
NAMES = ["wiki_3k", "wiki_15k", "wiki_ctx512", "wiki_12L512", "wiki_lr3e4", "wiki_lr3e3", "wiki_do0"]

print(f"{'运行':16}{'参数M':>8}{'层':>3}{'d':>5}{'ctx':>5}{'lr':>8}{'drop':>6}{'best_val':>10}{'@step':>7}{'末val':>8}")
print("-" * 88)
for n in NAMES:
    d = ROOT / n
    ck = d / "best.pt"
    if not ck.exists():
        continue
    c = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = c["config"]
    h = json.loads((d / "history.json").read_text(encoding="utf-8")) if (d / "history.json").exists() else []
    last = h[-1]["val"] if h else float("nan")
    # 参数量粗算：每层 (12 d^2 + 2d) + 词表*d
    d_ = cfg["n_embd"]
    params = cfg["n_layer"] * (12 * d_ * d_ + 13 * d_) + cfg["vocab_size"] * d_
    print(f"{n:16}{params/1e6:>8.2f}{cfg['n_layer']:>3}{d_:>5}{cfg['block_size']:>5}"
          f"{'—':>8}{cfg['dropout']:>6}{c['val']:>10.4f}{c['step']:>7}{last:>8.4f}")
