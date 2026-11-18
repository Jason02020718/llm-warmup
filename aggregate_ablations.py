# -*- coding: utf-8 -*-
"""汇总超参消融实验：每个子目录一个配置，读 best.pt 的 config 与 history.json。"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_gpt import GPT, GPTConfig  # noqa: E402

ROOT = Path(r"C:\dev\llm-warmup\out_gpt")

rows = []
for d in sorted(ROOT.glob("abl_*")):
    hist_p = d / "history.json"
    ckpt_p = d / "best.pt"
    if not hist_p.exists() or not ckpt_p.exists():
        print(f"跳过 {d.name}（缺文件）")
        continue
    hist = json.loads(hist_p.read_text(encoding="utf-8"))
    ckpt = torch.load(ckpt_p, map_location="cpu")
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg)
    params = model.num_params()

    # 取训练过程中最后一个 eval 点，和最优 val
    last = hist[-1]
    best = min(h["val"] for h in hist)
    rows.append({
        "tag": d.name.replace("abl_", ""),
        "n_layer": cfg.n_layer, "n_head": cfg.n_head, "n_embd": cfg.n_embd,
        "block": cfg.block_size, "dropout": cfg.dropout,
        "params_M": round(params / 1e6, 2),
        "steps": last["step"], "val_last": round(last["val"], 4),
        "val_best": round(best, 4), "train_last": round(last["train"], 4),
        "seconds": round(last["seconds"], 1),
    })

if not rows:
    print("没有找到 abl_* 目录")
    raise SystemExit(1)

rows.sort(key=lambda r: r["val_best"])
hdr = f"{'配置':14}{'层':>3}{'头':>3}{'d':>5}{'ctx':>5}{'参数M':>8}{'步数':>6}{'val_best':>10}{'val_last':>10}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['tag']:14}{r['n_layer']:>3}{r['n_head']:>3}{r['n_embd']:>5}{r['block']:>5}"
          f"{r['params_M']:>8.2f}{r['steps']:>6}{r['val_best']:>10.4f}{r['val_last']:>10.4f}")

with (ROOT / "ablation_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\n已写 {ROOT / 'ablation_summary.csv'}")

# 画图：val loss 曲线
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for d in sorted(ROOT.glob("abl_*")):
        hp = d / "history.json"
        if not hp.exists():
            continue
        hist = json.loads(hp.read_text(encoding="utf-8"))
        ax.plot([h["step"] for h in hist], [h["val"] for h in hist], "o-",
                ms=3, label=d.name.replace("abl_", ""))
    ax.set_xlabel("step")
    ax.set_ylabel("val loss")
    ax.set_title("Hyperparameter ablations")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "ablation_curves.png", dpi=130)
    print(f"已画 {ROOT / 'ablation_curves.png'}")
except Exception as exc:  # noqa: BLE001
    print(f"[!] 画图失败：{exc}")
