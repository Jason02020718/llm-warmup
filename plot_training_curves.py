# -*- coding: utf-8 -*-
"""
plot_training_curves.py  --  把训练历史画成 train / val 双曲线，看过拟合怎么发生的

用法：
    python plot_training_curves.py --dirs out_gpt out_gpt/long_6L384
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(d: str):
    p = Path(d) / "history.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["out_gpt", "out_gpt/long_6L384"])
    ap.add_argument("--out", default="out_gpt/train_curves.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    for d in args.dirs:
        hist = load(d)
        if not hist:
            print(f"[跳过] {d} 没有 history.json")
            continue
        name = Path(d).name
        steps = [h["step"] for h in hist]
        axes[0].plot(steps, [h["train"] for h in hist], "-", lw=1.8, label=f"{name} train")
        axes[0].plot(steps, [h["val"] for h in hist], "--", lw=1.8, label=f"{name} val")

        best = min(hist, key=lambda h: h["val"])
        axes[1].plot(steps, [h["val"] for h in hist], "-", lw=1.8, label=name)
        axes[1].scatter([best["step"]], [best["val"]], s=40, zorder=5)
        axes[1].annotate(f"{best['val']:.3f} @ {best['step']}",
                         (best["step"], best["val"]), textcoords="offset points",
                         xytext=(6, 6), fontsize=8)

    for ax, title in zip(axes, ["train vs val (overfitting)", "validation loss only"]):
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(title)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"[ok] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
