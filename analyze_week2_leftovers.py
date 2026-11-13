# -*- coding: utf-8 -*-
"""汇总种子实验 + 增强的性价比（本周第 5、6 项验证）。"""
import csv
import statistics as st
from pathlib import Path

p = Path(r"C:\dev\llm-warmup\results_cifar\runs.csv")
rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))

# ---- 1. 种子稳定性 ----
print("=" * 64)
print("第 5 项：SGD vs AdamW 是不是稳定结论（3 个种子，各 20 轮）")
print("=" * 64)
groups = {"adamw": [], "sgd": []}
for r in rows:
    tag = r["tag"]
    for opt in groups:
        if tag.startswith("seed") and tag.endswith(opt):
            groups[opt].append(float(r["test_acc"]))

print(f"{'优化器':10}{'种子数':>6}{'均值':>10}{'标准差':>10}{'最小':>9}{'最大':>9}")
stats = {}
for opt, vals in groups.items():
    if not vals:
        continue
    stats[opt] = (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)
    print(f"{opt.upper():10}{len(vals):>6}{st.mean(vals):>10.4f}"
          f"{stats[opt][1]:>10.4f}{min(vals):>9.4f}{max(vals):>9.4f}")

if len(stats) == 2:
    diff = stats["sgd"][0] - stats["adamw"][0]
    pooled = ((stats["sgd"][1] ** 2 + stats["adamw"][1] ** 2) / 2) ** 0.5
    print(f"\n  SGD - AdamW = {diff:+.4f}（约 {diff*100:+.2f} 个百分点）")
    print(f"  合并标准差  = {pooled:.4f}")
    print(f"  效应量 d    = {diff / max(pooled, 1e-9):.2f}")
    verdict = "稳定" if abs(diff) > 2 * pooled else "不稳定（差距小于 2 倍标准差）"
    print(f"  判断：{verdict}")

# ---- 2. 增强的性价比 ----
print()
print("=" * 64)
print("第 6 项：数据增强值不值（拿 GPU 时间换准确率）")
print("=" * 64)
base = {r["tag"]: r for r in rows}
pairs = [
    ("B 无正则 (none, wd=0)", "B_no_regularization"),
    ("B1 加增强 (basic, wd=0)", "B1_wd0_with_aug"),
    ("A 增强+wd (basic, wd=5e-4)", "A_baseline"),
    ("E 强增强 40 轮", "E_strong_aug"),
]
prev = None
print(f"{'配置':26}{'轮数':>5}{'耗时s':>8}{'acc':>9}{'相对上一行':>12}{'每分钟涨点':>12}")
for label, tag in pairs:
    r = base.get(tag)
    if not r:
        continue
    acc = float(r["test_acc"])
    sec = float(r["total_seconds"])
    delta = "" if prev is None else f"{(acc - prev[0])*100:+.2f}pt"
    per_min = "" if prev is None else f"{(acc - prev[0])*100 / max((sec - prev[1])/60, 1e-9):+.3f}"
    print(f"{label:26}{int(r['epochs']):>5}{sec:>8.0f}{acc:>9.4f}{delta:>12}{per_min:>12}")
    prev = (acc, sec)
