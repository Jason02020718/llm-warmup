# -*- coding: utf-8 -*-
"""
rnn_lstm_demo.py  --  手写 RNN / LSTM，测「梯度沿时间衰减」

任务：**记住第一个符号**
    输入序列长度 L，只有第 0 位是一个真实符号（0..7），其余全是空白。
    要求模型在最后一步输出第 0 位的那个符号。
    L 越大，越需要「把很早以前的信息带到底」——这正是 RNN 的死穴。

测两件事：
    1. 准确率随 L 怎么变
    2. ∂loss/∂h₀ 的范数随 L 怎么变（这就是梯度沿时间回传后还剩多少）

为什么自己写 cell 而不用 nn.RNN：
    nn.RNN 把时间步藏在 C++ 里，拿不到 h₀ 这个张量；手写之后 h₀ 是一个
    需要梯度的叶子节点，一行 torch.autograd.grad 就能量出 BPTT 的衰减。

用法：
    python rnn_lstm_demo.py                 # 训练 + 评测 + 画图
    python rnn_lstm_demo.py --quick         # 只跑短序列，冒烟用
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

import torch
import torch.nn as nn

VOCAB = 9          # 8 个真实符号 + 1 个空白
BLANK = 8
N_CLASS = 8


# --------------------------------------------------------------------------
# 手写 cell
# --------------------------------------------------------------------------
class ManualRNN(nn.Module):
    """最朴素的循环单元：h_t = tanh(x_t W_x + h_{t-1} W_h + b)"""

    def __init__(self, hidden: int):
        super().__init__()
        self.Wx = nn.Parameter(torch.randn(hidden, hidden) * 0.1)
        self.Wh = nn.Parameter(torch.randn(hidden, hidden) * 0.1)
        self.b = nn.Parameter(torch.zeros(hidden))

    def step(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x_t @ self.Wx + h @ self.Wh + self.b)


class ManualLSTM(nn.Module):
    """四个门：遗忘 f / 输入 i / 候选 g / 输出 o；细胞状态用加法更新。"""

    def __init__(self, hidden: int):
        super().__init__()
        self.Wx = nn.Parameter(torch.randn(hidden, 4 * hidden) * 0.1)
        self.Wh = nn.Parameter(torch.randn(hidden, 4 * hidden) * 0.1)
        self.b = nn.Parameter(torch.zeros(4 * hidden))
        self.hidden = hidden

    def step(self, x_t: torch.Tensor, state):
        h, c = state
        gates = x_t @ self.Wx + h @ self.Wh + self.b
        H = self.hidden
        i = torch.sigmoid(gates[:, :H])
        f = torch.sigmoid(gates[:, H:2 * H])
        g = torch.tanh(gates[:, 2 * H:3 * H])
        o = torch.sigmoid(gates[:, 3 * H:])
        c = f * c + i * g          # ← 加法，不是乘法。这一行就是 LSTM 的全部秘密
        h = o * torch.tanh(c)
        return h, c


class SeqModel(nn.Module):
    def __init__(self, cell_type: str, hidden: int = 64):
        super().__init__()
        self.cell_type = cell_type
        self.emb = nn.Embedding(VOCAB, hidden)
        self.cell = ManualRNN(hidden) if cell_type == "rnn" else ManualLSTM(hidden)
        self.out = nn.Linear(hidden, N_CLASS)

    def forward(self, x: torch.Tensor, keep_h0: bool = False):
        B, L = x.shape
        e = self.emb(x)
        dev = x.device
        h = torch.zeros(B, self.emb.embedding_dim, device=dev)
        if keep_h0:
            h.requires_grad_(True)          # 变成叶子，才能求 ∂loss/∂h₀
        h0 = h
        c = torch.zeros_like(h)
        for t in range(L):
            if self.cell_type == "rnn":
                h = self.cell.step(e[:, t], h)
            else:
                h, c = self.cell.step(e[:, t], (h, c))
        logits = self.out(h)
        return (logits, h0) if keep_h0 else logits


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
def make_batch(n: int, length: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.full((n, length), BLANK, dtype=torch.long)
    sym = torch.randint(0, N_CLASS, (n,))
    x[:, 0] = sym
    return x.to(device), sym.to(device)


@torch.no_grad()
def evaluate(model, length: int, device, n: int = 512) -> float:
    model.eval()
    x, y = make_batch(n, length, device)
    acc = (model(x).argmax(1) == y).float().mean().item()
    model.train()
    return acc


def measure_grad(model, length: int, device, n: int = 128) -> float:
    """返回 ‖∂loss/∂h₀‖ 的平均值：梯度从最后一步回传到第 0 步还剩多少。"""
    model.eval()
    x, y = make_batch(n, length, device)
    logits, h0 = model(x, keep_h0=True)
    loss = nn.functional.cross_entropy(logits, y)
    g = torch.autograd.grad(loss, h0)[0]
    model.train()
    return g.norm(dim=1).mean().item()


def train_model(cell_type: str, length: int, device, steps: int = 1000,
                batch: int = 64, lr: float = 1e-3, seed: int = 0):
    torch.manual_seed(seed)
    model = SeqModel(cell_type).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.functional.cross_entropy
    for _ in range(steps):
        x, y = make_batch(batch, length, device)
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description="RNN vs LSTM：记忆长度与梯度衰减")
    ap.add_argument("--lengths", default="5,15,30,60")
    ap.add_argument("--out-dir", default="./results_rnn")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lengths = [int(v) for v in args.lengths.split(",")]
    steps = 120 if args.quick else 1000
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"设备 {device.type.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"任务：记住第 0 位的符号，序列长度 {lengths}")
    print(f"随机猜的准确率 = 1/{N_CLASS} = {1/N_CLASS:.3f}\n")

    rows = []
    t0 = time.perf_counter()
    for cell in ("rnn", "lstm"):
        for L in lengths:
            t1 = time.perf_counter()
            model = train_model(cell, L, device, steps=steps)
            acc = evaluate(model, L, device)
            g = measure_grad(model, L, device)
            dt = time.perf_counter() - t1
            rows.append({"cell": cell, "length": L, "acc": acc, "grad_norm": g, "seconds": dt})
            print(f"  {cell:5} L={L:4d} | acc {acc:.3f} | ‖∂L/∂h₀‖ {g:.3e} | {dt:4.1f}s")

    print(f"\n总耗时 {time.perf_counter() - t0:.1f}s")

    # 落盘
    with (out / "rnn_lstm.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 画图
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for cell, color in (("rnn", "tab:red"), ("lstm", "tab:blue")):
            d = [r for r in rows if r["cell"] == cell]
            xs = [r["length"] for r in d]
            axes[0].plot(xs, [r["acc"] for r in d], "o-", color=color, label=cell.upper())
            axes[1].plot(xs, [r["grad_norm"] for r in d], "o-", color=color, label=cell.upper())

        axes[0].axhline(1 / N_CLASS, ls="--", color="gray", lw=1, label="chance = 1/8")
        axes[0].set_xlabel("sequence length")
        axes[0].set_ylabel("accuracy")
        axes[0].set_title("Can it remember the first token?")
        axes[0].set_ylim(0, 1.05)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        axes[1].set_yscale("log")
        axes[1].set_xlabel("sequence length")
        axes[1].set_ylabel(r"$\|\partial L/\partial h_0\|$")
        axes[1].set_title("Gradient reaching the first step")
        axes[1].legend()
        axes[1].grid(alpha=0.3, which="both")
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.tight_layout()
        fig.savefig(out / "rnn_lstm.png", dpi=130)
        plt.close(fig)
        print(f"[ok] 图已保存 {out / 'rnn_lstm.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[!] 画图失败：{type(exc).__name__}: {exc}")

    # 人话总结
    rnn = {r["length"]: r for r in rows if r["cell"] == "rnn"}
    lstm = {r["length"]: r for r in rows if r["cell"] == "lstm"}
    Ls = sorted(rnn)
    if len(Ls) >= 2:
        a, b = Ls[0], Ls[-1]
        drop_r = rnn[a]["grad_norm"] / max(rnn[b]["grad_norm"], 1e-30)
        drop_l = lstm[a]["grad_norm"] / max(lstm[b]["grad_norm"], 1e-30)
        print("\n===== 结论 =====")
        print(f"序列从 {a} 拉到 {b}：")
        print(f"  RNN  梯度衰减 {drop_r:,.0f} 倍，准确率 {rnn[a]['acc']:.3f} → {rnn[b]['acc']:.3f}")
        print(f"  LSTM 梯度衰减 {drop_l:,.0f} 倍，准确率 {lstm[a]['acc']:.3f} → {lstm[b]['acc']:.3f}")
        print(f"  准确率差：{lstm[b]['acc'] - rnn[b]['acc']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
