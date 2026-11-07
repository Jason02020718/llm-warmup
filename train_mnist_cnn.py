# -*- coding: utf-8 -*-
"""
train_mnist_cnn.py  --  手写训练循环的最小 CNN（对应学习计划 阶段一 / 5.1 第 7 步）

目标不是刷精度，而是把「下载数据 -> 训练 -> 评估」这条链路在本地打通，
并且记录每轮的耗时与显存峰值，作为后面所有训练的预算基准。

快速上手：
    python train_mnist_cnn.py                                  # 基线，3 轮
    python train_mnist_cnn.py --lr 1.0                         # 学习率过大，看它怎么炸
    python train_mnist_cnn.py --no-bn                          # 关掉 BatchNorm
    python train_mnist_cnn.py --no-dropout --epochs 15         # 看训练集爆打验证集 = 过拟合
    python train_mnist_cnn.py --epochs 1 --max-batches 20      # 20 秒冒烟测试链路是否通

产物：
    results/curve.png     训练/验证曲线
    results/runs.csv      每次运行的配置与结果（追加，用来做对比表）
    results/last_run.json 最后一次运行的完整结果
    results/best.pt       验证集上最好的权重
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

try:  # 别让中文在 GBK 控制台里抛 UnicodeEncodeError
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

try:
    from torchvision import datasets, transforms
except Exception as exc:  # noqa: BLE001
    print(f"[X] import torchvision 失败：{type(exc).__name__}: {exc}")
    print("    在 conda 环境里执行：python -m pip install torchvision")
    raise SystemExit(2)


# --------------------------------------------------------------------------
# 模型：2 个卷积块 + 2 层全连接，约 47 万参数
# --------------------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self, use_bn: bool = True, dropout: float = 0.25):
        super().__init__()

        def block(c_in: int, c_out: int) -> nn.Sequential:
            layers: list[nn.Module] = [nn.Conv2d(c_in, c_out, 3, padding=1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(c_out, c_out, 3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(block(1, 32), block(32, 64))  # 28 -> 14 -> 7
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass


def build_loaders(args: argparse.Namespace, device: torch.device):
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 全局均值/标准差
        ]
    )
    dl = not args.no_download
    full = datasets.MNIST(args.data_dir, train=True, download=dl, transform=tf)
    test = datasets.MNIST(args.data_dir, train=False, download=dl, transform=tf)

    n_val = max(1, int(len(full) * args.val_frac))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    if args.subset:
        from torch.utils.data import Subset

        k = min(args.subset, len(train_ds))
        train_ds = Subset(train_ds, range(k))
        k_val = min(max(args.subset // 5, 500), len(val_ds))
        val_ds = Subset(val_ds, range(k_val))

    pin = device.type == "cuda"
    common = dict(num_workers=args.num_workers, pin_memory=pin, drop_last=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, **common)
    test_loader = DataLoader(test, batch_size=512, shuffle=False, **common)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device, amp_ctx) -> tuple[float, float]:
    """返回 (平均 loss, 准确率)。"""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with amp_ctx():
            out = model(x)
            loss = criterion(out, y)
        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


def make_amp(device: torch.device, amp: str):
    """返回 (amp 上下文工厂, GradScaler)。不同 torch 版本签名不一样，所以包一层。"""
    from contextlib import nullcontext

    if amp == "none" or device.type != "cuda":
        return (lambda: nullcontext()), None

    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    need_scaler = dtype is torch.float16

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=need_scaler)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=need_scaler)

    def ctx():
        return torch.autocast(device_type="cuda", dtype=dtype)

    return ctx, scaler


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="MNIST 小 CNN：打通训练链路 + 记录显存与耗时")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    ap.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--no-bn", action="store_true", help="关掉 BatchNorm，用来做消融")
    ap.add_argument("--no-dropout", action="store_true", help="关掉 Dropout，用来复现过拟合")
    ap.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none",
                    help="混合精度；默认关闭，先看清 fp32 的基准")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--subset", type=int, default=0, help="只用前 N 个训练样本（冒烟测试用）")
    ap.add_argument("--max-batches", type=int, default=0, help="每个 epoch 最多跑几个 batch，0=不限")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0, help="Windows 上建议 0，避免多进程的坑")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--tag", default="", help="给这次运行起个名字，写进 runs.csv")
    ap.add_argument("--no-save", action="store_true", help="不写任何文件（只看屏显）")
    args = ap.parse_args()

    if args.no_dropout:
        args.dropout = 0.0
    use_bn = not args.no_bn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag or f"lr{args.lr:g}_ep{args.epochs}_bs{args.batch_size}" + ("" if use_bn else "_nobn") + ("" if args.dropout else "_nodrop")

    print("=" * 68)
    print(f"MNIST 小 CNN | tag={tag}")
    print(f"设备 {device.type.upper()}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "  <- 没有显卡，只能 CPU 慢跑"))
    print(f"超参 epochs={args.epochs} batch={args.batch_size} lr={args.lr:g} opt={args.optimizer} "
          f"bn={use_bn} dropout={args.dropout} amp={args.amp} scheduler={args.scheduler}")
    print("=" * 68)

    t_start = time.perf_counter()
    train_loader, val_loader, test_loader = build_loaders(args, device)
    print(f"数据就绪：train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"test={len(test_loader.dataset)}（本地缓存 {Path(args.data_dir).resolve()}）")

    model = SmallCNN(use_bn=use_bn, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{n_params:,}")

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "adamw":
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
        if args.scheduler == "cosine"
        else None
    )
    amp_ctx, scaler = make_amp(device, args.amp)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    history: list[dict] = []
    best_val_acc, best_state = -1.0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        run_loss, correct, n = 0.0, 0, 0
        diverged = False

        for step, (x, y) in enumerate(train_loader):
            if args.max_batches and step >= args.max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                out = model(x)
                loss = criterion(out, y)

            if not math.isfinite(loss.item()):
                print(f"\n[!] loss 变成 {loss.item()}，训练发散了。")
                print("    学习率太大是首要嫌疑，先降到 1e-3 再看。")
                diverged = True
                break

            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            run_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)

        if diverged:
            break

        train_loss, train_acc = run_loss / max(n, 1), correct / max(n, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, amp_ctx)
        if scheduler is not None:
            scheduler.step()

        dt = time.perf_counter() - t0
        peak = (
            torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else 0.0
        )
        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"{dt:5.1f}s | 显存峰值 {peak:.0f} MB"
        )
        history.append(
            dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                 val_loss=val_loss, val_acc=val_acc, seconds=dt, peak_mem_mb=peak)
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    total_time = time.perf_counter() - t_start

    # ---- 用最好的权重在测试集上评一次 ----
    test_acc, test_loss = float("nan"), float("nan")
    if best_state is not None:
        model.load_state_dict(best_state)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, amp_ctx)

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else 0.0
    final_train = history[-1]["train_loss"] if history else float("nan")
    final_val = history[-1]["val_loss"] if history else float("nan")

    print("-" * 68)
    print(f"[SUMMARY] tag={tag} test_acc={test_acc:.4f} best_val_acc={best_val_acc:.4f} "
          f"final_train_loss={final_train:.4f} final_val_loss={final_val:.4f} "
          f"peak_mem_MB={peak_mem:.0f} total_s={total_time:.1f} params={n_params}")

    gap = (final_val - final_train) if history else float("nan")
    if history and math.isfinite(gap):
        if math.isfinite(test_acc) and test_acc < 0.5:
            print(f"[诊断] 测试准确率只有 {test_acc:.3f}，等于随机猜（10 类 = 0.10）。"
                  "这不是过拟合，是训练根本没起步：学习率太大，或优化器没生效。")
        elif final_train > 0 and final_val > 2.0 * final_train and final_train < 0.02:
            print(f"[诊断] train_loss {final_train:.4f} 只有 val_loss {final_val:.4f} 的 1/{final_val/final_train:.1f}，"
                  "这是过拟合：训练集背下来了，验证集降不下去。")
        elif gap > 0.05:
            print(f"[诊断] val_loss 比 train_loss 高 {gap:.3f}，已经出现过拟合的迹象。"
                  "要么加正则，要么早停。")
        elif gap < -0.02:
            print("[诊断] val_loss 比 train_loss 低，通常说明正则（Dropout/BN）还在起作用，模型还没训够。")
        else:
            print("[诊断] train 与 val 贴合得不错，这个配置是健康的。")

    if history and history[0]["train_loss"] > 0 and final_train > history[0]["train_loss"] * 0.9:
        if args.max_batches or args.subset:
            print("[提醒] 这是冒烟测试（限了 batch 或样本数），loss 不降是正常的；正式跑一条 --epochs 3 再看。")
        else:
            print("[提醒] loss 几乎没降。先确认设备不是 CPU 慢到没训完，再确认优化器真的 step 了。")

    # ---- 落盘 ----
    if not args.no_save:
        _save_outputs(args, out_dir, tag, device, history, dict(
            tag=tag, test_acc=test_acc, test_loss=test_loss, best_val_acc=best_val_acc,
            final_train_loss=final_train, final_val_loss=final_val, peak_mem_mb=peak_mem,
            total_seconds=total_time, params=n_params, device=str(device),
            gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else "",
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            optimizer=args.optimizer, use_bn=use_bn, dropout=args.dropout, amp=args.amp,
        ))
        torch.save(model.state_dict(), out_dir / "best.pt")

    return 0


def _save_outputs(args, out_dir: Path, tag: str, device, history: list[dict], summary: dict) -> None:
    # 1) 曲线图
    if history:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator

            ep = [h["epoch"] for h in history]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            axes[0].plot(ep, [h["train_loss"] for h in history], "o-", label="train")
            axes[0].plot(ep, [h["val_loss"] for h in history], "s-", label="val")
            axes[0].set_xlabel("epoch")
            axes[0].set_ylabel("cross-entropy loss")
            axes[0].set_title(f"Loss  ({tag})")
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

            axes[1].plot(ep, [h["val_acc"] for h in history], "s-", color="tab:green", label="val acc")
            axes[1].plot(ep, [h["train_acc"] for h in history], "o--", color="tab:gray", label="train acc")
            axes[1].set_xlabel("epoch")
            axes[1].set_ylabel("accuracy")
            lo = min(min(h["val_acc"] for h in history), min(h["train_acc"] for h in history))
            axes[1].set_ylim(max(0.0, lo - 0.05), 1.02)
            axes[1].set_title("Accuracy")
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

            fig.tight_layout()
            fig.savefig(out_dir / f"curve_{tag}.png", dpi=130)
            fig.savefig(out_dir / "curve.png", dpi=130)
            plt.close(fig)
            print(f"[ok] 曲线已保存 {out_dir / f'curve_{tag}.png'}（同时更新 curve.png）")
        except Exception as exc:  # noqa: BLE001
            print(f"[!] 画图失败（不影响结果）：{type(exc).__name__}: {exc}")

    # 2) 追加 CSV
    csv_path = out_dir / "runs.csv"
    fields = [
        "time", "tag", "device", "gpu", "epochs", "batch_size", "lr", "optimizer",
        "use_bn", "dropout", "amp", "params", "final_train_loss", "final_val_loss",
        "best_val_acc", "test_acc", "peak_mem_mb", "total_seconds",
    ]
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": summary.get("device", ""),
        "gpu": summary.get("gpu", ""),
    }
    for k in fields:
        if k not in row:
            row[k] = summary.get(k, "")
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})
    print(f"[ok] 结果已追加 {csv_path}")

    # 3) 完整结果
    (out_dir / "last_run.json").write_text(
        json.dumps({"summary": summary, "history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
