# -*- coding: utf-8 -*-
"""
train_cifar10.py  --  CIFAR-10 上手写训练循环（阶段二主力脚本）

和第一周的 MNIST 脚本比，多了三件第二周要学的东西：
    1. 数据增强（RandomCrop / Flip / Cutout）—— 对抗过拟合的主力
    2. 优化器与调度的对照（AdamW vs SGD+momentum，cosine vs 常数）
    3. 逐层梯度范数记录 —— 用来回答「每一层的梯度流是什么样」

快速上手：
    python train_cifar10.py --epochs 3 --limit-batches 40        # 冒烟（1 分钟）
    python train_cifar10.py --epochs 30                          # 正式基线
    python train_cifar10.py --optimizer sgd --lr 0.1             # 换 SGD
    python train_cifar10.py --no-augment --wd 0                  # 拆掉所有正则，看过拟合

产物（默认 ./results_cifar）：
    runs.csv          每次运行的配置与结果（追加）
    curve_<tag>.png   训练/验证曲线
    gradflow.png      逐层梯度范数热图（层 × epoch）
    gradflow.csv      原始梯度数据
    last_run.json     完整结果
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

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cifar10_data import CLASSES, CIFAR10Dataset, build_transforms, ensure_data


# --------------------------------------------------------------------------
# 模型：三个卷积块（每块 2 个 3x3 卷积 + BN + Pool），全局平均池化后接一层线性
# --------------------------------------------------------------------------
class CIFARNet(nn.Module):
    def __init__(self, widths=(64, 128, 256), num_classes: int = 10,
                 dropout: float = 0.0, use_bn: bool = True):
        super().__init__()
        blocks = []
        in_ch = 3
        for w in widths:
            blocks.append(self._block(in_ch, w, use_bn))
            in_ch = w
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, num_classes)

    @staticmethod
    def _block(cin: int, cout: int, use_bn: bool) -> nn.Sequential:
        layers: list[nn.Module] = []
        for i in range(2):
            layers.append(nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))


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


def param_groups_report(model: nn.Module) -> list[str]:
    """把参数按模块归类，梯度热图的行就用这个顺序。"""
    groups: list[str] = []
    for name, _ in model.named_parameters():
        if not _.requires_grad:
            continue
        group = name.rsplit(".", 1)[0] if name.endswith(("weight", "bias")) else name
        if group not in groups:
            groups.append(group)
    return groups


def grad_norms(model: nn.Module) -> dict[str, dict[str, float]]:
    """每个参数组的 (梯度范数, 权重范数, 比值)。比值是判断梯度消失/爆炸的关键指标。"""
    out: dict[str, dict[str, float]] = {}
    for name, p in model.named_parameters():
        if p.grad is None or not p.requires_grad:
            continue
        group = name.rsplit(".", 1)[0] if name.endswith(("weight", "bias")) else name
        g = p.grad.detach().float().norm().item()
        w = p.detach().float().norm().item()
        d = out.setdefault(group, {"grad": 0.0, "weight": 0.0})
        # 同一组内各参数的平方和累加，最后开方
        d["grad"] += g * g
        d["weight"] += w * w
    for d in out.values():
        d["grad"] = math.sqrt(d["grad"])
        d["weight"] = math.sqrt(d["weight"])
        d["ratio"] = d["grad"] / max(d["weight"], 1e-12)
    return out


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_ctx):
    model.eval()
    total, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with amp_ctx():
            out = model(x)
            loss = criterion(out, y)
        total += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return total / max(n, 1), correct / max(n, 1)


def make_amp(device, amp: str):
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
    ap = argparse.ArgumentParser(description="CIFAR-10 手写训练循环 + 逐层梯度流")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=5e-4, help="weight decay")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--no-bn", action="store_true")
    ap.add_argument("--augment", choices=["none", "basic", "strong"], default="basic")
    ap.add_argument("--scheduler", choices=["none", "cosine", "step"], default="cosine")
    ap.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")
    ap.add_argument("--widths", default="64,128,256", help="三个卷积块的通道数")
    ap.add_argument("--data-root", default="./data/cifar10")
    ap.add_argument("--out-dir", default="./results_cifar")
    ap.add_argument("--val-size", type=int, default=5000)
    ap.add_argument("--limit-batches", type=int, default=0, help="每轮最多几个 batch，冒烟用")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    widths = tuple(int(x) for x in args.widths.split(","))
    use_bn = not args.no_bn
    tag = args.tag or f"{args.optimizer}_lr{args.lr:g}_wd{args.wd:g}_{args.augment}" + ("" if use_bn else "_nobn")

    print("=" * 70)
    print(f"CIFAR-10 | tag={tag}")
    print(f"设备 {device.type.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "  <- 没有显卡"))
    print(f"超参 epochs={args.epochs} bs={args.batch_size} lr={args.lr:g} opt={args.optimizer} "
          f"wd={args.wd:g} dropout={args.dropout} aug={args.augment} sched={args.scheduler} amp={args.amp}")
    print("=" * 70)

    t_start = time.perf_counter()
    print("加载数据 ...")
    data = ensure_data(args.data_root)

    import numpy as np

    x, y = data["x_train"], data["y_train"]
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(y))
    val_idx, tr_idx = idx[: args.val_size], idx[args.val_size:]
    train_ds = CIFAR10Dataset(x[tr_idx], y[tr_idx], build_transforms(True, args.augment))
    val_ds = CIFAR10Dataset(x[val_idx], y[val_idx], build_transforms(False))
    test_ds = CIFAR10Dataset(data["x_test"], data["y_test"], build_transforms(False))

    pin = device.type == "cuda"
    nw = args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=pin, drop_last=True,
                              persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=nw,
                            pin_memory=pin, persistent_workers=nw > 0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=nw,
                             pin_memory=pin, persistent_workers=nw > 0)
    print(f"数据就绪：train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    model = CIFARNet(widths=widths, dropout=args.dropout, use_bn=use_bn).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{n_params:,}")

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.wd, nesterov=True)

    steps_per_epoch = args.limit_batches or len(train_loader)
    total_steps = max(args.epochs * steps_per_epoch, 1)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(steps_per_epoch * 10, 1), gamma=0.3)
    else:
        scheduler = None

    amp_ctx, scaler = make_amp(device, args.amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    history: list[dict] = []
    grad_hist: list[dict] = []
    best_val, best_state = -1.0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        run_loss, correct, n, diverged = 0.0, 0, 0, False
        last_grad: dict[str, dict[str, float]] = {}

        for step, (xb, yb) in enumerate(train_loader):
            if args.limit_batches and step >= args.limit_batches:
                break
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                out = model(xb)
                loss = criterion(out, yb)
            if not math.isfinite(loss.item()):
                print(f"\n[!] loss={loss.item()}，训练发散了。先降学习率。")
                diverged = True
                break

            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                last_grad = grad_norms(model)      # 反归一化后再读，才是真实梯度
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                last_grad = grad_norms(model)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()
            run_loss += loss.item() * yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            n += yb.size(0)

        if diverged:
            break

        train_loss, train_acc = run_loss / max(n, 1), correct / max(n, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, amp_ctx)
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else 0.0

        print(f"epoch {epoch:02d}/{args.epochs} | train {train_loss:.4f} acc {train_acc:.4f} | "
              f"val {val_loss:.4f} acc {val_acc:.4f} | {dt:5.1f}s | {peak:.0f} MB")

        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc, seconds=dt, peak_mem_mb=peak,
                            lr=scheduler.get_last_lr()[0] if scheduler is not None else args.lr))
        if last_grad:
            row = {"epoch": epoch}
            for k, v in last_grad.items():
                row[k] = v["ratio"]
            grad_hist.append(row)
            worst = min(last_grad.items(), key=lambda kv: kv[1]["ratio"])
            print(f"          梯度/权重 最小：{worst[0]} = {worst[1]['ratio']:.2e}")

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    total_time = time.perf_counter() - t_start
    test_loss, test_acc = float("nan"), float("nan")
    if best_state is not None:
        model.load_state_dict(best_state)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, amp_ctx)

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else 0.0
    final_train = history[-1]["train_loss"] if history else float("nan")
    final_val = history[-1]["val_loss"] if history else float("nan")

    print("-" * 70)
    print(f"[SUMMARY] tag={tag} test_acc={test_acc:.4f} best_val_acc={best_val:.4f} "
          f"train_loss={final_train:.4f} val_loss={final_val:.4f} "
          f"peak_mem_MB={peak_mem:.0f} total_s={total_time:.1f} params={n_params}")

    gap = final_val - final_train
    if history and math.isfinite(gap):
        if math.isfinite(test_acc) and test_acc < 0.5:
            print(f"[诊断] 测试准确率 {test_acc:.3f}，接近随机猜（10 类=0.10）；训练没起步。")
        elif gap > 0.3:
            print(f"[诊断] val_loss 比 train_loss 高 {gap:.3f}，过拟合明显。加增强 / 权重衰减 / 早停。")
        elif final_train > 1.5:
            print("[诊断] train_loss 仍然很高，是欠拟合：模型太小、训练太短，或学习率太低。")
        else:
            print("[诊断] 训练与验证比较贴合。")

    if not args.no_save:
        _save(args, out_dir, tag, device, history, grad_hist, dict(
            tag=tag, test_acc=test_acc, best_val_acc=best_val,
            final_train_loss=final_train, final_val_loss=final_val,
            peak_mem_mb=peak_mem, total_seconds=total_time, params=n_params,
            device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else "",
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, optimizer=args.optimizer,
            wd=args.wd, dropout=args.dropout, augment=args.augment, scheduler=args.scheduler,
            amp=args.amp, use_bn=use_bn, widths=args.widths,
        ))
        torch.save(model.state_dict(), out_dir / f"best_{tag}.pt")

    return 0


def _save(args, out_dir: Path, tag: str, device, history, grad_hist, summary) -> None:
    # 曲线
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
            axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy"); axes[0].set_title(f"Loss ({tag})")
            axes[0].legend(); axes[0].grid(alpha=0.3)
            axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

            axes[1].plot(ep, [h["val_acc"] for h in history], "s-", color="tab:green", label="val acc")
            axes[1].plot(ep, [h["train_acc"] for h in history], "o--", color="tab:gray", label="train acc")
            axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].set_title("Accuracy")
            axes[1].legend(); axes[1].grid(alpha=0.3)
            axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
            fig.tight_layout()
            fig.savefig(out_dir / f"curve_{tag}.png", dpi=130)
            plt.close(fig)
            print(f"[ok] 曲线 {out_dir / f'curve_{tag}.png'}")
        except Exception as exc:  # noqa: BLE001
            print(f"[!] 画图失败：{type(exc).__name__}: {exc}")

    # 梯度流
    if grad_hist:
        layers = [k for k in grad_hist[0] if k != "epoch"]
        with (out_dir / f"gradflow_{tag}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["epoch"] + layers)
            w.writeheader(); w.writerows(grad_hist)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            g = np.array([[row.get(k, np.nan) for k in layers] for row in grad_hist], dtype=float)
            fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(layers)), 4.2))
            im = ax.imshow(np.log10(np.clip(g, 1e-12, None)).T, aspect="auto", cmap="viridis",
                           origin="lower")
            ax.set_xlabel("epoch")
            ax.set_ylabel("layer")
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels(layers, fontsize=7)
            ax.set_title(f"Grad/Weight norm ratio (log10)  {tag}")
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            fig.colorbar(im, ax=ax, label="log10(||grad|| / ||w||)")
            fig.tight_layout()
            fig.savefig(out_dir / f"gradflow_{tag}.png", dpi=130)
            fig.savefig(out_dir / "gradflow.png", dpi=130)
            plt.close(fig)
            print(f"[ok] 梯度流热图 {out_dir / f'gradflow_{tag}.png'}")
        except Exception as exc:  # noqa: BLE001
            print(f"[!] 梯度热图失败：{type(exc).__name__}: {exc}")

    # CSV
    fields = ["time", "tag", "device", "gpu", "epochs", "batch_size", "lr", "optimizer", "wd",
              "dropout", "augment", "scheduler", "amp", "use_bn", "widths", "params",
              "final_train_loss", "final_val_loss", "best_val_acc", "test_acc",
              "peak_mem_mb", "total_seconds"]
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S")}
    row.update({k: summary.get(k, "") for k in fields if k != "time"})
    p = out_dir / "runs.csv"
    new = not p.exists()
    with p.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})
    print(f"[ok] 追加 {p}")
    (out_dir / "last_run.json").write_text(
        json.dumps({"summary": summary, "history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
