# -*- coding: utf-8 -*-
"""
cifar10_data.py  --  自己写的 CIFAR-10 数据加载（不依赖 torchvision.datasets.CIFAR10）

为什么不用 torchvision 的现成类？
  1. 它要求官方 tar.gz 且校验每个 batch 文件的 MD5，国内下载源只有 12 KB/s，163MB 要三小时；
  2. 计划阶段二要「自己写数据加载」，这个文件就是那份作业。

数据从 HuggingFace 镜像拿（hf-mirror.com，实测 2+ MB/s），两个 parquet 文件：
    train-00000-of-00001.parquet   119.7 MB   50000 张
    test-00000-of-00001.parquet     23.9 MB   10000 张

第一次运行会下载 + 解码 + 缓存成 .npz，之后直接读缓存（约 0.5 秒）。

单独运行可以只准备数据并做体检：
    python cifar10_data.py --root ./data/cifar10
"""

from __future__ import annotations

import io
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

MIRRORS = [
    "https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/plain_text",
    "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text",
]
FILES = {
    "train": "train-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# CIFAR-10 通道均值/标准差（官方论文口径，按 0-1 归一化）
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"  下载 {dst.name} ...", flush=True)
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total and done % (16 << 20) < (1 << 20):
                print(f"    {done / 1048576:.0f}/{total / 1048576:.0f} MB", flush=True)
    tmp.replace(dst)


def _fetch_parquet(split: str, root: Path) -> Path:
    path = root / FILES[split]
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    last = None
    for base in MIRRORS:
        try:
            _download(f"{base}/{FILES[split]}", path)
            return path
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"  镜像失败 {base.split('/')[2]}: {type(exc).__name__}", flush=True)
    raise RuntimeError(f"两个镜像都下不动：{last}")


def _parquet_to_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """把 parquet 读成 (N,32,32,3) uint8 与 (N,) int64。"""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "需要 pyarrow：python -m pip install pyarrow -i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from exc

    from PIL import Image

    table = pq.read_table(path)
    cols = table.column_names
    img_col = "img" if "img" in cols else ("image" if "image" in cols else cols[0])
    lbl_col = "label" if "label" in cols else cols[-1]
    images = table.column(img_col).to_pylist()
    labels = np.asarray(table.column(lbl_col).to_pylist(), dtype=np.int64)

    xs = np.empty((len(images), 32, 32, 3), dtype=np.uint8)
    for i, item in enumerate(images):
        # HuggingFace 的 Image 特征是一个 {bytes, path} 结构
        raw = item["bytes"] if isinstance(item, dict) else item
        with Image.open(io.BytesIO(raw)) as im:
            xs[i] = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return xs, labels


def ensure_data(root: str | Path = "./data/cifar10", verbose: bool = True) -> dict:
    """返回 {'x_train','y_train','x_test','y_test'}，全部是 numpy 数组。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "cifar10.npz"

    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in ("x_train", "y_train", "x_test", "y_test")}

    out = {}
    for split, (xk, yk) in {"train": ("x_train", "y_train"), "test": ("x_test", "y_test")}.items():
        pq_path = _fetch_parquet(split, root)
        if verbose:
            print(f"  解码 {split}（{pq_path.stat().st_size / 1048576:.0f} MB）...", flush=True)
        x, y = _parquet_to_arrays(pq_path)
        out[xk], out[yk] = x, y

    np.savez_compressed(cache, **out)
    if verbose:
        print(f"  已缓存 → {cache}（{cache.stat().st_size / 1048576:.0f} MB）")
    return out


class CIFAR10Dataset:
    """最小的 Dataset 实现：只依赖 numpy + PIL，transform 走 torchvision。"""

    def __init__(self, x: np.ndarray, y: np.ndarray, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        from PIL import Image

        img = Image.fromarray(self.x[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.y[idx])


def build_transforms(train: bool, augment: str = "basic"):
    """train=True 时按 augment 强度加数据增强；验证/测试只做归一化。"""
    from torchvision import transforms

    normalize = transforms.Normalize(MEAN, STD)
    if not train:
        return transforms.Compose([transforms.ToTensor(), normalize])

    ops = []
    if augment in ("basic", "strong"):
        ops += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    ops.append(transforms.ToTensor())
    ops.append(normalize)
    if augment == "strong":
        ops.append(Cutout(8))
    return transforms.Compose(ops)


class Cutout:
    """随机挖一个正方形黑洞。CIFAR 上最便宜的正则，常能白捡 1-2 个点。"""

    def __init__(self, size: int = 8):
        self.size = size

    def __call__(self, img):
        import torch

        h, w = img.shape[1], img.shape[2]
        y = torch.randint(0, h, (1,)).item()
        x = torch.randint(0, w, (1,)).item()
        y1, y2 = max(0, y - self.size // 2), min(h, y + self.size // 2)
        x1, x2 = max(0, x - self.size // 2), min(w, x + self.size // 2)
        img = img.clone()
        img[:, y1:y2, x1:x2] = 0.0
        return img


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="准备并体检 CIFAR-10 数据")
    ap.add_argument("--root", default="./data/cifar10")
    args = ap.parse_args()

    print("准备 CIFAR-10 ...")
    d = ensure_data(args.root)
    for k in ("x_train", "y_train", "x_test", "y_test"):
        a = d[k]
        print(f"  {k:8} shape={a.shape}  dtype={a.dtype}")

    y = d["y_train"]
    print("\n类别分布（训练集）:")
    for i, name in enumerate(CLASSES):
        n = int((y == i).sum())
        print(f"  {i} {name:11} {n:6d}  {'█' * (n // 200)}")

    x = d["x_train"]
    print(f"\n像素范围: min={x.min()} max={x.max()}  mean={x.mean():.2f}")
    print(f"训练集内存: {x.nbytes / 1048576:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
