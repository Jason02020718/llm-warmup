# -*- coding: utf-8 -*-
"""
check_env.py  --  单卡游戏电脑的环境体检（对应学习计划 2.1 / 5.1）

在 conda 环境里、项目根目录下运行：
    python check_env.py
    python check_env.py --markdown results/env_report.md

输出内容：
    1. 系统 / Python 版本
    2. 驱动视角：显卡型号、显存总量/已用/空闲、驱动版本、温度与功耗（走 nvidia-smi，不依赖 torch）
    3. PyTorch 视角：版本、自带 CUDA、is_available、算力等级、此刻空闲显存
    4. 矩阵乘法实测吞吐（fp32 / tf32 / bf16 / fp16），作为后续训练的时间预算基准
    5. 按显存给出的选型建议（对应计划 2.2 节的表格）

设计原则：任何一段失败都不能让脚本崩掉，它要能告诉你"下一句该敲什么"。
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time

REPORT: list[str] = []


def say(line: str = "") -> None:
    print(line)
    REPORT.append(line)


def rule(title: str) -> None:
    say()
    say("=" * 66)
    say(title)
    say("=" * 66)


# --------------------------------------------------------------------------
# 1. nvidia-smi：驱动视角，不依赖 torch
# --------------------------------------------------------------------------
def nvidia_smi_info() -> dict:
    exe = shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
    if not os.path.exists(exe):
        return {"_error": "找不到 nvidia-smi，说明没装 NVIDIA 驱动，或者这不是 NVIDIA 显卡"}

    query = "name,memory.total,memory.used,memory.free,driver_version,temperature.gpu,power.draw,power.limit"
    try:
        kwargs = {}
        if platform.system() == "Windows":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW，不弹黑框
        out = subprocess.run(
            [exe, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace", **kwargs,
        )
        if out.returncode != 0:
            return {"_error": (out.stderr or "").strip() or "nvidia-smi 返回非 0"}
        line = out.stdout.strip().splitlines()[0]
        return dict(zip(query.split(","), [p.strip() for p in line.split(",")]))
    except Exception as exc:  # noqa: BLE001
        return {"_error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------
# 2. 吞吐实测
# --------------------------------------------------------------------------
def bench_matmul(torch, dtype, n: int = 4096, iters: int = 10, warmup: int = 3) -> float:
    """n x n 矩阵乘的平均吞吐，单位 TFLOPS。"""
    dev = torch.device("cuda")
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    for _ in range(warmup):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    del a, b
    torch.cuda.empty_cache()
    return 2.0 * (n ** 3) / dt / 1e12


def recommend(total_gb: float) -> tuple[str, str]:
    """按计划 2.2 节的表给出 (推荐模型, 说明)。"""
    if total_gb <= 6.5:
        return "Qwen2.5-0.5B / 1.5B，4bit QLoRA", "以跑通流程为主，别追效果"
    if total_gb <= 9.5:
        return "Qwen2.5-1.5B，4bit QLoRA", "7B 只能极端压缩（seq≤512 + batch=1），不推荐"
    if total_gb <= 14:
        return "Qwen2.5-7B 或 Llama-3.2-3B，4bit QLoRA", "最舒服的一档，可以拿来做正式项目"
    if total_gb <= 18:
        return "7B 宽裕，13B 可试", "可以把序列拉长做多轮对话"
    return "7B / 13B 4bit QLoRA", "接近入门工作站，但仍然不建议全参微调"


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="单卡环境体检")
    ap.add_argument("--markdown", default="", help="把报告另存为 markdown，如 results/env_report.md")
    ap.add_argument("--skip-bench", action="store_true", help="跳过吞吐实测，省 20 秒")
    args = ap.parse_args()

    say("单卡环境体检报告")
    say(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ---------- 系统 ----------
    rule("1. 系统与解释器")
    say(f"操作系统   : {platform.platform()}")
    say(f"Python     : {sys.version.split()[0]}")
    say(f"解释器路径 : {sys.executable}")
    say(f"CPU 逻辑核 : {os.cpu_count()}")

    # ---------- 驱动视角 ----------
    rule("2. 驱动视角（nvidia-smi，不依赖 PyTorch）")
    smi = nvidia_smi_info()
    smi_total_gb = None
    smi_free_gb = None
    if "_error" in smi:
        say(f"[!] 读取失败：{smi['_error']}")
    else:
        say(f"显卡型号    : {smi.get('name')}")
        try:
            total = float(smi.get("memory.total", "0"))
            used = float(smi.get("memory.used", "0"))
            free = float(smi.get("memory.free", "0"))
            smi_total_gb, smi_free_gb = total / 1024, free / 1024
            say(f"显存        : 总 {total / 1024:.2f} GB | 已用 {used / 1024:.2f} GB | 空闲 {free / 1024:.2f} GB")
        except ValueError:
            pass
        say(f"驱动版本    : {smi.get('driver_version')}")
        say(f"温度 / 功耗 : {smi.get('temperature.gpu')} C | {smi.get('power.draw')} W / {smi.get('power.limit')} W 上限")

    # ---------- PyTorch ----------
    rule("3. PyTorch 视角（真正决定能不能训）")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        say(f"[X] import torch 失败：{type(exc).__name__}: {exc}")
        say()
        say("这个环境还没装 PyTorch。回到执行手册「步骤 2」，在 Anaconda Prompt 里：")
        say("    conda activate llm")
        say("    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        if smi_total_gb:
            model, note = recommend(smi_total_gb)
            say()
            say(f"[参考] 你的显存 {smi_total_gb:.1f} GB，装好之后建议跑：{model}（{note}）")
        _write_markdown(args.markdown)
        return 2

    say(f"PyTorch     : {torch.__version__}")
    say(f"torch CUDA  : {torch.version.cuda}   <- 这是 wheel 自带的 runtime，不需要你另装 CUDA Toolkit")
    try:
        say(f"cuDNN       : {torch.backends.cudnn.version()}")
    except Exception:  # noqa: BLE001
        pass
    say(f"编译架构    : {torch.cuda.get_arch_list()}")

    available = torch.cuda.is_available()
    say(f"is_available() : {available}")

    if not available:
        say()
        say("[X] 用不上显卡。按顺序排查，前两条能解决 90% 的情况：")
        say("    1) 装成 CPU 版了：pip uninstall torch torchvision -y，再带 --index-url .../whl/cu128 重装")
        say("    2) 是在错的 conda 环境里跑的：先 conda activate llm，再 python check_env.py")
        say("    3) 驱动太旧：nvidia-smi 能跑说明驱动在，但版本过低仍需更新，装完重启")
        say("    4) 笔记本双显卡：设备管理器里确认独显没被禁用")
        _write_markdown(args.markdown)
        return 1

    say(f"设备数量    : {torch.cuda.device_count()}")
    say(f"设备名称    : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    say(f"算力等级    : sm_{props.major}{props.minor}，{props.multi_processor_count} 个 SM")
    say(f"显存总量    : {props.total_memory / 1024 ** 3:.2f} GB")

    try:
        free_b, total_b = torch.cuda.mem_get_info(0)
    except Exception:  # noqa: BLE001
        free_b = total_b = props.total_memory
    used_gb = (total_b - free_b) / 1024 ** 3
    say(f"此刻空闲    : {free_b / 1024 ** 3:.2f} GB（别的程序占了 {used_gb:.2f} GB）")

    if used_gb > 1.0:
        say()
        say(f"[!] 有 {used_gb:.2f} GB 正被桌面程序占着（浏览器 / 聊天软件 / 壁纸 / 录屏 / 游戏）。")
        say("    训练前关掉它们，能多留 1-2GB —— 这就是计划 5.3 里『看着显存够却 OOM』的元凶。")

    # ---------- 吞吐 ----------
    if not args.skip_bench:
        rule("4. 算力实测（4096x4096 矩阵乘，取平均）")
        results: dict[str, float] = {}
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            results["fp32"] = bench_matmul(torch, torch.float32)
            torch.backends.cuda.matmul.allow_tf32 = True
            results["tf32"] = bench_matmul(torch, torch.float32)
            results["bf16"] = bench_matmul(torch, torch.bfloat16)
            results["fp16"] = bench_matmul(torch, torch.float16)
        except Exception as exc:  # noqa: BLE001
            say(f"[!] 实测中断：{type(exc).__name__}: {exc}")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False

        for name, tf in results.items():
            say(f"{name:>5} : {tf:8.2f} TFLOPS")
        if results.get("fp32") and results.get("bf16"):
            say()
            say(f"bf16 / fp32 = {results['bf16'] / max(results['fp32'], 1e-9):.2f}x"
                "（明显大于 1 说明吃到了 Tensor Core；接近 1 也正常，矩阵不够大就喂不饱）")
        say()
        say("记住这个数：以后估『这轮训练要多久』，先算 FLOPs 再除以这里的吞吐，通常误差在 2 倍以内。")

    # ---------- 选型 ----------
    rule("5. 选型建议（对应学习计划 2.2）")
    total_gb = props.total_memory / 1024 ** 3
    model, note = recommend(total_gb)
    say(f"你的显存        : {total_gb:.1f} GB")
    say(f"推荐基座        : {model}")
    say(f"说明            : {note}")
    budget = max(free_b / 1024 ** 3 - 1.5, 0.5)
    say(f"训练显存预算    : 约 {budget:.1f} GB（空闲显存减掉 1.5GB 安全余量）")
    say()
    say("换算参考（估算值，以训练日志为准）：")
    say("  · 全参 7B 训练       : 需 28GB 以上 -> 本机做不到，也不建议做")
    say("  · 7B 4bit QLoRA 训练 : 权重约 4.5GB + 激活/优化器 2-4GB -> 本机可行（第 6-8 周）")
    say("  · 1.5B 4bit QLoRA    : 权重约 1.2GB + 约 2GB -> 非常宽裕")
    say("  · MNIST 这种小 CNN   : 不到 1GB -> 今天就是这一档，显卡基本在打瞌睡")

    rule("结论")
    name = torch.cuda.get_device_name(0)
    ok = "NVIDIA" in name.upper()
    say("环境可用：显卡已就绪，可以开始跑 MNIST 了。" if ok else "环境异常，按上面的排查步骤处理。")

    _write_markdown(args.markdown)
    return 0 if ok else 1


def _write_markdown(path: str) -> None:
    if not path:
        return
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# 单卡环境体检报告", ""]
    for line in REPORT:
        if set(line) == {"="}:  # 跳过分隔线
            continue
        lines.append(line)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[ok] 报告已保存到 {p}")


if __name__ == "__main__":
    raise SystemExit(main())
