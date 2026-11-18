# 一键复现（Windows PowerShell）
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File .\reproduce_all.ps1            # 全部跑一遍
#   powershell -ExecutionPolicy Bypass -File .\reproduce_all.ps1 -Week 4    # 只跑第 4 周
#
# 预估耗时（RTX 4080 Laptop 12GB）：
#   Week 1  约 3 分钟      Week 2  约 15 分钟
#   Week 3  约 5 分钟      Week 4  约 40 分钟（含 15000 步长训练）

param(
    [int]$Week = 0        # 0 = 全部
)

$ErrorActionPreference = "Continue"

function Need-Python {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Host "[X] 当前窗口没有 python。先 conda activate llm。" -ForegroundColor Red
        exit 1
    }
}

function Step($title) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor DarkGray
    Write-Host ("  " + $title) -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor DarkGray
}

Need-Python

if ($Week -eq 0 -or $Week -eq 1) {
    Step "Week 1 · 环境体检 + MNIST"
    python check_env.py --markdown results\env_report.md
    python train_mnist_cnn.py --epochs 3 --no-download
    powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1
}

if ($Week -eq 0 -or $Week -eq 2) {
    Step "Week 2 · CIFAR-10 + 消融"
    python cifar10_data.py --root .\data\cifar10
    powershell -ExecutionPolicy Bypass -File .\run_week2_experiments.ps1
}

if ($Week -eq 0 -or $Week -eq 3) {
    Step "Week 3 · RNN/LSTM + Attention"
    python rnn_lstm_demo.py --out-dir results_rnn
    python attention_demo.py
}

if ($Week -eq 0 -or $Week -eq 4) {
    Step "Week 4 · Tiny GPT"
    python tiny_gpt.py verify
    python tiny_gpt.py overfit
    python tiny_gpt.py train --max-steps 3000 --out-dir out_gpt
    python kv_cache_bench.py --ckpt out_gpt\best.pt
    python kv_cache_scaling.py --batch 8 --ctx-list "128,256,512"
    python sampling_demo.py --ckpt out_gpt\best.pt
    python tiny_gpt_anatomy.py
    python plot_training_curves.py --dirs out_gpt
}

Write-Host ""
Write-Host "全部完成。产物在 results\ results_cifar\ results_rnn\ results_attention\ out_gpt\" -ForegroundColor Green
