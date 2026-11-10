# run_week2_experiments.ps1
# 第二周五组对照实验（对应执行手册「五、四个实验」）
#
# 用法（在 conda 环境 llm 里，项目根目录）：
#     powershell -ExecutionPolicy Bypass -File .\run_week2_experiments.ps1
#
# 全程约 40-50 分钟（A/C/D 各 20 轮，B 20 轮，E 40 轮）。

$ErrorActionPreference = "Continue"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[X] 当前窗口里没有 python。先 conda activate llm 再运行本脚本。" -ForegroundColor Red
    exit 1
}

$runs = @(
    @{ tag = "A_baseline";        args = @("--epochs", "20") },
    @{ tag = "B_no_regularization"; args = @("--epochs", "20", "--augment", "none", "--wd", "0") },
    @{ tag = "C_sgd";             args = @("--epochs", "20", "--optimizer", "sgd", "--lr", "0.1", "--scheduler", "cosine") },
    @{ tag = "D_no_bn";           args = @("--epochs", "20", "--no-bn") },
    @{ tag = "E_strong_aug";      args = @("--epochs", "40", "--augment", "strong", "--wd", "1e-3") }
)

$sw = [System.Diagnostics.Stopwatch]::StartNew()
foreach ($r in $runs) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor DarkGray
    Write-Host ("  >>> 正在跑 " + $r.tag) -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor DarkGray

    $argv = @(".\train_cifar10.py", "--tag", $r.tag) + $r.args
    & python @argv
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("[!] " + $r.tag + " 退出码 " + $LASTEXITCODE + "，跳过继续") -ForegroundColor Yellow
    }
}

$sw.Stop()
Write-Host ""
Write-Host ("全部跑完，耗时 " + [math]::Round($sw.Elapsed.TotalMinutes, 1) + " 分钟") -ForegroundColor Green

if (Test-Path ".\results_cifar\runs.csv") {
    Write-Host ""
    Write-Host "===== 对比表（贴进 README）=====" -ForegroundColor Cyan
    Import-Csv ".\results_cifar\runs.csv" |
        Select-Object tag,
            @{ n = "opt"; e = { $_.optimizer } },
            @{ n = "lr"; e = { $_.lr } },
            @{ n = "wd"; e = { $_.wd } },
            @{ n = "aug"; e = { $_.augment } },
            @{ n = "bn"; e = { $_.use_bn } },
            @{ n = "train"; e = { $_.final_train_loss } },
            @{ n = "val"; e = { $_.final_val_loss } },
            @{ n = "test"; e = { $_.test_acc } },
            @{ n = "MB"; e = { $_.peak_mem_mb } } |
        Format-Table -AutoSize | Out-String -Width 220 | Write-Host

    Write-Host "接着回答四个问题（写进 README）：" -ForegroundColor Cyan
    Write-Host "  1) B 组拆掉正则后，train 与 val 差多少？这就是过拟合的量级。"
    Write-Host "  2) C 组 SGD 相比 AdamW，收敛曲线形状有什么不同？（前 5 轮谁快？）"
    Write-Host "  3) D 组关掉 BN 后，打开 results_cifar\gradflow.png 看最左边几层的颜色。"
    Write-Host "  4) E 组 40 轮 + 强增强，测试准确率比 A 高多少？代价是训练时间翻倍值不值？"
}
