# run_experiments.ps1
# 五组对照实验（对应执行手册「步骤 5」）
#
# 用法（在 conda 环境 llm 里，项目根目录）：
#     powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1
#
# 每组跑完会把一行追加进 results/runs.csv，最后自动打成一张对比表。

$ErrorActionPreference = "Continue"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[X] 当前窗口里没有 python。先 conda activate llm 再运行本脚本。" -ForegroundColor Red
    exit 1
}

$runs = @(
    @{ tag = "A_baseline";      args = @("--lr", "1e-3", "--epochs", "3") },
    @{ tag = "B_lr_too_high";   args = @("--lr", "1.0",  "--epochs", "3") },
    @{ tag = "C_lr_too_small";  args = @("--lr", "1e-5", "--epochs", "3") },
    @{ tag = "D_no_bn";         args = @("--lr", "1e-3", "--epochs", "3", "--no-bn") },
    @{ tag = "E_overfit";       args = @("--lr", "1e-3", "--epochs", "15", "--no-dropout") }
)

$sw = [System.Diagnostics.Stopwatch]::StartNew()
foreach ($r in $runs) {
    Write-Host ""
    Write-Host ("=" * 68) -ForegroundColor DarkGray
    Write-Host ("  >>> 正在跑 " + $r.tag) -ForegroundColor Cyan
    Write-Host ("=" * 68) -ForegroundColor DarkGray

    $argv = @(".\train_mnist_cnn.py", "--out-dir", "./results", "--tag", $r.tag) + $r.args
    & python @argv
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("[!] " + $r.tag + " 退出码 " + $LASTEXITCODE + "，跳过继续") -ForegroundColor Yellow
    }
}

$sw.Stop()
Write-Host ""
Write-Host ("全部跑完，耗时 " + [math]::Round($sw.Elapsed.TotalMinutes, 1) + " 分钟") -ForegroundColor Green

if (Test-Path ".\results\runs.csv") {
    Write-Host ""
    Write-Host "===== 对比表（把这张表贴进 README 第 5 项）=====" -ForegroundColor Cyan
    Import-Csv ".\results\runs.csv" |
        Select-Object tag,
            @{ n = "lr"; e = { $_.lr } },
            @{ n = "epochs"; e = { $_.epochs } },
            @{ n = "bn"; e = { $_.use_bn } },
            @{ n = "dropout"; e = { $_.dropout } },
            @{ n = "train_loss"; e = { $_.final_train_loss } },
            @{ n = "val_loss"; e = { $_.final_val_loss } },
            @{ n = "test_acc"; e = { $_.test_acc } },
            @{ n = "峰值MB"; e = { $_.peak_mem_mb } } |
        Format-Table -AutoSize | Out-String -Width 200 | Write-Host

    Write-Host "接着回答三个问题（写在 README 里）：" -ForegroundColor Cyan
    Write-Host "  1) B 组 loss 有没有变成 nan 或震荡？这印证了「学习率受 Hessian 最大特征值约束」这条结论。"
    Write-Host "  2) D 组去掉 BN 之后，收敛速度和最终精度各退了多少？"
    Write-Host "  3) E 组的 train_loss 与 val_loss 差距有多大？这条差距就是过拟合的量化定义。"
}
