# 第一周工具包 · 使用说明

这 5 个文件是《第一周执行手册》的配套工具，配合计划里「阶段一：体检、环境与最小可运行」使用。

## 文件清单

| 文件 | 干什么 | 什么时候用 |
|---|---|---|
| `check_env.py` | 体检脚本：打印显卡型号、显存、驱动、CUDA、算力等级，实测矩阵乘吞吐，并给出选型建议 | 装好 PyTorch 之后第一件事 |
| `train_mnist_cnn.py` | 手写训练循环的 MNIST 小 CNN，记录每轮耗时与显存峰值 | 步骤 4，今天的主角 |
| `run_experiments.ps1` | 依次跑 5 组对照实验，自动生成对比表 | 步骤 5 |
| `周报模板.md` | 四行周报模板 | 今天收工前 |
| `.gitignore` | 仓库忽略规则 | 建仓库时 |

## 快速开始

先确认在正确的环境里（命令行提示符前面应该有 `(llm)`）：

```powershell
conda activate llm
cd C:\dev\llm-warmup
python check_env.py --markdown results\env_report.md
```

看到 `is_available() : True` 和你的显卡型号，说明环境通了。接着：

```powershell
python -m pip install matplotlib
python train_mnist_cnn.py --epochs 3 --batch-size 128 --lr 1e-3
```

打开 `results\curve.png` 确认两条 loss 曲线都在下降。然后跑对照实验：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1
```

## 五个常用命令

```powershell
# 20 秒冒烟测试：只跑 20 个 batch，确认链路是通的
python train_mnist_cnn.py --epochs 1 --max-batches 20

# 学习率过大，看 loss 怎么炸
python train_mnist_cnn.py --lr 1.0 --epochs 3

# 关掉 BatchNorm，看收敛变慢多少
python train_mnist_cnn.py --no-bn --epochs 3

# 关掉 Dropout 并训 15 轮，看训练集怎么"背答案"
python train_mnist_cnn.py --no-dropout --epochs 15

# 全部参数
python train_mnist_cnn.py --help
```

## 两个容易踩的坑

**1. 装 matplotlib 要单独一条命令。**
`pip install torch torchvision matplotlib --index-url https://download.pytorch.org/whl/cu128` 会失败，因为 `--index-url` 把整个软件源换成了 PyTorch 的源，那个源里没有 matplotlib。正确做法是分两步：

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install matplotlib
```

**2. MNIST 下载不下来。**
`torchvision` 会从亚马逊的公共桶拉 4 个文件（共约 11MB）。如果卡住，可以手动下载后放进 `data\MNIST\raw\`：

```
train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
```

然后加 `--no-download` 参数运行。

## 参数速查

`train_mnist_cnn.py` 里对今天有用的几个：

| 参数 | 默认 | 作用 |
|---|---|---|
| `--epochs` | 3 | 训练轮数 |
| `--lr` | 1e-3 | 学习率 |
| `--batch-size` | 128 | 批大小 |
| `--optimizer` | adamw | `adamw` / `sgd` |
| `--scheduler` | none | `none` / `cosine` |
| `--dropout` | 0.25 | Dropout 概率 |
| `--no-bn` | 关 | 关掉 BatchNorm |
| `--no-dropout` | 关 | 关掉 Dropout |
| `--amp` | none | `bf16` / `fp16`，混合精度（12GB 卡上先不用） |
| `--subset` | 0 | 只用前 N 个样本，冒烟用 |
| `--max-batches` | 0 | 每轮最多几个 batch，冒烟用 |
| `--tag` | 自动 | 这次运行的名字，写进 `runs.csv` |
| `--no-save` | 关 | 不写文件，只看屏显 |
