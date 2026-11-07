# 单卡环境体检报告

单卡环境体检报告
生成时间：2026-11-07 17:36:17

1. 系统与解释器
操作系统   : Windows-10-10.0.26200-SP0
Python     : 3.11.18
解释器路径 : C:\Users\15102\miniconda3\envs\llm\python.exe
CPU 逻辑核 : 32

2. 驱动视角（nvidia-smi，不依赖 PyTorch）
显卡型号    : NVIDIA GeForce RTX 4080 Laptop GPU
显存        : 总 11.99 GB | 已用 1.35 GB | 空闲 10.36 GB
驱动版本    : 610.74
温度 / 功耗 : 46 C | 19.58 W / [N/A] W 上限

3. PyTorch 视角（真正决定能不能训）
PyTorch     : 2.11.0+cu128
torch CUDA  : 12.8   <- 这是 wheel 自带的 runtime，不需要你另装 CUDA Toolkit
cuDNN       : 91900
编译架构    : ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
is_available() : True
设备数量    : 1
设备名称    : NVIDIA GeForce RTX 4080 Laptop GPU
算力等级    : sm_89，58 个 SM
显存总量    : 11.99 GB
此刻空闲    : 10.80 GB（别的程序占了 1.19 GB）

[!] 有 1.19 GB 正被桌面程序占着（浏览器 / 聊天软件 / 壁纸 / 录屏 / 游戏）。
    训练前关掉它们，能多留 1-2GB —— 这就是计划 5.3 里『看着显存够却 OOM』的元凶。

4. 算力实测（4096x4096 矩阵乘，取平均）
 fp32 :    19.09 TFLOPS
 tf32 :    25.20 TFLOPS
 bf16 :    61.05 TFLOPS
 fp16 :    62.01 TFLOPS

bf16 / fp32 = 3.20x（明显大于 1 说明吃到了 Tensor Core；接近 1 也正常，矩阵不够大就喂不饱）

记住这个数：以后估『这轮训练要多久』，先算 FLOPs 再除以这里的吞吐，通常误差在 2 倍以内。

5. 选型建议（对应学习计划 2.2）
你的显存        : 12.0 GB
推荐基座        : Qwen2.5-7B 或 Llama-3.2-3B，4bit QLoRA
说明            : 最舒服的一档，可以拿来做正式项目
训练显存预算    : 约 9.3 GB（空闲显存减掉 1.5GB 安全余量）

换算参考（估算值，以训练日志为准）：
  · 全参 7B 训练       : 需 28GB 以上 -> 本机做不到，也不建议做
  · 7B 4bit QLoRA 训练 : 权重约 4.5GB + 激活/优化器 2-4GB -> 本机可行（第 6-8 周）
  · 1.5B 4bit QLoRA    : 权重约 1.2GB + 约 2GB -> 非常宽裕
  · MNIST 这种小 CNN   : 不到 1GB -> 今天就是这一档，显卡基本在打瞌睡

结论
环境可用：显卡已就绪，可以开始跑 MNIST 了。
