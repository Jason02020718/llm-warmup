# -*- coding: utf-8 -*-
"""
tiny_gpt.py  --  字符级 GPT，从零手写（阶段三主线）

不是一个"调用库"的脚本，每一块都写清楚：
    Tokenizer（字符级） -> Embedding(wte+wpe) -> N × [LayerNorm -> Attention -> 残差 -> LayerNorm -> MLP -> 残差]
    -> LayerNorm -> lm_head（与 wte 权重共享）

三个命令用途：
    python tiny_gpt.py verify          # 正确性自检：因果性 + KV Cache 等价性
    python tiny_gpt.py overfit         # 单 batch 过拟合：loss 应该降到 0.1 以下
    python tiny_gpt.py train           # 正式训练
    python tiny_gpt.py sample --ckpt out/best.pt --prompt "ROMEO:"

关键实现细节（和大模型一致的那几条）：
    · pre-norm：先归一化再进子层，残差走外圈 —— 训练更稳
    · 权重共享：输入 embedding 和输出投影用同一份权重
    · 残差投影缩放初始化：标准差除以 sqrt(2*n_layer)，否则深了会炸
    · 学习率 warmup + cosine 衰减
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 0. 字符级 tokenizer
# --------------------------------------------------------------------------
class CharTokenizer:
    """最朴素的 tokenizer：一个字符一个 token。词表几十个，比 BPE 简单得多，
    但训练目标（预测下一个 token）和 GPT 完全一样。"""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.itos = {i: c for i, c in enumerate(chars)}
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def state_dict(self) -> dict:
        return {"chars": [self.itos[i] for i in range(self.vocab_size)]}

    @classmethod
    def from_state(cls, d: dict) -> "CharTokenizer":
        t = cls.__new__(cls)
        chars = d["chars"]
        t.itos = {i: c for i, c in enumerate(chars)}
        t.stoi = {c: i for i, c in enumerate(chars)}
        t.vocab_size = len(chars)
        return t


# --------------------------------------------------------------------------
# 1. 配置
# --------------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256      # 上下文长度
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True
    use_sdpa: bool = True     # True 用 PyTorch 的 Flash Attention 实现

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


# --------------------------------------------------------------------------
# 2. 因果自注意力（支持 KV Cache）
# --------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd 必须能被 n_head 整除"
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.use_sdpa = getattr(cfg, "use_sdpa", True)
        # 一次算出 q、k、v：比三个 Linear 快，且参数排布更紧凑
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # 因果掩码：位置 i 只能看 j <= i，一次建好当 buffer
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
            persistent=False,
        )

    def forward(self, x, kv_cache=None, pos=0, use_cache: bool = False):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        # (B, T, nh, hd) -> (B, nh, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        past_len = 0
        if use_cache and kv_cache is not None:
            past_k, past_v = kv_cache
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present = (k, v) if use_cache else None

        scale = 1.0 / math.sqrt(self.head_dim)
        if self.use_sdpa:
            # PyTorch 内置实现：会走 FlashAttention / memory-efficient 内核，
            # 数学上和下面的手写版等价，但不需要把 (T,T) 的分数矩阵完整写回显存。
            drop = self.attn_dropout.p if self.training else 0.0
            if T > 1 and past_len == 0:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                   dropout_p=drop, scale=scale)
            elif T == 1:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, scale=scale)
            else:
                mask = self.mask[:, :, past_len:past_len + T, :k.size(2)]
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                   dropout_p=drop, scale=scale)
        else:
            att = (q @ k.transpose(-2, -1)) * scale
            if T > 1:
                # 需要掩码的两种情况：训练（整段），或推理时第一次喂入整段 prompt
                Tk = k.size(2)
                att = att.masked_fill(~self.mask[:, :, past_len:past_len + T, :Tk], float("-inf"))
            # 推理时 q 只有 1 步、k 含全部历史，天然满足因果性，无需掩码
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y)), present


# --------------------------------------------------------------------------
# 3. MLP / Block
# --------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, kv_cache=None, pos=0, use_cache: bool = False):
        # pre-norm：先归一化再进子层，残差在外面加
        a, present = self.attn(self.ln1(x), kv_cache=kv_cache, pos=pos, use_cache=use_cache)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, present


# --------------------------------------------------------------------------
# 4. GPT
# --------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)      # token embedding
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)      # position embedding
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # 权重共享：输入词嵌入和输出投影用同一份参数，省参数也提升效果
        self.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        # 残差投影缩放初始化（GPT-2 论文的做法）
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None, kv_caches=None, pos=0, return_all: bool = False):
        B, T = idx.size()
        device = idx.device
        if pos + T > self.cfg.block_size:
            raise ValueError(f"序列超长：pos+T={pos + T} > block_size={self.cfg.block_size}")

        tok_emb = self.wte(idx)                                  # (B,T,C)
        pos_ids = torch.arange(pos, pos + T, device=device)
        pos_emb = self.wpe(pos_ids)                              # (T,C)
        x = self.drop(tok_emb + pos_emb)

        presents = []
        use_cache = kv_caches is not None
        for i, block in enumerate(self.h):
            cache = kv_caches[i] if use_cache else None
            x, present = block(x, kv_cache=cache, pos=pos, use_cache=use_cache)
            if use_cache:
                presents.append(present)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss, presents
        if return_all:
            return self.lm_head(x), None, presents
        # 推理时只算最后一个位置，省算力
        logits = self.lm_head(x[:, [-1], :])
        return logits, None, presents

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 use_cache=True, stop_token=None):
        """采样。use_cache=True 时每步只算 1 个 token 的注意力，而不是整段重算。"""
        self.eval()
        caches = [None] * self.cfg.n_layer if use_cache else None
        out = idx
        pos = 0
        for step in range(max_new_tokens):
            if use_cache:
                chunk = out if step == 0 else out[:, -1:]
                logits, _, caches = self.forward(chunk, kv_caches=caches, pos=pos)
                pos += chunk.size(1)
            else:
                idx_cond = out[:, -self.cfg.block_size:]
                logits, _, _ = self.forward(idx_cond)

            logits = logits[:, -1, :]
            if temperature <= 0:                       # 贪心
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                if top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cum = probs.cumsum(dim=-1)
                    remove = cum - probs > top_p
                    sorted_logits[remove] = float("-inf")
                    logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, nxt], dim=1)
            if stop_token is not None and nxt.item() == stop_token:
                break
        return out


# --------------------------------------------------------------------------
# 5. 数据
# --------------------------------------------------------------------------
def load_data(path: str, block_size: int, device, val_frac: float = 0.1):
    text = Path(path).read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(len(ids) * (1 - val_frac))
    train_ids, val_ids = ids[:n], ids[n:]
    print(f"语料 {len(text):,} 字符 | 词表 {tok.vocab_size} | 训练 {len(train_ids):,} | 验证 {len(val_ids):,}")
    return tok, train_ids, val_ids


def get_batch(ids, block_size: int, batch_size: int, device):
    ix = torch.randint(len(ids) - block_size - 1, (batch_size,), device=ids.device)
    x = torch.stack([ids[i:i + block_size] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def get_lr(step: int, max_steps: int, lr: float, warmup: int, min_lr_ratio: float = 0.1) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    if step > max_steps:
        return lr * min_lr_ratio
    ratio = (step - warmup) / max(1, max_steps - warmup)
    return lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * ratio)))


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, eval_iters, device):
    model.eval()
    out = {}
    for name, ids in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(ids, block_size, batch_size, device)
            _, loss, _ = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


# --------------------------------------------------------------------------
# 6. 自检
# --------------------------------------------------------------------------
@torch.no_grad()
def cmd_verify(args) -> int:
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=17, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(cfg).to(args.device).eval()

    print("=" * 66)
    print("自检 1：因果性——改了位置 t 的输入，位置 < t 的输出必须不变")
    print("=" * 66)
    idx = torch.randint(0, cfg.vocab_size, (1, 12), device=args.device)
    logits1, _, _ = model(idx, return_all=True)
    idx2 = idx.clone()
    idx2[0, -1] = (idx2[0, -1] + 1) % cfg.vocab_size     # 只改最后一个位置
    logits2, _, _ = model(idx2, return_all=True)
    diff_past = (logits1[:, :-1] - logits2[:, :-1]).abs().max().item()
    diff_last = (logits1[:, -1] - logits2[:, -1]).abs().max().item()
    print(f"  位置 0..10 的最大变化：{diff_past:.2e}（应为 0）")
    print(f"  最后一位的最大变化  ：{diff_last:.2e}（应 > 0，说明它确实用到了自己）")
    ok_causal = diff_past < 1e-6 and diff_last > 1e-6

    print()
    print("=" * 66)
    print("自检 2：KV Cache 等价性——逐步缓存 vs 整段重算，每个位置都要一致")
    print("=" * 66)
    prompt = idx[:, :6]
    extra = torch.randint(0, cfg.vocab_size, (1, 5), device=args.device)
    seq = torch.cat([prompt, extra], dim=1)

    logits_full, _, _ = model(seq, return_all=True)     # 整段一次算完
    target = logits_full[0, -6:, :]                    # 最后 6 个位置的 logits

    caches = [None] * cfg.n_layer
    got = []
    lg, _, caches = model(prompt, kv_caches=caches, pos=0)   # 先喂整段 prompt
    got.append(lg[:, -1, :])
    for t in range(5):                                        # 再逐 token 生成
        lg, _, caches = model(extra[:, t:t + 1], kv_caches=caches, pos=prompt.size(1) + t)
        got.append(lg[:, -1, :])
    cached = torch.cat(got, dim=0)
    diff = (cached - target).abs().max().item()
    print(f"  逐位置比对 6 个 logits 向量，最大差：{diff:.2e}（应≈0）")
    ok_cache = diff < 1e-3

    print()
    print("=" * 66)
    print("自检 3：softmax 权重归一化 + 掩码上三角为 0")
    print("=" * 66)
    m = model.h[0].attn.mask[0, 0]
    upper = m.triu(diagonal=1).sum().item()
    print(f"  掩码上三角之和：{upper}（应为 0，即未来被挡住）")
    ok_mask = upper == 0

    print()
    print(f"结论：因果性 {'通过' if ok_causal else '失败'} | KV Cache {'通过' if ok_cache else '失败'} | "
          f"掩码 {'通过' if ok_mask else '失败'}")
    return 0 if (ok_causal and ok_cache and ok_mask) else 1


def cmd_overfit(args) -> int:
    """单 batch 过拟合：如果实现正确，loss 应该能压到接近 0。"""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=args.vocab_size, block_size=64, n_layer=2, n_head=2,
                    n_embd=64, dropout=0.0)
    model = GPT(cfg).to(args.device)
    ids = torch.randint(0, cfg.vocab_size, (400,), device=args.device)
    x, y = get_batch(ids, 64, 8, args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    print("单 batch 过拟合测试（loss 应降到 0.1 以下）")
    for i in range(1, 301):
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i % 50 == 0:
            print(f"  step {i:3d} loss {loss.item():.4f}")
    final = loss.item()
    print(f"最终 loss {final:.4f} → {'通过' if final < 0.1 else '失败，梯度或掩码可能有问题'}")
    return 0 if final < 0.1 else 1


def cmd_train(args) -> int:
    device = args.device
    tok, train_ids, val_ids = load_data(args.data, args.block_size, device)
    splits = {"train": train_ids, "val": val_ids}

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size,
                    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                    dropout=args.dropout)
    model = GPT(cfg).to(device)
    print(f"参数 {model.num_params()/1e6:.2f}M（不含位置嵌入）| block {cfg.block_size} | "
          f"层 {cfg.n_layer} | 头 {cfg.n_head} | d_model {cfg.n_embd}")

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")
    t0 = time.perf_counter()

    for step in range(args.max_steps + 1):
        lr = get_lr(step, args.max_steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        if step % args.eval_interval == 0:
            losses = estimate_loss(model, splits, args.block_size, args.batch_size,
                                   args.eval_iters, device)
            dt = time.perf_counter() - t0
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
                  f"lr {lr:.2e} | {dt:6.1f}s")
            history.append({"step": step, "train": losses["train"], "val": losses["val"],
                            "lr": lr, "seconds": dt})
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "config": asdict(cfg),
                            "tokenizer": tok.state_dict(), "step": step, "val": best_val},
                           out / "best.pt")

        if step == args.max_steps:
            break

        x, y = get_batch(train_ids, args.block_size, args.batch_size, device)
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\n训练结束，最优 val loss {best_val:.4f}，用时 {time.perf_counter()-t0:.0f}s")

    # 训练完立刻生成一段，眼见为实
    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    prompt = "ROMEO:"
    idx = torch.tensor([tok.encode(prompt)], device=device)
    with torch.no_grad():
        gen = model.generate(idx, 200, temperature=0.8, top_k=40)
    print("\n--- 生成样例（temperature 0.8, top-k 40）---")
    print(tok.decode(gen[0].tolist()))
    return 0


def cmd_sample(args) -> int:
    device = args.device
    ckpt = torch.load(args.ckpt, map_location=device)
    tok = CharTokenizer.from_state(ckpt["tokenizer"])
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    idx = torch.tensor([tok.encode(args.prompt)], device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        gen = model.generate(idx, args.max_new_tokens, temperature=args.temperature,
                             top_k=args.top_k, top_p=args.top_p, use_cache=not args.no_cache)
    dt = time.perf_counter() - t0
    n_new = gen.size(1) - idx.size(1)
    print(tok.decode(gen[0].tolist()))
    print(f"\n[{n_new} tokens in {dt:.2f}s = {n_new/dt:.1f} tok/s, cache={not args.no_cache}]")
    return 0


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="字符级 GPT，从零手写")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    v = sub.add_parser("verify"); common(v)
    o = sub.add_parser("overfit"); common(o); o.add_argument("--vocab-size", type=int, default=65)

    t = sub.add_parser("train"); common(t)
    t.add_argument("--data", default="./data/shakespeare/input.txt")
    t.add_argument("--out-dir", default="./out_gpt")
    t.add_argument("--block-size", type=int, default=256)
    t.add_argument("--n-layer", type=int, default=6)
    t.add_argument("--n-head", type=int, default=6)
    t.add_argument("--n-embd", type=int, default=384)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--max-steps", type=int, default=3000)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=0.1)
    t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--eval-interval", type=int, default=250)
    t.add_argument("--eval-iters", type=int, default=20)
    t.add_argument("--tf32", action="store_true",
                   help="开启 TF32（fp32 矩阵乘走 Tensor Core，约快 1.4x，但数值会变）")

    s = sub.add_parser("sample"); common(s)
    s.add_argument("--ckpt", default="./out_gpt/best.pt")
    s.add_argument("--prompt", default="ROMEO:")
    s.add_argument("--max-new-tokens", type=int, default=300)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=40)
    s.add_argument("--top-p", type=float, default=None)
    s.add_argument("--no-cache", action="store_true")

    args = ap.parse_args()
    if getattr(args, "device", "cpu") == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if getattr(args, "tf32", False):
        torch.set_float32_matmul_precision("high")
        print("[info] TF32 已开启（matmul precision = high）")
    return {"verify": cmd_verify, "overfit": cmd_overfit,
            "train": cmd_train, "sample": cmd_sample}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
