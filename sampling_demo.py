# -*- coding: utf-8 -*-
"""
sampling_demo.py  --  同一段 prompt，不同采样策略的差别

GPT 输出的其实是一个概率分布。怎么从这个分布里挑下一个词，决定了文本的"性格"：
    贪心(temperature=0)  永远选概率最大的：稳、重复、容易死循环
    temperature 低       接近贪心，但偶尔换词
    temperature 高       分布被压平，冷门词也能被选中：有创意，也会胡言乱语
    top-k                只在前 k 个候选里抽，砍掉长尾
    top-p (nucleus)      取累积概率到 p 的最小集合，长度自适应

用法：
    python sampling_demo.py --ckpt ./out_gpt/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

import torch

from tiny_gpt import GPT, GPTConfig, CharTokenizer

CASES = [
    ("贪心 (T=0)", {"temperature": 0.0}),
    ("低温 T=0.5", {"temperature": 0.5}),
    ("常溫 T=0.9", {"temperature": 0.9}),
    ("高温 T=1.4", {"temperature": 1.4}),
    ("top-k 10", {"temperature": 0.9, "top_k": 10}),
    ("top-p 0.9", {"temperature": 0.9, "top_p": 0.9}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./out_gpt/best.pt")
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--out", default="./out_gpt/sampling_demo.md")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    cfg = GPTConfig(**ckpt["config"])
    tok = CharTokenizer.from_state(ckpt["tokenizer"])
    model = GPT(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    idx = torch.tensor([tok.encode(args.prompt)], device=args.device)
    lines = [f"# 采样策略对比", "", f"prompt: `{args.prompt}`，各生成 {args.tokens} token", "",
             "| 策略 | 参数 | 输出（前 110 字符） |", "|---|---|---|"]

    print(f"prompt = {args.prompt!r}\n")
    for name, kw in CASES:
        torch.manual_seed(1234)          # 固定种子，让对比只反映策略差异
        with torch.no_grad():
            gen = model.generate(idx, args.tokens, use_cache=True, **kw)
        text = tok.decode(gen[0].tolist())
        body = text[len(args.prompt):][:110].replace("\n", " ⏎ ")
        param_str = ", ".join(f"{k}={v}" for k, v in kw.items())
        print(f"--- {name} ({param_str}) ---")
        print(body)
        print()
        lines.append(f"| {name} | `{param_str}` | {body} |")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] 已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
