# -*- coding: utf-8 -*-
"""
compare_checkpoints.py  --  不同训练步数的模型，同一 prompt 并排对比

用法：
    python compare_checkpoints.py --ckpts out_gpt/best.pt out_gpt/long_6L384/best.pt
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

from tiny_gpt import GPT, GPTConfig, tokenizer_from_state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=["out_gpt/best.pt", "out_gpt/long_6L384/best.pt"])
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_lines = ["# 不同训练步数的对比", "", f"prompt = `{args.prompt}`，T={args.temperature}，top-k={args.top_k}", ""]

    for path in args.ckpts:
        if not Path(path).exists():
            print(f"[跳过] {path} 不存在")
            continue
        ckpt = torch.load(path, map_location=args.device)
        cfg = GPTConfig(**ckpt["config"])
        tok = tokenizer_from_state(ckpt["tokenizer"])
        model = GPT(cfg).to(args.device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        idx = torch.tensor([tok.encode(args.prompt)], device=args.device)
        torch.manual_seed(2024)
        with torch.no_grad():
            gen = model.generate(idx, args.tokens, temperature=args.temperature,
                                 top_k=args.top_k, use_cache=True)
        text = tok.decode(gen[0].tolist())

        print("=" * 70)
        print(f"{path}  |  step {ckpt.get('step', '?')}  |  val {ckpt.get('val', float('nan')):.4f}")
        print("=" * 70)
        print(text)
        print()
        out_lines.append(f"## {Path(path).parent.name}（step {ckpt.get('step','?')}, val {ckpt.get('val',0):.4f}）")
        out_lines.append("```")
        out_lines.append(text)
        out_lines.append("```")
        out_lines.append("")

    Path("out_gpt/checkpoint_compare.md").write_text("\n".join(out_lines), encoding="utf-8")
    print("[ok] 已写入 out_gpt/checkpoint_compare.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
