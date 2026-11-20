# -*- coding: utf-8 -*-
"""
prepare_bpe.py  --  在 wikitext 上训练 BPE，并把全文编码成 ids.npy

产物：
    data/wikitext/bpe1024.json     tokenizer（merges 表）
    data/wikitext/bpe_ids.npy      全文 token id 序列（int32）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bpe_tokenizer import BPETokenizer  # noqa: E402

ROOT = Path(r"C:\dev\llm-warmup\data\wikitext")
TEXT = ROOT / "input.txt"
TOK_PATH = ROOT / "bpe1024.json"
IDS_PATH = ROOT / "bpe_ids.npy"
VOCAB = 1024


def main() -> int:
    text = TEXT.read_text(encoding="utf-8")
    print(f"语料 {len(text):,} 字符")

    t0 = time.perf_counter()
    if TOK_PATH.exists():
        tok = BPETokenizer.load(TOK_PATH)
        print(f"复用已有 tokenizer：词表 {tok.vocab_size}")
    else:
        print(f"训练 BPE（目标词表 {VOCAB}）...")
        tok = BPETokenizer.train(text, vocab_size=VOCAB)
        tok.save(TOK_PATH)
        print(f"tokenizer 已保存 {TOK_PATH}，耗时 {time.perf_counter()-t0:.0f}s")

    if IDS_PATH.exists():
        ids = np.load(IDS_PATH)
        print(f"复用已有 ids：{len(ids):,} 个 token")
    else:
        t1 = time.perf_counter()
        print("编码全文 ...")
        ids = np.asarray(tok.encode(text), dtype=np.int32)
        np.save(IDS_PATH, ids)
        print(f"编码完成 {len(ids):,} token，耗时 {time.perf_counter()-t1:.0f}s")

    n_bytes = len(text.encode("utf-8"))
    print()
    print("===== 对比 =====")
    print(f"  字符级：{len(text):,} token（1 字符 = 1 token）")
    print(f"  BPE   ：{len(ids):,} token（{n_bytes/len(ids):.2f} 字节/token）")
    print(f"  压缩比：{len(text)/len(ids):.2f}× —— 同样 256 的上下文能装下 {len(text)/len(ids):.1f} 倍的文本")

    # 抽样看几个 token 的原文
    print()
    print("  前 40 个 token 的原文：" + repr(tok.decode(ids[:40].tolist())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
