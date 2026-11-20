# -*- coding: utf-8 -*-
"""
bpe_tokenizer.py  --  从零实现 byte-level BPE（不用 tiktoken）

为什么不用现成库：这份实现只有 150 行，能讲清每个步骤；而 tokenizer 的行为
（合并顺序、预切分正则）直接影响 loss 可比性，自己写才知道自己在比什么。

算法（Sennrich/GPT-2 那一路）：
    1. 预切分：用正则把文本切成"词"，避免跨词合并（"a b" 不该合成一个 token）
    2. 统计词频，每个词表示成 byte 序列（UTF-8 字节，天然覆盖任意字符）
    3. 反复找出现次数最多的相邻字节对，合并成新 token，直到达到目标词表
    4. 编码时按"合并优先级"贪心合并

和字符级相比：词表 1024 而不是 1013 个字符，但同样一段文本的 token 数少 3–4 倍，
等于同样的上下文窗口能装下更多内容。

用法（一般由 prepare_bpe.py 调用）：
    tok = BPETokenizer.train(text, vocab_size=1024)
    ids = tok.encode(text)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# 预切分：字母串 / 数字串 / 标点串 / 空白，并保留前导空格（" word" 作为一个词）
PRETOKEN = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]], vocab_size: int):
        self.merges = [tuple(m) for m in merges]
        self.vocab_size = vocab_size
        self._rank = {pair: i for i, pair in enumerate(self.merges)}
        # 每个 token id 对应的字节串
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self._merge_to = {pair: 256 + i for i, pair in enumerate(self.merges)}

    # ------------------------------------------------------------------ 训练
    @classmethod
    def train(cls, text: str, vocab_size: int = 1024, max_words: int = 30000,
              min_freq: int = 2, verbose: bool = True) -> "BPETokenizer":
        assert 256 < vocab_size <= 256 + 20000
        words = Counter(PRETOKEN.findall(text))
        # 只保留高频词：低频词在编码时会退回字节级，不影响正确性，只影响压缩率
        items = [(list(w.encode("utf-8")), c) for w, c in words.most_common(max_words) if c >= min_freq]
        if verbose:
            covered = sum(c for _, c in items)
            total = sum(words.values())
            print(f"  唯一词 {len(words):,}，保留 {len(items):,} 个（覆盖 {covered/total*100:.1f}% 的词出现）")

        merges: list[tuple[int, int]] = []
        n_merges = vocab_size - 256
        for step in range(n_merges):
            pairs: Counter = Counter()
            for w, c in items:
                for i in range(len(w) - 1):
                    pairs[(w[i], w[i + 1])] += c
            if not pairs:
                break
            (a, b), cnt = pairs.most_common(1)[0]
            if cnt < 2:
                break
            new_id = 256 + len(merges)
            merges.append((a, b))
            for idx, (w, c) in enumerate(items):
                if len(w) < 2:
                    continue
                nw, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                        nw.append(new_id)
                        i += 2
                    else:
                        nw.append(w[i])
                        i += 1
                items[idx] = (nw, c)
            if verbose and (step + 1) % 128 == 0:
                print(f"  merge {step+1}/{n_merges}  最新对 {a},{b} 频次 {cnt}")
        if verbose:
            print(f"  完成：{len(merges)} 次合并，词表 {256 + len(merges)}")
        return cls(merges, 256 + len(merges))

    # ------------------------------------------------------------------ 编码
    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        rank = self._rank
        to = self._merge_to
        for word in PRETOKEN.findall(text):
            w = list(word.encode("utf-8"))
            while len(w) > 1:
                best_rank, best_i = None, -1
                for i in range(len(w) - 1):
                    r = rank.get((w[i], w[i + 1]))
                    if r is not None and (best_rank is None or r < best_rank):
                        best_rank, best_i = r, i
                if best_i < 0:
                    break
                pair = (w[best_i], w[best_i + 1])
                w[best_i:best_i + 2] = [to[pair]]
            ids.extend(w)
        return ids

    def decode(self, ids: list[int]) -> str:
        b = b"".join(self.vocab.get(i, b"?") for i in ids)
        return b.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ 存取
    def state_dict(self) -> dict:
        """存进 checkpoint 用，格式和 save() 一致，另加一个 type 标记。"""
        return {"type": "bpe", "merges": [list(m) for m in self.merges],
                "vocab_size": self.vocab_size}

    @classmethod
    def from_state(cls, d: dict) -> "BPETokenizer":
        return cls([tuple(m) for m in d["merges"]], d["vocab_size"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"merges": self.merges, "vocab_size": self.vocab_size}), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(m) for m in d["merges"]], d["vocab_size"])


if __name__ == "__main__":
    # 自测：小样本上训练，检查往返编解码
    sample = "the quick brown fox jumps over the lazy dog. " * 200
    tok = BPETokenizer.train(sample, vocab_size=300, min_freq=1, verbose=False)
    ids = tok.encode(sample[:200])
    back = tok.decode(ids)
    print("往返一致：", back == sample[:200])
    print("压缩率：", f"{len(sample[:200].encode())/len(ids):.2f} 字节/token")
    print("样例 id：", ids[:20])
