# -*- coding: utf-8 -*-
"""把 wikitext-2 的 parquet 转成纯文本 input.txt。"""
import io
from pathlib import Path

import pyarrow.parquet as pq

src = Path(r"C:\dev\llm-warmup\data\wikitext\wikitext2.parquet")
dst = Path(r"C:\dev\llm-warmup\data\wikitext\input.txt")

t = pq.read_table(src)
print("列：", t.column_names, "行数：", t.num_rows)
col = "text" if "text" in t.column_names else t.column_names[0]
texts = t.column(col).to_pylist()
text = "\n".join(x for x in texts if x)
dst.write_text(text, encoding="utf-8")

chars = sorted(set(text))
print(f"字符数 {len(text):,}")
print(f"词表大小 {len(chars)}")
print(f"前 200 字符：\n{text[:200]!r}")
