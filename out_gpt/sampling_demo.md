# 采样策略对比

prompt: `ROMEO:`，各生成 120 token

| 策略 | 参数 | 输出（前 110 字符） |
|---|---|---|
| 贪心 (T=0) | `temperature=0.0` |  ⏎ I will not stay thee for the world. ⏎  ⏎ BENVOLIO: ⏎ The seas the state of the world will be so. ⏎  ⏎ ROMEO: ⏎ And then t |
| 低温 T=0.5 | `temperature=0.5` |  ⏎ And thou shalt dispose thee the life of thy death ⏎ And this extremity of the strength of my father, ⏎ That he wo |
| 常溫 T=0.9 | `temperature=0.9` |  ⏎ And thou shalt disease thee speak again, ⏎ Which good doth to struck my will fence there; ⏎ The noble Tybalt to t |
| 高温 T=1.4 | `temperature=1.4` |  ⏎ Fault, myself: ⏎ Myself never shall perditatne ⏎ Dight outwittiest you. ⏎  ⏎ CENTIO: ⏎ Truly; ⏎ And I'll we go; make too, |
| top-k 10 | `temperature=0.9, top_k=10` |  ⏎ And thou shalt disease thee the life did make speak, ⏎ For these wanton will find them our fellow souls ⏎ Without |
| top-p 0.9 | `temperature=0.9, top_p=0.9` |  ⏎ And thou shalt disease thee speak again, ⏎ Which you are full of thee by this strength, ⏎ For every grace to the  |
