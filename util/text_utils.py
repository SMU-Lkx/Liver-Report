# util/text_utils.py
"""
文本端工具。
"""
from collections import Counter
import math

# ---------- char-level BLEU(无第三方依赖, corpus 级) ----------
def ngram_counts(chars, n):
    return Counter(tuple(chars[i:i + n]) for i in range(len(chars) - n + 1))


def char_bleu(refs_chars, hyps_chars, max_n=2):
    """简化版 corpus BLEU(几何平均 + brevity penalty), 字符级。"""
    p_ns = []
    for n in range(1, max_n + 1):
        match = total = 0
        for ref, hyp in zip(refs_chars, hyps_chars):
            rc, hc = ngram_counts(ref, n), ngram_counts(hyp, n)
            match += sum(min(c, hc[g]) for g, c in rc.items())
            total += sum(hc.values())
        p_ns.append((match + 1e-9) / (total + 1e-9))
    bp = 1.0
    r = sum(len(x) for x in refs_chars)
    c = sum(len(x) for x in hyps_chars)
    if c < r:
        bp = math.exp(1 - r / max(c, 1))
    return bp * math.exp(sum(math.log(p) for p in p_ns) / max_n)
