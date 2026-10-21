# util/text_utils.py
"""
文本端工具。
"""
from collections import Counter
import math

# ---------- char-level BLEU(corpus 级) ----------
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

# ---------- char-level ROUGE-L ----------
def _lcs_len(a, b):
    n = len(b)
    dp = [0] * (n + 1)
    for i in range(1, len(a) + 1):
        prev, ai = 0, a[i - 1]
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if ai == b[j - 1] else max(dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def char_rouge_l(refs_chars, hyps_chars):
    """corpus 级 ROUGE-L (LCS 的 P/R 调和平均), 字符级。"""
    tp = sum(_lcs_len(r, h) for r, h in zip(refs_chars, hyps_chars))
    R = tp / max(sum(len(r) for r in refs_chars), 1)
    P = tp / max(sum(len(h) for h in hyps_chars), 1)
    return 2 * R * P / (R + P) if (R + P) > 0 else 0.0
