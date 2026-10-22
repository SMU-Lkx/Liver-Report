# eval/text_histogram.py
"""
生成文本频次直方图，分析模型输出是否坍塌（只输出模板、套话）
基于 eval_{model}.py 的结果做二次分析:
  GEN 来源: {eval_dir}/out_sample/{pid}/gen.txt   (逐例落盘的生成文本)
  REF 来源: data.xlsx 按 pid 回查 (与评估脚本同一数据源)
输出 (全部写回 --eval_dir):
  text_hist_metrics.json      标量: 精确级/签名级 distinct 数、top1 覆盖率等
  eval_csv/text_hist_gen.csv  GEN 全量直方图 (不限 top-N, 含每模板平均 nll)
  eval_csv/text_hist_ref.csv  REF 全量直方图
  text_hist.png               双面板条形图 (签名级 top-K, 左GEN右REF)
用法:
  python eval/text_histogram.py \
      --eval_dir model_save/261020_CT2Rep/eval \
      --data data/data.xlsx \
      --split_csv data/split_idx/260925_SSL/test_idx.csv
"""
from __future__ import annotations

import argparse, json, os, re, sys
from collections import Counter
from os.path import join
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
FP = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


# ---------------- 签名归一化 (数值槽 mask, 精确级/签名级共用) ----------------
RE_CTVAL = re.compile(r"CT值约?\s*\d+(?:\.\d+)?(?:\s*[-~—]\s*\d+(?:\.\d+)?)?\s*(?:Hu|HU|hu)")
RE_SIZE  = re.compile(r"(\d+(?:\.\d+)?)\s*(?:mm|cm|MM|CM)")
RE_NUM   = re.compile(r"\d+(?:\.\d+)?")

def normalize_template(text: str) -> str:
    """把数值槽替换为占位符, 得到'句式骨架'. 数值正确性由组B指标单独测, 这里只关心句式."""
    t = RE_CTVAL.sub("CT值<NUM>", text)
    t = RE_SIZE.sub("<SIZE>", t)
    t = RE_NUM.sub("<NUM>", t)
    return t.strip()


# ---------------- 数据加载 ----------------
def load_gen_texts(eval_dir: str, pids: list[str]) -> dict[str, str]:
    out_root = join(eval_dir, "out_sample")
    gens, missing = {}, []
    for pid in pids:
        p = join(out_root, str(pid), "gen.txt")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                gens[str(pid)] = f.read().strip()
        else:
            missing.append(str(pid))
    if missing:
        print(f"[WARN] {len(missing)} 例缺 gen.txt, 已跳过: {missing[:5]}...")
    return gens

def load_ref_texts(data_path: str, split_csv: str) -> dict[str, str]:
    pids = pd.read_csv(split_csv)["id"].astype(str).tolist()
    df = pd.read_excel(data_path)
    df["id"] = df["id"].astype(str)
    df = df.set_index("id")
    refs = {}
    for pid in pids:
        if pid in df.index and isinstance(df.loc[pid, "report_desc"], str):
            refs[pid] = df.loc[pid, "report_desc"].strip()
    return refs

def load_case_nll(eval_dir: str) -> dict[str, float]:
    """可选: join test_cases.csv 的逐例 nll, 用于看'每个模板的置信度'. 文件不存在则返回空."""
    p = join(eval_dir, "eval_csv", "test_cases.csv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    df["pid"] = df["pid"].astype(str)
    return dict(zip(df["pid"], df["nll"]))


# ---------------- 直方图构建 ----------------
def build_hist(texts: dict[str, str], nll_map: dict[str, float], level: str) -> pd.DataFrame:
    """level: 'exact' 整串原文 | 'signature' 归一化骨架"""
    key_fn = (lambda t: t) if level == "exact" else normalize_template
    counter = Counter(key_fn(t) for t in texts.values())
    n = sum(counter.values())
    rows = []
    for key, cnt in counter.most_common():
        pids_k = [pid for pid, t in texts.items() if key_fn(t) == key]
        nlls = [nll_map[pid] for pid in pids_k if pid in nll_map]
        rows.append({
            f"{level}_key": key,
            "count": cnt,
            "pct": round(cnt / max(n, 1), 4),
            "avg_nll": round(sum(nlls) / len(nlls), 4) if nlls else None,
        })
    return pd.DataFrame(rows)

def hist_metrics(df: pd.DataFrame, n_total: int, level: str) -> dict:
    m = {
        f"n_total": n_total,
        f"n_distinct_{level}": len(df),
        f"distinct_ratio_{level}": round(len(df) / max(n_total, 1), 4),
    }
    if len(df):
        m[f"top1_count_{level}"] = int(df.iloc[0]["count"])
        m[f"top1_coverage_{level}"] = round(float(df.iloc[0]["pct"]), 4)
        m[f"top1_{level}_preview"] = str(df.iloc[0][f"{level}_key"])[:60]
    return m


# ---------------- 画图 ----------------
def plot_hist(gen_df: pd.DataFrame, ref_df: pd.DataFrame, out_png: str, topk: int):
    def _panel(ax, df, title):
        k = min(topk, len(df))
        top = df.iloc[:k].iloc[::-1]  # 条形图自下而上频次递增
        labels = [str(x)[:32] + ("…" if len(str(x)) > 32 else "") for x in top[df.columns[0]]]
        ypos = range(len(top))
        ax.barh(ypos, top["count"], color="#4878cf")
        ax.set_yticks(list(ypos))
        ax.set_yticklabels(labels, fontproperties=FP, fontsize=7)  # ← 逐元素指定字体
        for i, (c, p) in enumerate(zip(top["count"], top["pct"])):
            ax.text(c + max(top["count"]) * 0.01, i, f"{c} ({p:.0%})",
                    va="center", fontsize=8)
        ax.set_title(f"{title} (top {k}, n={int(df['count'].sum())})", fontsize=11)
        ax.tick_params(axis="y", labelsize=7)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, topk * 0.32)))
    _panel(axes[0], gen_df, "GEN")
    _panel(axes[1], ref_df, "REF")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", help="eval_{model}.py 的输出目录, 如 model_save/261020_CT2Rep/eval",
                    default='/home/longkaixing/Liver/model_save/261020_CT2Rep/eval')
    ap.add_argument("--data",
                    default=str(PROJECT_ROOT / "data/data.xlsx"))
    ap.add_argument("--split_csv",
                    default=str(PROJECT_ROOT / "data/split_idx/260925_SSL/test_idx.csv"))
    ap.add_argument("--topk", type=int, help="PNG 条形图展示的签名数 (CSV 仍保留全量)",
                    default=20)
    cli = ap.parse_args()

    pids = pd.read_csv(cli.split_csv)["id"].astype(str).tolist()
    gens = load_gen_texts(cli.eval_dir, pids)
    refs = load_ref_texts(cli.data, cli.split_csv)
    nll_map = load_case_nll(cli.eval_dir)
    common = [p for p in pids if p in gens and p in refs]
    print(f"[Load] test={len(pids)} | GEN={len(gens)} | REF={len(refs)} | 共同={len(common)}")

    os.makedirs(join(cli.eval_dir, "eval_csv"), exist_ok=True)
    metrics = {}

    for tag, texts in (("gen", {p: gens[p] for p in common}),
                       ("ref", {p: refs[p] for p in common})):
        for level in ("exact", "signature"):
            df = build_hist(texts, nll_map if tag == "gen" else {}, level)
            df.to_csv(join(cli.eval_dir, "eval_csv", f"text_hist_{tag}_{level}.csv"),
                      index=False, encoding="utf-8-sig")
            metrics.update({f"{tag}_{k}": v for k, v in
                            hist_metrics(df, len(texts), level).items()})
            if level == "signature":
                plot_df = df
        # 图: 签名级双面板 (GEN vs REF 在同一张图上对比)
        if tag == "gen":
            gen_sig_df = plot_df
        else:
            ref_sig_df = plot_df

    png_path = join(cli.eval_dir, "text_hist.png")
    plot_hist(gen_sig_df, ref_sig_df, png_path, cli.topk)

    metrics["n_missing_gen"] = len(pids) - len(gens)
    metrics["n_missing_ref"] = len(pids) - len(refs)
    metrics["note"] = "exact=整串原文去重; signature=数值槽mask后的句式骨架; avg_nll 来自 test_cases.csv(若有)"
    with open(join(cli.eval_dir, "text_hist_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)

    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    print(f"-> {cli.eval_dir}/text_hist_metrics.json, eval_csv/text_hist_*.csv, text_hist.png")


if __name__ == "__main__":
    main()