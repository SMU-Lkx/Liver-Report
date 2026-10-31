# eval/eval_MRI2Rep.py
"""
eval_MRI2Rep.py — MRI2Rep-RLC 渲染报告评估。

主指标:  官方结构化指标（LiverAcc / Sen / Spe / pair-F1 micro / triplet-F1 micro，
         口径与训练 validate 完全一致——复用 train_MRI2Rep.parse_sequence/f1_*）。
辅指标:  corpus 级 char-BLEU2/4 + char-ROUGE-L（渲染稿 vs GT渲染稿；
         模板主导、仅作 NLG 层面对齐，对应官方 evaluate_nlg.py 的位置）。
输出结构与 eval_CT2Rep.py 完全对齐:
  eval/test_metrics.json / test_metrics.csv（经 util.val_utils.eval_outputs）
  eval/eval_csv/test_cases.csv（逐例: nll/序列匹配/报告长度/pair-F1）
无模型、无 NPU，纯 CPU 秒级。

用法:
  python -m eval.eval_MRI2Rep \
      --reports-csv model_save/261029_MRI2Rep_4p/eval/rlc_pred/test_reports.csv
"""
from __future__ import annotations

import argparse
import csv, os, json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.Dataset.Multiphase_dataset import RLCTokenizer
from train.train_MRI2Rep import parse_sequence, f1_from_sets, f1_from_counters
from util.text_utils import sacre_bleu_zh, char_rouge_l
from util.val_utils import eval_outputs

_TOK = RLCTokenizer()


def to_ids(seq_str: str) -> list[int]:
    """序列字符串 -> token id 列表；词表外 token（如解码出的 Unknown）直接丢弃。"""
    return [_TOK.vocab[p.strip()] for p in str(seq_str).split(",")
            if p.strip() in _TOK.vocab]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-csv",
                    default='/home/longkaixing/Liver/model_save/261029_MRI2Rep_4p/eval/rlc_pred/test_reports.csv')
    ap.add_argument("--tag",
                    default="test")
    args = ap.parse_args()

    reports_path = Path(args.reports_csv)
    with open(reports_path, encoding="utf-8-sig") as f:
        cases = list(csv.DictReader(f))
    if not cases:
        raise SystemExit(f"空文件: {reports_path}")

    eval_dir = str(reports_path.parent.parent)          # model_save/{save_name}/eval
    os.makedirs(eval_dir, exist_ok=True)

    # ---------- 结构化指标（主） ----------
    n = len(cases)
    liver_correct = no_lesion_correct = 0
    case_tp = case_fn = case_tn = case_fp = 0
    pair_tp = pair_fp = pair_fn = 0
    trip_tp = trip_fp = trip_fn = 0
    rows = []
    for r in cases:
        gt = parse_sequence(to_ids(r["gt"]))          # 序列字符串 -> id -> 容错解析
        pred = parse_sequence(to_ids(r["pred"]))
        liver_correct += int(gt["liver"] is not None and pred["liver"] == gt["liver"])
        gt_no, pred_no = len(gt["lesions"]) == 0, len(pred["lesions"]) == 0
        no_lesion_correct += int(gt_no == pred_no)
        if gt_no:
            case_tn += int(pred_no); case_fp += int(not pred_no)
        else:
            case_fn += int(pred_no); case_tp += int(not pred_no)
        p_pairs = {(t, p) for t, p, _ in pred["lesions"]}
        g_pairs = {(t, p) for t, p, _ in gt["lesions"]}
        case_pair = f1_from_sets(p_pairs, g_pairs)
        pair_tp += len(p_pairs & g_pairs)
        pair_fp += len(p_pairs - g_pairs)
        pair_fn += len(g_pairs - p_pairs)
        _, t_tp, t_fp, t_fn = f1_from_counters(
            Counter(pred["lesions"]), Counter(gt["lesions"]))
        trip_tp += t_tp; trip_fp += t_fp; trip_fn += t_fn
        rows.append({
            "pid": r["pid"], "tier": r["tier"],
            "nll": r.get("nll", ""),
            "gen_len": len(r["pred_report"]), "ref_len": len(r["gt_report"]),
            "gt_lesions": len(gt["lesions"]), "pred_lesions": len(pred["lesions"]),
            "liver_match": int(pred["liver"] == gt["liver"]),
            "pair_f1": round(case_pair, 4),
        })

    sen = case_tp / (case_tp + case_fn) if case_tp + case_fn else 0.0
    spe = case_tn / (case_tn + case_fp) if case_tn + case_fp else 0.0
    p_den = 2 * pair_tp + pair_fp + pair_fn
    t_den = 2 * trip_tp + trip_fp + trip_fn

    # ---------- 文本指标（辅，corpus 级） ----------
    refs_l = [r["gt_report"] for r in cases]
    hyps_l = [r["pred_report"] for r in cases]
    ref_lens = [max(len(r["gt_report"]), 1) for r in cases]
    nlls = [float(r["nll"]) for r in cases if str(r.get("nll", "")).strip()]

    metrics = {
        "reports_csv": str(reports_path),
        "n_test": n,
        "test_nll": round(float(np.average(nlls, weights=ref_lens)), 4) if nlls else None,
        # ---- 结构化（主指标） ----
        "liver_acc": round(liver_correct / n, 4),
        "no_lesion_acc": round(no_lesion_correct / n, 4),
        "case_sen": round(sen, 4),
        "case_spe": round(spe, 4),
        "pair_f1_micro": round(2 * pair_tp / p_den, 4) if p_den else 0.0,
        "triplet_f1_micro": round(2 * trip_tp / t_den, 4) if t_den else 0.0,
        # ---- 文本（辅指标） ----
        "char_bleu2": round(sacre_bleu_zh(refs_l, hyps_l, max_n=2), 4),
        "char_bleu4": round(sacre_bleu_zh(refs_l, hyps_l, max_n=4), 4),
        "char_rouge_l": round(char_rouge_l([list(s) for s in refs_l],
                                           [list(s) for s in hyps_l]), 4),
        "avg_gen_len": round(float(np.mean([len(s) for s in hyps_l])), 1),
        "avg_ref_len": round(float(np.mean([len(s) for s in refs_l])), 1),
        "note": "主指标为官方结构化指标；BLEU/ROUGE 为渲染稿vs GT渲染稿的模板主导辅指标",
    }

    eval_outputs(metrics, rows, eval_dir, tag=args.tag)
    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    print(f"-> {eval_dir}/test_metrics.json, eval_csv/, out_sample/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())