# Models/MRI2Rep/infer/render_report.py
"""
render_report.py — RLC 序列 -> 中文报告文本（自包含单文件，合并自 rlc_render.py）。

输入: eval/rlc_pred/test_pred.csv（infer_MRI2Rep.py 输出，列含 epoch,pid,tier,nll,gt,pred,...）
输出:
  eval/rlc_pred/{split}_reports.csv   列: epoch,pid,tier,nll,gt,pred,gt_report,pred_report
  eval/out_sample/{split}_all.txt     [GEN]/[REF] 成对全文（对齐 eval_CT2Rep 的 all.txt 形式）

方法边界（与 MRI2Rep §3.3 同构）：
  - 确定性查表填槽，无 LLM、无网络；变体抽取用固定 seed 的 random.Random。
  - 只生成"检查所见"(desc 风格)：模板中不得出现"考虑/拟/建议"等诊断意见(impression)
    口吻；文本参照物只用 liver_desc，不用 report_impression。
  - 句式锚定本科室 liver_desc 高频措辞（2042 例挖掘）：
      中文报告无"洗脱"表述，APHE 统一用"强化减退/呈相对低密度"；
      "肝脏轮廓光整，边缘规则，肝叶比例正常。"、"肝实质内未见异常密度灶及异常强化灶。"
      "动脉期边缘结节样强化，门脉期及延迟期向心性填充" 均为原句式。
  - GT 与预测同渲染器、同 seed 规则（seed=int(pid)），保证 LLM-Eval 同分布。

用法:
  python -m eval.render_report \
      --pred-csv model_save/261029_MRI2Rep_4p/eval/rlc_pred/test_pred.csv --split-name test
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# RLC 词表子集（与 RLCTokenizer.VOCAB 一致，仅渲染所需条目）
# ---------------------------------------------------------------------------
LIVER_TOKENS = {"Fibrosis/Cirrhosis", "NoFibrosis/Cirrhosis"}
TYPE_TOKENS = {"APHE_WO", "RIM_ATYP", "APHE_NoWO", "HEM", "CYST"}
POS_TOKENS = {"L_LAT", "L_MED", "R_ANT", "R_POST", "R_JUNCTION",
              "CAUDATE", "L_DIFFUSE", "R_DIFFUSE", "DIFFUSE"}
QTY_TOKENS = {"1", "2", "GE3", "Multiple"}

POS_CN = {   # 科室惯用: 肝右前叶/肝右后叶/肝左外叶/肝左内叶/肝尾状叶
    "L_LAT": "左外叶", "L_MED": "左内叶",
    "R_ANT": "右前叶", "R_POST": "右后叶",
    "R_JUNCTION": "右叶段间交界区", "CAUDATE": "尾状叶",
    "L_DIFFUSE": "左叶弥漫分布", "R_DIFFUSE": "右叶弥漫分布",
    "DIFFUSE": "内弥漫分布",
}
DIFFUSE_POS = {"L_DIFFUSE", "R_DIFFUSE", "DIFFUSE"}
QTY_CN = {"1": "见一枚", "2": "见两枚", "GE3": "见数枚", "Multiple": "见多发"}

LIVER_TEMPLATES = {
    "NoFibrosis/Cirrhosis": [
        "肝脏轮廓光整，边缘规则，肝叶比例正常。",
        "肝脏轮廓光整，边缘规则，肝叶比例未见异常。",
    ],
    "Fibrosis/Cirrhosis": [
        "肝脏轮廓欠光整，边缘呈波浪状，肝叶比例失常。",
        "肝脏体积缩小，轮廓欠光整，边缘呈波浪状，肝实质密度欠均匀。",
    ],
}

# 每个病灶类型 = (形态描述, 强化方式) 两段式；仅"检查所见"语言，
# 诊断信息由强化方式描述承载，与 liver_desc / report_impression 的分工一致。
LESION_TEMPLATES = {
    "CYST": [
        ("{loc}{qty}类圆形低密度影，边界清楚", "增强扫描未见强化"),
        ("{loc}{qty}囊状低密度影，边界清", "增强扫描各期均无强化"),
    ],
    "HEM": [
        ("{loc}{qty}类圆形稍低密度影",
         "增强扫描动脉期边缘结节样强化，门脉期及延迟期向心性填充"),
        ("{loc}{qty}类圆形低密度影",
         "动态增强扫描呈渐进性、向心性强化，强化程度与同层面大血管相近"),
    ],
    "APHE_WO": [
        ("{loc}{qty}异常强化结节", "动脉期明显强化，门脉期及延迟期强化减退"),
        ("{loc}{qty}结节状异常强化灶",
         "动脉期明显强化，门脉期及延迟期强化减退呈相对低密度"),
    ],
    "APHE_NoWO": [
        ("{loc}{qty}动脉期强化灶", "门脉期及延迟期未见明确强化减退"),
        ("{loc}{qty}动脉期异常强化结节",
         "门脉期及延迟期呈等/稍高密度，未见明确强化减退"),
    ],
    "RIM_ATYP": [
        ("{loc}{qty}稍低密度影，边界欠清", "增强扫描呈环形强化"),
        ("{loc}{qty}异常强化灶", "增强后呈边缘/环形强化"),
    ],
}

TAIL_NO_LESION = "肝实质内未见异常密度灶及异常强化灶。"
TAIL_WITH_LESION = "余肝实质内未见明确异常强化灶。"

# 防漂移守卫：模板中不得混入诊断意见口吻（impression 属于 report_impression 层）
import re as _re
_BAD = _re.compile(r"考虑|拟|建议|可能|性质待定|符合|转移瘤|恶性")
for _pool in (LIVER_TEMPLATES,):
    for _sents in _pool.values():
        for _s in _sents:
            assert not _BAD.search(_s), f"impression 口吻混入肝脏模板: {_s}"
for _tpls in LESION_TEMPLATES.values():
    for _d, _e in _tpls:
        assert not _BAD.search(_d + _e), f"impression 口吻混入病灶模板: {_d},{_e}"


# ---------------------------------------------------------------------------
# 渲染核心
# ---------------------------------------------------------------------------
def _parse(sequence: str) -> tuple[str, list[tuple[str, str, str]]]:
    """序列字符串 -> (liver_token, [(type,pos,qty), ...])，结构与 RLCTokenizer 校验一致。"""
    parts = [p.strip() for p in str(sequence).split(",") if p.strip()]
    if not parts or parts[0] not in LIVER_TOKENS:
        raise ValueError(f"非法 RLC 序列（肝脏背景缺失）：{sequence!r}")
    liver, rest = parts[0], parts[1:]
    if "NO_LESION" in rest:
        rest = [p for p in rest if p != "NO_LESION"]
    if len(rest) % 3 != 0:
        raise ValueError(f"RLC 病灶字段不是完整三元组：{sequence!r}")
    lesions = []
    for i in range(0, len(rest), 3):
        t, p, q = rest[i:i + 3]
        if t not in TYPE_TOKENS or p not in POS_TOKENS or q not in QTY_TOKENS:
            raise ValueError(f"RLC 含非法 token ({t},{p},{q})：{sequence!r}")
        lesions.append((t, p, q))
    if len(lesions) > 8:
        raise ValueError(f"RLC 病灶数超过官方上限 8：{sequence!r}")
    return liver, lesions


def render_sequence(sequence: str, seed: int | None = 42) -> str:
    """RLC 序列字符串 -> 中文肝脏报告。同一 seed 输出确定（变体可复现）。"""
    rng = random.Random(seed)
    liver, lesions = _parse(sequence)
    sentences = [rng.choice(LIVER_TEMPLATES[liver])]
    for t, p, q in lesions:
        desc, enh = rng.choice(LESION_TEMPLATES[t])
        loc = "肝" + POS_CN[p]
        qty = "" if p in DIFFUSE_POS else QTY_CN[q]   # 弥漫位置省略数量词
        sentences.append(f"{desc.format(loc=loc, qty=qty)}，{enh}。")
    sentences.append(TAIL_NO_LESION if not lesions else TAIL_WITH_LESION)
    return "".join(sentences)


# ---------------------------------------------------------------------------
# 批量渲染
# ---------------------------------------------------------------------------
def _seed_of(pid: str) -> int:
    return int(pid) if str(pid).isdigit() else 42


def _tolerant_render(sequence: str, seed: int) -> str:
    """与 train_MRI2Rep.parse_sequence 同口径的容错渲染：
    滑窗 salvage 完整 (type,pos,qty) 三元组，残缺尾部/乱码 token 丢弃。"""
    parts = [p.strip() for p in str(sequence).split(",") if p.strip()]
    if not parts or parts[0] not in LIVER_TEMPLATES:
        raise ValueError(f"非法 RLC 序列（肝脏背景缺失）：{sequence!r}")
    liver = parts[0]
    rest = [p for p in parts[1:] if p in TYPE_TOKENS | POS_TOKENS | QTY_TOKENS]
    lesions = []
    i = 0
    while i < len(rest) - 2:
        t, p, q = rest[i:i + 3]
        if t in TYPE_TOKENS and p in POS_TOKENS and q in QTY_TOKENS:
            lesions.append((t, p, q))
            i += 3
        else:
            i += 1
    rng = random.Random(seed)
    sentences = [rng.choice(LIVER_TEMPLATES[liver])]
    for t, p, q in lesions:
        desc, enh = rng.choice(LESION_TEMPLATES[t])
        loc = "肝" + POS_CN[p]
        qty = "" if p in DIFFUSE_POS else QTY_CN[q]
        sentences.append(f"{desc.format(loc=loc, qty=qty)}，{enh}。")
    sentences.append(TAIL_NO_LESION if not lesions else TAIL_WITH_LESION)
    return "".join(sentences)


def safe_render(seq: str, seed: int, stats: dict | None = None) -> str:
    """严格渲染优先；序列残缺（如三元组缺数量 token）时按容错口径 salvage 渲染，
    与评估端 parse_sequence 的计数口径一致，并在文首加不完整标记。"""
    seq = (seq or "").strip()
    if not seq:
        return "[空序列：模型未输出有效肝脏背景]"
    try:
        return render_sequence(seq, seed=seed)
    except ValueError:
        pass
    try:
        if stats is not None:
            stats["truncated"] = stats.get("truncated", 0) + 1
        return "【序列不完整，已按可解析部分渲染】" + _tolerant_render(seq, seed)
    except ValueError as exc:
        if stats is not None:
            stats["failed"] = stats.get("failed", 0) + 1
        return f"[渲染失败: {exc}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv",
                    default='/home/longkaixing/Liver/model_save/261029_MRI2Rep_4p/eval/rlc_pred/test_pred.csv')
    ap.add_argument("--split-name",
                    default="test")
    args = ap.parse_args()

    pred_path = Path(args.pred_csv)
    with open(pred_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"空文件: {pred_path}")

    stats = {}
    out_rows = []
    for r in rows:
        seed = _seed_of(r["pid"])
        out_rows.append({
            "epoch": r.get("epoch", -1), "pid": r["pid"], "tier": r["tier"],
            "nll": r.get("nll", ""),
            "gt": r["gt"], "pred": r["pred"],
            "gt_report": safe_render(r["gt"], seed),
            "pred_report": safe_render(r["pred"], seed, stats),
        })
    if stats:
        print(f"[Render] 容错触发: {stats}")

    reports_csv = pred_path.with_name(f"{args.split_name}_reports.csv")
    with open(reports_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    out_root = pred_path.parent.parent / "out_sample"
    out_root.mkdir(parents=True, exist_ok=True)
    all_txt = out_root / f"{args.split_name}_all.txt"
    with open(all_txt, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(f"===== pid={r['pid']} | tier={r['tier']} | nll={r['nll']} =====\n")
            f.write(f"[GEN]\n{r['pred_report']}\n\n[REF]\n{r['gt_report']}\n\n\n")

    print(f"[Render] {len(out_rows)} cases -> {reports_csv}")
    print(f"[Render] all.txt -> {all_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
