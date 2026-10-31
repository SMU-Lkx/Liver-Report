# Models/MRI2Rep/model/RLC.py
"""
RLC.py — 中文肝脏 CT 报告 → MRI2Rep RLC 标签（官方流程的中文适配版）

方法边界：
  - 严格保留官方 Step 1 → Step 2 → Step 3、三轮投票、2/3 恢复、低一致性病
    例五轮重跑，以及 build_accurate_dataset.py 的分级接受规则。
  - 仅将三份提示词翻译为中文，并适配 data.xlsx / split CSV 的输入输出。
  - 不加入 evidence、JSON schema、逐字证据 QA、JSON mode 或 self-repair。

输入：
  1. data.xlsx：至少包含 id、report_desc、report_impression 三列。
  2. split_dir/{train,val,test}_idx.csv：至少包含 id 列。
  3. DeepSeek API key：环境变量 DEEPSEEK_API_KEY，或 ~/.deepseek_api_key。

主要输出（out_dir）：
  logs/batch_run_{split}/{pid}/final_result.json
      官方字段：id、organ_status、tumors、consistent、consistency_details、_meta。
  logs/batch_run_{split}/{pid}/extraction_log.json
      三步提示词、原始响应和提取结果，仅用于复现审计。
  logs/batch_run_{split}/Summary.csv
      官方后处理格式：id、consistency、cost、sequence。
  logs/merged_summary_recovered.csv、logs/rerun_ids.txt
      recover 命令输出：接受原 3/3 一致及可恢复的 >=2/3 病例，其余列入重跑。
  logs/accurate_summary.csv、logs/rerun_needed.txt
      build 命令输出：按官方 Tier 1–4 规则接受；数量为训练词表的 1/2/GE3/Multiple。

使用顺序：
  # 1. train 先试运行 20 例；正式提取固定为 Step 3 三轮
  python RLC.py extract --data data/data.xlsx --split-dir data/split_idx/260925_SSL \
    --out-dir data/prompt/rlc_reports --splits train --limit 20

  # 2. 人工审核后全量三 split；已完成的 20 例默认断点跳过
  python RLC.py extract --data data/data.xlsx --split-dir data/split_idx/260925_SSL \
    --out-dir data/prompt/rlc_reports --splits train val test

  # 3. 无需 API：恢复原 3/3 与 >=2/3 病例，并生成低一致性重跑清单
  python RLC.py recover --out-dir data/prompt/rlc_reports

  # 4. 仅对清单病例重跑 Step 3，默认五轮；保留 liver 状态并备份旧结果
  python RLC.py rerun --data data/data.xlsx --split-dir data/split_idx/260925_SSL \
    --out-dir data/prompt/rlc_reports --rounds 5

  # 5. 重跑后再次 recover，再按官方分级接受策略生成训练用 accurate_summary.csv
  python RLC.py recover --out-dir data/prompt/rlc_reports
  python RLC.py build --out-dir data/prompt/rlc_reports

依赖：pandas、openpyxl、litellm。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from litellm import completion
import pandas as pd

CONSISTENCY_ROUNDS = 3
SYSTEM_PROMPT = (
    "You are a helpful medical AI assistant specialized in MRI report analysis. "
    "Follow instructions strictly."
)

LIVER_TYPES = {"Fibrosis/Cirrhosis", "NoFibrosis/Cirrhosis"}
VALID_TYPES = {"APHE_WO", "RIM_ATYP", "APHE_NoWO", "HEM", "CYST"}
VALID_POS = {
    "L_LAT", "L_MED", "R_ANT", "R_POST", "R_JUNCTION",
    "CAUDATE", "L_DIFFUSE", "R_DIFFUSE", "DIFFUSE",
}
VALID_QTY = {"1", "2", "GE3", "Multiple"}
VOCAB = [
    "PAD", "BOS", "EOS", "NO_LESION",
    "Fibrosis/Cirrhosis", "NoFibrosis/Cirrhosis",
    "APHE_WO", "RIM_ATYP", "APHE_NoWO", "HEM", "CYST",
    "L_LAT", "L_MED", "R_ANT", "R_POST", "R_JUNCTION",
    "CAUDATE", "L_DIFFUSE", "R_DIFFUSE", "DIFFUSE",
    "1", "2", "GE3", "Multiple",
]
assert len(VOCAB) == 24

TYPE_PRIORITY = {
    "APHE_WO": 1, "APHE_NoWO": 2, "RIM_ATYP": 3,
    "HEM": 4, "CYST": 5, "FAT": 6, "OTHER": 99, "Unknown": 100,
}
POS_PRIORITY = {
    "CAUDATE": 1, "L_LAT": 2, "L_MED": 3, "R_ANT": 4, "R_POST": 5,
    "R_JUNCTION": 6, "L_DIFFUSE": 7, "R_DIFFUSE": 8,
    "DIFFUSE": 9, "Unknown": 99,
}

# 以下为官方三份 prompt 的中文直译；输出标记和闭集 token 保持英文原样。
PROMPT_STEP_1_ORGAN = r"""
分析以下影像报告文本，以确定肝实质的状况。

任务：
判断肝脏是否表现为“Fibrosis/Cirrhosis”或“NoFibrosis/Cirrhosis”。

标准：
- 查找“肝硬化”“肝硬化性”“表面结节状”“纤维化”“粗糙回声质地”等关键词，或明确提到“慢性肝病”。
- 如果存在这些表现，则分类为“Fibrosis/Cirrhosis”。
- 如果肝脏被描述为“无异常”“正常”或“光滑”，则分类为“NoFibrosis/Cirrhosis”。

输入报告：
```
{report_text}
```

输出格式：
请先输出你的推理，然后另起一行，以“FINAL_RESULT:”开头输出最终分类。

示例：
REASONING: 报告提到表面结节状，这是进展期纤维化的征象。
FINAL_RESULT: Fibrosis/Cirrhosis
""".strip()

PROMPT_STEP_2_ORGAN_CHECK = r"""
你是一名质量保证审核员。请审查原始报告和此前提取的器官状态。

原始报告：
```
{report_text}
```

此前提取的状态：
```
{previous_status}
```

任务：
根据报告验证此前的状态是否正确。
1. 如果报告写明“肝硬化”或其征象，状态必须为“Fibrosis/Cirrhosis”。
2. 如果报告写明肝脏正常，状态必须为“NoFibrosis/Cirrhosis”。

输出格式：
请给出你的复核推理，然后另起一行，以“CONFIRMED_RESULT:”开头输出最终纠正后的结果。

示例：
VERIFICATION_REASONING: 报告明确写明“肝硬化形态学征象”，因此此前的提取是正确的。
CONFIRMED_RESULT: Fibrosis/Cirrhosis
""".strip()

PROMPT_STEP_3_TUMOR_EXTRACTION = r"""
分析影像报告，提取肝脏内所有局灶性发现、观察和病灶的结构化列表。

目标：
捕获报告中提到的肝实质内每一个不同的视觉观察。必须根据影像特征，将每一个发现分类为以下 5 种允许类型之一。

映射规则（严格遵守）：

1. 位置映射（基于 Couinaud 分段）：
- 第 2、3 段 -> L_LAT
- 第 4 段 -> L_MED
- 第 5、8 段、 Dome -> R_ANT
- 第 6、7 段 -> R_POST
- 第 1 段 -> CAUDATE
- 交界处（5/6、6/7、5/8）-> R_JUNCTION
- 左叶（泛指）-> L_DIFFUSE
- 右叶（泛指）-> R_DIFFUSE
- 未指明 -> DIFFUSE

2. 类型映射（严格限于以下 5 类）：
- CYST：单纯囊肿、多囊肝、液性聚集、血清肿。
- HEM：血管瘤（周边结节样强化、填充）。
- APHE_WO：动脉期高强化（APHE）伴洗脱。（典型 HCC 模式，LR-5）。
- APHE_NoWO：动脉期高强化不伴洗脱。（包括 LR-3、LR-4、快速填充型血管瘤、THAD、分流、灌注异常和性质未定的动脉期强化灶）。
- RIM_ATYP：环形强化、周边洗脱、靶样表现、胆管细胞癌特征、转移瘤、治疗后病灶/瘢痕，或任何不符合以上类别的其他非典型发现。

3. 数量映射：
- 明确数字（1、2、3……）-> 写出该数字。
- “多个/多发”“无数”“许多”“病灶”（未给出数字的复数）等表述 -> Multiple

任务：
1. 将报告内容拆分为单独的句子。
2. 分析每个句子，检查其是否描述了肝脏局灶性观察。
3. 如果识别到发现，根据上述规则确定其类型、位置和数量。
4. 忽略明确写明属于其他器官的发现（例如肾囊肿）。

输入报告：
```
{report_text}
```

输出格式：
严格遵循以下格式进行推理和输出。分析部分不要输出 Markdown 代码块，只输出纯文本。

Step 1: Sentence Analysis
（遍历报告中的每一个句子）
Sentence 1: [原文] -> [推理：未发现病灶 / 发现……]
Sentence 2: [原文] -> [推理：……]
……

Step 2: Structured Output
（对于 Step 1 中识别出的每个有效发现，输出单独一行）
TUMOR_RECORD: Type | Location | Quantity
TUMOR_RECORD: Type | Location | Quantity

输出示例：
Step 1: Sentence Analysis
Sentence 1: 肝脏呈现肝硬化形态学征象。 -> 背景信息，无局灶性病灶。
Sentence 2: 第 8 段见一 1.2 cm 观察灶，动脉期强化但无洗脱。 -> 发现病灶。特征：APHE + 无洗脱。映射为 APHE_NoWO。位置：第 8 段（R_ANT）。
Sentence 3: 第 4 段见楔形灌注异常。 -> 发现观察灶（THAD）。按规则映射为 APHE_NoWO。位置：第 4 段（L_MED）。

Step 2: Structured Output
TUMOR_RECORD: APHE_NoWO | R_ANT | 1
TUMOR_RECORD: APHE_NoWO | L_MED | 1
""".strip()


# ------------------------------- 通用工具 -------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def read_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    key_file = Path.home() / ".deepseek_api_key"
    if not key and key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("未找到 DeepSeek API key：请设置 DEEPSEEK_API_KEY 或写入 ~/.deepseek_api_key")
    return key


def response_content(response: Any) -> str:
    if response is None:
        return ""
    content = response.choices[0].message.content
    return content.strip() if content else ""


def response_dict(response: Any) -> Any:
    if response is None:
        return None
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    if isinstance(response, dict):
        return response
    return repr(response)


# ------------------------------- DeepSeek -------------------------------
class DeepSeekClient:
    """与官方 utils/llm.py 相同：litellm + DeepSeek，无 JSON mode。"""

    def __init__(self, args: argparse.Namespace):
        self.model = args.model
        self.base_url = args.base_url
        self.api_key = read_api_key()
        self.max_retries = args.max_retries
        if args.proxy.lower() in {"", "none", "off"}:
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("HTTPS_PROXY", None)
            os.environ.pop("http_proxy", None)
            os.environ.pop("https_proxy", None)
        else:
            os.environ["HTTP_PROXY"] = args.proxy
            os.environ["HTTPS_PROXY"] = args.proxy

    def chat(self, prompt: str, temperature: float) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    api_key=self.api_key,
                    base_url=self.base_url,
                    temperature=temperature,
                    drop_params=True,
                )
            except Exception as exc:
                # 仅重试网络/服务调用，不重写提示词，也不进行输出自我修复。
                last_error = exc
                if attempt >= self.max_retries:
                    break
                sleep_s = min(20.0, 2.0 ** (attempt - 1)) + random.random()
                logging.warning("DeepSeek 调用失败（%d/%d）：%s；%.1fs 后重试",
                                attempt, self.max_retries, exc, sleep_s)
                time.sleep(sleep_s)
        raise RuntimeError(f"DeepSeek 调用在 {self.max_retries} 次尝试后仍失败：{last_error}")


class CaseLogger:
    def __init__(self, case_dir: Path, pid: str):
        self.case_dir = case_dir
        self.pid = pid
        self.entries: list[dict] = []

    def log_step(self, name: str, prompt: str, response: Any, extracted_info: str) -> None:
        self.entries.append({
            "step": name,
            "prompt": prompt,
            "content": response_content(response),
            "response": response_dict(response),
            "extracted_info": extracted_info,
        })

    def save(self, filename: str = "extraction_log.json") -> None:
        write_json(self.case_dir / filename, {
            "id": self.pid,
            "created_at": utc_now(),
            "steps": self.entries,
        })


# ------------------------- 官方三步提取与多数投票 -------------------------
def run_step_1_organ(report_text: str, client: DeepSeekClient, logger: CaseLogger) -> str:
    prompt = PROMPT_STEP_1_ORGAN.replace("{report_text}", report_text)
    response = client.chat(prompt, temperature=0.0)
    content = response_content(response)
    match = re.search(r"FINAL_RESULT:\s*(.+)", content)
    status = match.group(1).strip() if match else "Unknown"
    logger.log_step("step_1_organ", prompt, response, extracted_info=status)
    return status


def run_step_2_verification(
    report_text: str, previous_status: str, client: DeepSeekClient, logger: CaseLogger,
) -> str:
    prompt = (PROMPT_STEP_2_ORGAN_CHECK
              .replace("{report_text}", report_text)
              .replace("{previous_status}", previous_status))
    response = client.chat(prompt, temperature=0.0)
    content = response_content(response)
    match = re.search(r"CONFIRMED_RESULT:\s*(.+)", content)
    final_status = match.group(1).strip() if match else previous_status
    logger.log_step("step_2_verification", prompt, response, extracted_info=final_status)
    return final_status


def run_step_3_tumor_extraction(
    report_text: str, client: DeepSeekClient, logger: CaseLogger,
    rounds: int = CONSISTENCY_ROUNDS,
) -> tuple[list[str], bool, dict]:
    if rounds < 1:
        raise ValueError("Step 3 rounds 必须 >= 1")
    prompt = PROMPT_STEP_3_TUMOR_EXTRACTION.replace("{report_text}", report_text)
    extraction_results: list[list[str]] = []
    normalized_results_for_voting: list[str] = []

    for i in range(1, rounds + 1):
        temperature = 0.2 if rounds > 1 else 0.0
        response = client.chat(prompt, temperature=temperature)
        content = response_content(response)
        records = re.findall(r"TUMOR_RECORD:\s*(.+)", content)
        record_str = "\n".join(records) if records else "NONE"
        logger.log_step(f"step_3_extraction_run_{i}", prompt, response,
                        extracted_info=record_str)
        extraction_results.append(records)
        sorted_records = sorted(r.strip() for r in records)
        normalized_results_for_voting.append(json.dumps(sorted_records))

    vote_counts = Counter(normalized_results_for_voting)
    winner_key, winner_count = vote_counts.most_common(1)[0]
    final_records: list[str] = []
    for idx, key in enumerate(normalized_results_for_voting):
        if key == winner_key:
            final_records = extraction_results[idx]
            break
    is_consistent = winner_count == rounds
    consistency_stats = {
        "total_rounds": rounds,
        "winner_count": winner_count,
        "details": dict(vote_counts),
    }
    return final_records, is_consistent, consistency_stats


def parse_tumor_records(raw_records: list[str]) -> list[dict]:
    structured = []
    for record in raw_records:
        if "NONE" in record:
            continue
        parts = [p.strip() for p in record.split("|")]
        if len(parts) == 3:
            structured.append({
                "type": parts[0],
                "position": parts[1],
                "quantity": parts[2],
            })
    return structured


# ------------------------- 官方 post.py 确定性后处理 -------------------------
def calculate_merged_quantity(quantities: list[str]) -> str:
    total = 0
    for quantity in quantities:
        q_str = str(quantity).strip()
        if "multiple" in q_str.lower():
            return "Multiple"
        try:
            total += int(q_str)
        except ValueError:
            return "Multiple"
    return str(total)


def merge_tumors(tumor_list: list[dict]) -> list[dict]:
    if not tumor_list:
        return []
    grouped: dict[tuple[str, str], list[str]] = {}
    for tumor in tumor_list:
        tumor_type = tumor.get("type", "Unknown").strip()
        position = tumor.get("position", "Unknown").strip()
        quantity = tumor.get("quantity", "1").strip()
        grouped.setdefault((tumor_type, position), []).append(quantity)

    merged = []
    for (tumor_type, position), quantities in grouped.items():
        merged.append({
            "type": tumor_type,
            "position": position,
            "quantity": calculate_merged_quantity(quantities),
        })
    merged.sort(key=lambda x: (
        TYPE_PRIORITY.get(x["type"], 100),
        POS_PRIORITY.get(x["position"], 100),
    ))
    return merged


def format_sequence(organ_status: str, tumors: list[dict]) -> str:
    parts = [organ_status]
    for tumor in tumors:
        parts.extend([tumor["type"], tumor["position"], tumor["quantity"]])
    return ", ".join(parts)


# ------------------------------- 数据读取 -------------------------------
def load_source_data(data_path: str) -> pd.DataFrame:
    df = pd.read_excel(data_path, dtype={"id": str})
    required = {"id", "report_desc", "report_impression"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"data.xlsx 缺少列：{sorted(missing)}")
    df = df.copy()
    df["id"] = df["id"].astype(str).str.strip()
    if df["id"].duplicated().any():
        logging.warning("data.xlsx 存在重复 id；每个 id 仅使用第一行")
        df = df.drop_duplicates("id", keep="first")
    return df.set_index("id", drop=False)


def clean_text(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def row_to_record(pid: str, split: str, row: pd.Series) -> dict:
    desc = clean_text(row["report_desc"])
    impression = clean_text(row["report_impression"])
    if not desc and not impression:
        raise ValueError(f"pid={pid} 的 report_desc 和 report_impression 均为空")
    report_text = f"检查所见：\n{desc}\n\n诊断意见：\n{impression}"
    return {
        "pid": str(pid), "split": split,
        "report_desc": desc, "report_impression": impression,
        "report_text": report_text,
    }


def load_records(data_path: str, split_dir: str, split: str) -> list[dict]:
    split_csv = Path(split_dir) / f"{split}_idx.csv"
    if not split_csv.exists():
        raise FileNotFoundError(split_csv)
    split_ids = (pd.read_csv(split_csv, dtype={"id": str})["id"]
                 .astype(str).str.strip().tolist())
    source = load_source_data(data_path)
    records = []
    for pid in split_ids:
        if pid not in source.index:
            logging.warning("split=%s 的 pid=%s 不在 data.xlsx 中，跳过", split, pid)
            continue
        try:
            records.append(row_to_record(pid, split, source.loc[pid]))
        except ValueError as exc:
            logging.warning("%s，跳过", exc)
    return records


def find_record_by_id(data_path: str, split_dir: str, pid: str) -> dict:
    source = load_source_data(data_path)
    if pid not in source.index:
        raise FileNotFoundError(f"data.xlsx 中找不到 pid={pid}")
    split = "unknown"
    for candidate in ("train", "val", "test"):
        split_csv = Path(split_dir) / f"{candidate}_idx.csv"
        if split_csv.exists():
            ids = (pd.read_csv(split_csv, dtype={"id": str})["id"]
                   .astype(str).str.strip())
            if pid in set(ids):
                split = candidate
                break
    return row_to_record(pid, split, source.loc[pid])


# ------------------------------- 单例提取 -------------------------------
def process_record(record: dict, client: DeepSeekClient, logs_root: Path,
                   overwrite: bool = False) -> str:
    pid = record["pid"]
    case_dir = logs_root / f"batch_run_{record['split']}" / pid
    result_path = case_dir / "final_result.json"
    error_path = case_dir / "error.json"
    if result_path.exists() and not overwrite:
        return "skipped"

    case_dir.mkdir(parents=True, exist_ok=True)
    logger = CaseLogger(case_dir, pid)
    try:
        status = run_step_1_organ(record["report_text"], client, logger)
        final_status = run_step_2_verification(
            record["report_text"], status, client, logger)
        raw_tumors, consistent, stats = run_step_3_tumor_extraction(
            record["report_text"], client, logger, rounds=CONSISTENCY_ROUNDS)
        structured = parse_tumor_records(raw_tumors)
        final_data = {
            "id": pid,
            "organ_status": final_status,
            "tumors": structured,
            "consistent": consistent,
            "consistency_details": stats,
            "_meta": {
                "model": client.model,
                "created_at": utc_now(),
                "total_cost_usd": 0.0,
            },
        }
        write_json(result_path, final_data)
        logger.save()
        error_path.unlink(missing_ok=True)
        return "consistent" if consistent else "voted"
    except Exception as exc:
        logger.save()
        write_json(error_path, {
            "id": pid, "split": record["split"],
            "created_at": utc_now(), "error": str(exc),
        })
        logging.error("pid=%s 处理失败：%s", pid, exc)
        return "failed"


def result_files(logs_root: Path, split: str | None = None) -> list[str]:
    pattern = str(logs_root / (f"batch_run_{split}" if split else "batch_run_*")
                  / "*" / "final_result.json")
    return sorted(glob.glob(pattern))


def write_summary(logs_root: Path, split: str) -> Path:
    files = result_files(logs_root, split)
    rows = []
    for file_path in files:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        tumors = merge_tumors(data.get("tumors", []))
        rows.append({
            "id": data.get("id", "Unknown"),
            "consistency": "Yes" if data.get("consistent", False) else "No",
            "cost": f"{data.get('_meta', {}).get('total_cost_usd', 0.0):.6f}",
            "sequence": format_sequence(data.get("organ_status", "Unknown"), tumors),
        })
    rows.sort(key=lambda x: x["id"])
    output = logs_root / f"batch_run_{split}" / "Summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "consistency", "cost", "sequence"])
        writer.writeheader()
        writer.writerows(rows)
    return output


def command_extract(args: argparse.Namespace, logs_root: Path) -> None:
    client = DeepSeekClient(args)
    selected_pids: set[str] | None = None
    if args.pids_file:
        selected_pids = {
            line.strip() for line in Path(args.pids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    for split in args.splits:
        records = load_records(args.data, args.split_dir, split)
        if selected_pids is not None:
            records = [r for r in records if r["pid"] in selected_pids]
        if args.limit > 0:
            records = records[:args.limit]
        logging.info("split=%s：本次选中 %d 例", split, len(records))
        counts: Counter = Counter()
        for idx, record in enumerate(records, 1):
            status = process_record(record, client, logs_root, overwrite=args.overwrite)
            counts[status] += 1
            logging.info("split=%s [%d/%d] pid=%s：%s",
                         split, idx, len(records), record["pid"], status)
        summary_path = write_summary(logs_root, split)
        logging.info("split=%s 状态统计：%s；Summary：%s", split, dict(counts), summary_path)


# ------------------------- 官方 recover_2of3.py -------------------------
def recover_2of3(logs_root: Path, min_winner: int = 2) -> dict:
    rows_yes_original = []
    rows_recovered = []
    rerun_ids = []
    files = result_files(logs_root)
    for file_path in files:
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logging.warning("无法读取 %s：%s", file_path, exc)
            continue
        case_id = data.get("id", "Unknown")
        consistent = data.get("consistent", False)
        organ = data.get("organ_status", "Unknown")
        tumors = merge_tumors(data.get("tumors", []))
        sequence = format_sequence(organ, tumors)
        cost = data.get("_meta", {}).get("total_cost_usd", 0.0)
        details = data.get("consistency_details", {})
        winner_count = details.get("winner_count", 0)
        rounds = details.get("total_rounds", 3)
        row = {
            "id": case_id, "consistency": "Yes", "cost": f"{cost:.6f}",
            "sequence": sequence, "winner_count": winner_count,
            "total_rounds": rounds,
        }
        if consistent:
            rows_yes_original.append(row)
        elif winner_count >= min_winner:
            rows_recovered.append(row)
        else:
            rerun_ids.append(case_id)

    all_rows = sorted(rows_yes_original + rows_recovered, key=lambda x: x["id"])
    out_csv = logs_root / "merged_summary_recovered.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "consistency", "cost", "sequence", "winner_count", "total_rounds",
        ])
        writer.writeheader()
        writer.writerows(all_rows)
    rerun_txt = logs_root / "rerun_ids.txt"
    rerun_txt.write_text("\n".join(rerun_ids), encoding="utf-8")
    stats = {
        "found": len(files),
        "originally_consistent": len(rows_yes_original),
        "recovered": len(rows_recovered),
        "need_rerun": len(rerun_ids),
        "total_usable": len(all_rows),
        "out_csv": str(out_csv),
        "rerun_txt": str(rerun_txt),
    }
    logging.info("recover 完成：%s", json.dumps(stats, ensure_ascii=False))
    return stats


# -------------------------- 官方 rerun_failed.py --------------------------
def find_case_dir(logs_root: Path, case_id: str) -> Path | None:
    matches = glob.glob(str(logs_root / "batch_run_*" / case_id))
    return Path(matches[0]) if matches else None


def rerun_case(case_id: str, args: argparse.Namespace, logs_root: Path,
               client: DeepSeekClient | None) -> str | None:
    case_dir = find_case_dir(logs_root, case_id)
    if case_dir is None:
        logging.warning("[SKIP] 找不到病例目录：%s", case_id)
        return None
    result_path = case_dir / "final_result.json"
    backup_path = case_dir / "final_result.json.backup"
    try:
        record = find_record_by_id(args.data, args.split_dir, case_id)
    except Exception as exc:
        logging.warning("[SKIP] %s：%s", case_id, exc)
        return None
    if args.rerun_dry_run:
        logging.info("[DRY] 将以 %d 轮仅重跑 Step 3：%s", args.rounds, case_id)
        return "dry_run"
    if client is None:
        raise RuntimeError("实际重跑需要 DeepSeek client")

    if result_path.exists() and not backup_path.exists():
        shutil.copy2(result_path, backup_path)
    source_path = backup_path if backup_path.exists() else result_path
    if not source_path.exists():
        logging.warning("[SKIP] 找不到旧结果：%s", case_id)
        return None
    with source_path.open(encoding="utf-8") as f:
        old_data = json.load(f)

    logger = CaseLogger(case_dir, case_id)
    try:
        organ_status = old_data.get("organ_status", "Unknown")
        raw_tumors, consistent, stats = run_step_3_tumor_extraction(
            record["report_text"], client, logger, rounds=args.rounds)
        structured = parse_tumor_records(raw_tumors)
        final_data = {
            "id": case_id,
            "organ_status": organ_status,
            "tumors": structured,
            "consistent": consistent,
            "consistency_details": stats,
            "_meta": old_data.get("_meta", {}),
        }
        write_json(result_path, final_data)
        logger.save("rerun_log.json")
        logging.info("%s 重跑结果：winner_count=%s/%s，consistent=%s",
                     case_id, stats.get("winner_count", 0), args.rounds, consistent)
        return "consistent" if consistent else "failed"
    except Exception as exc:
        logging.error("%s 重跑失败：%s", case_id, exc)
        logger.save("rerun_log.json")
        if backup_path.exists():
            shutil.copy2(backup_path, result_path)
        return None


def command_rerun(args: argparse.Namespace, logs_root: Path) -> None:
    rerun_txt = Path(args.rerun_txt) if args.rerun_txt else logs_root / "rerun_ids.txt"
    if not rerun_txt.exists():
        raise FileNotFoundError(f"{rerun_txt} 不存在；请先运行 recover")
    ids = [line.strip() for line in rerun_txt.read_text(encoding="utf-8").splitlines()
           if line.strip()]
    if args.limit > 0:
        ids = ids[:args.limit]
    client = None if args.rerun_dry_run else DeepSeekClient(args)
    counts: Counter = Counter()
    for idx, case_id in enumerate(ids, 1):
        logging.info("重跑 [%d/%d] %s", idx, len(ids), case_id)
        result = rerun_case(case_id, args, logs_root, client)
        counts[result or "skipped"] += 1
    logging.info("重跑完成：%s；请再次运行 recover 和 build", dict(counts))


# ---------------------- 官方 build_accurate_dataset.py ----------------------
def qty_normalize(quantity: Any) -> str:
    q = str(quantity).strip()
    if "multiple" in q.lower():
        return "Multiple"
    try:
        number = int(q)
        if number <= 0:
            return "1"
        if number == 1:
            return "1"
        if number == 2:
            return "2"
        return "GE3"
    except ValueError:
        return "Multiple"


def merge_tumors_accurate(tumor_list: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for tumor in tumor_list:
        tumor_type = tumor.get("type", "Unknown").strip()
        position = tumor.get("position", "Unknown").strip()
        quantity = tumor.get("quantity", "1").strip()
        grouped.setdefault((tumor_type, position), []).append(quantity)
    merged = []
    for (tumor_type, position), quantities in grouped.items():
        total = 0
        final_quantity = None
        for quantity in quantities:
            if "multiple" in quantity.lower():
                final_quantity = "Multiple"
                break
            try:
                total += int(quantity)
            except ValueError:
                final_quantity = "Multiple"
                break
        if final_quantity is None:
            final_quantity = qty_normalize(total)
        merged.append({
            "type": tumor_type, "position": position, "quantity": final_quantity,
        })
    merged.sort(key=lambda x: (
        TYPE_PRIORITY.get(x["type"], 99), POS_PRIORITY.get(x["position"], 99)))
    return merged


def is_vocab_valid(tumors: list[dict]) -> bool:
    # 与官方一致：检查 type/position 及多位置异常字符串，不额外增加证据或 QA 规则。
    for tumor in tumors:
        if "," in tumor.get("position", ""):
            return False
        if tumor.get("type") not in VALID_TYPES:
            return False
        if tumor.get("position") not in VALID_POS:
            return False
    return True


def classify_disagreement(details: dict) -> tuple[int, str, int]:
    variants = details.get("details", {})
    if not variants:
        return 0, "unknown", 999
    winner_key = max(variants, key=variants.get)
    winner_count = variants[winner_key]
    winner_set = set(json.loads(winner_key))
    max_diff = 0
    diff_type = "qty_only"
    for key in variants:
        if key == winner_key:
            continue
        dissenter_set = set(json.loads(key))
        added = dissenter_set - winner_set
        removed = winner_set - dissenter_set
        n_diff = len(added) + len(removed)
        max_diff = max(max_diff, n_diff)
        winner_pairs = {tuple(r.split(" | ")[:2]) for r in winner_set}
        dissenter_pairs = {tuple(r.split(" | ")[:2]) for r in dissenter_set}
        if winner_pairs != dissenter_pairs:
            diff_type = "large" if n_diff >= 3 else "small"
    return winner_count, diff_type, max_diff


def build_accurate_dataset(logs_root: Path, max_diff_accept: int = 2) -> dict:
    accepted = []
    rerun = []
    tier_counts: Counter = Counter()
    files = result_files(logs_root)
    for file_path in files:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        case_id = data.get("id", "Unknown")
        organ = data.get("organ_status", "Unknown")
        tumors = merge_tumors_accurate(data.get("tumors", []))
        sequence = format_sequence(organ, tumors)
        details = data.get("consistency_details", {})
        consistent = data.get("consistent", False)
        winner_count, diff_type, max_diff = classify_disagreement(details)

        if not is_vocab_valid(tumors):
            tier_counts["rejected_vocab"] += 1
            rerun.append(case_id)
        elif consistent:
            accepted.append({"id": case_id, "tier": 1, "wc": winner_count, "sequence": sequence})
            tier_counts[1] += 1
        elif winner_count >= 2:
            if diff_type == "qty_only":
                tier = 2
            elif max_diff == 1:
                tier = 3
            elif max_diff <= max_diff_accept:
                tier = 4
            else:
                tier_counts["rejected_diff"] += 1
                rerun.append(case_id)
                continue
            accepted.append({"id": case_id, "tier": tier, "wc": winner_count,
                             "sequence": sequence})
            tier_counts[tier] += 1
        else:
            tier_counts["rejected_1of3"] += 1
            rerun.append(case_id)

    accepted.sort(key=lambda x: x["id"])
    out_csv = logs_root / "accurate_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "tier", "wc", "sequence"])
        writer.writeheader()
        writer.writerows(accepted)
    rerun_txt = logs_root / "rerun_needed.txt"
    rerun_txt.write_text("\n".join(sorted(rerun)), encoding="utf-8")
    stats = {
        "found": len(files), "accepted": len(accepted), "rerun": len(rerun),
        "tier_counts": dict(tier_counts),
        "out_csv": str(out_csv), "rerun_txt": str(rerun_txt),
    }
    logging.info("build 完成：%s", json.dumps(stats, ensure_ascii=False))
    return stats


# ------------------------------- 命令行 -------------------------------
def parse_args() -> argparse.Namespace:  # todo
    parser = argparse.ArgumentParser(description="中文报告 → MRI2Rep RLC（官方流程中文适配）")
    parser.add_argument("--command", choices=["extract", "recover", "rerun", "build"],
                        help='按顺序执行，build后再一次recover、build更新汇总表格',
                        default="build")
    parser.add_argument("--data",
                        default="/home/longkaixing/Liver/data/data.xlsx")
    parser.add_argument("--split-dir",
                        default="/home/longkaixing/Liver/data/split_idx/260925_SSL")
    parser.add_argument("--out-dir",
                        default="/home/longkaixing/Liver/Models/MRI2Rep/RLC_report")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"],
                        default=["test"])
    parser.add_argument("--limit", type=int,
                        help="extract：每个 split 取前 N 例；rerun：前 N 个待重跑病例；0=全部",
                        default=0)
    parser.add_argument("--pids-file", help="extract 可选病例清单，每行一个 id",
                        default="")
    parser.add_argument("--overwrite", help="重新提取已有 final_result 的病例",
                        default=False)
    parser.add_argument("--model",
                        default="deepseek/deepseek-chat")
    parser.add_argument("--base-url",
                        default="https://api.deepseek.com")
    parser.add_argument("--proxy", help="设为 none 可禁用代理",
                        default="http://127.0.0.1:7898")
    parser.add_argument("--max-retries", type=int, help="仅用于网络/API 调用失败",
                        default=3)
    parser.add_argument("--min-winner", type=int, help="recover 最低 winner_count",
                        default=2)
    parser.add_argument("--rounds", type=int, help="rerun 的 Step 3 轮数，官方默认 5",
                        default=5)
    parser.add_argument("--rerun-txt", help="默认 out_dir/logs/rerun_ids.txt",
                        default="")
    parser.add_argument("--rerun-dry-run", help="仅打印重跑计划，不调用 API",
                        default=False)
    parser.add_argument("--max-diff-accept", type=int, help="build 接受 2/3 结果的最大 token diff，官方默认 2",
                        default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    logs_root = out_dir / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "RLC.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("命令=%s，输出目录=%s", args.command, out_dir)
    logging.info("本次参数：%s", vars(args))

    if args.command == "extract":  # 调用LLM从报告生成 RLC 标签
        command_extract(args, logs_root)
    elif args.command == "recover":  # 根据投票一致性恢复可接受病例，并生成重跑清单（整理投票结果，不再调用LLM）（接受3/3和至少2/3的病例，生成低一致性清单）
        recover_2of3(logs_root, min_winner=args.min_winner)
    elif args.command == "rerun":  # 对低一致性病例重新执行 Step 3 分析影像报告，提取出结构化列表
        command_rerun(args, logs_root)
    elif args.command == "build":  # 按官方分级规则筛选并生成 RLC 最终标签表
        build_accurate_dataset(logs_root, max_diff_accept=args.max_diff_accept)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
