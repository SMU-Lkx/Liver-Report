# train/infer_MRI2Rep.py
"""
Stage-B 测试集推理：加载 best checkpoint，对 test split 逐例生成 RLC 序列并落盘。

输出格式与训练期 out_sample/val_predictions.csv 完全一致，可直接喂
Models/MRI2Rep/infer/render_reports.py 渲染中文报告。

用法：
  python -m train.infer_MRI2Rep
   --cfg config/MRI2Rep/261029_MRI2Rep_4p.yaml
   --ckpt model_save/261029_MRI2Rep_4p/ckpt/<best文件名>
   --out  model_save/261029_MRI2Rep_4p/out_sample/test_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os, tqdm
import sys
from os.path import join
from pathlib import Path

os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
CANN_PYTHON_PATH = "/home/HwHiAiUser/Ascend/ascend-toolkit/8.2.RC1/python/site-packages"
os.environ["PYTHONPATH"] = CANN_PYTHON_PATH + (
    ":" + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else ""
)
sys.path.insert(0, CANN_PYTHON_PATH)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.Dataset.Multiphase_dataset import (
    RLC_MAX_LEN, RLCTokenizer, normalize_rlc_phases, MultiphaseRLC, rlc_collate,
)
from Models.MRI2Rep.model import build_generator
from train.train_MRI2Rep import parse_sequence  # 复用官方容错解析
from util.train_utils import (CkptManager, load_config, log_setup,
    save_config, setup_seed, amp_context)

PAD, BOS, EOS = 0, 1, 2


def build_dataset(args, split: str, tok: RLCTokenizer) -> MultiphaseRLC:
    pid_csv = join(args.sampler, f"{split}_idx.csv")
    if not os.path.exists(pid_csv):
        raise FileNotFoundError(pid_csv)
    return MultiphaseRLC(
        data_path=args.data, pid_csv=pid_csv, rlc_csv=args.RLC_csv, tokenizer=tok,
        cache_dir=args.RLC_cache_dir, max_len=RLC_MAX_LEN,
        shape=tuple(args.RLC_shape), phases=args.RLC_phase,
        image_cache_dir=args.cache_dir, build_if_missing=True,
        verify_paths=bool(args.verify_paths),
    )





VOCAB_SIZE = 24


@torch.no_grad()
def infer(model, loader, device, tok: RLCTokenizer, use_bf16: bool = True, epoch_tag: int = -1):
    model.eval()
    dump_rows, n_cases = [], 0
    metrics_sum = {k: 0.0 for k in
                   ["liver_correct", "no_lesion_correct", "case_tp", "case_fn", "case_tn", "case_fp"]}
    pair_tp = pair_fp = pair_fn = 0
    for batch in tqdm.tqdm(loader):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with amp_context(use_bf16):
            out = model(x, y[:, :-1])
            logits = out[0] if isinstance(out, (tuple, list)) else out
            tok_nll = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE).float(), y[:, 1:].reshape(-1),
                ignore_index=PAD, reduction="none",
            ).view(y.shape[0], -1).sum(dim=1)
            n_valid = (y[:, 1:] != PAD).sum(dim=1).clamp(min=1)
            case_nll = (tok_nll / n_valid).cpu().tolist()
            preds = model.generate(x, max_len=RLC_MAX_LEN)
        pred_np, gt_np = preds.cpu().numpy(), y.cpu().numpy()
        for i, (pred_seq, gt_seq) in enumerate(zip(pred_np, gt_np)):
            pred, gt = parse_sequence(pred_seq), parse_sequence(gt_seq)
            n_cases += 1
            metrics_sum["liver_correct"] += int(
                gt["liver"] is not None and pred["liver"] == gt["liver"])
            gt_no, pred_no = len(gt["lesions"]) == 0, len(pred["lesions"]) == 0
            metrics_sum["no_lesion_correct"] += int(gt_no == pred_no)
            metrics_sum[{"TN": "case_tn", "FP": "case_fp", "FN": "case_fn", "TP": "case_tp"}[
                ("T" if pred_no else "F") + ("N" if gt_no else "P")]] += 1
            p_pairs = {(t, p) for t, p, _ in pred["lesions"]}
            g_pairs = {(t, p) for t, p, _ in gt["lesions"]}
            pair_tp += len(p_pairs & g_pairs)
            pair_fp += len(p_pairs - g_pairs)
            pair_fn += len(g_pairs - p_pairs)
            dump_rows.append({
                "epoch": epoch_tag, "pid": batch["pid"][i], "tier": batch["tier"][i],
                "nll": round(case_nll[i], 4),
                "gt": batch["sequence"][i],
                "pred": tok.decode(pred_seq.tolist()),
                "gt_lesions": len(gt["lesions"]), "pred_lesions": len(pred["lesions"]),
                "liver_match": int(pred["liver"] == gt["liver"]),
            })
    m = metrics_sum
    sen = m["case_tp"] / (m["case_tp"] + m["case_fn"]) if m["case_tp"] + m["case_fn"] else 0.0
    spe = m["case_tn"] / (m["case_tn"] + m["case_fp"]) if m["case_tn"] + m["case_fp"] else 0.0
    micro_den = 2 * pair_tp + pair_fp + pair_fn
    pair_micro = 2 * pair_tp / micro_den if micro_den else 0.0
    logging.info(f"[Test] n={n_cases} | LiverAcc={m['liver_correct']/n_cases:.4f} | "
                 f"Sen={sen:.4f} | Spe={spe:.4f} | pairF1_micro={pair_micro:.4f}")
    return dump_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",
                        default="/home/longkaixing/Liver/config/MRI2Rep/261029_MRI2Rep_4p.yaml")
    parser.add_argument("--ckpt", help="best checkpoint 路径",
                        default='/home/longkaixing/Liver/model_save/261029_MRI2Rep_4p/ckpt/best_model.pt')
    parser.add_argument("--out", help="输出 CSV（test_pred.csv）",
                        default='/home/longkaixing/Liver/model_save/261029_MRI2Rep_4p/eval/rlc_pred/test_pred.csv')
    parser.add_argument("--split",
                        default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="npu:2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.cfg)
    for k, v in config.items():
        setattr(args, k, v)
    args.RLC_phase = normalize_rlc_phases(args.RLC_phase)
    args.RLC_shape = tuple(int(v) for v in args.RLC_shape)
    args.in_ch = len(args.RLC_phase) + 1
    args.max_len = RLC_MAX_LEN

    save_dir = str(Path(args.out).parent)
    os.makedirs(save_dir, exist_ok=True)
    log_setup(save_dir, "infer", name="infer")
    logging.info(f"[Config] saved to {save_config(args.cfg, save_dir, 'infer')}")

    import torch_npu
    torch_npu.npu.set_device(args.device)
    device = torch.device(args.device)
    setup_seed(args.seed)

    tok = RLCTokenizer()
    test_ds = build_dataset(args, args.split, tok)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=rlc_collate)
    logging.info(f"[Dataset] {args.split}={len(test_ds)}")

    model = build_generator(args, tok.vocab_size, pad_id=PAD, bos_id=BOS, eos_id=EOS).to(device)
    ckpt_mgr = CkptManager(str(Path(args.ckpt).parent))
    ckpt = ckpt_mgr.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    logging.info(f"[Ckpt] {args.ckpt} | epoch={ckpt.get('epoch', -1)} | "
                 f"val_pair_f1_lesion1p={ckpt.get('val_pair_f1_lesion1p', float('nan')):.6f}")

    epoch_tag = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1
    rows = infer(model, loader, device, tok, bool(args.use_bf16), epoch_tag)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logging.info(f"[Dump] {len(rows)} cases -> {args.out}")


if __name__ == "__main__":
    main()
