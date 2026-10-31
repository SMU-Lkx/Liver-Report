# train/train_MRI2Rep.py
"""
Stage-B 主训练：[RLC_phase..., Mask] -> MRI2Rep RLC 24-token 序列。

依赖 Multiphase_dataset.py 中的 RLCTokenizer、MultiphaseRLC、rlc_collate。
不再读取中文字符 vocab，不计算 char_bleu；验证采用官方结构化指标。
最佳模型按病灶阳性病例的 pair_f1_lesion1p 选择，避免无病灶多数类造成虚高。
"""
from __future__ import annotations

import argparse
import csv
import datetime
import logging
import os
import random
import setproctitle
import sys
import time
from collections import Counter
from os.path import join
from pathlib import Path

import numpy as np

# --- NPU environment ---
os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
os.environ["ASCEND_LOG_DEVICE_FLUSH_TIMEOUT"] = "0"
os.environ["ASCEND_LAUNCH_BLOCKING"] = "0"
os.environ["NPU_FUZZY_COMPILE"] = "0"
os.environ["ASCEND_PRECISION_MODE"] = "allow_mix_precision"
os.environ.setdefault("TMPDIR", "/var/tmp/longkaixing")
CANN_PYTHON_PATH = "/home/HwHiAiUser/Ascend/ascend-toolkit/8.2.RC1/python/site-packages"
os.environ["PYTHONPATH"] = CANN_PYTHON_PATH + (
    ":" + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else ""
)
sys.path.insert(0, CANN_PYTHON_PATH)

import torch
import torch.nn as nn
import torch_npu
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.Dataset.Multiphase_dataset import (
    RLC_MAX_LEN,
    # RLC_SHAPE,
    RLCTokenizer,
    normalize_rlc_phases,
    MultiphaseRLC,
    rlc_collate,
)
from Models.MRI2Rep.model import build_generator, load_pretrained_backbone
from util.train_utils import (CkptManager, EpochCSVLogger, load_config, log_setup,
    resume_from_ckpt, save_config, setup_seed, amp_context)

PAD, BOS, EOS, NO_LESION = 0, 1, 2, 3
OFFSET_LIVER, OFFSET_TUMOR = 4, 6
OFFSET_POS, OFFSET_QTY, VOCAB_SIZE = 11, 20, 24
DEFAULT_RLC_CSV = "/home/longkaixing/Liver/Models/MRI2Rep/RLC_report/logs/accurate_summary.csv"


def build_dataset(args, split: str, tok: RLCTokenizer) -> MultiphaseRLC:
    pid_csv = join(args.sampler, f"{split}_idx.csv")
    if not os.path.exists(pid_csv):
        raise FileNotFoundError(pid_csv)
    return MultiphaseRLC(
        data_path=args.data,
        pid_csv=pid_csv,
        rlc_csv=args.RLC_csv,
        tokenizer=tok,
        cache_dir=args.RLC_cache_dir,
        max_len=RLC_MAX_LEN,
        shape=tuple(args.RLC_shape),
        phases=args.RLC_phase,
        image_cache_dir=args.cache_dir,
        build_if_missing=True,
        verify_paths=bool(args.verify_paths),
    )


# ------------------------- 官方病灶分层采样 -------------------------
class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_ratio=0.2):
        super().__init__()
        self.primary_indices = list(primary_indices)
        self.secondary_indices = list(secondary_indices)
        self.batch_size = int(batch_size)
        self.secondary_batch_size = int(round(self.batch_size * secondary_ratio))
        self.primary_batch_size = self.batch_size - self.secondary_batch_size
        if self.secondary_batch_size <= 0 or self.primary_batch_size <= 0:
            raise ValueError("secondary_ratio 与 batch_size 组合无效")

    def __iter__(self):
        primary = self.primary_indices[:]
        secondary = self.secondary_indices[:]
        random.shuffle(primary)
        random.shuffle(secondary)
        sec_iter = iter(secondary)

        def next_secondary(k):
            nonlocal sec_iter
            out = []
            while len(out) < k:
                try:
                    out.append(next(sec_iter))
                except StopIteration:
                    random.shuffle(secondary)
                    sec_iter = iter(secondary)
            return out

        for start in range(0, len(primary), self.primary_batch_size):
            p_batch = primary[start:start + self.primary_batch_size]
            if len(p_batch) < self.primary_batch_size:
                break
            batch = p_batch + next_secondary(self.secondary_batch_size)
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def _is_rare_sequence(sequence: str) -> bool:
    parts = [part.strip() for part in sequence.split(",") if part.strip()]
    rest = parts[1:]
    for i in range(0, len(rest), 3):
        if i + 2 >= len(rest):
            break
        tumor, position, quantity = rest[i:i + 3]
        if tumor == "HEM" or position in {"CAUDATE", "L_DIFFUSE"}:
            return True
        if quantity == "GE3":
            return True
    return False


def build_lesion_sampler(train_ds, args):
    if not bool(args.use_lesion_sampler):
        return None, "使用普通随机打乱。"
    lesion_indices, no_lesion_indices, rare_indices, common_indices = [], [], [], []
    for idx, rec in enumerate(train_ds.records):
        parts = [p.strip() for p in rec["sequence"].split(",") if p.strip()]
        if len(parts) > 1:
            lesion_indices.append(idx)
        else:
            no_lesion_indices.append(idx)
        if _is_rare_sequence(rec["sequence"]):
            rare_indices.append(idx)
        else:
            common_indices.append(idx)
    if not lesion_indices or not no_lesion_indices:
        return None, "病灶或无病灶样本不足，使用普通随机打乱。"

    if rare_indices and common_indices:
        ratio = float(args.rare_sampling_ratio)
        target_rare = max(
            len(rare_indices),
            int(len(common_indices) * ratio / max(1e-6, 1.0 - ratio)),
        )
        repeats = max(1, (target_rare + len(rare_indices) - 1) // len(rare_indices))
        inflated = (rare_indices * repeats)[:target_rare]
        lesion_pool = common_indices + inflated
        desc = (f"分层采样：common={len(common_indices)}，"
                f"inflated_rare={len(inflated)}，no_lesion={len(no_lesion_indices)}")
    else:
        lesion_pool = lesion_indices
        desc = f"分层采样：lesion={len(lesion_pool)}，no_lesion={len(no_lesion_indices)}"
    try:
        sampler = TwoStreamBatchSampler(
            lesion_pool,
            no_lesion_indices,
            int(args.batch_size),
            secondary_ratio=float(args.no_lesion_ratio),
        )
    except ValueError as exc:
        return None, f"{exc}；使用普通随机打乱。"
    return sampler, desc


# ------------------------- 官方类别损失加权 -------------------------
def build_class_weights(train_ds, tok: RLCTokenizer, args, device):
    if not bool(args.use_class_weights):
        return None
    counts = np.zeros(VOCAB_SIZE, dtype=np.float64)
    for rec in train_ds.records:
        ids = tok.encode(rec["sequence"], max_len=RLC_MAX_LEN)
        counts += np.bincount(ids, minlength=VOCAB_SIZE)

    beta = float(getattr(args, "class_weight_beta", 0.999))
    weights = np.ones(VOCAB_SIZE, dtype=np.float32)
    for lo, hi in ((OFFSET_TUMOR, OFFSET_POS), (OFFSET_POS, OFFSET_QTY),
                   (OFFSET_QTY, VOCAB_SIZE)):
        cnt = counts[lo:hi]
        nonzero = cnt > 0
        w = np.zeros_like(cnt, dtype=np.float64)
        if nonzero.any():
            w[nonzero] = (1.0 - beta) / (1.0 - np.power(beta, cnt[nonzero]))
            w = w * (nonzero.sum() / w.sum())
        weights[lo:hi] = w.astype(np.float32)
    logging.info("[Loss] 已按训练集统计 RLC type/position/quantity 类别权重")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def forward_outputs(model, x, dec_input):
    output = model(x, dec_input)
    if isinstance(output, (tuple, list)):
        return output[0], output[1]
    return output, None


# ------------------------- 训练与 RLC 解析 -------------------------
def train_one_epoch(model, loader, optimizer, device, args, epoch, loss_fn):
    model.train()
    seq_sum, aux_sum, batches = 0.0, 0.0, 0
    ss_prob = 0.0
    ss_max = float(getattr(args, "ss_max_prob", 0.3))
    ss_start = int(getattr(args, "ss_start_epoch", 10))
    if ss_max > 0 and epoch > ss_start:
        ss_prob = min(
            ss_max,
            ss_max * (epoch - ss_start) / max(1, int(getattr(args, "ss_warmup_epochs", 20))),
        )
    word_dropout = float(getattr(args, "word_dropout_p", 0.15))
    aux_weight = float(getattr(args, "aux_weight", 0.5))

    pbar = tqdm(loader, desc=f"RLC Ep {epoch:02d}/{args.epoch}" +
                (f" ss={ss_prob:.2f}" if ss_prob else ""))
    for batch in pbar:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        dec_input, target = y[:, :-1].clone(), y[:, 1:]

        if ss_prob > 0:
            ss_mask = torch.rand(x.shape[0], device=device) < ss_prob
            if ss_mask.any():
                with torch.no_grad():
                    model.eval()
                    gen = model.generate(x[ss_mask], max_len=y.shape[1])
                    model.train()
                tgt_len = dec_input.shape[1]
                if gen.shape[1] - 1 >= tgt_len:
                    gen_input = gen[:, :tgt_len]
                else:
                    pad = torch.full(
                        (gen.shape[0], tgt_len - gen.shape[1] + 1),
                        PAD, dtype=torch.long, device=device,
                    )
                    gen_input = torch.cat([gen[:, :-1], pad], dim=1)
                dec_input[ss_mask] = gen_input

        if word_dropout > 0:
            dropout_mask = (torch.rand_like(dec_input.float()) < word_dropout)
            dropout_mask[:, 0] = False
            dropout_mask &= dec_input != PAD
            dec_input = dec_input.masked_fill(dropout_mask, PAD)

        optimizer.zero_grad(set_to_none=True)
        with amp_context(bool(args.use_bf16)):
            logits, aux_logit = forward_outputs(model, x, dec_input)
            seq_nll = loss_fn(logits.reshape(-1, VOCAB_SIZE), target.reshape(-1))
            if aux_logit is not None and aux_weight > 0:
                has_lesion = (target >= OFFSET_TUMOR).any(dim=1).float()
                aux_loss = nn.functional.binary_cross_entropy_with_logits(
                    aux_logit, has_lesion,
                )
                loss = seq_nll + aux_weight * aux_loss
            else:
                aux_loss = seq_nll.new_zeros(())
                loss = seq_nll

        loss.backward()
        if float(args.max_grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
        optimizer.step()
        seq_sum += float(seq_nll.detach().item())
        aux_sum += float(aux_loss.detach().item())
        batches += 1
        pbar.set_postfix(seq_nll=f"{seq_sum / batches:.4f}", aux=f"{aux_sum / batches:.4f}")
    return seq_sum / max(batches, 1), aux_sum / max(batches, 1)


def parse_sequence(seq):
    """与官方 engine.py 一致：容错解析肝脏背景和病灶三元组。"""
    seq = [int(t) for t in seq if int(t) not in (PAD, BOS, EOS)]  # 过滤补齐PAD、起止BOS/EOS符号
    info = {"liver": None, "lesions": [], "no_lesion": False}
    if NO_LESION in seq:  # 含 NO_LESION（无病灶时），置 no_lesion=True 并删除该 token。
        info["no_lesion"] = True
        seq = [t for t in seq if t != NO_LESION]
        # 有效 token 若在 id 4–5（Fibrosis/Cirrhosis 或 NoFibrosis/Cirrhosis），就认领为肝脏状态并消费掉
    if seq and OFFSET_LIVER <= seq[0] < OFFSET_TUMOR:
        info["liver"] = seq[0]
        seq = seq[1:]
    i = 0
    # 滑窗(连续3个token)判断：模型输出重复（例如重复了一个type token），则尝试错开一位后重新对齐
    while i < len(seq) - 2:
        tumor, position, quantity = seq[i:i + 3]
        if OFFSET_TUMOR <= tumor < OFFSET_POS <= position < OFFSET_QTY <= quantity < VOCAB_SIZE:
            info["lesions"].append((tumor, position, quantity))
            i += 3
        else:
            i += 1
    return info


def f1_from_sets(pred_set, gt_set):
    if not pred_set and not gt_set:
        return 1.0
    tp = len(pred_set & gt_set)
    fp, fn = len(pred_set - gt_set), len(gt_set - pred_set)
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def f1_from_counters(pred_ctr: Counter, gt_ctr: Counter):
    if not pred_ctr and not gt_ctr:
        return 1.0, 0, 0, 0
    tp = sum(min(pred_ctr.get(k, 0), gt_ctr.get(k, 0))
             for k in set(pred_ctr) | set(gt_ctr))
    fp = sum(pred_ctr.values()) - tp
    fn = sum(gt_ctr.values()) - tp
    denom = 2 * tp + fp + fn
    return (2 * tp / denom if denom else 0.0), tp, fp, fn


@torch.no_grad()
def validate(model, loader, device, args, epoch, tok=None, out_dir=""):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    nll_sum, batches, n_cases = 0.0, 0, 0
    liver_correct = 0
    no_lesion_correct = 0
    type_f1, pair_f1, triplet_f1 = [], [], []
    pair_zero, pair_positive = [], []
    triplet_zero, triplet_positive = [], []
    pair_tp = pair_fp = pair_fn = 0
    trip_tp = trip_fp = trip_fn = 0
    case_tp = case_fn = case_tn = case_fp = 0
    max_batches = int(args.val_max_batches)
    dump_rows = []

    for batch in tqdm(loader, desc="Val RLC", leave=False):
        if 0 < max_batches <= batches:
            break
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with amp_context(bool(args.use_bf16)):
            logits, _ = forward_outputs(model, x, y[:, :-1])
            loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), y[:, 1:].reshape(-1))
        nll_sum += float(loss.item())
        batches += 1

        preds = model.generate(x, max_len=min(RLC_MAX_LEN, y.shape[1] + 2))
        pred_np = preds.cpu().numpy()
        gt_np = y.cpu().numpy()
        for case_i, (pred_seq, gt_seq) in enumerate(zip(pred_np, gt_np)):
            pred = parse_sequence(pred_seq)
            gt = parse_sequence(gt_seq)
            n_cases += 1
            if out_dir:
                dump_rows.append({
                    "epoch": epoch,
                    "pid": batch["pid"][case_i],
                    "tier": batch["tier"][case_i],
                    "gt": batch["sequence"][case_i],
                    "pred": tok.decode(pred_seq.tolist()) if tok is not None else "",
                    "gt_lesions": len(gt["lesions"]),
                    "pred_lesions": len(pred["lesions"]),
                    "liver_match": int(pred["liver"] == gt["liver"]),
                })
            if gt["liver"] is not None and pred["liver"] == gt["liver"]:
                liver_correct += 1

            gt_no = len(gt["lesions"]) == 0
            pred_no = len(pred["lesions"]) == 0
            no_lesion_correct += int(gt_no == pred_no)
            if gt_no:
                if pred_no:
                    case_tn += 1
                else:
                    case_fp += 1
            else:
                if pred_no:
                    case_fn += 1
                else:
                    case_tp += 1

            pred_types = {t for t, _, _ in pred["lesions"]}
            gt_types = {t for t, _, _ in gt["lesions"]}
            type_f1.append(f1_from_sets(pred_types, gt_types))

            pred_pairs = {(t, p) for t, p, _ in pred["lesions"]}
            gt_pairs = {(t, p) for t, p, _ in gt["lesions"]}
            current_pair = f1_from_sets(pred_pairs, gt_pairs)
            pair_f1.append(current_pair)
            pair_tp += len(pred_pairs & gt_pairs)
            pair_fp += len(pred_pairs - gt_pairs)
            pair_fn += len(gt_pairs - pred_pairs)

            current_trip, tp, fp, fn = f1_from_counters(
                Counter(pred["lesions"]), Counter(gt["lesions"]),
            )
            triplet_f1.append(current_trip)
            trip_tp += tp
            trip_fp += fp
            trip_fn += fn
            if gt_no:
                pair_zero.append(current_pair)
                triplet_zero.append(current_trip)
            else:
                pair_positive.append(current_pair)
                triplet_positive.append(current_trip)

    if out_dir and dump_rows:
        os.makedirs(out_dir, exist_ok=True)
        out_path = join(out_dir, "val_pred.csv")
        write_header = not os.path.exists(out_path)
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(dump_rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(dump_rows)
        n_gt_lesion = sum(r["gt_lesions"] > 0 for r in dump_rows)
        n_pred_lesion = sum(r["pred_lesions"] > 0 for r in dump_rows)
        logging.info(
            f"[ValDump] {len(dump_rows)} cases -> {out_path} | "
            f"GT有病灶={n_gt_lesion} 预测有病灶={n_pred_lesion}"
        )
        shown = 0
        for r in dump_rows:
            if r["gt_lesions"] > 0 and shown < 3:
                logging.info(f"  [Sample+] pid={r['pid']} GT : {r['gt']}")
                logging.info(f"            pid={r['pid']} Pred: {r['pred']}")
                shown += 1
        for r in dump_rows:
            if r["gt_lesions"] == 0 and shown < 5:
                logging.info(f"  [Sample0] pid={r['pid']} GT : {r['gt']} | Pred: {r['pred']}")
                shown += 1

    pair_micro_den = 2 * pair_tp + pair_fp + pair_fn
    trip_micro_den = 2 * trip_tp + trip_fp + trip_fn
    return {  # 官方评估指标
        "nll": nll_sum / max(batches, 1),  # 负对数似然损失
        "liver_acc": liver_correct / max(n_cases, 1),  # 肝脏病变token准确率
        "no_lesion_acc": no_lesion_correct / max(n_cases, 1),  # 有无病灶准确率
        "type_f1": float(np.mean(type_f1)) if type_f1 else 0.0,  # 病灶类型的f1
        "pair_f1": float(np.mean(pair_f1)) if pair_f1 else 0.0,  # (病灶类型type, 位置position) token对 的f1
        "pair_f1_micro": 2 * pair_tp / pair_micro_den if pair_micro_den else 0.0,
        "triplet_f1": float(np.mean(triplet_f1)) if triplet_f1 else 0.0,  # (病灶,位置,数量) token三元组的f1
        "triplet_f1_micro": 2 * trip_tp / trip_micro_den if trip_micro_den else 0.0,
        "case_sen": case_tp / (case_tp + case_fn) if case_tp + case_fn else 0.0,  # GT 有病灶的病例中，模型检出 ≥1 个病灶的比例
        "case_spe": case_tn / (case_tn + case_fp) if case_tn + case_fp else 0.0,  # GT 无病灶的病例中，模型也输出无病灶的比例
        "pair_f1_lesion0": float(np.mean(pair_zero)) if pair_zero else 0.0,  # 仅在 GT 无病灶病例上的 pair-F1
        "pair_f1_lesion1p": float(np.mean(pair_positive)) if pair_positive else 0.0,  # 仅在 GT 有病灶病例上的 pair-F1（最佳模型选择依据）
        "triplet_f1_lesion0": float(np.mean(triplet_zero)) if triplet_zero else 0.0,
        "triplet_f1_lesion1p": float(np.mean(triplet_positive)) if triplet_positive else 0.0,
        "evaluated": n_cases,
    }


# ------------------------------- 主流程 -------------------------------
def main():
    setproctitle.setproctitle("lkxMRI2Rep10/31")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str,  # todo
        default="/home/longkaixing/Liver/config/MRI2Rep/261029_MRI2Rep_4p.yaml")
    args, _ = parser.parse_known_args()
    config = load_config(args.cfg)
    for key, value in config.items():
        setattr(args, key, value)

    # 期相由 YAML 控制；最后一个输入通道固定为 Venous liver mask。
    args.RLC_phase = normalize_rlc_phases(args.RLC_phase)
    args.RLC_shape = tuple(int(v) for v in args.RLC_shape)
    args.in_ch = len(args.RLC_phase) + 1
    args.max_len = RLC_MAX_LEN

    save_path = join(args.save_dir, args.save_name)
    os.makedirs(save_path, exist_ok=True)
    log_setup(save_path, "rlc", name=args.save_name)
    logging.info(f"[Config] saved to {save_config(args.cfg, save_path, args.save_name)}")
    logging.info(f"[RLC] labels={args.RLC_csv}")
    logging.info(
        f"[RLC] labels_cache={args.RLC_cache_dir}; "
        f"image_cache={args.cache_dir}; "
        f"phases={args.RLC_phase}; shape={args.RLC_shape}; "
        f"in_ch={args.in_ch}; max_len={RLC_MAX_LEN}"
    )

    torch_npu.npu.set_device(args.device)
    device = torch.device(args.device)
    setup_seed(int(args.seed))

    tok = RLCTokenizer()
    logging.info(f"[Tokenizer] fixed RLC vocab_size={tok.vocab_size}")
    train_ds = build_dataset(args, "train", tok)
    val_ds = build_dataset(args, "val", tok)
    if not len(train_ds) or not len(val_ds):
        raise RuntimeError(f"RLC 数据为空：Train={len(train_ds)}，Val={len(val_ds)}")

    batch_sampler, sampler_desc = build_lesion_sampler(train_ds, args)
    logging.info(f"[Sampler] {sampler_desc}")
    loader_kwargs = dict(
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=rlc_collate,
    )
    if batch_sampler is not None:
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=int(args.batch_size), shuffle=True,
            drop_last=True, **loader_kwargs,
        )
    val_loader = DataLoader(
        val_ds, batch_size=int(args.batch_size), shuffle=False, **loader_kwargs,
    )
    logging.info(f"[Dataset] Train={len(train_ds)} | Val={len(val_ds)}")

    model = build_generator(
        args, tok.vocab_size, pad_id=PAD, bos_id=BOS, eos_id=EOS,
    ).to(device)
    load_pretrained_backbone(
        model, args.pretrained_backbone, map_location=device,
        input_phases=args.RLC_phase,
    )

    aux_weight = float(args.aux_weight)
    has_aux_head = hasattr(model, "aux_lesion_head")
    if aux_weight > 0 and not has_aux_head:
        logging.warning(
            "[Loss] 当前模型没有 aux_lesion_head；本次仅使用序列 CE，"
            "这不包含官方辅助病灶分类损失。"
        )

    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    aux_params = list(model.aux_lesion_head.parameters()) if has_aux_head else []
    aux_ids = {id(p) for p in aux_params}
    other_params = [
        p for p in model.parameters()
        if id(p) not in backbone_ids and id(p) not in aux_ids
    ]
    param_groups = [
        {"params": backbone_params, "lr": float(args.lr_backbone),
         "weight_decay": float(args.weight_decay), "group_name": "backbone"},
        {"params": other_params, "lr": float(args.lr_decoder),
         "weight_decay": float(args.weight_decay), "group_name": "decoder"},
    ]
    if aux_params:
        param_groups.append({
            "params": aux_params,
            "lr": float(args.lr_aux),
            "weight_decay": 0.0,
            "group_name": "aux",
        })
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(args.epoch), eta_min=float(args.min_lr),
    )
    weight_tensor = build_class_weights(train_ds, tok, args, device)
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=PAD,
        weight=weight_tensor,
        label_smoothing=float(args.label_smoothing),
    )

    ckpt_mgr = CkptManager(
        join(save_path, "ckpt"),
        max_keep_epochs=int(args.max_keep_epochs),
    )
    start_epoch, best_f1, no_improve = 1, 0.0, 0
    if bool(args.resume):
        ckpt = ckpt_mgr.load_last(map_location=device)
        if ckpt is not None:
            start_epoch, best_f1 = resume_from_ckpt(ckpt, model, optimizer, scheduler)
            no_improve = int(ckpt.get("no_improve", 0))
            logging.info(f"[Resume] epoch={start_epoch}，best_pair_f1={best_f1:.6f}")

    csv_logger = EpochCSVLogger(
        save_path, "rlc_metrics.csv", resume=bool(args.resume),
    )
    start_time = time.time()
    min_epoch_best = int(args.min_epoch_best)  # todo
    min_f1_for_best = float(args.min_lesion1p_f1_for_best)

    for epoch in range(start_epoch, int(args.epoch) + 1):
        train_nll, train_aux = train_one_epoch(
            model, train_loader, optimizer, device, args, epoch, loss_fn,
        )
        metrics = validate(
            model, val_loader, device, args, epoch,
            tok=tok, out_dir=join(save_path, "out_sample"),
        )
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        logging.info(
            f"[Ep {epoch:02d}] train_nll={train_nll:.4f} | aux={train_aux:.4f} | "
            f"val_nll={metrics['nll']:.4f} | LiverAcc={metrics['liver_acc']:.4f} | "
            f"Sen={metrics['case_sen']:.4f} | Spe={metrics['case_spe']:.4f} | "
            f"LesionF1={metrics['pair_f1_lesion1p']:.4f} | "
            f"TripletF1={metrics['triplet_f1_lesion1p']:.4f} | "
            f"lr={lr_now:.2e} | {elapsed}"
        )
        csv_logger.add_epoch(
            epoch,
            train_nll=train_nll,
            train_aux=train_aux,
            val_nll=metrics["nll"],
            val_liver_acc=metrics["liver_acc"],
            val_no_lesion_acc=metrics["no_lesion_acc"],
            val_type_f1=metrics["type_f1"],
            val_pair_f1=metrics["pair_f1"],
            val_pair_f1_micro=metrics["pair_f1_micro"],
            val_triplet_f1=metrics["triplet_f1"],
            val_triplet_f1_micro=metrics["triplet_f1_micro"],
            val_case_sen=metrics["case_sen"],
            val_case_spe=metrics["case_spe"],
            val_pair_f1_lesion0=metrics["pair_f1_lesion0"],
            val_pair_f1_lesion1p=metrics["pair_f1_lesion1p"],
            val_triplet_f1_lesion0=metrics["triplet_f1_lesion0"],
            val_triplet_f1_lesion1p=metrics["triplet_f1_lesion1p"],
            lr=lr_now,
        )

        qualified = (
            epoch >= min_epoch_best and metrics["pair_f1_lesion1p"] >= min_f1_for_best
        )
        improved = qualified and metrics["pair_f1_lesion1p"] > best_f1
        if improved:
            best_f1 = float(metrics["pair_f1_lesion1p"])
            no_improve = 0
            ckpt_mgr.save_best(model, extra={
                "epoch": epoch,
                "val_pair_f1_lesion1p": best_f1,
                "val_triplet_f1_lesion1p": metrics["triplet_f1_lesion1p"],
            })
            logging.info(f"  -> BEST | val_pair_f1_lesion1p={best_f1:.6f}")
        elif epoch >= min_epoch_best:
            no_improve += 1

        ckpt_mgr.save_epoch(
            epoch, model, optimizer=optimizer, scheduler=scheduler,
            best_val=best_f1, extra={"no_improve": no_improve},
        )
        patience = int(args.early_stop)
        if 0 < patience <= no_improve:
            logging.info(f"Early stopping after {patience} non-improving epochs.")
            break

    logging.info(f"[Done] best val_pair_f1_lesion1p={best_f1:.6f} | output: {save_path}")


if __name__ == "__main__":
    main()