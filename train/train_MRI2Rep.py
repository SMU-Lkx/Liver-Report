# train/train_MRI2Rep.py
"""
Stage-B 主训练: 四期相 -> 自回归生成 report_desc (MRI2Rep 式 baseline).
本脚本只含训练/验证循环; 模型构建在 Models/MRI2Rep/model/.
"""
from __future__ import annotations

import argparse, setproctitle
import datetime
import logging
import os
import sys
import time
from os.path import join
from pathlib import Path

# --- NPU environment ---
os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
os.environ["ASCEND_LOG_DEVICE_FLUSH_TIMEOUT"] = "0"
os.environ["ASCEND_LAUNCH_BLOCKING"] = "0"
os.environ["NPU_FUZZY_COMPILE"] = "0"
os.environ["ASCEND_PRECISION_MODE"] = "allow_mix_precision"
os.environ.setdefault("TMPDIR", "/var/tmp/longkaixing")
CANN_PYTHON_PATH = "/home/HwHiAiUser/Ascend/ascend-toolkit/8.2.RC1/python/site-packages"
os.environ["PYTHONPATH"] = CANN_PYTHON_PATH + (":" + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else "")
sys.path.insert(0, CANN_PYTHON_PATH)

import torch
import torch.nn as nn
import torch_npu
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.Dataset.Multiphase_dataset import CharTokenizer, MultiphaseReport, multiphase_collate
from Models.MRI2Rep.model import build_generator, load_pretrained_backbone
from util.train_utils import (CkptManager, EpochCSVLogger, load_config, log_setup, resume_from_ckpt,
                              save_config, setup_seed, amp_context)
from util.text_utils import ngram_counts, char_bleu

PAD, BOS, EOS = 0, 1, 2


def build_dataset(args, split, tok):
    pid_csv = join(args.sampler, f"{split}_idx.csv")
    if not os.path.exists(pid_csv):
        raise FileNotFoundError(pid_csv)
    return MultiphaseReport(
        data_path=args.data, pid_csv=pid_csv, tokenizer=tok,
        cache_dir=args.cache_dir, max_len=int(args.max_len),
        mask_mode=args.mask_mode,
        verify_paths=bool(getattr(args, "verify_paths", False)),
    )


def train_one_epoch(model, loader, optimizer, device, args, epoch, tok):
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD,
                                  label_smoothing=float(args.label_smoothing))
    sums = {"nll": 0.0, "n": 0}
    ss_prob = 0.0
    if int(args.ss_max_prob) > 0 and epoch > int(args.ss_start_epoch):
        ss_prob = min(float(args.ss_max_prob),
                      float(args.ss_max_prob) * (epoch - int(args.ss_start_epoch)) / max(1, int(args.ss_warmup_epochs)))
    wd = float(args.word_dropout_p)

    pbar = tqdm(loader, desc=f"TextGen Ep {epoch:02d}/{args.epoch}" + (f" ss={ss_prob:.2f}" if ss_prob else ""))
    for batch in pbar:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        dec_input, target = y[:, :-1], y[:, 1:]

        # word dropout
        if wd > 0:
            m = (torch.rand_like(dec_input.float()) < wd) & (dec_input != PAD)
            m[:, 0] = False
            dec_input = dec_input.masked_fill(m, PAD)

        # scheduled sampling (MRI2Rep 式: 逐样本替换解码输入)
        if ss_prob > 0:
            ss_mask = torch.rand(x.shape[0], device=device) < ss_prob
            if ss_mask.any():
                model.eval()
                gen = model.generate(x[ss_mask], max_len=y.shape[1])
                model.train()
                tgt_len = dec_input.shape[1]
                g = gen[:, 1:tgt_len + 1]
                if g.shape[1] < tgt_len:
                    g = torch.cat([g, torch.full((g.shape[0], tgt_len - g.shape[1]), PAD,
                                                 dtype=torch.long, device=device)], dim=1)
                dec_input = dec_input.clone()
                dec_input[ss_mask] = g

        with amp_context(bool(args.use_bf16)):
            logits = model(x, dec_input)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if float(args.max_grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
        optimizer.step()
        sums["nll"] += float(loss.detach().item())
        sums["n"] += 1
        n = max(sums["n"], 1)
        pbar.set_postfix(nll=f"{sums['nll']/n:.4f}")
    return sums["nll"] / max(sums["n"], 1)


@torch.no_grad()
def validate(model, loader, device, args, tok):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    nll, n = 0.0, 0
    refs, hyps = [], []
    gen_batches = 0
    for batch in tqdm(loader, desc="Val", leave=False):
        if n >= int(args.val_max_batches):
            break
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with amp_context(bool(args.use_bf16)):
            logits = model(x, y[:, :-1])
            nll += float(loss_fn(logits.reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1)).item())
        if gen_batches < int(args.val_generate_batches):
            gen = model.generate(x, max_len=int(args.max_len))
            for i in range(x.shape[0]):
                hyps.append(tok.decode(gen[i].cpu().tolist()))
                refs.append(tok.decode(y[i].cpu().tolist()))
            gen_batches += 1
        n += 1
    bleu = char_bleu([list(r) for r in refs], [list(h) for h in hyps]) if refs else 0.0
    return {"nll": nll / max(n, 1), "bleu2": bleu, "evaluated": n}


def main():
    setproctitle.setproctitle("lkxMRI2Rep10/20")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str,
        default="/home/longkaixing/Liver/config/MRI2Rep/261015_MRI2Rep.yaml")
    args, _ = parser.parse_known_args()
    config = load_config(args.cfg)
    for k, v in config.items():
        setattr(args, k, v)

    save_path = join(args.save_dir, args.save_name)
    os.makedirs(save_path, exist_ok=True)
    log_setup(save_path, "textgen", name=args.save_name)
    logging.info(f"[Config] saved to {save_config(args.cfg, save_path, args.save_name)}")

    torch_npu.npu.set_device(args.device)
    device = torch.device(args.device)
    setup_seed(int(args.seed))

    tok = CharTokenizer.load(join(args.cache_dir, "vocab.json"))
    logging.info(f"[Tokenizer] vocab_size={tok.vocab_size}")

    train_ds = build_dataset(args, "train", tok)
    val_ds = build_dataset(args, "val", tok)
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True,
                              num_workers=int(args.num_workers), pin_memory=True,
                              collate_fn=multiphase_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False,
                            num_workers=int(args.num_workers), pin_memory=True,
                            collate_fn=multiphase_collate)
    logging.info(f"[Dataset] Train={len(train_ds)} | Val={len(val_ds)}")

    model = build_generator(args, tok.vocab_size, pad_id=PAD, bos_id=BOS, eos_id=EOS).to(device)
    load_pretrained_backbone(model, getattr(args, "pretrained_backbone", ""), map_location=device)

    backbone_params = list(model.backbone.parameters())
    bb_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in bb_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": float(args.lr_backbone),
         "weight_decay": float(args.weight_decay), "group_name": "backbone"},
        {"params": other_params,    "lr": float(args.lr_decoder),
         "weight_decay": float(args.weight_decay), "group_name": "decoder"},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(args.epoch), eta_min=float(args.min_lr))

    ckpt_mgr = CkptManager(join(save_path, "ckpt"),
                           max_keep_epochs=int(getattr(args, "max_keep_epochs", 0)))
    start_epoch, best_val, no_improve = 1, float("inf"), 0
    if bool(getattr(args, "resume", False)):
        ckpt = ckpt_mgr.load_last(map_location=device)
        if ckpt is not None:
            start_epoch, best_val = resume_from_ckpt(ckpt, model, optimizer, scheduler)
            no_improve = int(ckpt.get("no_improve", 0))
            logging.info(f"[Resume] from epoch {start_epoch}, best_val={best_val:.6f}")

    csv_logger = EpochCSVLogger(save_path, "textgen_metrics.csv",
                                resume=bool(getattr(args, "resume", False)))
    start_time = time.time()

    for epoch in range(start_epoch, int(args.epoch) + 1):
        tr_nll = train_one_epoch(model, train_loader, optimizer, device, args, epoch, tok)
        va = validate(model, val_loader, device, args, tok)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        logging.info(f"[Ep {epoch:02d}] train_nll={tr_nll:.4f} | val_nll={va['nll']:.4f} "
                     f"bleu2={va['bleu2']:.4f} | lr={lr_now:.2e} | {elapsed}")
        csv_logger.add_epoch(epoch, train_nll=tr_nll, val_nll=va["nll"],
                             val_bleu2=va["bleu2"], lr=lr_now)

        ckpt_mgr.save_epoch(epoch, model, optimizer=optimizer, scheduler=scheduler,
                            best_val=best_val, extra={"no_improve": no_improve})

        if va["nll"] < best_val:
            best_val, no_improve = float(va["nll"]), 0
            ckpt_mgr.save_best(model, extra={"epoch": epoch, "val_nll": best_val})
            logging.info(f"  -> BEST | val_nll={best_val:.6f}")
        else:
            no_improve += 1
            if no_improve >= int(args.early_stop):
                logging.info(f"Early stopping after {args.early_stop} epochs.")
                break

    logging.info(f"[Done] best val_nll={best_val:.6f} | output: {save_path}")


if __name__ == "__main__":
    main()
