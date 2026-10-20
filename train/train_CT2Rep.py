# train/train_CT2Rep.py
"""
CT2Rep baseline 训练: 四期相 -> 自回归生成 report_desc.
与 train_MRI2Rep.py 共享: 数据/缓存/split/tokenizer/char-BLEU 函数 -> 指标口径完全一致.
方法本体差异(忠实原论文): CTViT visual extractor, Adam(lr_ve/lr_ed, betas 0.9/0.99),
StepLR, batch=1, RM+MCLN decoder, 从零训练(无预训练).
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

# --- NPU environment (与 Stage-B 完全相同) ---
os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
os.environ["ASCEND_LOG_DEVICE_FLUSH_TIMEOUT"] = "0"
os.environ["ASCEND_LAUNCH_BLOCKING"] = "0"
os.environ["NPU_FUZZY_COMPILE"] = "0"
os.environ["ASCEND_PRECISION_MODE"] = "allow_mix_precision"
CANN_PYTHON_PATH = "/home/HwHiAiUser/Ascend/ascend-toolkit/8.2.RC1/python/site-packages"
os.environ["PYTHONPATH"] = CANN_PYTHON_PATH + (":" + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else "")
sys.path.insert(0, CANN_PYTHON_PATH)

import torch
import torch.nn.functional as F
import torch_npu
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# CT2Rep 是脚本式 repo(无 __init__.py), 靠 namespace package + path 注入导入
CT2REP_CODE = PROJECT_ROOT / "Models" / "CT2Rep" / "CT2Rep"
sys.path.insert(0, str(CT2REP_CODE))

from data.Dataset.Multiphase_dataset import CharTokenizer, MultiphaseReport, multiphase_collate
from Models.CT2Rep.model.CT2Rep import CT2Rep, CharTokenizerAdapter
from util.train_utils import (CkptManager, EpochCSVLogger, load_config, log_setup, resume_from_ckpt,
                              save_config, setup_seed, amp_context)
from util.text_utils import ngram_counts, char_bleu

PAD, BOS, EOS = 0, 1, 2

def build_dataset(args, split, tok):
    pid_csv = join(args.sampler, f"{split}_idx.csv")
    return MultiphaseReport(                       # ← 原装的, 和 Stage-B 同一个类
        data_path=args.data, pid_csv=pid_csv, tokenizer=tok,
        cache_dir=args.cache_dir, max_len=int(args.max_len),
        mask_mode=args.mask_mode,
        verify_paths=bool(getattr(args, "verify_paths", False)),
    )

def train_one_epoch(model, loader, optimizer, device, args, epoch):
    model.train()
    sums = {"nll": 0.0, "n": 0}
    pbar = tqdm(loader, desc=f"CT2Rep Ep {epoch:02d}/{args.epoch}")
    for batch in pbar:                             # ← 和 Stage-B 同款循环
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with amp_context(bool(args.use_bf16)):
            out = model(x, y)                      # (B,L-1,V) log-probs, 内部已移位
            loss = F.nll_loss(out.reshape(-1, out.size(-1)), y[:, 1:].reshape(-1),
                              ignore_index=PAD, label_smoothing=float(args.label_smoothing))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if float(args.max_grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
        optimizer.step()
        sums["nll"] += float(loss.detach().item()); sums["n"] += 1
        pbar.set_postfix(nll=f"{sums['nll']/max(sums['n'],1):.4f}")
    return sums["nll"] / max(sums["n"], 1)

@torch.no_grad()
def validate(model, loader, device, args, tok_adapt):
    model.eval()
    nll, n = 0.0, 0
    refs, hyps = [], []
    gen_batches = 0
    for batch in tqdm(loader, desc="Val", leave=False):
        if n >= int(args.val_max_batches):
            break
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with amp_context(bool(args.use_bf16)):
            out = model(x, y)
            nll += float(F.nll_loss(out.reshape(-1, out.size(-1)),
                                    y[:, 1:].reshape(-1), ignore_index=PAD).item())
        if gen_batches < int(args.val_generate_batches):
            seq = model.generate(x)                # greedy
            for i in range(x.size(0)):
                hyps.append(tok_adapt.decode(seq[i].cpu().tolist()))
                refs.append(tok_adapt.decode(y[i].cpu().tolist()))
            gen_batches += 1
        n += 1
    bleu = char_bleu([list(r) for r in refs], [list(h) for h in hyps]) if refs else 0.0
    return {"nll": nll / max(n, 1), "bleu2": bleu, "evaluated": n}

def main():
    setproctitle.setproctitle("lkxCT2Rep10/20")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str,
        default="/home/longkaixing/Liver/config/CT2Rep/261020_CT2Rep.yaml")
    args, _ = parser.parse_known_args()
    config = load_config(args.cfg)
    for k, v in config.items():
        setattr(args, k, v)

    # ---- 日志保存 ----
    save_path = join(args.save_dir, args.save_name)
    os.makedirs(save_path, exist_ok=True)
    log_setup(save_path, "ct2rep", name=args.save_name)
    logging.info(f"[Config] saved to {save_config(args.cfg, save_path, args.save_name)}")
    # ---- 设置设备与随机种子 ----
    torch_npu.npu.set_device(args.device)
    device = torch.device(args.device)
    setup_seed(int(args.seed))
    # ---- 字符词元化 ----
    tok = CharTokenizer.load(join(args.cache_dir, "vocab.json"))
    tok_adapt = CharTokenizerAdapter(tok)
    logging.info(f"[Tokenizer] vocab_size={tok.vocab_size} (adapter idx2token={len(tok_adapt.idx2token)})")
    # ---- 构建数据集 ----
    train_ds = build_dataset(args, "train", tok)
    val_ds = build_dataset(args, "val", tok)
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True,
                              num_workers=int(args.num_workers), pin_memory=True,
                              collate_fn=multiphase_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False,
                            num_workers=int(args.num_workers), pin_memory=True,
                            collate_fn=multiphase_collate)
    logging.info(f"[Dataset] Train={len(train_ds)} | Val={len(val_ds)}")
    # ---- 模型构建 ----
    model = CT2Rep(args, tok_adapt).to(device)     # args 需携带模型结构字段(见 yaml)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"[Model] trainable params: {n / 1e6:.2f}M")

    # 论文: 视觉提取器 lr=5e-5, 其余 lr=1e-4, Adam(betas 0.9/0.99), StepLR(gamma 0.1)
    ve_params = [p for n, p in model.named_parameters()
                 if n.startswith("ctvit") or n.startswith("fc.")]
    ve_ids = {id(p) for p in ve_params}
    other_params = [p for n, p in model.named_parameters()
                    if not (n.startswith("ctvit") or n.startswith("fc."))]
    optimizer = torch.optim.Adam([
        {"params": ve_params,    "lr": float(args.lr_ve)},
        {"params": other_params, "lr": float(args.lr_ed)},
    ], betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.step_size), gamma=0.1)
    # ---- 断点恢复 ----
    ckpt_mgr = CkptManager(join(save_path, "ckpt"),
                           max_keep_epochs=int(getattr(args, "max_keep_epochs", 0)))
    start_epoch, best_val, no_improve = 1, float("inf"), 0
    if bool(getattr(args, "resume", False)):
        ckpt = ckpt_mgr.load_last(map_location=device)
        if ckpt is not None:
            start_epoch, best_val = resume_from_ckpt(ckpt, model, optimizer, scheduler)
            no_improve = int(ckpt.get("no_improve", 0))
            logging.info(f"[Resume] from epoch {start_epoch}, best_val={best_val:.6f}")

    csv_logger = EpochCSVLogger(save_path, "ct2rep_metrics.csv",
                                resume=bool(getattr(args, "resume", False)))
    # ---- 训练循环 ----
    start_time = time.time()
    for epoch in range(start_epoch, int(args.epoch) + 1):
        tr_nll = train_one_epoch(model, train_loader, optimizer, device, args, epoch)
        va = validate(model, val_loader, device, args, tok_adapt)
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
