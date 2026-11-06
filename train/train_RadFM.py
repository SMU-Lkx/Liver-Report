# train/train_RadFM.py
"""RadFM 复现训练（均衡重训版）：冻结视觉编码器，LoRA 微调 LLM（batch=1）。

与 train_CT2Rep.py 同构：每 epoch 训练 -> val 验证 -> save_best/early_stop。
类别均衡：异常病例按 abnormal_weight 加权采样（思想同 RadFM 官方 RadMD 构造时的
pos/neg balance——generative models are sensitive to data imbalances）。
选择指标：select_metric = val_nll(协议对齐) 或 val_distinct(反坍缩信号)，均记录进 CSV。
"""
import argparse, datetime, logging, os, re, sys, time
import setproctitle
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
import torch_npu
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.Dataset.Multiphase_dataset import MultiphaseReport, multiphase_collate, CharTokenizer
from Models.RadFM import LiverRadFM, LoRAAdapterShim, get_tokenizer, batch_to_inputs
from util.train_utils import (load_config, save_config, setup_seed, amp_context,
                              log_setup, CkptManager, EpochCSVLogger, resume_from_ckpt,
                              text_sampling)

# ---- 异常病例的文本启发式（仅用于采样加权；先剔否定再匹配，避免"未见异常"误报）----
RE_NEG = re.compile(r"(未见|未见明确|未见明显|无明确|没有)[^。；]*")
RE_LESION = re.compile(r"结节|肿块|团块|低密度|稍低密度|等密度|高密度|异常强化|强化灶|结石|扩张|积液|"
                       r"占位|病灶|增厚|钙化|密度减低|密度降低|密度不均|变形|淋巴结|术后改变")

def is_abnormal(text: str) -> bool:
    return bool(RE_LESION.search(RE_NEG.sub("", text)))


def build_dataset(args, split, tok):
    return MultiphaseReport(
        data_path=args.data, pid_csv=join(args.sampler, f"{split}_idx.csv"), tokenizer=tok,
        cache_dir=args.cache_dir, cache_shape=tuple(args.shape), cache_norm=args.norm,
        mask_mode=args.mask_mode, build_if_missing=False)


def train_one_epoch(model, loader, optimizer, args, epoch, tok, img_toks):
    model.train()
    run_loss, run_acc, n = 0.0, 0.0, 0
    pbar = tqdm(loader, desc=f"RadFM Ep {epoch:02d}/{args.epochs}")
    for batch in pbar:
        inputs = batch_to_inputs(batch, tok, img_toks, args.phases,
                                 max_len=int(args.prompt_max_len))
        with amp_context(bool(args.use_bf16)):
            loss, acc = model(inputs)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), float(args.max_grad_norm))
        optimizer.step()
        run_loss += float(loss.item())
        run_acc += float(acc.item()) if torch.is_tensor(acc) else float(acc)
        n += 1
        pbar.set_postfix(loss=f"{run_loss/n:.4f}", acc=f"{run_acc/n:.4f}")
    return run_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, args, epoch, save_path, tok, img_toks):
    """val_nll(全量前向) + 前 save_sample_num 例生成 -> distinct 数(反坍缩信号) + 样例归档。"""
    model.eval()
    nll_sum, n = 0.0, 0
    gens = []
    for batch in tqdm(loader, desc="Val", leave=False):
        if 0 < int(args.val_max_batches) <= n:
            break
        inputs = batch_to_inputs(batch, tok, img_toks, args.phases,
                                 max_len=int(args.prompt_max_len))
        with amp_context(bool(args.use_bf16)):
            loss, _ = model(inputs)
        nll_sum += float(loss.item()); n += 1
        if len(gens) < int(args.save_sample_num):
            new_tokens = model.generate(inputs, max_new_tokens=int(args.max_new_tokens))
            gen = tok.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            gens.append({"pid": batch["pid"][0], "gen": gen, "ref": batch["text"][0]})
    distinct = len(set(g["gen"] for g in gens))
    if gens:
        text_sampling(gens, save_path, epoch)          # -> out_sample/val_epXXXX.*
        logging.info(f"[ValSample ep{epoch}] distinct={distinct}/{len(gens)} | {gens[0]['gen'][:60]}...")
    return {"nll": nll_sum / max(n, 1), "distinct": distinct, "evaluated": n}


def main():
    setproctitle.setproctitle("lkxRadFM-balance")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",  # todo
                        default="/home/longkaixing/Liver/config/RadFM/261103_RadFM.yaml")
    args, _ = parser.parse_known_args()
    config = load_config(args.cfg)
    for key, value in config.items():
        setattr(args, key, value)

    save_path = join(args.save_dir, args.save_name)
    os.makedirs(save_path, exist_ok=True)
    log_setup(save_path, "radfm", name=args.save_name)
    logging.info(f"[Config] saved to {save_config(args.cfg, save_path, args.save_name)}")

    args.device = f"npu:{args.device}" if isinstance(args.device, int) else args.device
    torch_npu.npu.set_device(args.device)
    device = torch.device(args.device)
    setup_seed(int(args.seed))

    tok, img_toks = get_tokenizer(args.lang_model_path)
    char_tok = CharTokenizer.load(join(args.cache_dir, "vocab.json"))

    train_ds = build_dataset(args, "train", char_tok)
    val_ds = build_dataset(args, "val", char_tok)
    # ---- 类别均衡采样：异常病例加权（同 RadMD 均衡思想的数据级方案）----
    weights = [float(args.abnormal_weight) if is_abnormal(r["text"]) else 1.0
               for r in train_ds.records]
    n_ab = sum(1 for w in weights if w > 1.0)
    logging.info(f"[Balance] train={len(weights)} | 异常(加权)={n_ab} ({n_ab/len(weights):.1%}) "
                 f"| abnormal_weight={args.abnormal_weight}")
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights),
                                    num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=1, sampler=sampler,
                              num_workers=int(args.num_workers), collate_fn=multiphase_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=int(args.num_workers), collate_fn=multiphase_collate)
    logging.info(f"[Dataset] Train={len(train_ds)} (加权采样) | Val={len(val_ds)}")

    model = LiverRadFM(vars(args)).to(device)
    model.train()
    logging.info(f"[Model] trainable={sum(p.numel() for p in model.trainable_parameters())/1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=float(args.lr), weight_decay=0.0)
    epochs = int(args.epochs)
    total_steps = epochs * len(train_loader)
    warm = int(0.03 * total_steps)

    def lr_lambda(step):
        if step < warm:
            return step / max(1, warm)
        p = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + torch.cos(torch.tensor(3.1415926 * p)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_mgr = CkptManager(join(save_path, "ckpt"),
                           max_keep_epochs=int(args.max_keep_epochs),
                           save_freq=int(args.save_freq))
    adapter = LoRAAdapterShim(model.model.lang_model)
    csv_logger = EpochCSVLogger(save_path, "radfm_metrics.csv", resume=bool(args.resume))

    lower_better = bool(args.select_metric == "val_nll")   # val_distinct 越大越好
    best_val = float("inf") if lower_better else 0.0
    start_epoch, no_improve = 1, 0
    if bool(args.resume):
        ckpt = ckpt_mgr.load_last(map_location=device)
        if ckpt is not None:
            start_epoch, best_val = resume_from_ckpt(ckpt, adapter, optimizer, scheduler)
            no_improve = int(ckpt.get("no_improve", 0))
            logging.info(f"[Resume] from epoch {start_epoch}, best({args.select_metric})={best_val}")

    start_time = time.time()
    for epoch in range(start_epoch, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, args, epoch, tok, img_toks)
        va = validate(model, val_loader, args, epoch, save_path, tok, img_toks)
        score = va["nll"] if lower_better else va["distinct"]
        improved = (score < best_val) if lower_better else (score > best_val)
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        logging.info(f"[Ep {epoch:02d}] loss={tr_loss:.4f} | val_nll={va['nll']:.4f} "
                     f"val_distinct={va['distinct']}/{len(range(min(int(args.save_sample_num), va['evaluated'])))} "
                     f"| lr={lr_now:.2e} | {elapsed}")
        csv_logger.add_epoch(epoch, train_loss=f"{tr_loss:.4f}", val_nll=f"{va['nll']:.4f}",
                             val_distinct=va["distinct"], lr=lr_now)

        ckpt_mgr.save_epoch(epoch, adapter, optimizer=optimizer, scheduler=scheduler,
                            best_val=best_val, extra={"no_improve": no_improve})
        if improved:
            best_val, no_improve = float(score), 0
            ckpt_mgr.save_best(adapter, extra={"epoch": epoch, args.select_metric: best_val})
            logging.info(f"  -> BEST ({args.select_metric}={best_val:.4f})")
        else:
            no_improve += 1
            if 0 < int(args.early_stop) <= no_improve:
                logging.info(f"Early stopping after {args.early_stop} epochs.")
                break

    logging.info(f"[Done] best {args.select_metric}={best_val:.4f} | output: {save_path}")


if __name__ == "__main__":
    main()