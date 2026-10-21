# eval/eval_CT2Rep.py
"""
CT2Rep baseline 测试集评估: 指定 ckpt -> test 全体逐例生成 + 逐例损失 + corpus级文本指标（整个测试集所有 n-gram 匹配数加总再算，
而不是每个per-case 分数取平均——模板语料下 per-case 平均会被大量"正常模板全对"的短例虚高，corpus 级更诚实）.
输出:
test_metrics.json / test_metrics.csv / test_cases.csv / out_sample/test_epXXXX.txt+html
诊断指标 (Sen/Spe/mAUC/Slot-F1): 待 Layer1 解析器就绪后统一补充.
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time
from os.path import join
from pathlib import Path

# --- NPU environment ---
os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
os.environ.setdefault("TMPDIR", "/var/tmp/longkaixing")
CANN_PYTHON_PATH = "/home/HwHiAiUser/Ascend/ascend-toolkit/8.2.RC1/python/site-packages"
os.environ["PYTHONPATH"] = CANN_PYTHON_PATH + (":" + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else "")
sys.path.insert(0, CANN_PYTHON_PATH)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_npu
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.Dataset.Multiphase_dataset import CharTokenizer, MultiphaseReport, multiphase_collate
from Models.CT2Rep.model.CT2Rep import CT2Rep, CharTokenizerAdapter
from util.train_utils import load_config, setup_seed, amp_context, text_sampling
from util.text_utils import sacre_bleu_zh, char_rouge_l
from util.val_utils import eval_outputs

PAD = 0


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str,
        default="/home/longkaixing/Liver/config/CT2Rep/261020_CT2Rep.yaml")
    parser.add_argument("--ckpt", type=str,
        default="/home/longkaixing/Liver/model_save/261020_CT2Rep/ckpt/best_model.pt")
    parser.add_argument("--device", type=str,
        default='npu:0')
    cli, _ = parser.parse_known_args()

    config = load_config(cli.cfg)
    args = argparse.Namespace(**{**vars(cli), **config})

    torch_npu.npu.set_device(args.device)
    device = torch.device(args.device)
    setup_seed(int(args.seed))

    tok = CharTokenizer.load(join(args.cache_dir, "vocab.json"))
    tok_adapt = CharTokenizerAdapter(tok)

    test_ds = MultiphaseReport(
        data_path=args.data, pid_csv=join(args.sampler, f"test_idx.csv"),
        tokenizer=tok, cache_dir=args.cache_dir, max_len=int(args.max_len),
        mask_mode=args.mask_mode,
        verify_paths=args.verify_paths)

    loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=int(args.num_workers), pin_memory=True,
        collate_fn=multiphase_collate)
    print(f"[Test] {len(test_ds)} cases")

    model = CT2Rep(args, tok_adapt).to(device)
    ckpt = torch.load(cli.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    epoch = int(ckpt.get("epoch", 0))
    print(f"[Ckpt] {cli.ckpt} | epoch={epoch} val_nll={ckpt.get('val_nll')}")

    save_path = str(Path(cli.ckpt).resolve().parents[1])   # model_save/{save_name}/
    eval_dir = join(save_path, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    # ---- 评估循环 ----
    out_root = join(eval_dir, "out_sample")
    os.makedirs(out_root, exist_ok=True)
    all_txt = join(out_root, f"test_ep{epoch:04d}_all.txt")

    records, rows = [], []
    t0 = time.time()
    pbar = tqdm(loader, desc="Test")
    for batch in pbar:
        pid = batch["pid"][0]
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with amp_context(bool(args.use_bf16)):
            out = model(x, y)
            nll = float(F.nll_loss(out.reshape(-1, out.size(-1)),
                                   y[:, 1:].reshape(-1), ignore_index=PAD,
                                   reduction="sum").item())
            n_valid = int((y[:, 1:] != PAD).sum().item())
            gen_path = join(out_root, str(pid), "gen.txt")
            if os.path.exists(gen_path):                 # ← 断点续跑: 已生成则跳过采样
                with open(gen_path, encoding="utf-8") as f:
                    gen = f.read()
            else:
                seq = model.generate(x)
                gen = tok_adapt.decode(seq[0].cpu().tolist())
        ref = tok_adapt.decode(y[0].cpu().tolist())

        # 逐例保存: gen.txt/diff.html/meta.json
        text_sampling([{"pid": pid, "gen": gen, "ref": ref}], eval_dir, epoch=epoch, prefix="test")
        # all.txt (GEN/REF 成对便于比较)
        with open(all_txt, "a", encoding="utf-8") as f:
            f.write(f"===== pid={pid} | gen_len={len(gen)} ref_len={len(ref)} =====\n")
            f.write(f"[GEN]\n{gen}\n\n[REF]\n{ref}\n\n\n")

        records.append({"pid": pid, "gen": gen, "ref": ref})
        rows.append({"pid": pid, "nll": round(nll / max(n_valid, 1), 4),
                     "gen_len": len(gen), "ref_len": len(ref)})
        pbar.set_postfix(nll=f"{rows[-1]['nll']:.3f}")
    gen_time = time.time() - t0

    # ---- corpus 级指标 ----
    rc = [list(r["ref"]) for r in records]
    hc = [list(r["gen"]) for r in records]
    refs_l = [r["ref"] for r in records]
    hyps_l = [r["gen"] for r in records]
    st = text_sampling(records, eval_dir, epoch=epoch, prefix="test")
    metrics = {
        "cfg": cli.cfg, "ckpt": cli.ckpt, "epoch": epoch,
        "n_test": len(records),
        "test_nll": round(sum(r["nll"] * r["ref_len"] for r in rows) / max(sum(r["ref_len"] for r in rows), 1), 4),
        "char_bleu2": round(sacre_bleu_zh(refs_l, hyps_l, max_n=2), 4),
        "char_bleu4": round(sacre_bleu_zh(refs_l, hyps_l, max_n=4), 4),
        "char_rouge_l": round(char_rouge_l(rc, hc), 4),
        "avg_gen_len": st["avg_gen_len"], "avg_ref_len": st["avg_ref_len"],
        "rep_ratio": st["rep_ratio"],
        "total_time_s": round(gen_time, 1),
        "time_s_per_case": round(gen_time / max(len(records), 1), 1),
    }

    # ---- 输出: json + 汇总csv + 逐例csv ----
    out_root = eval_outputs(metrics, rows, eval_dir, tag='test')
    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    print(f"-> {out_root}/test_metrics.json, eval_csv/, out_sample/")


if __name__ == "__main__":
    main()
