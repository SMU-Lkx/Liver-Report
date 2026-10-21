# util/val_utils.py
import torch
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
import numpy as np
import json, csv, os
from os.path import join


def evaluate_multilabel(model, dataloader, device, diagnosis_list, auc_threshold=0.6):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch["image_feat"].to(device))
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["diag_labels"].cpu().numpy())

    all_probs = np.concatenate(all_probs)      # [N, C]
    all_labels = np.concatenate(all_labels)    # [N, C]
    preds = (all_probs > 0.5).astype(int)

    # ---------- 原有全局指标 ----------
    micro_f1 = f1_score(all_labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
    macro_precision = precision_score(all_labels, preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, preds, average="macro", zero_division=0)

    # Specificity = TN / (TN + FP) = 1 - FP/(TN+FP)
    # 对多标签，macro_specificity 需要对每个标签分别计算
    specificities = []
    for i in range(all_labels.shape[1]):
        tn = ((preds[:, i] == 0) & (all_labels[:, i] == 0)).sum()
        fp = ((preds[:, i] == 1) & (all_labels[:, i] == 0)).sum()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    macro_specificity = np.mean(specificities)

    # Exact Match Accuracy：所有标签都预测正确的样本比例
    exact_match_acc = accuracy_score(all_labels, preds) * 100  # 注意：这个对多标签是"所有位都对"

    # ---------- 每类 AUC（新增）----------
    per_class_auc = {}
    bad_classes = []

    for i, name in enumerate(diagnosis_list):
        y_true, y_prob = all_labels[:, i], all_probs[:, i]
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        auc = roc_auc_score(y_true, y_prob)
        per_class_auc[name] = auc
        if auc < auc_threshold:
            bad_classes.append((name, auc, int(y_true.sum())))

    mean_auc = np.mean(list(per_class_auc.values()))

    # ---------- 打印 ----------
    print(f"\n{'='*60}")
    print(f"【AUC < {auc_threshold} 的类别】")
    bad_classes.sort(key=lambda x: x[1])
    for name, auc, n_pos in bad_classes:
        print(f"  {name:<15} AUC={auc:.4f}  (正样本: {n_pos})")

    valid = sorted(per_class_auc.items(), key=lambda x: -x[1])
    print(f"\n【AUC Top 3】")
    for name, auc in valid[:3]:
        print(f"  {name:<15} AUC={auc:.4f}")

    return {
        "exact_match_acc": exact_match_acc,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_specificity": macro_specificity,
        "mean_auc": mean_auc,
        "per_class_auc": per_class_auc,  # 新增，供分析用
    }


def eval_outputs(metrics, rows, save_path, tag="test"):
    """
    评估结果统一落盘 (所有模型的评估脚本共用):
      {save_path}/{tag}_metrics.json       汇总指标
      {save_path}/eval_csv/{tag}_metrics.csv  汇总指标(表格)
      {save_path}/eval_csv/{tag}_cases.csv   逐例记录
    Args:
        metrics: dict              汇总指标 (键=列名)
        rows:    list[dict]        逐例记录, 如 [{"pid":.., "nll":.., ...}]
        tag:     str               文件名前缀, 如 "test" / "val" / "test_seed42"
    """
    csv_dir = join(save_path, "eval_csv")
    os.makedirs(csv_dir, exist_ok=True)

    with open(join(save_path, f"{tag}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)

    with open(join(csv_dir, f"{tag}_metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        w.writeheader()
        w.writerow(metrics)

    if rows:
        with open(join(csv_dir, f"{tag}_cases.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
