# CT2Rep 复现说明（Models/CT2Rep/README_Exp.md）

> 文献：CT2Rep: Automated Radiology Report Generation for 3D Medical Imaging（MICCAI 2024）
> 官方代码：https://github.com/ibrahimethemhamamci/CT2Rep
> 复现目标：四期相肝脏 CT → 自回归生成中文 report_desc（自由文本 baseline）
> 实验日期：2026-10-20 ~ 2026-10-21

## 1. 代码结构

```
Models/CT2Rep/
├── model/
│   ├── CT2Rep.py
│   ├── ctvit.py            # 3D ViT（spatial+causal transformer）+ VQ 量化
│   └── attention.py        # Attention/Transformer/ContinuousPositionBias
├── modules/                # 文本侧（官方 R2Gen 代码）
│   ├── encoder_decoder.py  # Transformer encoder + RM decoder + MCLN
│   ├── att_model.py        # AttModel（采样/beam/greedy 逻辑）
│   ├── caption_model.py    # beam_search 基础设施
│   └── utils.py            # repeat_tensors 等
└── README_Exp.md           # 本文件
```

训练入口 `train/train_CT2Rep.py`，评估入口 `eval/eval_CT2Rep.py`。

## 2. 相对官方代码的适配修改

| # | 适配点 | 原因 | 位置 |
|---|---|---|---|
| 1 | CTViT 输入 `channels=4` | 四期相拼接（官方单通道胸 CT） | `model/CT2Rep.py` 构造参数 |
| 2 | `image_size=224, patch_size=16, temporal_patch_size=8` | 我们的 (4,224,224,160) → token 网格 20×14×14=3920（官方 480/24/12 → 20×20×20） | 同上 |
| 3 | `AvgPool3d((20,14,14))` | 官方 VisualExtractor 硬编码 kernel=20，须匹配新 token 网格 | `model/CT2Rep.py` |
| 5 | **NPU patch**：删除 `attention.py` 中 `device=torch.device('cuda')` 硬编码（encode/decode 均用 `tokens.device`） | NPU 无 CUDA | `model/attention.py` |
| 6 | **tokenizer 整体替换**：官方 R2Gen 词级 tokenizer → `CharTokenizerAdapter`（暴露 `idx2token`/`decode`/`decode_batch`，id 约定 PAD=0/BOS=1/EOS=2 与官方兼容） | 中文无空格，词级不可用 | `model/CT2Rep.py` |
| 7 | 数据/指标/训练循环按本项目规范重写（MultiphaseReport + multiphase_collate + CkptManager + sacrebleu） | 标准化复现流程 | `train/`、`eval/` |
| 8 | HU 窗统一 [-1000,1000]→x*2-1 | 对比实验统一口径（官方 [-1000,200]） | `model/CT2Rep.py::_encode` |

未改动官方方法本体：CTViT 双轴注意力、VQ、RM+MCLN 解码器、greedy 采样、
Adam(lr_ve=5e-5/lr_ed=1e-4, betas 0.9/0.99) + StepLR(γ=0.1)、batch=1、从零训练。

## 3. 训练

```bash
cd /home/longkaixing/Liver && nohup .../python3.9 train/train_CT2Rep.py \
  --cfg config/CT2Rep/261020_CT2Rep.yaml > train_ct2rep.log 2>&1 &
```

- 配置：`config/CT2Rep/261020_CT2Rep.yaml`
- 规模：train 1633 / val 204 / test 205（split_idx/260925_SSL）
- 成本：batch=1 → 1633 步/epoch

## 4. 评估

```bash
nohup .../python3.9 eval/eval_CT2Rep.py --cfg ... --ckpt model_save/261020_CT2Rep/ckpt/best_model.pt &
```

- 输出：model_save/{save_name}/：
```
eval/
├── test_metrics.json / test_metrics.csv     # 汇总指标
├── eval_csv/test_cases.csv                  # 逐例 (pid, nll, gen_len, ref_len)
└── out_sample/{pid}/{gen.txt, diff.html, meta.json}
    └── test_ep{best_model:04d}_all.txt                # GEN/REF 成对全文索引
```

## 5. 实验日志路径

| 内容 | 路径 |
|---|---|
| 训练日志 | `model_save/261020_CT2Rep/261020_CT2Rep_ct2rep.txt`（log_setup） |
| 训练曲线 | `model_save/261020_CT2Rep/train_csv/ct2rep_metrics.csv` |
| 权重 | `model_save/261020_CT2Rep/ckpt/{best_model.pt(epoch7), last_model.pt, epoch*.pt}` |
| 评估产物 | `model_save/261020_CT2Rep/eval/`（见上节） |
| 配置存档 | `model_save/261020_CT2Rep/261020_CT2Rep_*.yaml` |
