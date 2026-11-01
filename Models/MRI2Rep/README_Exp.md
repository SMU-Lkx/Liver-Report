# MRI2Rep 复现说明（Models/MRI2Rep/README_Exp.md）

> 文献：MRI2Rep: Autoregressive Structured Report Generation for 3D Liver MRI（MICCAI 2026）  
> 官方代码：https://github.com/Alena-Xinran/MRI2Rep  
> 复现目标：四期相肝脏 CT → RLC 结构化标签（24-token 闭集序列）→ 确定性模板渲染中文报告  
> 实验日期：2026-10-29 ~ 2026-11-01  

## 1. 代码结构

```
Models/MRI2Rep/
├── model/
│   ├── RLC.py              # 中文报告 → RLC 标签（官方流程中文适配：Step1/2/3、三轮投票、
│   │                       #   recover 2/3、低一致五轮 rerun、Tier 1-4 分级接受）
│   ├── backbone.py         # 3D CNN（Conv3d-IN-GELU×4，in_ch 可配），官方同构
│   ├── recon.py            # NCAD2V 重建预训练模型（trilinear 双卷积解码，MRI2Rep pretrain_pv 同构）
│   ├── report_generator.py # Backbone + visual projector + Transformer 自回归解码器
│   └── losses.py           # masked MSE + avg_pool3d SSIM（预训练损失，NPU版）
├── infer/
│   ├── infer_MRI2Rep.py    # test 推理：best ckpt → 逐例生成 RLC 序列 + 逐例 NLL → eval/rlc_pred/
│   └── render_report.py    # RLC 序列 → 中文报告（自包含：词形/句式池/容错渲染）
└── README_Exp.md           # 本文件
```

训练入口 `train/train_MRI2Rep.py`，评估入口 `eval/eval_MRI2Rep.py`。

数据侧：`data/Dataset/Multiphase_dataset.py` 的 `MultiphaseRLC`（复用 CT2Rep 影像缓存 + 独立 RLC 标签缓存）。

## 2. 相对官方代码的适配修改

| # | 适配点 | 原因 | 位置 |
|---|---|---|---|
| 1 | 输入 4 期相 + mask 五通道 `[NC,Art,Ven,Del,MASK]`，网格 224×224×160 | 官方仅 ART/PV/mask 三通道 192×192×96 是其数据选择，架构本身不限期相；与 Stage-0 CT2Rep 缓存同网格可直接复用 | `config/MRI2Rep/261029_MRI2Rep_4p.yaml` |
| 2 | NCAD2V 预训练 stem 按期相索引复制、mask 卷积核零初始化、其余 strict-load | 预训练 backbone 为四通道，须适配五通道输入 | `model/report_generator.py::load_pretrained_backbone` |
| 3 | **RLC 提示词中文化 + DeepSeek 后端**（官方 Claude-3.5） | 中文报告 + 项目已有 litellm/DeepSeek 管线；论文 Table 1 已验证后端间 RLC 保留率一致 | `model/RLC.py` |
| 4 | 未实现 `aux_lesion_head`，`aux_weight: 0.0` | 辅助病灶分类头为可选项，序列 CE + 类别加权已覆盖主损失 | `train/train_MRI2Rep.py` |
| 5 | 训练目标 = per-class 加权 CE（effective number, β=0.999）+ word dropout + scheduled sampling；双流水采样器（病灶流 + 20% 无病灶流，稀有病例过采样 0.4） | 与官方 oversampling + class-reweighted loss 同构 | `train/train_MRI2Rep.py` |
| 6 | 最佳模型按 **病灶阳性子集 pair-F1** 选择，且 `epoch≥15 ∧ F1≥0.02` 才允许保存 best | 无病灶占多数，全集指标虚高；防早期坍塌模型被存为 best | `train/train_MRI2Rep.py` |
| 7 | **中文渲染模板，只生成"检查所见"**（无"考虑/拟/建议"等诊断意见口吻） | 本项目只评估肝脏所见层（liver_desc 层）；官方句式含 "consistent with definite HCC" 轻结论尾，此处刻意剥离 | `infer/render_report.py` |
| 8 | 评估 = 官方结构化指标（主）+ corpus 级 BLEU/ROUGE（辅） | 结构化指标为主；文本指标双方同渲染器同分布 | `eval/eval_MRI2Rep.py` |

未改动官方方法本体：RLC 三步提取与三轮投票、recover/rerun/Tier 1-4 接受规则、
24-token 词表与序列语法、自回归编码器-解码器、greedy 解码、确定性模板渲染。

## 3. RLC 标签构建（LLM 提取管线）

```bash
python Models/MRI2Rep/RLC.py extract --splits train val test   # Step1/2 + Step3×3 投票
python Models/MRI2Rep/RLC.py recover                           # 接受 3/3 与可恢复 ≥2/3
python Models/MRI2Rep/RLC.py rerun --rounds 5                  # 低一致病例重跑 Step3
python Models/MRI2Rep/RLC.py recover && python Models/MRI2Rep/RLC.py build
```

- 后端：`deepseek/deepseek-chat`（API key 读环境变量或 ~/.deepseek_api_key）
- 产出：`RLC_report/logs/accurate_summary.csv`（id, tier, wc, sequence）
- 覆盖：train 1633 中接受 **1622**（Tier1=1473 / 2=38 / 3=53 / 4=58，rejected_diff=11）；
  val 204、test 205 全量提取；test 最终入评 **201**（RLC 管线排除 4 例）

## 4. 训练

```bash
cd /home/longkaixing/Liver && nohup python -m train.train_MRI2Rep   --cfg config/MRI2Rep/261029_MRI2Rep_4p.yaml > log_261029_MRI2Rep_4p.txt 2>&1 &
```

- 配置：`config/MRI2Rep/261029_MRI2Rep_4p.yaml`（in_ch=5, base=32, d_model=256, dec_layers=4,
  batch=4, lr_backbone=5e-6, lr_decoder=5e-5, cosine, bf16, NPU）
- 预训练 backbone：`model_save/261016_pretrain_NCAD2V/backbone_NCAD2V_best.pt`（NC+A+D→V 重建）
- 成本：782 步/epoch（病灶分层采样），约 40 min/epoch
- 结果：best epoch **33**（val_pair_f1_lesion1p=**0.0706**），epoch 45 early stop（patience=10）
- 训练期验证附带逐例 RLC 转储：`model_save/261029_MRI2Rep_4p/out_sample/val_pred.csv`

## 5. 推理 / 渲染 / 评估

```bash
# 1) test 推理（NPU）→ eval/rlc_pred/test_pred.csv（RLC 序列 + 逐例 NLL + 结构化指标日志）
python -m Models.MRI2Rep.infer.infer_MRI2Rep   --cfg config/MRI2Rep/261029_MRI2Rep_4p.yaml   --ckpt model_save/261029_MRI2Rep_4p/ckpt/best_model.pt   --out model_save/261029_MRI2Rep_4p/eval/rlc_pred/test_pred.csv

# 2) 渲染（CPU 秒级）→ rlc_pred/test_reports.csv + eval/out_sample/test_all.txt
python -m eval.render_report   --pred-csv model_save/261029_MRI2Rep_4p/eval/rlc_pred/test_pred.csv --split-name test

# 3) 评估（CPU）→ eval/test_metrics.json / eval_csv/test_cases.csv
python -m eval.eval_MRI2Rep   --reports-csv model_save/261029_MRI2Rep_4p/eval/rlc_pred/test_reports.csv --tag test
```

渲染说明：GT/预测同渲染器、seed=int(pid)（变体可复现）；模型输出残缺三元组（提前 EOS）
时按评估端同款滑窗容错 salvage 渲染，文首加"【序列不完整…】"标记，运行日志打印
`容错触发: {truncated: N}`——N 即解码早期终止频率，可作错误分析指标。

差异分析：敏感性偏低但特异性接近，病灶刻画 F1 差距明显。主要量级因素——batch 4 vs 官方 12、
训练样本 1622 vs 3830、本队列为增强 CT 病灶富集人群（GT 病灶阳性率约 58% vs 官方分层覆盖）。
Liver-Acc 高于官方与本队列肝硬化标注较集中有关。BLEU/ROUGE 为模板主导辅指标，仅作 NLG 层面对齐。

## 6. 实验日志路径

| 内容 | 路径 |
|---|---|
| 训练日志 | `model_save/261029_MRI2Rep_4p/261029_MRI2Rep_4p_rlc.txt`（log_setup） |
| 训练曲线 | `model_save/261029_MRI2Rep_4p/train_csv/rlc_metrics.csv` |
| 权重 | `model_save/261029_MRI2Rep_4p/ckpt/{best_model.pt(epoch33), last_model.pt, epoch*.pt}` |
| 推理产物 | `model_save/261029_MRI2Rep_4p/eval/rlc_pred/{test_pred.csv, test_reports.csv}` |
| 评估产物 | `model_save/261029_MRI2Rep_4p/eval/{test_metrics.json, eval_csv/test_cases.csv, out_sample/test_all.txt}` |
| RLC 提取日志 | `Models/MRI2Rep/RLC_report/logs/`（batch_run_*、accurate_summary.csv、RLC.log） |
| 配置存档 | `model_save/261029_MRI2Rep_4p/eval/rlc_pred/infer_*.yaml` 及 save_dir 下 config 副本 |
