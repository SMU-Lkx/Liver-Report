# Liver-Report：肝病影像报告生成

基于私有肝脏影像数据（CT），系统性复现并对比现有三维医学影像报告生成方法，为肝病报告生成模型建立基线。

## 目录结构

```
Liver/
├── config/                    # 各模型配置文件
│   └── {YYMMDD}_{模型名称}_{可选}.yaml
├── data/                      # 数据相关（私有数据，不进仓库）
│   └── Dataset/               # 数据集加载函数，命名 XXX_dataset.py
├── eval/                      # 评估脚本，命名 eval_{模型名称}.py
├── model_save/                # 模型权重（不进仓库）
│   └── {YYMMDD}_{模型名称}_{可选}/ 
│      ├── eval/               # 评估结果（val_utils.py/eval_outputs）
│      │   └── out_sample/     # 评估过程生成的报告文本（train_utils.py/text_sampling）
│      └── train_csv/          # 训练过程的损失、指标记录（EpochCSVLogger）
├── Models/                    # 待对比 / 复现的方法
│   ├── Astra/
│   ├── CT2Rep/
│   ├── M3D/
│   ├── Merlin/
│   ├── MRI2Rep/
│   └── RadFM/
├── train/                     # 训练脚本，命名 train_{模型名称}.py
├── util/                      # 工具函数
│   ├── text_utils.py          # 字符评估工具：ngram 计数、字符级 BLEU 等
│   ├── train_utils.py         # 配置加载/保存、CSV 训练记录器、随机种子、日志记录、断点保存恢复等
│   └── val_utils.py           # 评估结果保存工具
├── README.md
└── 肝病报告生成-对比实验.xlsx    # 对比实验结果记录表
```

> **注意**：`data/`、`model_save/` 及各模型权重均通过 `.gitignore` 排除，仓库只同步代码与配置。

## 对比方法

| 方法 | 论文 | 官方代码 | 复现状态 |
|------|------|----------|----------|
| **M3D** | [M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models](https://arxiv.org/abs/2404.00578) | [BAAI-DCAI/M3D](https://github.com/BAAI-DCAI/M3D) | 待复现 |
| **CT2Rep** | [CT2Rep: Automated Radiology Report Generation for 3D Medical Imaging](https://arxiv.org/abs/2403.06801) (MICCAI 2024) | [ibrahimethemhamamci/CT2Rep](https://github.com/ibrahimethemhamamci/CT2Rep) | 已复现 |
| **Merlin** | [Merlin: a computed tomography vision–language foundation model and dataset](https://arxiv.org/abs/2406.06512) (Nature 2026) | [StanfordMIMI/Merlin](https://github.com/StanfordMIMI/Merlin) | 待复现（zero-shot） |
| **RadFM** | [Towards Generalist Foundation Model for Radiology by Leveraging Web-scale 2D&3D Medical Data](https://doi.org/10.1038/s41467-025-62385-7) (Nature Communications 2025) | [chaoyi-wu/RadFM](https://github.com/chaoyi-wu/RadFM) | 待复现 |
| **MRI2Rep** | [MRI2Rep: Autoregressive Structured Report Generation for 3D Liver MRI](https://arxiv.org/abs/2606.25279) (MICCAI 2026) | [Alena-Xinran/MRI2Rep](https://github.com/Alena-Xinran/MRI2Rep) | 正在复现 |
| **Astra** | [Astra: a generalizable report generation foundation model for 3D computed tomography](https://arxiv.org/abs/2605.31437) | [zh-Wang-Med/Astra](https://github.com/zh-Wang-Med/Astra) | 待复现（zero-shot） |

> 标注 *zero-shot* 的方法直接在私有数据上推理评估，不微调；其余方法需迁移/复现训练流程。

## 复现安排

| 方法        | 复现方式                                               | 负责人 | 时间                  |
|-------------|--------------------------------------------------------|--------|-----------------------|
| **M3D**     | 利用公开权重，冻结视觉编码器，训练perceiver + LLM-LoRA | 李传浩 | 2026.10.23~2026.11.12 |
| **CT2Rep**  | 端到端从头训练                                         | 龙楷兴 | 2026.10.20~2026.10.22 |
| **Merlin**  | 利用公开权重，冻结视觉编码器，训练adapter + LoRA       | 杨兰   | 2026.11.01~2026.11.12 |
| **RadFM**   | 利用公开权重，冻结视觉编码器，训练perceiver + LLM-LoRA | 杨兰   | 2026.11.13~2026.11.22 |
| **MRI2Rep** | 端到端从头训练                                         | 龙楷兴 | 2026.10.23~2026.11.01 |
| **Astra**   | 利用公开权重，冻结视觉编码器，训练perceiver + LLM-LoRA | 杨兰   | 2026.10.23~2026.11.01 |


## 复现进度

详细指标记录见 `肝病报告生成-对比实验.xlsx`：

| 方法    | BLEU2 ↑ | BLEU4 ↑ | ROUGE-L ↑ | Sen ↑ | Spe ↑ | mAUC ↑ | LLM-Eval ↑ |
|------   |---------|---------|-----------|-------|-------|--------|------------|
| M3D     |         |         |           |       |       |        |            |
| CT2Rep  | 0.6217  | 0.5637 | 0.6718 | | | | |
| Merlin  |         | | | | | | |
| RadFM   |         | | | | | | |
| MRI2Rep |         | | | | | | |
| Astra   |         | | | | | | |

## 使用约定

- **配置文件**：`config/{YYMMDD}_{模型名称}_{可选}.yaml`，如 `260915_Merlin_IntraPhase.yaml`
- **模型输出**：`model_save/{YYMMDD}_{模型名称}_{可选}/`，权重与日志统一放此处
- **数据路径**：通过配置文件指定
- **评估指标**：文本生成指标统一调用 `util/text_utils.py`；训练入口统一使用 `util/train_utils.py` 中的记录器与断点管理

## 协作规范
1. `master` 分支为保护分支，成员不直接推送，通过 Pull Request 合并
2. 每个方法在 `Models/{方法名}/` 下独立开发，复现完成后在对应目录补充 `README_Exp.md`，记录：
   - 复现步骤
   - 复现指标与论文指标对比、实验日志路径
3. 提交信息简述改动内容，如 `add: CT2Rep 数据加载适配多期相CT`
