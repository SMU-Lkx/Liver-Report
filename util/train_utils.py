# util/train_utils.py
"""
训练实用工具：配置保存、配置加载、断点保存、断点恢复、损失函数、损失写入csv管理类、文本生成可视化
"""
import os, csv, yaml, shutil, datetime, logging, re, json
from os.path import join
from collections import OrderedDict
from easydict import EasyDict
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch_npu
import difflib


FIRST_PRINT = True # 只打印一次的flag

# -------- 加载配置文件 --------
def load_config(config_path):
    """
    载入配置文件，返回字典形式。
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return EasyDict(config)

# -------- 保存配置文件 --------
def save_config(config_path, save_dir, savename='config'):
    """
    将配置文件保存为 YAML 格式，文件名包含当前时间。
    :param config_path: 配置字典的路径。
    :param save_dir: 保存目录。
    :param savename: 文件名前缀，默认为 "config"。
    """
    # 获取当前时间并格式化为字符串
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 构造文件名
    filename = f"{savename}_{current_time}.yaml"
    # 构造完整的保存路径
    save_path = os.path.join(save_dir, filename)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    shutil.copy(config_path, save_path)

    return save_path


class EpochCSVLogger:
    """
    极简Epoch级CSV记录器，支持断点恢复覆盖
    """
    def __init__(self, save_path, filename, resume=False):
        save_dir = join(save_path, 'train_csv')
        os.makedirs(save_dir, exist_ok=True)
        self.filepath = join(save_dir, filename)
        self.resume = resume and os.path.exists(self.filepath)

        # 存储所有数据：OrderedDict保证epoch顺序
        self.data = OrderedDict()

        # 恢复模式：加载已有数据
        if self.resume:
            self._load()

    def _load(self):
        """加载已有CSV数据到内存"""
        with open(self.filepath, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epoch = int(row['epoch'])
                # 删除epoch列，其余作为该epoch的数据
                row.pop('epoch')
                self.data[epoch] = row

    def add_epoch(self, epoch, **kwargs):
        """
        记录/覆盖一个epoch的数据

        Args:
            epoch: 轮次（主键）
            **kwargs: 任意列名和值
        """
        # 字符串化所有值，保持原样
        row_data = {k: str(v) for k, v in kwargs.items()}

        if epoch in self.data:
            # 覆盖：更新已有epoch
            self.data[epoch].update(row_data)
        else:
            # 新增：创建新epoch
            self.data[epoch] = row_data

        # 立即写入磁盘（防崩溃丢失）
        self._flush()

    def _flush(self):
        """刷新到文件"""
        if not self.data:
            return

        # 收集所有列名（按首次出现的epoch顺序）
        all_keys = set()
        for row in self.data.values():
            all_keys.update(row.keys())
        fieldnames = ['epoch'] + sorted(all_keys)  # epoch第一，其余按字母序

        # 写入文件
        with open(self.filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for epoch, row in self.data.items():
                row_with_epoch = {'epoch': epoch}
                row_with_epoch.update(row)
                writer.writerow(row_with_epoch)


def add_column_titles(grid_np, titles, fontsize=None, font_color=(0, 255, 255)):
    """
    grid_tensor: [3, H, W]  已拼好的大图
    titles: List[str]  长度=列数
    返回: PIL.Image  带文字标注
    """
    pil = Image.fromarray(grid_np)   # 已经是 (H,W,3) 的 uint8
    draw = ImageDraw.Draw(pil)
    font = fontsize if fontsize is None else ImageFont.load_default()

    ncol = len(titles)
    w, h = pil.size          # (W, H)
    col_w = w // ncol
    y = 5                    # 顶部留边
    for idx, title in enumerate(titles):
        x = idx * col_w
        draw.text((x, y), title, font=font, fill=font_color)

    return pil


def plot_intensity_histogram(cond_HU, fake_HU, real_HU, vis_dir, epoch,
                              hu_min=-1000, hu_max=1000, num_bins=500,
                              smooth_window=5):
    """
    绘制CTA、生成NCCT、真实NCCT的CT值强度分布直方图（连续折线形式）

    Args:
        cond_HU: CTA输入 (B,1,H,W,D)
        fake_HU: 生成的NCCT (B,1,H,W,D)
        real_HU: 真实的NCCT (B,1,H,W,D)
        vis_dir: 可视化输出目录
        epoch: 当前轮次
        hu_min, hu_max: HU值范围
        num_bins: 直方图bin数量
        smooth_window: 平滑窗口大小（奇数）
    """

    # 配置参数（参考 Histogram.py）
    SMOOTHING = True

    # 颜色配置（参考 Histogram.py 风格）
    COLORS = {
        'CTA': '#A5A5A5',  # 灰色 - 输入
        'Real_NCCT': '#2ecc71',  # 绿色 - 真实目标
        'Fake_NCCT': '#7030A0'  # 紫色 - 生成结果
    }

    # 获取中心切片索引
    sl = cond_HU.size(-1) // 2

    # 准备数据：分别提取切片和体积
    # 切片数据 (取每个样本的中心切片)
    cond_slice = cond_HU[:, 0, :, :, sl].numpy().flatten()  # (B*H*W,)
    fake_slice = fake_HU[:, 0, :, :, sl].numpy().flatten()
    real_slice = real_HU[:, 0, :, :, sl].numpy().flatten()

    # 体积数据 (整个subvolume)
    cond_vol = cond_HU[:, 0].numpy().flatten()
    fake_vol = fake_HU[:, 0].numpy().flatten()
    real_vol = real_HU[:, 0].numpy().flatten()

    # 过滤HU范围并计算直方图
    def _compute_histogram(data, hu_min, hu_max, num_bins):
        """计算直方图，返回bin中心和概率密度"""
        # 过滤范围
        mask = (data >= hu_min) & (data <= hu_max)
        filtered = data[mask]

        if len(filtered) == 0:
            return None, None, 0

        # 计算直方图
        bins = np.linspace(hu_min, hu_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]

        counts, _ = np.histogram(filtered, bins=bins)
        density = counts / (counts.sum() * bin_width) if counts.sum() > 0 else counts

        return bin_centers, density, len(filtered)

    def _smooth_curve(x, y, window_size):
        """移动平均平滑"""
        if window_size < 3 or window_size % 2 == 0:
            window_size = max(3, window_size + 1)
        half_window = window_size // 2
        y_smooth = np.copy(y).astype(float)
        for i in range(half_window, len(y) - half_window):
            y_smooth[i] = np.mean(y[i - half_window:i + half_window + 1])
        return x, y_smooth

    # 数据配置列表
    datasets = [
        ('Slice', cond_slice, fake_slice, real_slice, f'epoch{epoch:04d}_Histogram_slice.png'),
        ('Volume', cond_vol, fake_vol, real_vol, f'epoch{epoch:04d}_Histogram_volume.png')
    ]

    for data_type, cond_data, fake_data, real_data, filename in datasets:
        fig, ax = plt.subplots(figsize=(12, 7))

        stats_info = []

        # 绘制三条曲线
        for name, data, color in [
            ('CTA (Input)', cond_data, COLORS['CTA']),
            ('Real NCCT (Target)', real_data, COLORS['Real_NCCT']),
            ('Generated NCCT', fake_data, COLORS['Fake_NCCT'])
        ]:
            bin_centers, density, pixel_count = _compute_histogram(
                data, hu_min, hu_max, num_bins
            )

            if bin_centers is None:
                continue

            # 平滑处理
            if SMOOTHING:
                bin_centers, density = _smooth_curve(bin_centers, density, smooth_window)

            # 绘制折线
            ax.plot(bin_centers, density,
                    label=name,
                    color=color,
                    linewidth=2.5,
                    linestyle='-',
                    alpha=0.9)

            # 记录统计信息
            stats_info.append({
                'name': name,
                'mean': np.mean(data),
                'std': np.std(data),
                'count': pixel_count
            })

        # 添加参考线
        ax.axvline(x=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, label='Water (0 HU)')
        ax.axvline(x=50, color='lightcoral', linestyle=':', linewidth=2, alpha=0.8, label='Vessel (~50 HU)')

        # 设置图形属性
        ax.set_xlabel('HU Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        ax.set_title(f'CT Intensity Distribution - {data_type} Level\n'
                     f'Epoch {epoch} (HU: [{hu_min}, {hu_max}])',
                     fontsize=13, fontweight='bold', pad=15)

        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([hu_min, hu_max])
        ax.set_ylim(bottom=0)

        # 添加统计信息文本框
        stats_text = 'Statistics:\n'
        for s in stats_info:
            stats_text += f'{s["name"].split()[0]}: μ={s["mean"]:.1f}, σ={s["std"]:.1f}\n'

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()

        # 保存
        save_path = join(vis_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # print(f'[Histogram] Saved {data_type} distribution plot: {save_path}')


def setup_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ------ 日志记录 ------
def log_setup(savepath, log, name=None):
    if name is None:
        name = 'hetero'
    ldir = savepath
    if not os.path.exists(ldir):
        os.makedirs(ldir)

    lfile = name + '_' + log + '.txt' if log else name + '.txt'
    lfile = os.path.join(ldir, lfile)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        filename=lfile,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.getLogger('').addHandler(console)


class CkptManager:
    """
    统一断点目录结构: {ckpt_dir}/
      last_model.pt        每 epoch 完整更新（model+optimizer+scheduler），断点恢复用
      epoch{:04d}.pt       每 epoch 模型权重快照（不含优化器），阶段性记录用
      best_model.pt        最优模型（由训练脚本按验证指标触发 save_best）
    max_keep_epochs: 保留最近 N 个 epoch 快照，0 = 全部保留。
    """
    def __init__(self, ckpt_dir: str, max_keep_epochs: int = 0, save_freq: int = 1):
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        self.max_keep_epochs = int(max_keep_epochs)
        self.save_freq = max(1, int(save_freq))

    @property
    def last_path(self) -> str:
        return join(self.ckpt_dir, "last_model.pt")

    def epoch_path(self, epoch: int) -> str:
        return join(self.ckpt_dir, f"epoch{epoch:04d}.pt")

    def save_epoch(self, epoch, model, optimizer=None, scheduler=None,
                   best_val=None, extra=None):
        payload = {"epoch": int(epoch), "model": model.state_dict()}
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        if best_val is not None:
            payload["best_val_loss"] = float(best_val)
        if extra:
            payload.update(extra)
        torch.save(payload, self.last_path)
        if epoch % self.save_freq == 0 or epoch == 1:
            torch.save({"epoch": int(epoch), "model": model.state_dict()},
                       self.epoch_path(epoch))
            self._gc_epochs()

    def save_best(self, model, extra=None):
        payload = {"model": model.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, join(self.ckpt_dir, "best_model.pt"))

    def load_last(self, map_location=None):
        if not os.path.exists(self.last_path):
            return None
        return torch.load(self.last_path, map_location=map_location)

    def _gc_epochs(self):
        if self.max_keep_epochs <= 0:
            return
        epochs = sorted(
            int(m.group(1)) for f in os.listdir(self.ckpt_dir)
            if (m := re.match(r"epoch(\d+)\.pt", f))
        )
        for e in epochs[:-self.max_keep_epochs]:
            os.remove(self.epoch_path(e))


def resume_from_ckpt(ckpt, model, optimizer=None, scheduler=None):
    """恢复 model/optimizer/scheduler，返回 (start_epoch, best_val_loss)。"""
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt["epoch"]) + 1, float(ckpt.get("best_val_loss", float("inf")))


def amp_context(enabled: bool):
    return torch_npu.npu.amp.autocast(enabled=enabled, dtype=torch.bfloat16)

# ---- 文本生成可视化 ----
def _repetition_ratio(text: str, n: int = 8) -> float:
    """复读检测: 最长重复子串占全文比例 (greedy 死循环报警器)"""
    if len(text) < 2 * n:
        return 0.0
    seen, best = {}, 0
    for i in range(len(text) - n + 1):
        g = text[i:i + n]
        if g in seen:
            best = max(best, i + n - seen[g])
        else:
            seen[g] = i
    return best / len(text)


def text_sampling(records, save_path, epoch, prefix="val"):
    """
    生成样本归档 (训练期 val / 评估期 test 共用):
      每例目录: {save_path}/out_sample/{pid}/gen.txt + diff.html + meta.json
                (医生原文不回存, 经 pid 回查 data.xlsx; 同 pid 新 epoch 覆盖)
      全量索引: {save_path}/out_sample/{prefix}_ep{epoch:04d}_all.txt
                (GEN/REF 成对, 供人快速翻阅 diff)
    Args:
        records: [{"pid": str, "gen": str, "ref": str}, ...]
    Returns:
        stats: {"n", "avg_gen_len", "avg_ref_len", "rep_ratio"}
    """
    out_root = join(save_path, "out_sample")
    os.makedirs(out_root, exist_ok=True)

    stats = {
        "n": len(records),
        "avg_gen_len": round(float(np.mean([len(r["gen"]) for r in records])), 1) if records else 0.0,
        "avg_ref_len": round(float(np.mean([len(r["ref"]) for r in records])), 1) if records else 0.0,
        "rep_ratio": round(float(np.mean([_repetition_ratio(r["gen"]) for r in records])), 3) if records else 0.0,
    }

    sm = difflib.SequenceMatcher(autojunk=False)
    for r in records:
        d = join(out_root, str(r["pid"]))
        os.makedirs(d, exist_ok=True)

        with open(join(d, "gen.txt"), "w", encoding="utf-8") as f:
            f.write(r["gen"])

        # 字符级 diff (灰=一致, 红=REF有GEN无, 绿=GEN多写, 黄=改写)
        sm.set_seqs(r["ref"], r["gen"])
        frag = []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                frag.append(f'<span style="color:#adb5bd">{r["ref"][i1:i2]}</span>')
            elif op == "delete":
                frag.append(f'<span style="background:#ffd7d5">{r["ref"][i1:i2]}</span>')
            elif op == "insert":
                frag.append(f'<span style="background:#d3f9d8">{r["gen"][j1:j2]}</span>')
            else:
                frag.append(f'<span style="background:#ffe066">{r["ref"][i1:i2]}&#8594;{r["gen"][j1:j2]}</span>')
        with open(join(d, "diff.html"), "w", encoding="utf-8") as f:
            f.write("<meta charset='utf-8'><style>"
                    "body{font-family:sans-serif;max-width:1100px;margin:auto}"
                    "p{line-height:1.9;font-size:15px}</style>"
                    f"<h3>pid={r['pid']} | gen {len(r['gen'])} / ref {len(r['ref'])} chars"
                    f" | ep{epoch:04d}</h3><p>{''.join(frag)}</p>")

        with open(join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"pid": r["pid"], "epoch": int(epoch), "prefix": prefix,
                        "gen_len": len(r["gen"]), "ref_len": len(r["ref"]),
                        "rep_ratio": round(_repetition_ratio(r["gen"]), 3)},
                       f, ensure_ascii=False, indent=1)

    # 全量索引 (GEN/REF 成对, 自包含阅读)
    if records:
        with open(join(out_root, f"{prefix}_ep{epoch:04d}_all.txt"), "w", encoding="utf-8") as f:
            for r in records:
                f.write(f"===== pid={r['pid']} | gen_len={len(r['gen'])} ref_len={len(r['ref'])} =====\n")
                f.write(f"[GEN]\n{r['gen']}\n\n[REF]\n{r['ref']}\n\n\n")

    return stats
