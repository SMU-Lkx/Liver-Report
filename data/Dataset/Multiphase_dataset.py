# data/Dataset/Multiphase_dataset.py
"""
Multiphase liver CT + liver mask + report-text dataset (MRI2Rep-style baseline). v3

几何处理（重要）:
  - 各期相先做 nib.as_closest_canonical(RAS 规范化), 再重采样到统一目标网格;
  - 目标网格以 Venous 的解剖范围为世界参考(F不变), 体素尺寸统一缩放到 (224,224,160);
  - 期相覆盖不到的 z 区域填空气(-1000 -> 0), 保证"同一 voxel = 同一解剖位置";
  - 构建时自动计算各期 mask 与 Venous mask 的 Dice, <0.3 告警。

显示配方: plt.imshow(x[ch, :, :, k], origin="lower", cmap="gray")
  -> 面朝上, 屏右=患者左侧(放射科常规)。

Cache: per-patient folder {cache_dir}/{pid}/{x.npy, mask.npy, y.npy, meta.json}
  - x.npy    : uint8 (4, 224, 224, 160), HU window [-1000,1000] -> [0,255]
  - mask.npy : uint8 (M, 224, 224, 160)  (M=1 'venous' | M=4 'stack' | 0 'none')
  - y.npy    : int64  (L,)  char ids (BOS + report_desc + EOS)
  - manifest.json at cache root: [{id, cache_dir, split, text, impression}]

Pretrain mode: MultiphasePretrain returns (x_in, target_V, mask), x_in 的 V 通道已置零。

RLC mode: MultiphaseRLC 使用独立缓存，读取 accurate_summary.csv，
  返回 x=[Arterial,Venous,Venous_mask]（3通道）和官方24-token序列；
  默认网格为192×192×96，不读取或覆盖自由文本 y.npy。
"""
from __future__ import annotations

import json
import os, tqdm
from pathlib import Path
from typing import Dict, List, Sequence
from collections import Counter
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from nibabel.processing import resample_from_to
from torch.utils.data import Dataset

PHASE_ORDER = ["NC", "Arterial", "Venous", "Delay"]
VENOUS_IDX = PHASE_ORDER.index("Venous")
IMG_COLS = {"NC": "NC_img", "Arterial": "Arterial_img",
            "Venous": "Venous_img", "Delay": "Delay_img"}
MASK_COLS = {"NC": "NC_mask", "Arterial": "Arterial_mask",
             "Venous": "Venous_mask", "Delay": "Delay_mask"}

HU_MIN, HU_MAX = -1000.0, 1000.0
SHAPE = (224, 224, 160)
DEFAULT_CACHE = "/data/longkaixing/Liver/Multiphase"

# 官方 MRI2Rep-RLC：仅使用 ART、PV 和肝脏 mask，不使用 NC/Delay。
RLC_PHASE_ORDER = ["Arterial", "Venous"]
RLC_SHAPE = (192, 192, 96)
RLC_MAX_LESIONS = 8
RLC_MAX_LEN = 2 + 1 + 3 * RLC_MAX_LESIONS  # BOS/EOS + liver + 8 个三元组
DEFAULT_RLC_CACHE = "/data/longkaixing/Liver/Multiphase_RLC"


def normalize_rlc_phases(phases: Sequence[str]) -> tuple:
    """校验 YAML 中的 RLC_phase；返回与模型通道一致的期相顺序。"""
    if isinstance(phases, str):
        phases = phases.replace("，", ",").split(",")
    phases = tuple(str(phase).strip() for phase in phases)
    if not phases or any(phase not in IMG_COLS for phase in phases):
        raise ValueError(f"RLC_phase 必须从 {PHASE_ORDER} 中选择，收到 {phases}")
    if len(set(phases)) != len(phases):
        raise ValueError(f"RLC_phase 不得包含重复期相：{phases}")
    return phases


# 检查缓存文件齐全
def _cache_ready(cache_dir: str, pid: str) -> bool:
    d = _patient_dir(cache_dir, pid)
    return all(os.path.exists(os.path.join(d, f))
               for f in ("x.npy", "mask.npy", "y.npy", "meta.json"))

# ---------------------------------------------------------------------------
# preprocessing: RAS canonical -> resample to common world grid
# ---------------------------------------------------------------------------
def _target_affine(ref_img, shape=SHAPE) -> np.ndarray:
    """以参考期相(Venous, 已 RAS 规范化)的解剖范围构造目标网格 affine."""
    zooms = np.array(ref_img.header.get_zooms()[:3], dtype=np.float64)
    v2w = ref_img.affine
    rot = v2w[:3, :3] @ np.diag(1.0 / zooms)                 # 方向余弦(去除体素尺度)
    t_zoom = zooms * np.array(ref_img.shape[:3]) / np.array(shape)  # 目标体素尺寸(FOV不变)
    t = np.eye(4)
    t[:3, :3] = rot @ np.diag(t_zoom)
    t[:3, 3] = v2w[:3, 3]                                    # 与参考相同的世界原点
    return t


def prep_volume_to_grid(nii_path: str, target_aff: np.ndarray) -> np.ndarray:
    """RAS 规范化 + 重采样到统一网格 -> [1,H,W,D] float32 [0,1]; 覆盖外填空气."""
    img = nib.as_closest_canonical(nib.load(nii_path))
    out = resample_from_to(img, (SHAPE, target_aff), order=1,
                           mode="constant", cval=-1000.0)
    x = np.clip(out.get_fdata().astype(np.float32), HU_MIN, HU_MAX)
    x = (x - HU_MIN) / (HU_MAX - HU_MIN)
    return x[None]


def prep_mask_to_grid(nii_path: str, target_aff: np.ndarray) -> np.ndarray:
    """RAS 规范化 + 最近邻重采样 -> [1,H,W,D] uint8 {0,1}; 覆盖外填 0."""
    img = nib.as_closest_canonical(nib.load(nii_path))
    out = resample_from_to(img, (SHAPE, target_aff), order=0,
                           mode="constant", cval=0.0)
    return (out.get_fdata() > 0.5).astype(np.uint8)[None]


def _mask_dice(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum()
    return float(2.0 * inter / (m1.sum() + m2.sum() + 1e-8))


# ---------------------------------------------------------------------------
# char-level tokenizer
# ---------------------------------------------------------------------------
class CharTokenizer:
    PAD, BOS, EOS = 0, 1, 2

    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.inv = {i: c for c, i in vocab.items()}

    @classmethod
    def build(cls, texts: Sequence[str], min_freq: int = 1) -> "CharTokenizer":
        """统计报告的所有字符出现频率，每个汉字/符号=一个字符"""
        cnt = Counter("".join(texts))
        chars = sorted(c for c, n in cnt.items() if n >= min_freq and c.strip())
        return cls({c: i + 3 for i, c in enumerate(chars)})

    def save(self, path: str) -> None:
        """统计的字符保存为vocab.json"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + 3

    def encode(self, text: str, max_len: int) -> np.ndarray:
        """加入起止字符BOS/EOC"""
        ids = [self.BOS] + [self.vocab.get(c, self.PAD) for c in text] + [self.EOS]
        return np.asarray(ids[:max_len], dtype=np.int64)

    def decode(self, ids: Sequence[int]) -> str:
        """去掉起止字符BOS/EOC"""
        return "".join(self.inv.get(int(t), "") for t in ids
                       if int(t) not in (self.PAD, self.BOS, self.EOS))


# ---------------------------------------------------------------------------
# records (shared by both datasets)
# ---------------------------------------------------------------------------
def load_records(data_path: str, pid_csv: str,
                 require_phases: Sequence[str] = tuple(PHASE_ORDER),
                 verify_paths: bool = False) -> List[dict]:
    df = pd.read_excel(data_path)
    df["id"] = df["id"].astype(str)
    wanted = set(pd.read_csv(pid_csv)["id"].astype(str))
    df = df[df["id"].isin(wanted)]

    records, skipped = [], {"missing_phase": 0, "no_text": 0}
    for _, row in df.iterrows():
        pid = row["id"]
        imgs, masks = {}, {}
        ok = True
        for ph in PHASE_ORDER:
            p = row.get(IMG_COLS[ph])
            if pd.isna(p) or not str(p).strip() or (verify_paths and not Path(str(p)).exists()):
                ok = False
                break
            imgs[ph] = str(p)
            mp = row.get(MASK_COLS[ph])
            masks[ph] = None if pd.isna(mp) else str(mp)
        if not ok or any(ph not in imgs for ph in require_phases):
            skipped["missing_phase"] += 1
            continue
        text = "" if pd.isna(row.get("report_desc")) else str(row["report_desc"]).strip()
        if not text:
            skipped["no_text"] += 1
            continue
        records.append({
            "pid": pid, "imgs": imgs, "masks": masks, "text": text,
            "impression": "" if pd.isna(row.get("report_impression")) else str(row["report_impression"]),
        })
    print(f"[load_records] {Path(pid_csv).name}: {len(records)} patients | "
          f"skipped: {skipped['missing_phase']} missing-phase, {skipped['no_text']} no-text")
    return records


# ---------------------------------------------------------------------------
# cache IO (.npy + mmap)
# ---------------------------------------------------------------------------
def _patient_dir(cache_dir: str, pid: str) -> str:
    return os.path.join(cache_dir, str(pid))


def write_cache(rec: dict, tokenizer: CharTokenizer, cache_dir: str,
                max_len: int, mask_mode: str) -> None:
    out = _patient_dir(cache_dir, rec["pid"])
    os.makedirs(out, exist_ok=True)

    ref_img = nib.as_closest_canonical(nib.load(rec["imgs"]["Venous"]))
    target_aff = _target_affine(ref_img)

    vols, msks = {}, {}
    for ph in tqdm.tqdm(PHASE_ORDER):
        vols[ph] = prep_volume_to_grid(rec["imgs"][ph], target_aff)[0]
        msks[ph] = (prep_mask_to_grid(rec["masks"][ph], target_aff)[0]
                    if rec["masks"][ph] else np.zeros(SHAPE, np.uint8))
    x = np.stack([vols[ph] for ph in PHASE_ORDER], axis=0)   # (4,H,W,D) float32

    # 对齐质检: 各期 mask 与 Venous mask 的 Dice(覆盖差异会拉低, 不应极低)
    ref_m = msks["Venous"]
    dices = {ph: _mask_dice(msks[ph], ref_m) for ph in PHASE_ORDER if ph != "Venous"}
    if dices and min(dices.values()) < 0.3:
        print(f"[WARN] pid={rec['pid']}: low cross-phase mask Dice={dices}, "
              f"check world alignment of phase affines!")

    if mask_mode == "none":
        mask = np.zeros((0,) + SHAPE, np.uint8)
    else:
        m_ph = PHASE_ORDER if mask_mode == "stack" else ["Venous"]
        mask = np.stack([msks[ph] for ph in m_ph], axis=0)

    y = tokenizer.encode(rec["text"], max_len)
    np.save(os.path.join(out, "x.npy"), (x * 255).astype(np.uint8))
    np.save(os.path.join(out, "mask.npy"), mask)
    np.save(os.path.join(out, "y.npy"), y)
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"text": rec["text"], "impression": rec["impression"]}, f, ensure_ascii=False)


def read_cache(cache_dir: str, pid: str):
    d = _patient_dir(cache_dir, pid)
    x = np.load(os.path.join(d, "x.npy"), mmap_mode="r")
    mask = np.load(os.path.join(d, "mask.npy"), mmap_mode="r")
    y = np.load(os.path.join(d, "y.npy"), mmap_mode="r")
    return x, mask, y


# ---------------------------------------------------------------------------
# main dataset (text generation)
# ---------------------------------------------------------------------------
class MultiphaseReport(Dataset):
    def __init__(self, data_path: str, pid_csv: str, tokenizer: CharTokenizer,
                 cache_dir: str = DEFAULT_CACHE, max_len: int = 768,
                 mask_mode: str = "venous", build_if_missing: bool = True,
                 **rec_kwargs):
        assert mask_mode in {"venous", "stack", "none"}
        self.tok, self.max_len = tokenizer, int(max_len)
        self.cache_dir, self.mask_mode, self.build = cache_dir, mask_mode, build_if_missing
        self.records = load_records(data_path, pid_csv, **rec_kwargs)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        if not _cache_ready(self.cache_dir, rec["pid"]):
            if not self.build:
                raise FileNotFoundError(rec["pid"])
            write_cache(rec, self.tok, self.cache_dir, self.max_len, self.mask_mode)
        x, mask, y = read_cache(self.cache_dir, rec["pid"])
        with open(os.path.join(_patient_dir(self.cache_dir, rec["pid"]), "meta.json"),
                  encoding="utf-8") as f:
            meta = json.load(f)
        return {
            "pid": rec["pid"],
            "x": torch.from_numpy(np.asarray(x, dtype=np.float32) / 255.0),  # (4,H,W,D)
            "mask": torch.from_numpy(np.asarray(mask, dtype=np.float32)) if mask.shape[0] else None,
            "y": torch.from_numpy(np.asarray(y, dtype=np.int64)),
            "text": meta["text"],
        }


def multiphase_collate(batch: List[dict]) -> dict:
    PAD = CharTokenizer.PAD
    x = torch.stack([b["x"] for b in batch])
    mask = torch.stack([b["mask"] for b in batch]) if batch[0]["mask"] is not None else None
    L = max(b["y"].numel() for b in batch)
    y = torch.full((len(batch), L), PAD, dtype=torch.long)
    for i, b in enumerate(batch):
        y[i, : b["y"].numel()] = b["y"]
    return {"x": x, "mask": mask, "y": y,
            "pid": [b["pid"] for b in batch], "text": [b["text"] for b in batch]}


# ---------------------------------------------------------------------------
# MRI2Rep-RLC dataset: [Arterial, Venous, Venous mask] -> closed-set sequence
# ---------------------------------------------------------------------------
class RLCTokenizer:
    """官方24-token词表；NO_LESION可解码，但不允许写入GT序列。"""
    PAD, BOS, EOS = 0, 1, 2
    VOCAB = [
        "PAD", "BOS", "EOS", "NO_LESION",
        "Fibrosis/Cirrhosis", "NoFibrosis/Cirrhosis",
        "APHE_WO", "RIM_ATYP", "APHE_NoWO", "HEM", "CYST",
        "L_LAT", "L_MED", "R_ANT", "R_POST", "R_JUNCTION",
        "CAUDATE", "L_DIFFUSE", "R_DIFFUSE", "DIFFUSE",
        "1", "2", "GE3", "Multiple",
    ]
    LIVER_TOKENS = {"Fibrosis/Cirrhosis", "NoFibrosis/Cirrhosis"}

    def __init__(self):
        assert len(self.VOCAB) == 24
        self.vocab = {token: idx for idx, token in enumerate(self.VOCAB)}
        self.inv = {idx: token for token, idx in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.VOCAB)

    def encode(self, sequence: str, max_len: int = RLC_MAX_LEN) -> np.ndarray:
        parts = [part.strip() for part in str(sequence).split(",") if part.strip()]
        if not parts or parts[0] not in self.LIVER_TOKENS:
            raise ValueError(f"非法 RLC 肝脏背景：{sequence!r}")
        if "NO_LESION" in parts:
            raise ValueError("GT 不应包含 NO_LESION；无病灶应为 [BOS, liver, EOS]")
        if (len(parts) - 1) % 3 != 0:
            raise ValueError(f"RLC 病灶字段不是完整三元组：{sequence!r}")
        n_lesions = (len(parts) - 1) // 3
        if n_lesions > RLC_MAX_LESIONS:
            raise ValueError(f"RLC 病灶数 {n_lesions} 超过官方上限 {RLC_MAX_LESIONS}")
        unknown = [token for token in parts if token not in self.vocab]
        if unknown:
            raise ValueError(f"RLC 含词表外 token：{unknown}")
        ids = [self.BOS] + [self.vocab[token] for token in parts] + [self.EOS]
        if len(ids) > max_len:
            raise ValueError(f"RLC 长度 {len(ids)} 超过 max_len={max_len}")
        return np.asarray(ids, dtype=np.int64)

    def decode(self, ids: Sequence[int]) -> str:
        tokens = []
        for value in ids:
            idx = int(value)
            if idx == self.EOS:
                break
            if idx not in (self.PAD, self.BOS):
                tokens.append(self.inv.get(idx, "Unknown"))
        return ", ".join(tokens)


def _normalize_rlc_id(value) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def load_rlc_records(data_path: str, pid_csv: str, rlc_csv: str,
                     tokenizer: RLCTokenizer, phases: Sequence[str] = tuple(RLC_PHASE_ORDER),
                     verify_paths: bool = False) -> List[dict]:
    """按 split 顺序联合所选期相的原始影像路径与 build 接受的 RLC 标签。"""
    phases = normalize_rlc_phases(phases)
    source = pd.read_excel(data_path)
    labels = pd.read_csv(rlc_csv, dtype={"id": str})
    required_source = {"id", "Venous_mask"} | {IMG_COLS[phase] for phase in phases}
    required_labels = {"id", "sequence"}
    if required_source - set(source.columns):
        raise ValueError(f"data.xlsx 缺少列：{sorted(required_source - set(source.columns))}")
    if required_labels - set(labels.columns):
        raise ValueError(f"RLC 标签缺少列：{sorted(required_labels - set(labels.columns))}")

    source["id"] = source["id"].map(_normalize_rlc_id)
    labels["id"] = labels["id"].map(_normalize_rlc_id)
    if labels["id"].duplicated().any():
        raise ValueError(f"{rlc_csv} 存在重复 id，无法唯一对应标签")
    source = source.drop_duplicates("id", keep="first").set_index("id", drop=False)
    labels = labels.set_index("id", drop=False)

    split_ids = pd.read_csv(pid_csv, dtype={"id": str})["id"].map(_normalize_rlc_id)
    split_ids = list(dict.fromkeys(pid for pid in split_ids if pid))
    records = []
    skipped = Counter({
        "missing_data": 0, "missing_rlc": 0, "missing_phase": 0, "invalid_rlc": 0,
    })
    for pid in split_ids:
        if pid not in source.index:
            skipped["missing_data"] += 1
            continue
        if pid not in labels.index:
            skipped["missing_rlc"] += 1
            continue
        row = source.loc[pid]
        label = labels.loc[pid]
        sequence = "" if pd.isna(label["sequence"]) else str(label["sequence"]).strip()
        try:
            tokenizer.encode(sequence)
        except ValueError:
            skipped["invalid_rlc"] += 1
            continue

        imgs = {phase: row.get(IMG_COLS[phase]) for phase in phases}
        mask = row.get(MASK_COLS["Venous"])
        paths = list(imgs.values()) + [mask]
        if any(pd.isna(path) or not str(path).strip() for path in paths):
            skipped["missing_phase"] += 1
            continue
        if verify_paths and any(not Path(str(path)).exists() for path in paths):
            skipped["missing_phase"] += 1
            continue
        records.append({
            "pid": pid,
            "imgs": {phase: str(imgs[phase]) for phase in phases},
            "mask": str(mask),
            "sequence": sequence,
            "tier": int(label["tier"]) if "tier" in labels.columns and not pd.isna(label["tier"]) else -1,
            "wc": int(label["wc"]) if "wc" in labels.columns and not pd.isna(label["wc"]) else -1,
        })
    print(f"[load_rlc_records] {Path(pid_csv).name}: {len(records)} patients | skipped: {dict(skipped)}")
    return records

# ---------------------------------------------------------------------------
# pretrain dataset: reconstruct Venous from NC+A+D (V channel zeroed)
# ---------------------------------------------------------------------------
def _prep_rlc_volume(nii_path: str, target_aff: np.ndarray, shape) -> np.ndarray:
    img = nib.as_closest_canonical(nib.load(nii_path))
    out = resample_from_to(img, (shape, target_aff), order=1,
                           mode="constant", cval=-1000.0)
    x = np.clip(out.get_fdata().astype(np.float32), HU_MIN, HU_MAX)
    return ((x - HU_MIN) / (HU_MAX - HU_MIN))[None]


def _prep_rlc_mask(nii_path: str, target_aff: np.ndarray, shape) -> np.ndarray:
    img = nib.as_closest_canonical(nib.load(nii_path))
    out = resample_from_to(img, (shape, target_aff), order=0,
                           mode="constant", cval=0.0)
    return (out.get_fdata() > 0.5).astype(np.uint8)[None]


def _rlc_cache_ready(cache_dir: str, pid: str, sequence: str, shape,
                     phases: Sequence[str], image_cache_dir: str = "") -> bool:
    d = _patient_dir(cache_dir, pid)
    files = ("y.npy", "meta.json") if image_cache_dir else ("x.npy", "mask.npy", "y.npy", "meta.json")
    if not all(os.path.exists(os.path.join(d, name)) for name in files):
        return False
    try:
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        external = os.path.abspath(image_cache_dir) if image_cache_dir else ""
        return (
            meta.get("sequence") == sequence
            and tuple(meta.get("shape", ())) == tuple(shape)
            and tuple(meta.get("phases", ())) == tuple(phases)
            and meta.get("image_cache_dir", "") == external
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def write_rlc_cache(rec: dict, tokenizer: RLCTokenizer, cache_dir: str,
                    max_len: int = RLC_MAX_LEN, shape=RLC_SHAPE,
                    phases: Sequence[str] = tuple(RLC_PHASE_ORDER),
                    image_cache_dir: str = "") -> None:
    """写 RLC 标签；外接 CT2Rep 缓存时只写 y/meta，绝不改写其影像或中文标签。"""
    phases = normalize_rlc_phases(phases)
    out = _patient_dir(cache_dir, rec["pid"])
    os.makedirs(out, exist_ok=True)
    y = tokenizer.encode(rec["sequence"], max_len=max_len)

    if not image_cache_dir:
        ref_img = nib.as_closest_canonical(nib.load(rec["imgs"]["Venous"]))
        target_aff = _target_affine(ref_img, shape=shape)
        vols = [_prep_rlc_volume(rec["imgs"][phase], target_aff, shape)[0]
                for phase in phases]
        mask = _prep_rlc_mask(rec["mask"], target_aff, shape)
        x = np.stack(vols, axis=0)
        np.save(os.path.join(out, "x.npy"), (x * 255).astype(np.uint8))
        np.save(os.path.join(out, "mask.npy"), mask)

    np.save(os.path.join(out, "y.npy"), y)
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "sequence": rec["sequence"], "tier": rec["tier"], "wc": rec["wc"],
            "shape": list(shape), "phases": list(phases), "mask": "Venous",
            "image_cache_dir": os.path.abspath(image_cache_dir) if image_cache_dir else "",
        }, f, ensure_ascii=False)


class MultiphaseRLC(Dataset):
    """返回 x=[RLC_phase..., Venous MASK] 和固定词表的 RLC token IDs。"""

    def __init__(self, data_path: str, pid_csv: str, rlc_csv: str,
                 tokenizer: RLCTokenizer = None, cache_dir: str = DEFAULT_RLC_CACHE,
                 max_len: int = RLC_MAX_LEN, shape=RLC_SHAPE,
                 phases: Sequence[str] = tuple(RLC_PHASE_ORDER),
                 image_cache_dir: str = "", build_if_missing: bool = True,
                 verify_paths: bool = False):
        self.tok = tokenizer or RLCTokenizer()
        self.cache_dir = cache_dir
        self.image_cache_dir = image_cache_dir or ""
        self.phases = normalize_rlc_phases(phases)
        self.max_len = int(max_len)
        self.shape = tuple(int(v) for v in shape)
        if len(self.shape) != 3:
            raise ValueError(f"shape 必须为三维，收到 {self.shape}")
        if self.image_cache_dir:
            if os.path.abspath(self.image_cache_dir) == os.path.abspath(self.cache_dir):
                raise ValueError("CT2Rep 影像 cache_dir 与 RLC_cache_dir 必须不同")
            if self.phases != tuple(PHASE_ORDER) or self.shape != SHAPE:
                raise ValueError(
                    f"复用 CT2Rep 缓存要求 RLC_phase={PHASE_ORDER}、RLC_shape={SHAPE}；"
                    f"当前 phases={self.phases}、shape={self.shape}"
                )
        self.build = bool(build_if_missing)
        self.records = load_rlc_records(
            data_path, pid_csv, rlc_csv, self.tok, phases=self.phases,
            verify_paths=verify_paths,
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        if not _rlc_cache_ready(
            self.cache_dir, rec["pid"], rec["sequence"], self.shape,
            self.phases, self.image_cache_dir,
        ):
            if not self.build:
                raise FileNotFoundError(
                    f"RLC 缓存缺失或配置已更新：{_patient_dir(self.cache_dir, rec['pid'])}"
                )
            write_rlc_cache(
                rec, self.tok, self.cache_dir, self.max_len, self.shape,
                self.phases, self.image_cache_dir,
            )

        label_dir = _patient_dir(self.cache_dir, rec["pid"])
        if self.image_cache_dir:
            image_dir = _patient_dir(self.image_cache_dir, rec["pid"])
            image_files = [os.path.join(image_dir, name) for name in ("x.npy", "mask.npy")]
            if not all(os.path.exists(path) for path in image_files):
                raise FileNotFoundError(
                    f"CT2Rep 影像缓存缺失：{image_dir}；请先构建 CT2Rep 缓存。"
                    "RLC 不会向该目录写入任何文件。"
                )
            x = np.load(image_files[0], mmap_mode="r")
            mask = np.load(image_files[1], mmap_mode="r")
            y = np.load(os.path.join(label_dir, "y.npy"), mmap_mode="r")
        else:
            x, mask, y = read_cache(self.cache_dir, rec["pid"])

        expected_x = (len(self.phases),) + self.shape
        expected_mask = (1,) + self.shape
        if tuple(x.shape) != expected_x or tuple(mask.shape) != expected_mask:
            raise ValueError(
                f"pid={rec['pid']} 缓存形状不匹配：x={x.shape}（期望 {expected_x}），"
                f"mask={mask.shape}（期望 {expected_mask}）；请检查期相、mask_mode 和网格。"
            )
        x = np.asarray(x, dtype=np.float32) / 255.0
        mask = np.asarray(mask, dtype=np.float32)
        model_input = np.concatenate([x, mask], axis=0)
        return {
            "pid": rec["pid"],
            "x": torch.from_numpy(model_input),
            "y": torch.from_numpy(np.array(y, dtype=np.int64, copy=True)),
            "sequence": rec["sequence"],
            "tier": rec["tier"],
            "wc": rec["wc"],
        }

def rlc_collate(batch: List[dict]) -> dict:
    x = torch.stack([item["x"] for item in batch])
    max_seq_len = max(item["y"].numel() for item in batch)
    y = torch.full((len(batch), max_seq_len), RLCTokenizer.PAD, dtype=torch.long)
    for idx, item in enumerate(batch):
        y[idx, :item["y"].numel()] = item["y"]
    return {
        "x": x, "y": y,
        "pid": [item["pid"] for item in batch],
        "sequence": [item["sequence"] for item in batch],
        "tier": [item["tier"] for item in batch],
        "wc": [item["wc"] for item in batch],
    }


class MultiphasePretrain(Dataset):
    def __init__(self, data_path: str, pid_csv: str, tokenizer: CharTokenizer,
                 cache_dir: str = DEFAULT_CACHE, max_len: int = 768,
                 mask_mode: str = "venous", **rec_kwargs):
        self.ds = MultiphaseReport(data_path, pid_csv, tokenizer, cache_dir, max_len, mask_mode, **rec_kwargs)
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.ds.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.ds.records[idx]
        if not _cache_ready(self.cache_dir, rec["pid"]):
            write_cache(rec, self.ds.tok, self.cache_dir, self.ds.max_len, self.ds.mask_mode)
        x, mask, _ = read_cache(self.cache_dir, rec["pid"])
        x = np.asarray(x, dtype=np.float32) / 255.0
        x_in = x.copy()
        x_in[VENOUS_IDX] = 0.0                        # 置零目标期相
        return {
            "pid": rec["pid"],
            "x_in": torch.from_numpy(x_in),           # (4,H,W,D), V 通道 = 0
            "target": torch.from_numpy(x[VENOUS_IDX:VENOUS_IDX + 1].copy()),  # (1,H,W,D)
            "mask": torch.from_numpy(np.asarray(mask, dtype=np.float32)) if mask.shape[0] else None,
        }


def pretrain_collate(batch: List[dict]) -> dict:
    return {
        "x_in": torch.stack([b["x_in"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]) if batch[0]["mask"] is not None else None,
        "pid": [b["pid"] for b in batch],
    }


# ---------------------------------------------------------------------------
# cache builder + manifest
# ---------------------------------------------------------------------------
def build_all(data_path: str, split_dir: str, cache_dir: str,
              tokenizer: CharTokenizer, max_len: int = 768,
              mask_mode: str = "venous", verify_paths: bool = False) -> None:
    manifest = []
    for split in ("train", "val", "test"):
        csv = os.path.join(split_dir, f"{split}_idx.csv")
        if not os.path.exists(csv):
            print(f"[build_all] skip missing {csv}")
            continue
        recs = load_records(data_path, csv, verify_paths=verify_paths)
        for rec in recs:
            if not _cache_ready(cache_dir, rec["pid"]):
                write_cache(rec, tokenizer, cache_dir, max_len, mask_mode)
            manifest.append({"id": rec["pid"], "cache_dir": _patient_dir(cache_dir, rec["pid"]),
                             "split": split, "text": rec["text"], "impression": rec["impression"]})
        print(f"[build_all] {split}: {len(recs)} patients cached")
    with open(os.path.join(cache_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"[build_all] manifest -> {os.path.join(cache_dir, 'manifest.json')}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/longkaixing/Liver/data/data.xlsx")
    ap.add_argument("--split-dir", default="/home/longkaixing/Liver/data/split_idx/260925_SSL")
    ap.add_argument("--out", default=DEFAULT_CACHE)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--mask-mode", default="venous", choices=["venous", "stack", "none"])
    ap.add_argument("--verify-paths", action="store_true")
    args = ap.parse_args()

    df = pd.read_excel(args.data)
    df["id"] = df["id"].astype(str)
    train_pids = set(pd.read_csv(os.path.join(args.split_dir, "train_idx.csv"))["id"].astype(str))
    tok = CharTokenizer.build([str(t) for t in df.loc[df["id"].isin(train_pids), "report_desc"].dropna()])
    tok.save(os.path.join(args.out, "vocab.json"))
    print(f"[tokenizer] vocab_size={tok.vocab_size}")
    # 构建缓存
    build_all(args.data, args.split_dir, args.out, tok,
              max_len=args.max_len, mask_mode=args.mask_mode,
              verify_paths=args.verify_paths)