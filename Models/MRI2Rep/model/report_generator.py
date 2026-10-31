# Models/MRI2Rep/model/report_generator.py
"""Backbone + visual token projector + Transformer decoder (MRI2Rep 同构, 自由文本版)."""
import torch
import torch.nn as nn

from .backbone import ConvBackbone3D


class VisualProjector(nn.Module):
    """(B, C, H, W, D) -> (B, gH*gW*gD, d_model)"""

    def __init__(self, in_ch: int, d_model: int, vis_grid=(7, 7, 5)):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(tuple(vis_grid))
        self.proj = nn.Linear(in_ch, d_model)
        self.n_tokens = vis_grid[0] * vis_grid[1] * vis_grid[2]
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))

    def forward(self, f):
        f = self.pool(f).flatten(2).transpose(1, 2)   # (B, N, C)
        return self.proj(f) + self.pos


class MRI2RepTextGen(nn.Module):
    def __init__(self, in_ch=4, base_channels=32, d_model=256, nhead=8,
                 enc_layers=2, dec_layers=4, dim_ff=512, dropout=0.2,
                 vis_grid=(7, 7, 5), vocab_size=1024, max_len=768,
                 pad_id=0, bos_id=1, eos_id=2):
        super().__init__()
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id
        self.max_len = max_len

        self.backbone = ConvBackbone3D(in_ch, base_channels)
        self.vis_proj = VisualProjector(base_channels * 8, d_model, vis_grid)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        self.vis_encoder = nn.TransformerEncoder(enc_layer, enc_layers)

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)  # (24,256)的词向量（输入token id，输出对应id的向量）
        self.dec_pos = nn.Parameter(torch.zeros(1, max_len, d_model))  # 位置编码
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)
        self.out = nn.Linear(d_model, vocab_size)
        self.drop = nn.Dropout(dropout)

    def encode(self, x):
        f = self.backbone(x)
        tok = self.vis_proj(f)
        return self.vis_encoder(tok)

    def decode(self, ys, memory):
        # 从24个词的闭式集(RLCTokenizer-VOCAB)中取概率最大的词(id)
        L = ys.shape[1]
        mask = torch.triu(torch.full((L, L), float("-inf"), device=ys.device), diagonal=1)
        t = self.drop(self.embed(ys) + self.dec_pos[:, :L])
        h = self.decoder(t, memory, tgt_mask=mask)  # (B,L,256)词向量解码。
        return self.out(h)  # (B,L,256) → (B,L,24logits) 24个词的各个概率

    def forward(self, x, dec_input):
        memory = self.encode(x)
        return self.decode(dec_input, memory)

    @torch.no_grad()
    def generate(self, x, max_len=None):
        """推理时逐个token生成（自回归生成）"""
        max_len = max_len or self.max_len
        memory = self.encode(x)
        B = x.shape[0]
        ys = torch.full((B, 1), self.bos_id, dtype=torch.long, device=x.device)  # ys=前缀BOS
        for _ in range(max_len):
            logits = self.decode(ys, memory)
            nxt = logits[:, -1].argmax(-1, keepdim=True)  # next token，取最大概率的token。
            ys = torch.cat([ys, nxt], dim=1)  #　往前缀后面拼
            if (nxt == self.eos_id).all():
                break
        return ys


def build_generator(cfg, vocab_size: int, pad_id=0, bos_id=1, eos_id=2) -> MRI2RepTextGen:
    return MRI2RepTextGen(
        in_ch=int(cfg.in_ch), base_channels=int(cfg.base_channels),
        d_model=int(cfg.d_model), nhead=int(cfg.nhead),
        enc_layers=int(cfg.enc_layers), dec_layers=int(cfg.dec_layers),
        dim_ff=int(cfg.dim_ff), dropout=float(cfg.dropout),
        vis_grid=tuple(int(v) for v in cfg.vis_grid),
        vocab_size=vocab_size, max_len=int(cfg.max_len),
        pad_id=pad_id, bos_id=bos_id, eos_id=eos_id)


def load_pretrained_backbone(model, ckpt_path, map_location="cpu", input_phases=None):
    import os, logging
    if not ckpt_path or not os.path.exists(ckpt_path):
        logging.info(f"[Backbone] no pretrained checkpoint at {ckpt_path}, training from scratch.")
        return 0
    sd = torch.load(ckpt_path, map_location=map_location)
    pretrained_n = sum(v.numel() for v in sd.values())

    # 本地 NCAD2V 预训练通道为 NC/ART/PV/Delay。RLC 在其后追加 mask，
    # 按名称选取预训练图像通道，mask 卷积核置零，避免直接 strict-load 通道冲突。
    key = "stem.0.weight"
    if input_phases is not None and key in sd:
        src = sd[key]
        dst = model.backbone.state_dict()[key]
        if tuple(src.shape) != tuple(dst.shape):
            pretrained_phases = ["NC", "Arterial", "Venous", "Delay"]
            expected_dst_ch = len(input_phases) + 1
            compatible = (
                src.ndim == 5 and dst.ndim == 5
                and src.shape[1] == len(pretrained_phases)
                and dst.shape[1] == expected_dst_ch
                and src.shape[0] == dst.shape[0]
                and src.shape[2:] == dst.shape[2:]
            )
            if not compatible or any(phase not in pretrained_phases for phase in input_phases):
                raise RuntimeError(
                    f"预训练输入层不兼容：checkpoint={tuple(src.shape)}，"
                    f"model={tuple(dst.shape)}，RLC_phase={tuple(input_phases)}"
                )
            indices = [pretrained_phases.index(phase) for phase in input_phases]
            expanded = src.new_zeros(dst.shape)
            expanded[:, :len(input_phases)] = src[:, indices]
            sd = dict(sd)
            sd[key] = expanded
            logging.info(
                f"[Backbone] adapted input channels {src.shape[1]} -> {dst.shape[1]}; "
                f"image phases={tuple(input_phases)}, appended MASK weight=0."
            )

    missing, unexpected = model.backbone.load_state_dict(sd, strict=True)
    logging.info(f"[Backbone] loaded {pretrained_n/1e6:.2f}M params from {ckpt_path} "
                 f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return pretrained_n