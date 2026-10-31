# Models/MRI2Rep/model/recon.py
"""Reconstruction decoder + NCAD2V pretrain model + factory (官方化 v4).

相对 v3 的改动:
  1. decoder 末尾加 nn.Sigmoid() —— 对齐官方 ReconDecoder3D (target 归一化到 [0,1],
     有界输出显著稳定 from-scratch 训练早期动力学);
  2. mask 不再只用于 loss, 而作为输入通道 (官方 [ART, 0, MASK] 设计):
     use_mask_channel=True 时输入为 [NC, A, V=0, D, MASK], in_ch = in_ch + 1;
  3. 保留 trilinear(内部 fp16 岛) + 双卷积 decoder —— NPU dtype 全链路合法且无棋盘格,
     容量不低于官方三层 ConvTranspose.

dtype 说明: NPU(CANN 8.2.RC1) 上 Conv3D 反向仅支持 fp16/bf16, Upsample3d 反向
不支持 bf16 —— fp16 是两者唯一公约数。故三线性插值在 fp16 岛内执行
(bf16 进 -> fp16 插值 -> 输出交还 autocast), 全链路可微、全分辨率监督.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

from .backbone import ConvBackbone3D


class UpTrilinear2x(nn.Module):
    """bf16 进, 内部 fp16 三线性插值, 输出 fp16 (下游 conv 由 autocast 接管).

    反向链路 dtype 合法性: 下游 conv 反向 -> grad bf16 -> cast 回 fp16 ->
    UpsampleTrilinear3dGrad(fp16, 支持) -> grad fp16 -> cast 回 bf16 ->
    上游 Conv3DBackpropInput(bf16, 支持).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch_npu.npu.amp.autocast(enabled=False):
            y = F.interpolate(x.to(torch.float16), scale_factor=2.0,
                              mode="trilinear", align_corners=False)
        return y


class ReconDecoder(nn.Module):
    """trilinear(内部fp16) + 双卷积, 28 -> 56 -> 112 -> 224 全分辨率重建, 末尾 Sigmoid."""

    @staticmethod
    def _block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            UpTrilinear2x(),
            nn.Conv3d(cin, cout, 3, padding=1), nn.InstanceNorm3d(cout), nn.GELU(),
            nn.Conv3d(cout, cout, 3, padding=1), nn.InstanceNorm3d(cout), nn.GELU(),
        )

    def __init__(self, base_channels: int = 32, out_ch: int = 1):
        super().__init__()
        b = base_channels
        self.up1 = self._block(b * 8, b * 4)     # 28 -> 56
        self.up2 = self._block(b * 4, b * 2)     # 56 -> 112
        self.up3 = self._block(b * 2, b)         # 112 -> 224
        self.head = nn.Sequential(
            nn.Conv3d(b, out_ch, 3, padding=1),
            nn.Sigmoid(),   # 对齐官方: target 归一化到 [0,1], 输出有界
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        f = self.up1(f)
        f = self.up2(f)
        f = self.up3(f)
        return self.head(f)


class NCAD2VReconModel(nn.Module):
    """输入 NC+A+D（V 通道已被数据集置零, 可选拼 mask 通道），重建 V 期相.

    mask 作为输入通道时: x_in 布局为 [NC, A, V=0, D, MASK], in_ch 需 +1.
    """

    def __init__(self, in_ch: int = 4, base_channels: int = 32):
        super().__init__()
        self.backbone = ConvBackbone3D(in_ch, base_channels)
        self.decoder = ReconDecoder(base_channels, out_ch=1)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.backbone(x_in))

    def backbone_state_dict(self):
        return self.backbone.state_dict()


def build_model(cfg) -> NCAD2VReconModel:
    in_ch = int(cfg.in_ch) + (1 if bool(getattr(cfg, "use_mask_channel", True)) else 0)
    return NCAD2VReconModel(in_ch=in_ch, base_channels=int(cfg.base_channels))