# Models/MRI2Rep/model/backbone.py
"""3D CNN backbone, ported from MRI2Rep (Conv3d-IN-GELU x4), in_ch configurable."""
import torch
import torch.nn as nn


class ConvINAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__(
            nn.Conv3d(cin, cout, kernel_size=3, stride=stride, padding=1),
            nn.InstanceNorm3d(cout),
            nn.GELU(),
        )


class ConvBackbone3D(nn.Module):
    """(B, in_ch, H, W, D) -> (B, base*8, H/8, W/8, D/8)."""

    def __init__(self, in_ch: int = 4, base_channels: int = 32):
        super().__init__()
        b = base_channels
        self.stem   = ConvINAct(in_ch, b,     2)
        self.stage2 = ConvINAct(b,     b * 2, 2)
        self.stage3 = ConvINAct(b * 2, b * 4, 2)
        self.stage4 = ConvINAct(b * 4, b * 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
