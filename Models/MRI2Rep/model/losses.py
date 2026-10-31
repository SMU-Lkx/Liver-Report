# Models/MRI2Rep/model/losses.py
"""masked MSE + 3D SSIM (肝 mask 内, 全分辨率) —— 与 MRI2Rep 官方 pretrain_pv.py 同构.

SSIM 用 avg_pool3d 做窗口统计(单算子, NPU fp32 前向/反向均合法),
不再使用 conv3d 高斯窗 —— 后者在 NPU(CANN 8.2.RC1)上 fp32 反向不支持
(Conv3DBackpropInput), 会迫使 loss 只能放在 autocast 内计算.
loss_fn 约定: (pred, target, mask) -> (total, {"mse": float, "ssim": float}),
内部一律 .float() 计算, 与 autocast 解耦.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor = None, eps: float = 1e-8) -> torch.Tensor:
    pred, target = pred.float(), target.float()
    sq = (pred - target) ** 2
    if mask is None:
        return sq.mean()
    m = mask.float()
    return (sq * m).sum() / (m.sum() + eps)


class SSIM3D(nn.Module):
    """官方实现: avg_pool3d 作为窗口均值估计器 (win=7, 与 pretrain_pv.py 一致).

    注意: avg_pool3d 不在 autocast 转换名单内, 输入 fp32 就以 fp32 全程计算,
    NPU 上 AvgPool3D 前向/反向 dtype 支持均合法.
    """

    def __init__(self, win: int = 7):
        super().__init__()
        if win % 2 == 0:
            raise ValueError(f"SSIM3D win must be odd, got {win}")
        self.win = int(win)
        self.pad = self.win // 2
        self.C1, self.C2 = 0.01 ** 2, 0.03 ** 2

    def _lmean(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(x, kernel_size=self.win, stride=1, padding=self.pad)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        x, y = pred.float(), target.float()
        mu_x, mu_y = self._lmean(x), self._lmean(y)
        mu_xx, mu_yy, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
        sx = self._lmean(x * x) - mu_xx
        sy = self._lmean(y * y) - mu_yy
        sxy = self._lmean(x * y) - mu_xy
        ssim_map = ((2 * mu_xy + self.C1) * (2 * sxy + self.C2)) / \
                   ((mu_xx + mu_yy + self.C1) * (sx + sy + self.C2) + 1e-8)
        ssim_map = ssim_map.clamp(-1.0, 1.0)
        if mask is None:
            return ssim_map.mean()
        m = mask.float()
        return (ssim_map * m).sum() / (m.sum() + 1e-8)


def build_loss(cfg):
    """返回 loss_fn(pred_fullres, target, mask) -> (total, parts dict)。"""
    ssim = SSIM3D(win=int(getattr(cfg, "ssim_win", 7)))
    w_mse = float(cfg.mse_weight)
    w_ssim = float(cfg.ssim_weight)

    def loss_fn(pred, target, mask=None):
        mse = masked_mse(pred, target, mask)
        s = ssim(pred, target, mask)
        total = w_mse * mse + w_ssim * (1.0 - s)
        return total, {"mse": float(mse.detach().item()),
                       "ssim": float(s.detach().item())}

    return loss_fn