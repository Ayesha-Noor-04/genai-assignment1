"""L_UDAE = alpha * L1(x, x^) + (1 - alpha) * (1 - SSIM(x, x^))   (assignment, Task 1)."""
import torch
import torch.nn as nn
from pytorch_msssim import ssim


class L1SSIMLoss(nn.Module):
    def __init__(self, alpha: float = 0.8):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        pred, target = pred.float(), target.float()  # keep the loss in fp32 even under AMP
        l1 = (pred - target).abs().mean()
        s = ssim(pred, target, data_range=1.0, size_average=True)
        loss = self.alpha * l1 + (1.0 - self.alpha) * (1.0 - s)
        return loss, l1.detach(), s.detach()