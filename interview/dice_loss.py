import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, ignore_idx=None):
        super().__init__()
        self.ignore_idx = ignore_idx

    def forward(self, preds:torch.Tensor, target:torch.Tensor, smooth=1e-8):
        """

        Args:
            pred: logit, [B, C, H, W]
            target:[B, H, W]

        Returns:

        """
        b, num_classes = preds.shape[:2]

        # prob
        probs = torch.softmax(preds, dim=1)
        probs = probs.permute(0, 2, 3, 1).reshape(-1, num_classes)  # [B*H*W, C]

        # target
        target = target.reshape(-1).long()
        target_onehot = F.one_hot(target, num_classes) # [B*H*W, C]

        # valid mask
        if self.ignore_idx is not None:
            valid = target != self.ignore_idx
            probs = probs[valid]
            target_onehot = target_onehot[valid]

        # dice
        # (2*intersection) / (pred_sum, target_sum)
        intersection = (probs * target_onehot).sum(0)
        total_area = probs.sum(0) + target_onehot.sum(0)
        dice = (2 * intersection + smooth) / (total_area + smooth)  # [1, C]

        loss = 1 - dice.mean()  # [1, C]

        return loss


if __name__ == "__main__":
    preds = torch.randn(2, 4, 8, 8, requires_grad=True)
    target = torch.randint(0, 4, (2, 8, 8))

    criterion = DiceLoss(ignore_idx=255)
    loss = criterion(preds, target)

