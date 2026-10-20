import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    """Safe binary dice loss with eps protection against division by zero."""
    def __init__(self, smooth=1.0, eps=1e-7):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth + self.eps

        dice_score = 2.0 * num / den
        loss_avg = 1.0 - dice_score.mean()
        # Bound loss between [0, 1] to prevent negative loss values
        return torch.clamp(loss_avg, min=0.0, max=1.0)


class RegionDecomposedLoss(nn.Module):
    """
    Sub-region decomposed loss evaluating:
      - Region 0: Enhancing Tumor (ET)
      - Region 1: Whole Tumor (WT)
      - Region 2: Tumor Core (TC)
    Provides individual losses L_ET, L_WT, L_TC for isolated outer-loop meta-updates,
    plus total segmentation loss for inner-loop model training.
    """
    def __init__(self, smooth=1.0):
        super(RegionDecomposedLoss, self).__init__()
        self.dice = BinaryDiceLoss(smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predict_logits, target):
        """
        predict_logits: (N, 3, D, H, W)
        target: (N, 3, D, H, W)
        """
        assert predict_logits.shape == target.shape, "Shape mismatch between prediction and target"
        probs = torch.sigmoid(predict_logits)

        # Region 0: ET
        dice_ET = self.dice(probs[:, 0], target[:, 0])
        bce_ET = self.bce(predict_logits[:, 0], target[:, 0].float())
        loss_ET = dice_ET + bce_ET

        # Region 1: WT
        dice_WT = self.dice(probs[:, 1], target[:, 1])
        bce_WT = self.bce(predict_logits[:, 1], target[:, 1].float())
        loss_WT = dice_WT + bce_WT

        # Region 2: TC
        dice_TC = self.dice(probs[:, 2], target[:, 2])
        bce_TC = self.bce(predict_logits[:, 2], target[:, 2].float())
        loss_TC = dice_TC + bce_TC

        total_loss = (loss_ET + loss_WT + loss_TC) / 3.0
        return total_loss, (loss_ET, loss_WT, loss_TC)
