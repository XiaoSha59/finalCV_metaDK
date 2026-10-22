import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BRATS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "brats"))
for p in [PARENT_DIR, BRATS_DIR, os.path.join(BRATS_DIR, "process")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from losses.loss_dual import DiceLoss4BraTS, BCELoss4BraTS
except ImportError:
    from process.losses.loss_dual import DiceLoss4BraTS, BCELoss4BraTS


class ConsistencyRegularizationLoss(nn.Module):
    """
    Consistency Regularization Loss (L_CR) between Student (missing modalities)
    and Teacher (complete 4 modalities, detached).
    
    Inspired by MGML (Meta-Guided Multi-Modal Learning) and Mean-Teacher principles.
    Combines Soft-Dice Consistency and Temperature-scaled Soft-BCE Consistency
    for multi-region brain tumor segmentation (ET, TC, WT).
    """
    def __init__(self, temperature=4.0, lambda_dice=1.0, lambda_bce=0.5, eps=1e-5):
        super().__init__()
        self.temperature = float(temperature)
        self.lambda_dice = float(lambda_dice)
        self.lambda_bce = float(lambda_bce)
        self.eps = eps

    def forward(self, student_logits, teacher_logits):
        """
        student_logits: (B, 3, D, H, W)
        teacher_logits: (B, 3, D, H, W) - detached target
        """
        teacher_target = teacher_logits.detach()

        # 1. Soft-Dice Consistency
        # Measures overlap of predicted continuous confidence regions
        p_s = torch.sigmoid(student_logits)
        p_t = torch.sigmoid(teacher_target)

        # Batch & spatial reduction over (D, H, W) per channel
        dims = (2, 3, 4)
        intersection = torch.sum(p_s * p_t, dim=dims)
        cardinality = torch.sum(p_s * p_s, dim=dims) + torch.sum(p_t * p_t, dim=dims)
        dice_per_channel = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        loss_dice = 1.0 - dice_per_channel.mean()

        # 2. Temperature-scaled Soft BCE Consistency
        # Smoothed distributions reveal inter-channel confidence structures
        p_t_temp = torch.sigmoid(teacher_target / self.temperature)
        loss_bce = F.binary_cross_entropy_with_logits(
            student_logits / self.temperature,
            p_t_temp
        ) * (self.temperature ** 2)

        # Total Consistency Loss
        total_cr = self.lambda_dice * loss_dice + self.lambda_bce * loss_bce
        return total_cr, loss_dice, loss_bce


__all__ = ["ConsistencyRegularizationLoss", "DiceLoss4BraTS", "BCELoss4BraTS"]
