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


class UncertaintyWeightedKDLoss(nn.Module):
    """
    Uncertainty-Aware Evidential Knowledge Distillation Loss for BraTS 3D Segmentation.
    
    Inspired by Evidential Deep Learning (Sensoy et al., NeurIPS) & Uncertainty-guided KD (MICCAI/TMI).
    Measures Teacher's prediction uncertainty via voxel-wise Shannon entropy across tumor channels (ET, WT, TC).
    Regions with high Teacher uncertainty (entropy -> ln 2) receive lower distillation weight,
    preventing the Student from memorizing Teacher hallucinations or ambiguous boundaries.
    """
    def __init__(self, tau_unc=0.5, lambda_dice=1.0, lambda_bce=0.5, eps=1e-5):
        super().__init__()
        self.tau_unc = float(tau_unc)
        self.lambda_dice = float(lambda_dice)
        self.lambda_bce = float(lambda_bce)
        self.eps = eps

    def compute_teacher_confidence(self, teacher_logits):
        """
        teacher_logits: (B, 3, D, H, W)
        returns confidence weight W in [0, 1] of shape (B, 1, D, H, W)
        """
        p_t = torch.sigmoid(teacher_logits.detach())
        # Multi-label binary entropy across channels
        entropy = - (p_t * torch.log(p_t + self.eps) + (1.0 - p_t) * torch.log(1.0 - p_t + self.eps))
        mean_entropy = entropy.mean(dim=1, keepdim=True)  # (B, 1, D, H, W)
        
        # Exponential confidence weight: high certainty -> W ~ 1.0; high entropy -> W ~ 0.2
        confidence = torch.exp(- mean_entropy / self.tau_unc)
        return confidence

    def forward(self, student_logits, teacher_logits):
        """
        student_logits: (B, 3, D, H, W)
        teacher_logits: (B, 3, D, H, W) - detached target
        """
        t_target = teacher_logits.detach()
        confidence = self.compute_teacher_confidence(t_target)  # (B, 1, D, H, W)

        p_s = torch.sigmoid(student_logits)
        p_t = torch.sigmoid(t_target)

        dims = (2, 3, 4)
        # 1. Uncertainty-weighted Soft Dice
        w_inter = torch.sum(confidence * p_s * p_t, dim=dims)
        w_card = torch.sum(confidence * (p_s * p_s + p_t * p_t), dim=dims)
        w_dice_channel = (2.0 * w_inter + self.eps) / (w_card + self.eps)
        loss_dice = 1.0 - w_dice_channel.mean()

        # 2. Uncertainty-weighted BCE
        bce_voxel = F.binary_cross_entropy_with_logits(student_logits, p_t, reduction='none')
        loss_bce = (confidence * bce_voxel).sum() / (confidence.expand_as(bce_voxel).sum() + self.eps)

        total_ukd = self.lambda_dice * loss_dice + self.lambda_bce * loss_bce
        return total_ukd, loss_dice, loss_bce, confidence.mean()


__all__ = ["UncertaintyWeightedKDLoss", "DiceLoss4BraTS", "BCELoss4BraTS"]
