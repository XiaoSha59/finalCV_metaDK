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


class RegionDecoupledUncertaintyKDLoss(nn.Module):
    """
    Region-Decoupled Evidential Uncertainty Knowledge Distillation Loss for BraTS 3D Segmentation.
    
    Directly addresses and rectifies the fatal limitation of Job 7:
    - Job 7 averaged entropy across all 3 channels, causing the blurry boundaries of Whole Tumor (WT)
      to be severely penalized, dropping WT Dice from 85.90% to 83.21%.
    - V11 decouples uncertainty per sub-region:
        * ET (k=0): tau_ET = 0.5 (sharp confidence filter: prevents hallucinating ET without T1ce)
        * WT (k=1): tau_WT = 1.0 (relaxed confidence filter: protects blurry WT edema boundaries)
        * TC (k=2): tau_TC = 0.5 (sharp confidence filter: ensures solid Tumor Core precision)
    """
    def __init__(self, tau_ET=0.5, tau_WT=1.0, tau_TC=0.5, lambda_dice=1.0, lambda_bce=0.5, eps=1e-5):
        super().__init__()
        self.tau_ET = float(tau_ET)
        self.tau_WT = float(tau_WT)
        self.tau_TC = float(tau_TC)
        self.lambda_dice = float(lambda_dice)
        self.lambda_bce = float(lambda_bce)
        self.eps = eps

    def compute_channel_confidence(self, teacher_logits):
        """
        Computes region-specific confidence masks:
        teacher_logits: (B, 3, D, H, W) where:
          - Channel 0: ET (Enhancing Tumor)
          - Channel 1: WT (Whole Tumor)
          - Channel 2: TC (Tumor Core)
        Returns:
          confidence: (B, 3, D, H, W) in [0, 1]
        """
        p_t = torch.sigmoid(teacher_logits.detach())
        # Voxel-wise binary entropy per channel
        entropy = - (p_t * torch.log(p_t + self.eps) + (1.0 - p_t) * torch.log(1.0 - p_t + self.eps))

        conf_ET = torch.exp(- entropy[:, 0:1] / self.tau_ET)
        conf_WT = torch.exp(- entropy[:, 1:2] / self.tau_WT)
        conf_TC = torch.exp(- entropy[:, 2:3] / self.tau_TC)

        return torch.cat([conf_ET, conf_WT, conf_TC], dim=1)  # (B, 3, D, H, W)

    def forward(self, student_logits, teacher_logits, kd_weights_module=None, mode_split=None):
        """
        student_logits: (B, 3, D, H, W)
        teacher_logits: (B, 3, D, H, W) - detached target
        kd_weights_module: KD_Weights_Region module (optional, for region weighting)
        mode_split: list of available modality indices (e.g., [0, 3])
        """
        t_target = teacher_logits.detach()
        confidence = self.compute_channel_confidence(t_target)  # (B, 3, D, H, W)

        p_s = torch.sigmoid(student_logits)
        p_t = torch.sigmoid(t_target)

        dims = (2, 3, 4)
        # 1. Uncertainty-weighted Soft Dice per channel
        w_inter = torch.sum(confidence * p_s * p_t, dim=dims)
        w_card = torch.sum(confidence * (p_s * p_s + p_t * p_t), dim=dims)
        w_dice_channel = (2.0 * w_inter + self.eps) / (w_card + self.eps)
        loss_dice_per_channel = 1.0 - w_dice_channel  # (B, 3)

        # 2. Uncertainty-weighted BCE per channel
        bce_voxel = F.binary_cross_entropy_with_logits(student_logits, p_t, reduction='none')
        # Sum over spatial dims for each channel
        loss_bce_per_channel = (confidence * bce_voxel).sum(dim=dims) / (confidence.sum(dim=dims) + self.eps)  # (B, 3)

        # Total UKD loss per channel (ET, WT, TC)
        loss_ukd_per_channel = self.lambda_dice * loss_dice_per_channel + self.lambda_bce * loss_bce_per_channel  # (B, 3)

        # Optional region weighting modulation from meta-learned W matrix
        if kd_weights_module is not None and mode_split is not None:
            norm_W = kd_weights_module.get_normalized_weights()  # (3, 4)
            # Modality coverage for each region: sum of weights of available modalities
            region_coverage = norm_W[:, mode_split].sum(dim=1)  # (3,)
            region_coverage_norm = region_coverage / (region_coverage.sum() + self.eps)
            # Weighted channel distillation
            total_ukd = (loss_ukd_per_channel * region_coverage_norm.unsqueeze(0)).sum(dim=1).mean()
        else:
            total_ukd = loss_ukd_per_channel.mean()

        return (
            total_ukd,
            loss_dice_per_channel[:, 0].mean(),  # ET Dice loss
            loss_dice_per_channel[:, 1].mean(),  # WT Dice loss
            loss_dice_per_channel[:, 2].mean(),  # TC Dice loss
            confidence[:, 0].mean(),             # ET mean confidence
            confidence[:, 1].mean(),             # WT mean confidence
            confidence[:, 2].mean()              # TC mean confidence
        )


__all__ = ["RegionDecoupledUncertaintyKDLoss", "DiceLoss4BraTS", "BCELoss4BraTS"]
