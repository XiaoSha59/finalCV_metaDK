"""
Class-Decoupled Uncertainty Knowledge Distillation (CDU-KD) Loss
================================================================
Core innovation of Plan 1 Extended:
Resolves the Enhancing Tumor (ET) gradient starvation problem in multi-modal
brain tumor segmentation under missing modalities.

Mathematical Formulation:
-------------------------
Standard MetaKD applies a unified entropy threshold tau to all classes:
    M(x) = exp(- H(p_t(x)) / tau)
Since Whole Tumor (WT/Edema) has fuzzy, high-entropy boundaries, global filtering
penalizes WT severely. Conversely, Enhancing Tumor (ET) is small and sparse,
causing its signal to be wiped out if thresholding is too aggressive or uniform.

CDU-KD decouples the confidence estimation per anatomical class:
    1. Channel 0 (ET): Filtered with adaptive tau_et (0.50 -> 0.25).
    2. Channel 1 (WT): Full confidence 1.0 (tau_wt = None), preserving 100% of the edema boundary.
    3. Channel 2 (TC): Filtered with sharp boundary constraint (tau_tc = 0.50).

Loss Composition:
-----------------
    L_CDUKD = lambda_dice * L_Dice_Decoupled + lambda_bce * L_BCE_Decoupled
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss4BraTS(nn.Module):
    """
    Soft Dice loss for 3D multi-label binary segmentation (ET, WT, TC).
    Computes per-channel Dice score across spatial dimensions (D, H, W).
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        """
        Args:
            logits:  (B, 3, D, H, W) raw model predictions.
            targets: (B, 3, D, H, W) binary ground truth masks.
        """
        probs = torch.sigmoid(logits)
        dims = (2, 3, 4)
        intersection = torch.sum(probs * targets, dim=dims)
        cardinality = torch.sum(probs * probs + targets * targets, dim=dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()


class BCELoss4BraTS(nn.Module):
    """
    Binary Cross Entropy with Logits for BraTS multi-label targets.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets.float())


class ClassDecoupledUncertaintyKDLoss(nn.Module):
    """
    Class-Decoupled Uncertainty Knowledge Distillation (CDU-KD) Loss.

    Args:
        tau_et (float): Temperature for Enhancing Tumor (ET) confidence mask. Default: 0.25.
        tau_tc (float): Temperature for Tumor Core (TC) confidence mask. Default: 0.50.
        tau_wt (float or None): Temperature for Whole Tumor (WT). None denotes full preservation (conf=1.0).
        lambda_dice (float): Weight for weighted soft dice distillation loss. Default: 1.0.
        lambda_bce (float): Weight for weighted BCE distillation loss. Default: 0.5.
        eps (float): Numerical stability epsilon. Default: 1e-5.
    """
    def __init__(self, tau_et=0.25, tau_tc=0.5, tau_wt=None, lambda_dice=1.0, lambda_bce=0.5, eps=1e-5):
        super().__init__()
        self.tau_et = float(tau_et)
        self.tau_tc = float(tau_tc)
        self.tau_wt = float(tau_wt) if tau_wt is not None else None
        self.lambda_dice = float(lambda_dice)
        self.lambda_bce = float(lambda_bce)
        self.eps = eps

    def compute_channel_confidences(self, teacher_logits):
        """
        Estimates per-voxel predictive confidence from the Teacher network
        independently for each anatomical region.

        Args:
            teacher_logits: (B, 3, D, H, W) logits from Teacher (Full modalities 0,1,2,3).
        Returns:
            confidence: (B, 3, D, H, W) decoupled confidence weights.
        """
        p_t = torch.sigmoid(teacher_logits.detach())

        # Per-channel Shannon entropy: H(p) = - [p log(p) + (1-p) log(1-p)]
        entropy = - (p_t * torch.log(p_t + self.eps) + (1.0 - p_t) * torch.log(1.0 - p_t + self.eps))

        # Channel 0: Enhancing Tumor (ET)
        conf_et = torch.exp(- entropy[:, 0:1] / self.tau_et)

        # Channel 1: Whole Tumor (WT) - Full preservation to protect edema boundaries
        if self.tau_wt is None:
            conf_wt = torch.ones_like(entropy[:, 1:2])
        else:
            conf_wt = torch.exp(- entropy[:, 1:2] / self.tau_wt)

        # Channel 2: Tumor Core (TC)
        conf_tc = torch.exp(- entropy[:, 2:3] / self.tau_tc)

        confidence = torch.cat([conf_et, conf_wt, conf_tc], dim=1)
        return confidence

    def forward(self, student_logits, teacher_logits):
        """
        Computes decoupled knowledge distillation loss between Student and Teacher.

        Args:
            student_logits: (B, 3, D, H, W) logits predicted by Student under missing modality.
            teacher_logits: (B, 3, D, H, W) logits predicted by Teacher with all 4 modalities.
        Returns:
            total_loss: scalar combined distillation loss.
            loss_dice:  scalar weighted Dice distillation loss.
            loss_bce:   scalar weighted BCE distillation loss.
            mean_confs: tuple of (mean_conf_et, mean_conf_wt, mean_conf_tc).
        """
        t_target = teacher_logits.detach()
        confidence = self.compute_channel_confidences(t_target)

        p_s = torch.sigmoid(student_logits)
        p_t = torch.sigmoid(t_target)

        dims = (2, 3, 4)
        # 1. Channel-decoupled Weighted Soft Dice Loss
        w_inter = torch.sum(confidence * p_s * p_t, dim=dims)
        w_card = torch.sum(confidence * (p_s * p_s + p_t * p_t), dim=dims)
        dice_channels = (2.0 * w_inter + self.eps) / (w_card + self.eps)
        loss_dice = 1.0 - dice_channels.mean()

        # 2. Channel-decoupled Weighted BCE Loss
        bce_voxels = F.binary_cross_entropy_with_logits(student_logits, p_t, reduction='none')
        loss_bce = (confidence * bce_voxels).sum() / (confidence.sum() + self.eps)

        total_loss = self.lambda_dice * loss_dice + self.lambda_bce * loss_bce

        mean_conf_et = confidence[:, 0].mean()
        mean_conf_wt = confidence[:, 1].mean()
        mean_conf_tc = confidence[:, 2].mean()

        return total_loss, loss_dice, loss_bce, (mean_conf_et, mean_conf_wt, mean_conf_tc)


__all__ = [
    "ClassDecoupledUncertaintyKDLoss",
    "DiceLoss4BraTS",
    "BCELoss4BraTS"
]
