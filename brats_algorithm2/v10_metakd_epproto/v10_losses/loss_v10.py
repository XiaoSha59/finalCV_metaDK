import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
V10_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
ALG2_DIR = os.path.abspath(os.path.join(V10_DIR, ".."))
REPO_DIR = os.path.abspath(os.path.join(ALG2_DIR, ".."))
BRATS_DIR = os.path.abspath(os.path.join(REPO_DIR, "brats"))

for p in [CURRENT_DIR, V10_DIR, ALG2_DIR, REPO_DIR, BRATS_DIR, os.path.join(BRATS_DIR, "process")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from losses.loss_dual import DiceLoss4BraTS, BCELoss4BraTS
except ImportError:
    from process.losses.loss_dual import DiceLoss4BraTS, BCELoss4BraTS


class DecoupledUncertaintyWeightedKDLoss(nn.Module):
    """
    Region-Decoupled Evidential Uncertainty Knowledge Distillation Loss.
    
    Upgrades Job 7's loss by:
    1. Decoupling uncertainty estimation per anatomical sub-region (ET, WT, TC).
    2. Protecting Whole Tumor (WT) via Bounded Uncertainty:
       W_WT = max(exp(-H_WT / tau), min_wt_conf)
       This prevents boundary collapse at the fuzzy edema perimeter.
    3. Preserving sharp confidence on Enhancing Tumor (ET) and Tumor Core (TC).
    """
    def __init__(self, tau_unc=0.5, min_wt_conf=0.70, lambda_dice=1.0, lambda_bce=0.5, eps=1e-5):
        super().__init__()
        self.tau_unc = float(tau_unc)
        self.min_wt_conf = float(min_wt_conf)
        self.lambda_dice = float(lambda_dice)
        self.lambda_bce = float(lambda_bce)
        self.eps = eps

    def compute_decoupled_confidence(self, teacher_logits):
        """
        teacher_logits: (B, 3, D, H, W)
        returns confidence weight W of shape (B, 3, D, H, W) in [0, 1]
        """
        p_t = torch.sigmoid(teacher_logits.detach())
        # Channel-wise entropy: (B, 3, D, H, W)
        entropy = - (p_t * torch.log(p_t + self.eps) + (1.0 - p_t) * torch.log(1.0 - p_t + self.eps))
        
        # Raw exponential confidence
        confidence = torch.exp(- entropy / self.tau_unc)  # (B, 3, D, H, W)

        # Protect channel 1 (WT / Whole Tumor): Bounded margin to prevent edema under-segmentation
        conf_et = confidence[:, 0:1]
        conf_wt = torch.clamp(confidence[:, 1:2], min=self.min_wt_conf, max=1.0)
        conf_tc = confidence[:, 2:3]

        decoupled_conf = torch.cat([conf_et, conf_wt, conf_tc], dim=1)
        return decoupled_conf

    def forward(self, student_logits, teacher_logits):
        t_target = teacher_logits.detach()
        conf = self.compute_decoupled_confidence(t_target)  # (B, 3, D, H, W)

        p_s = torch.sigmoid(student_logits)
        p_t = torch.sigmoid(t_target)

        dims = (2, 3, 4)
        # 1. Decoupled Uncertainty Soft Dice
        w_inter = torch.sum(conf * p_s * p_t, dim=dims)
        w_card = torch.sum(conf * (p_s * p_s + p_t * p_t), dim=dims)
        w_dice_channel = (2.0 * w_inter + self.eps) / (w_card + self.eps)
        loss_dice = 1.0 - w_dice_channel.mean()

        # 2. Decoupled Uncertainty BCE
        bce_voxel = F.binary_cross_entropy_with_logits(student_logits, p_t, reduction='none')
        loss_bce = (conf * bce_voxel).sum() / (conf.sum() + self.eps)

        total_ukd = self.lambda_dice * loss_dice + self.lambda_bce * loss_bce
        return total_ukd, loss_dice, loss_bce, conf.mean()


class SemanticPrototypeAlignmentLoss(nn.Module):
    """
    Class Semantic Prototype Alignment Loss (ProtoKD from Job 9).
    Extracts class semantic prototypes for ET, WT, TC via masked average pooling over bottleneck features.
    Aligns student's semantic cluster centers with teacher prototypes while repelling distinct class prototypes.
    """
    def __init__(self, lambda_sep=0.1, margin=0.2, eps=1e-5):
        super().__init__()
        self.lambda_sep = float(lambda_sep)
        self.margin = float(margin)
        self.eps = eps

    def compute_class_prototypes(self, features, masks):
        B, C, D, H, W = features.shape
        K = masks.shape[1]
        prototypes = []

        for k in range(K):
            m = masks[:, k:k+1, :, :, :]
            m_sum = m.sum()
            if m_sum < 1.0:
                prototypes.append(None)
                continue
            proto = (features * m).sum(dim=(0, 2, 3, 4)) / (m_sum + self.eps)
            proto = F.normalize(proto, p=2, dim=0)
            prototypes.append(proto)

        return prototypes

    def forward(self, student_features, teacher_features, labels):
        t_feats = teacher_features.detach()
        B, C, D, H, W = student_features.shape

        if labels.shape[2:] != (D, H, W):
            masks = F.interpolate(labels.float(), size=(D, H, W), mode='nearest')
        else:
            masks = labels.float()

        proto_s = self.compute_class_prototypes(student_features, masks)
        proto_t = self.compute_class_prototypes(t_feats, masks)

        loss_align = torch.tensor(0.0, device=student_features.device)
        valid_align_count = 0

        # 1. Intra-class Prototype Alignment (Cosine Distance)
        for k in range(len(proto_s)):
            if proto_s[k] is not None and proto_t[k] is not None:
                cos_sim = torch.dot(proto_s[k], proto_t[k])
                loss_align = loss_align + (1.0 - cos_sim)
                valid_align_count += 1

        if valid_align_count > 0:
            loss_align = loss_align / valid_align_count

        # 2. Inter-class Prototype Separation (Repel different tumor structures)
        loss_sep = torch.tensor(0.0, device=student_features.device)
        valid_sep_count = 0

        for i in range(len(proto_s)):
            for j in range(len(proto_t)):
                if i != j and proto_s[i] is not None and proto_t[j] is not None:
                    inter_sim = torch.dot(proto_s[i], proto_t[j])
                    loss_sep = loss_sep + torch.clamp(inter_sim - self.margin, min=0.0)
                    valid_sep_count += 1

        if valid_sep_count > 0:
            loss_sep = loss_sep / valid_sep_count

        total_proto = loss_align + self.lambda_sep * loss_sep
        return total_proto, loss_align, loss_sep


class TopologicalContainmentLoss(nn.Module):
    """
    Differentiable Biological Containment Penalty.
    Enforces the anatomical BraTS hierarchy: ET subset of TC subset of WT during backpropagation.
    Penalizes:
      1. P_ET > P_TC
      2. P_TC > P_WT
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        probs = torch.sigmoid(logits)
        p_et = probs[:, 0:1]
        p_wt = probs[:, 1:2]
        p_tc = probs[:, 2:3]

        viol_et_tc = F.relu(p_et - p_tc)
        viol_tc_wt = F.relu(p_tc - p_wt)

        loss = viol_et_tc.mean() + viol_tc_wt.mean()
        return loss


__all__ = [
    "DecoupledUncertaintyWeightedKDLoss",
    "SemanticPrototypeAlignmentLoss",
    "TopologicalContainmentLoss",
    "DiceLoss4BraTS",
    "BCELoss4BraTS",
]
