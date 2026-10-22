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


class SemanticPrototypeAlignmentLoss(nn.Module):
    """
    Semantic Prototype Alignment Loss (ProtoKD) for Incomplete Multi-Modal Brain Tumor Segmentation.
    
    Inspired by ProtoKD (MICCAI 2023 / IEEE TMI 2024).
    Extracts class semantic prototypes for ET (Enhancing Tumor), WT (Whole Tumor), and TC (Tumor Core)
    via masked average pooling over bottleneck features.
    Aligns the student's semantic cluster centers with teacher prototypes while repelling distinct class prototypes.
    """
    def __init__(self, lambda_sep=0.1, margin=0.2, eps=1e-5):
        super().__init__()
        self.lambda_sep = float(lambda_sep)
        self.margin = float(margin)
        self.eps = eps

    def compute_class_prototypes(self, features, masks):
        """
        features: (B, C, D, H, W)
        masks: (B, K, D, H, W) binary or continuous masks for K classes
        Returns: prototypes dict mapping class_idx -> (C,) normalized tensor or None
        """
        B, C, D, H, W = features.shape
        K = masks.shape[1]
        prototypes = []

        for k in range(K):
            m = masks[:, k:k+1, :, :, :]  # (B, 1, D, H, W)
            m_sum = m.sum()
            if m_sum < 1.0:
                prototypes.append(None)
                continue
            # Weighted average pooling over batch and spatial dims
            proto = (features * m).sum(dim=(0, 2, 3, 4)) / (m_sum + self.eps)  # (C,)
            proto = F.normalize(proto, p=2, dim=0)
            prototypes.append(proto)

        return prototypes

    def forward(self, student_features, teacher_features, labels):
        """
        student_features: (B, C, D, H, W)
        teacher_features: (B, C, D, H, W) - detached target
        labels: (B, 3, D_orig, H_orig, W_orig) or downsampled
        """
        t_feats = teacher_features.detach()
        B, C, D, H, W = student_features.shape

        # Downsample labels to match feature spatial dimensions
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


__all__ = ["SemanticPrototypeAlignmentLoss", "DiceLoss4BraTS", "BCELoss4BraTS"]
