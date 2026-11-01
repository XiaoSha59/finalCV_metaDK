"""
Uncertainty-Modulated Knowledge Distillation for AudioVision-MNIST
==================================================================
Formulates Uncertainty-Aware Cross-Modal Distillation between Spoken Digit (Audio)
and Handwritten Digit (Vision) under missing modality scenarios.

Mathematical Formulation:
-------------------------
1. Full-Modality Teacher Probability:
   Given paired image x_v and audio x_a, the complete teacher generates joint logits:
   p_T = Softmax(fused_classifier(Concat(f_v, f_a))) in R^10.

2. Teacher Predictive Entropy:
   H(p_T) = - sum_{c=0}^9 p_{T,c} * log(p_{T,c} + eps)
   Max entropy for 10 classes is log(10) ≈ 2.3026.

3. Confidence Weight:
   u_k = exp(- H(p_{T,k}) / tau) in (0, 1].
   Samples with clear, unambiguous audio pronunciation and legible handwriting
   yield low entropy -> u_k ≈ 1.0 (high distillation gradient).
   Ambiguous samples (e.g. garbled noise or messy handwriting) yield high entropy -> u_k -> 0
   (prunes noisy cross-modal alignment signals).

4. Uncertainty-Weighted Cross-Modal Distillation:
   - Pairwise Feature Distillation: L_feat = (w_v/w_a + w_a/w_v) * sum_k (u_k * paired_k * ||f_v - f_a||_1) / sum_k (u_k * paired_k)
   - Soft Logit Distillation: L_logit = T^2 * sum_k (u_k * KL(Softmax(z_S/T) || Softmax(z_T/T))) / sum_k u_k
   - L_UKD = lambda_kd_feat * L_feat + lambda_kd_logit * L_logit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyCrossModalKDLoss(nn.Module):
    def __init__(self, tau=1.0, temperature_kd=4.0, lambda_kd_feat=1.0, lambda_kd_logit=0.5, eps=1e-6):
        super().__init__()
        self.tau = float(tau)
        self.temperature_kd = float(temperature_kd)
        self.lambda_kd_feat = float(lambda_kd_feat)
        self.lambda_kd_logit = float(lambda_kd_logit)
        self.eps = eps

    def compute_confidence(self, teacher_logits):
        """
        teacher_logits: (B, 10) logits from full audio+vision forward pass
        """
        probs = F.softmax(teacher_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + self.eps), dim=-1)  # (B,)
        conf = torch.exp(-entropy / self.tau)
        return conf, entropy

    def forward(self, img_feat, aud_feat, w_v, w_a, teacher_logits=None, student_logits=None, paired_mask=None):
        """
        img_feat: (B, 160)
        aud_feat: (B, 160)
        w_v, w_a: scalar importance weights from KDWeights2D
        teacher_logits: (B, 10) optional full-modality teacher logits
        student_logits: (B, 10) optional unimodal student logits
        paired_mask: (B,) boolean mask of samples having both visual and audio present
        """
        B = img_feat.size(0)

        if teacher_logits is not None:
            conf, _ = self.compute_confidence(teacher_logits.detach())
        else:
            conf = torch.ones(B, device=img_feat.device)

        # 1. Uncertainty-weighted feature L1 distance on paired samples
        pairwise_l1 = torch.abs(img_feat - aud_feat).mean(dim=-1)  # (B,)
        ratio_weight = (w_v / (w_a + self.eps) + w_a / (w_v + self.eps))

        if paired_mask is not None:
            mask = paired_mask.float() * conf
            if mask.sum() > 0:
                feat_loss = ratio_weight * (pairwise_l1 * mask).sum() / (mask.sum() + self.eps)
            else:
                feat_loss = torch.tensor(0.0, device=img_feat.device)
        else:
            feat_loss = ratio_weight * (pairwise_l1 * conf).sum() / (conf.sum() + self.eps)

        # 2. Uncertainty-weighted logit soft distillation
        logit_loss = torch.tensor(0.0, device=img_feat.device)
        if teacher_logits is not None and student_logits is not None:
            T = self.temperature_kd
            p_s = F.log_softmax(student_logits / T, dim=-1)
            p_t = F.softmax(teacher_logits.detach() / T, dim=-1)
            kl = F.kl_div(p_s, p_t, reduction='none').sum(dim=-1)  # (B,)
            logit_loss = (kl * conf).sum() / (conf.sum() + self.eps) * (T * T)

        total_kd = self.lambda_kd_feat * feat_loss + self.lambda_kd_logit * logit_loss
        return total_kd, conf.mean()
