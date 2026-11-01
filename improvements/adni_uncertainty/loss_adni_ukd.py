"""
Class-Decoupled Uncertainty KD Loss for ADNI Multimodal Classification
======================================================================
Formulates Uncertainty-Aware Knowledge Distillation for multi-class
tabular classification (CN: Cognitively Normal, MCI: Mild Cognitive Impairment, AD: Alzheimer's Disease).

Mathematical Formulation:
-------------------------
1. Predictive Entropy:
   For sample k, let p_k = Softmax(logits_k) in R^3.
   H(p_k) = - sum_{c=1}^3 p_{k,c} * log(p_{k,c} + eps)

2. Class-Decoupled Confidence Modulation:
   - CN (c=0): tau_cn = 0.50 (clean normal cognitive profile)
   - MCI (c=1): tau_mci = 0.80 or bounded min_conf = 0.50 (inherently transitional/borderline class)
   - AD (c=2): tau_ad = 0.50 (clear pathology)

   u_k = exp(- H(p_k) / tau_{c_k}), where c_k = argmax(p_k).
   For MCI, confidence is bounded to avoid starving transitional cases:
   u_k = max(u_k, min_mci_conf) if c_k == 1.

3. Uncertainty-Weighted Pairwise Feature Distillation:
   L_UKD = sum_k u_k * [ sum_{i,j} (w*_i / w*_j) * delta_i * delta_j * ||f_i - f_j||_1 ] / sum_k u_k
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassDecoupledUncertaintyLossADNI(nn.Module):
    def __init__(self, tau_cn=0.5, tau_mci=0.8, tau_ad=0.5, min_mci_conf=0.5, eps=1e-6):
        super().__init__()
        self.tau_cn = float(tau_cn)
        self.tau_mci = float(tau_mci)
        self.tau_ad = float(tau_ad)
        self.min_mci_conf = float(min_mci_conf)
        self.eps = eps

    def compute_sample_confidence(self, teacher_logits):
        """
        teacher_logits: (N, 3) logits from full/observed modalities
        Returns:
            conf: (N,) confidence weights in (0, 1]
            entropy: (N,) Shannon entropy values
        """
        probs = F.softmax(teacher_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + self.eps), dim=-1)  # (N,)

        pred_classes = torch.argmax(probs, dim=-1)  # (N,)
        taus = torch.where(
            pred_classes == 1,
            torch.tensor(self.tau_mci, device=teacher_logits.device),
            torch.where(
                pred_classes == 0,
                torch.tensor(self.tau_cn, device=teacher_logits.device),
                torch.tensor(self.tau_ad, device=teacher_logits.device)
            )
        )

        conf = torch.exp(-entropy / taus)

        # Protect transitional MCI class from excessive pruning
        is_mci = (pred_classes == 1)
        if is_mci.any():
            conf[is_mci] = torch.clamp(conf[is_mci], min=self.min_mci_conf)

        return conf, entropy

    def forward(self, feats, active, iwv_weights, teacher_logits=None):
        """
        feats: (N, n_mod, latent_dim)
        active: (N, n_mod) boolean mask of active modalities
        iwv_weights: (n_mod,) softmax importance weights
        teacher_logits: (N, 3) optional teacher predictions for entropy calculation
        """
        N, n_mod, d = feats.shape
        active = active.float()

        if teacher_logits is not None:
            conf, _ = self.compute_sample_confidence(teacher_logits.detach())
        else:
            conf = torch.ones(N, device=feats.device)

        loss = torch.tensor(0.0, device=feats.device)
        total_weight = torch.tensor(0.0, device=feats.device)

        # Pairwise distillation across active modalities
        for i in range(n_mod):
            for j in range(n_mod):
                if i == j:
                    continue
                pair_mask = active[:, i] * active[:, j]  # (N,)
                valid = pair_mask > 0
                if not valid.any():
                    continue

                diff = torch.abs(feats[:, i] - feats[:, j]).mean(dim=-1)  # (N,) mean L1 per sample
                ratio = iwv_weights[i] / (iwv_weights[j] + self.eps)

                weighted_diff = (diff * pair_mask * conf)
                loss = loss + ratio * weighted_diff.sum()
                total_weight = total_weight + (pair_mask * conf).sum()

        if total_weight > 0:
            loss = loss / (total_weight + self.eps)

        return loss, conf.mean()
