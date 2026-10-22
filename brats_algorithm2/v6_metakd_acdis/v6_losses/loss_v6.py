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


class AnatomicalConsistencyLoss(nn.Module):
    """
    Anatomical Consistency ConsTraints (ACCT) Loss from ACDIS (ECAI 2024 / arXiv:2408.13733).
    
    Decomposes knowledge distillation into:
    1. Local Variance Loss (L_Var, Eq. 1):
       Enforces mono-modal/student representations to match the information richness
       and boundary contrast of the multi-modal teacher representation.
       
    2. Local Covariance Loss (L_Covar, Eq. 3):
       Measures anatomical gradient alignment and variation trends using L1-normalized covariance:
       L_Covar = 1 - (|cov(f_s, f_t)| + eps) / (sigma_s * sigma_t + eps)
       
    Total:
       L_ACCT = L_Var + L_Covar
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, student_ft, teacher_ft):
        """
        student_ft: (B, C, D, H, W)
        teacher_ft: (B, C, D, H, W) - multi-modal feature target
        """
        target = teacher_ft.detach()

        # Sigmoid normalization as specified in ACDIS Section 3.1.1
        f_s = torch.sigmoid(student_ft)
        f_t = torch.sigmoid(target)

        # Spatial dimensions (D, H, W)
        dims = (2, 3, 4)
        mu_s = f_s.mean(dim=dims, keepdim=True)
        mu_t = f_t.mean(dim=dims, keepdim=True)

        delta_s = f_s - mu_s
        delta_t = f_t - mu_t

        var_s = (delta_s ** 2).mean(dim=dims)
        var_t = (delta_t ** 2).mean(dim=dims)

        sigma_s = torch.sqrt(var_s + self.eps)
        sigma_t = torch.sqrt(var_t + self.eps)

        # 1. Variance Loss (Eq. 1)
        l_var = 1.0 - (2.0 * sigma_s * sigma_t + self.eps) / (var_s + var_t + self.eps)

        # 2. Covariance Loss with L1 norm on gradient magnitude (Eq. 3)
        cov = (delta_s * delta_t).mean(dim=dims)
        l_covar = 1.0 - (torch.abs(cov) + self.eps) / (sigma_s * sigma_t + self.eps)

        total_acct = l_var.mean() + l_covar.mean()
        return total_acct, l_var.mean(), l_covar.mean()


__all__ = ["AnatomicalConsistencyLoss", "DiceLoss4BraTS", "BCELoss4BraTS"]
