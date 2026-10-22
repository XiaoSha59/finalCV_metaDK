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


class PromptDiversityRegularizer(nn.Module):
    """
    Prompt Orthogonality / Diversity Regularizer to prevent prompt collapse.
    Ensures that learnable modality prompts P_m for FLAIR, T1, T1ce, T2 remain distinct.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, prompts):
        """
        prompts: (4, C, 1, 1, 1)
        """
        flat_p = prompts.view(4, -1)
        norm_p = F.normalize(flat_p, p=2, dim=1)
        sim_matrix = torch.matmul(norm_p, norm_p.t())  # (4, 4)
        eye = torch.eye(4, device=prompts.device)
        off_diag = sim_matrix - eye
        loss_ortho = (off_diag ** 2).sum() / 12.0  # Mean of off-diagonal squared similarities
        return loss_ortho


__all__ = ["PromptDiversityRegularizer", "DiceLoss4BraTS", "BCELoss4BraTS"]
