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


class MoFeRankingLoss(nn.Module):
    """
    More vs. Fewer (MoFe) Ranking Loss from SimMLM (ICCV 2025, Eq. 3).
    
    Formula:
        L_MoFe(o+, o-, y) = max(0, l_task(o+, y) - l_task(o-, y))
        
    Enforces that the task loss with more modalities (o+) must be at least
    no higher than with fewer modalities (o-). Penalizes any instance where
    removing a modality artificially lowers the loss or having more modalities
    degrades performance.
    """
    def __init__(self, margin=0.0):
        super().__init__()
        self.margin = margin

    def forward(self, loss_more, loss_fewer):
        """
        loss_more: task loss with richer/complete modality set (scalar or per-batch)
        loss_fewer: task loss with poorer/missing modality set (scalar or per-batch)
        """
        # Penalize if loss_more > loss_fewer - margin
        diff = loss_more - loss_fewer + self.margin
        return torch.relu(diff)


__all__ = ["MoFeRankingLoss", "DiceLoss4BraTS", "BCELoss4BraTS"]
