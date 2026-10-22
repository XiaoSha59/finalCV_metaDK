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
    from models.dual_net import DualNet, KD_Weights, U_Res3D_enc, U_Res3D_dec
except ImportError:
    from process.models.dual_net import DualNet, KD_Weights, U_Res3D_enc, U_Res3D_dec

# DualNet_V4 retains the exact 100% pure backbone of MetaKD DualNet
DualNet_V4 = DualNet

__all__ = ["DualNet_V4", "DualNet", "KD_Weights", "U_Res3D_enc", "U_Res3D_dec"]
