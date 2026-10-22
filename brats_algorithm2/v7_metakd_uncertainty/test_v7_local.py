import os
import sys
import torch
import torch.nn as nn

if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    nn.Module.cuda = lambda self, *args, **kwargs: self

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ALG2_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
BRATS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "brats"))
for p in [CURRENT_DIR, ALG2_DIR, BRATS_DIR, os.path.join(BRATS_DIR, "process")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from v7_models.dual_net_v7 import DualNet_V7
from v7_losses.loss_v7 import UncertaintyWeightedKDLoss, DiceLoss4BraTS, BCELoss4BraTS


def test_v7_forward_and_loss():
    print("=== Testing V7: MetaKD-Uncertainty Local Smoke Test ===")
    B, C, D, H, W = 1, 4, 32, 32, 32
    images = torch.randn(B, C, D, H, W)
    labels = (torch.rand(B, 3, D, H, W) > 0.8).float()

    model = DualNet_V7(num_classes=3)
    model.eval()

    # 1. Forward complete mode (Teacher)
    with torch.no_grad():
        preds_tea, mode_split_t, kd_tea = model(images, mode="0,1,2,3")
    print(f"[OK] Teacher forward (complete): shape={preds_tea.shape}")

    # 2. Forward missing mode (Student: only FLAIR, T2)
    preds_stu, mode_split_s, kd_stu = model(images, mode="0,3")
    print(f"[OK] Student forward (0,3 missing): shape={preds_stu.shape}")

    # 3. Uncertainty-Weighted KD Loss
    ukd_loss = UncertaintyWeightedKDLoss(tau_unc=0.5)
    loss_val, l_dice, l_bce, mean_conf = ukd_loss(preds_stu, preds_tea)
    print(f"[OK] UKD Loss: total={loss_val.item():.4f}, dice={l_dice.item():.4f}, bce={l_bce.item():.4f}, conf={mean_conf.item():.4f}")

    # 4. Backward check
    loss_val.backward()
    print("[OK] Backward pass successful, gradients computed.")
    print("=== All V7 Tests PASSED! ===")


if __name__ == "__main__":
    test_v7_forward_and_loss()
