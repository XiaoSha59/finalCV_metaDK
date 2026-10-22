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

from v9_models.dual_net_v9 import DualNet_V9
from v9_losses.loss_v9 import SemanticPrototypeAlignmentLoss


def test_v9_forward_and_loss():
    print("=== Testing V9: MetaKD-Proto Local Smoke Test ===")
    B, C, D, H, W = 1, 4, 32, 32, 32
    images = torch.randn(B, C, D, H, W)
    labels = (torch.rand(B, 3, D, H, W) > 0.8).float()

    model = DualNet_V9(num_classes=3)
    model.eval()

    # 1. Forward complete mode (Teacher) with feature return
    with torch.no_grad():
        preds_tea, mode_t, kd_tea, feat_tea = model(images, mode="0,1,2,3", return_feat=True)
    print(f"[OK] Teacher forward: preds={preds_tea.shape}, feats={feat_tea.shape}")

    # 2. Forward missing mode (Student: 0, 2)
    preds_stu, mode_s, kd_stu, feat_stu = model(images, mode="0,2", return_feat=True)
    print(f"[OK] Student forward: preds={preds_stu.shape}, feats={feat_stu.shape}")

    # 3. Prototype Alignment Loss
    loss_fn = SemanticPrototypeAlignmentLoss(lambda_sep=0.1, margin=0.2)
    total_proto, l_align, l_sep = loss_fn(feat_stu, feat_tea, labels)
    print(f"[OK] Proto Loss: total={total_proto.item():.4f}, align={l_align.item():.4f}, sep={l_sep.item():.4f}")

    # 4. Backward check
    total_proto.backward()
    print("[OK] Backward pass successful, gradients computed.")
    print("=== All V9 Tests PASSED! ===")


if __name__ == "__main__":
    test_v9_forward_and_loss()
