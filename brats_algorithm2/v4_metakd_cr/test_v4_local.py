import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix cuda stubs for local machine without GPU if needed
if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    nn.Module.cuda = lambda self, *args, **kwargs: self

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ALG2_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
REPO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
BRATS_DIR = os.path.abspath(os.path.join(REPO_DIR, "brats"))

for p in [CURRENT_DIR, ALG2_DIR, BRATS_DIR, os.path.join(BRATS_DIR, "process"), REPO_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from v4_models.dual_net_v4 import DualNet_V4
from v4_losses.loss_v4 import ConsistencyRegularizationLoss, DiceLoss4BraTS, BCELoss4BraTS
from v4_utils.postprocess_v4 import adaptive_postprocess_case


class MockArgs:
    norm_cfg = 'IN'
    activation_cfg = 'LeakyReLU'
    num_classes = 3
    weight_std = False
    self_att = False
    cross_att = False


def run_smoke_test():
    print(">>> [TEST 1/3] Instantiating DualNet_V4 & Losses...")
    args = MockArgs()
    model = DualNet_V4(
        args=args,
        norm_cfg=args.norm_cfg,
        activation_cfg=args.activation_cfg,
        num_classes=args.num_classes,
        weight_std=args.weight_std,
        self_att=args.self_att,
        cross_att=args.cross_att
    )
    model.train()

    loss_D = DiceLoss4BraTS()
    loss_BCE = BCELoss4BraTS()
    loss_CR = ConsistencyRegularizationLoss(temperature=4.0, lambda_dice=1.0, lambda_bce=0.5)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Patch size: B=2, C=4, D=32, H=64, W=64
    images = torch.randn(2, 4, 32, 64, 64)
    labels = (torch.rand(2, 3, 32, 64, 64) > 0.8).float()

    print(">>> [TEST 2/3] Forward Pass: Student (missing) + Teacher (full) + Loss...")
    optimizer.zero_grad()

    # 1. Student Forward
    student_preds, mode_split, totKDLoss = model(images, mode="random")
    assert student_preds.shape == (2, 3, 32, 64, 64), f"Unexpected student_preds shape: {student_preds.shape}"

    # 2. Supervised Loss
    seg_loss = loss_D(student_preds, labels) + loss_BCE(student_preds, labels)
    assert not torch.isnan(seg_loss), "seg_loss is NaN!"

    # 3. Teacher Forward & Consistency Regularization
    with torch.no_grad():
        teacher_preds, _, _ = model(images, mode="0,1,2,3")
    assert teacher_preds.shape == (2, 3, 32, 64, 64), f"Unexpected teacher_preds shape: {teacher_preds.shape}"

    l_cr, l_dice, l_bce = loss_CR(student_preds, teacher_preds)
    assert not torch.isnan(l_cr), "l_cr is NaN!"

    total_loss = seg_loss + 0.1 * totKDLoss + 0.05 * l_cr
    print(f"    Mode: {mode_split}")
    print(f"    seg_loss: {seg_loss.item():.4f}")
    print(f"    kd_loss:  {totKDLoss.item():.4f}")
    print(f"    cr_loss:  {l_cr.item():.4f} (dice={l_dice.item():.4f}, bce={l_bce.item():.4f})")
    print(f"    total_loss: {total_loss.item():.4f}")

    total_loss.backward()
    optimizer.step()
    print("    Backward & Optimizer Step: PASSED with zero NaNs!")

    print(">>> [TEST 3/3] Adaptive Postprocessing...")
    mock_pred_binary = (torch.sigmoid(student_preds).detach().numpy() > 0.5).astype('uint8')
    cleaned = adaptive_postprocess_case(mock_pred_binary[0], min_speckle_size=30)
    assert cleaned.shape == (3, 32, 64, 64), f"Unexpected cleaned shape: {cleaned.shape}"
    print("    Adaptive Postprocessing: PASSED!")

    print("\n>>> ALL SMOKE TESTS PASSED CLEANLY FOR METAKD-CR (V4)! <<<\n")


if __name__ == '__main__':
    run_smoke_test()
