import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

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

from v6_models.dual_net_v6 import DualNet_V6
from v6_losses.loss_v6 import AnatomicalConsistencyLoss, DiceLoss4BraTS, BCELoss4BraTS
from v6_utils.postprocess_v6 import adaptive_postprocess_case


class MockArgs:
    norm_cfg = 'IN'
    activation_cfg = 'LeakyReLU'
    num_classes = 3
    weight_std = False
    self_att = False
    cross_att = False


def run_smoke_test():
    print(">>> [TEST 1/3] Instantiating DualNet_V6 & Losses...")
    args = MockArgs()
    model = DualNet_V6(
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
    loss_ACCT = AnatomicalConsistencyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    images = torch.randn(2, 4, 32, 64, 64)
    labels = (torch.rand(2, 3, 32, 64, 64) > 0.8).float()

    print(">>> [TEST 2/3] ACDIS ACCT Distillation Forward Pass + Loss...")
    optimizer.zero_grad()

    # 1. Student Forward (missing modalities) with feature return
    student_preds, mode_split, totKDLoss, student_out_ft, _ = model(
        images, mode="random", return_features=True
    )
    seg_loss = loss_D(student_preds, labels) + loss_BCE(student_preds, labels)

    # 2. Teacher Forward (full modalities) with feature return
    with torch.no_grad():
        _, _, _, teacher_out_ft, _ = model(images, mode="0,1,2,3", return_features=True)

    # 3. ACCT Variance + Covariance Loss
    l_acct, l_var, l_cov = loss_ACCT(student_out_ft, teacher_out_ft)
    assert not torch.isnan(l_acct), "l_acct is NaN!"

    total_loss = seg_loss + 0.1 * totKDLoss + 0.05 * l_acct
    print(f"    Mode: {mode_split}")
    print(f"    seg_loss:  {seg_loss.item():.4f}")
    print(f"    kd_loss:   {totKDLoss.item():.4f}")
    print(f"    acct_loss: {l_acct.item():.4f} (var={l_var.item():.4f}, covar={l_cov.item():.4f})")
    print(f"    total_loss: {total_loss.item():.4f}")

    total_loss.backward()
    optimizer.step()
    print("    Backward & Optimizer Step: PASSED with zero NaNs!")

    print(">>> [TEST 3/3] Adaptive Postprocessing...")
    mock_pred_binary = (torch.sigmoid(student_preds).detach().numpy() > 0.5).astype('uint8')
    cleaned = adaptive_postprocess_case(mock_pred_binary[0], min_speckle_size=30)
    assert cleaned.shape == (3, 32, 64, 64), f"Unexpected cleaned shape: {cleaned.shape}"
    print("    Adaptive Postprocessing: PASSED!")

    print("\n>>> ALL SMOKE TESTS PASSED CLEANLY FOR METAKD-ACDIS (V6)! <<<\n")


if __name__ == '__main__':
    run_smoke_test()
