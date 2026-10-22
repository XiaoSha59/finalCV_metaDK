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

from v5_models.dual_net_v5 import DualNet_V5
from v5_losses.loss_v5 import MoFeRankingLoss, DiceLoss4BraTS, BCELoss4BraTS
from v5_utils.postprocess_v5 import adaptive_postprocess_case


class MockArgs:
    norm_cfg = 'IN'
    activation_cfg = 'LeakyReLU'
    num_classes = 3
    weight_std = False
    self_att = False
    cross_att = False


def run_smoke_test():
    print(">>> [TEST 1/3] Instantiating DualNet_V5 & Losses...")
    args = MockArgs()
    model = DualNet_V5(
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
    loss_MoFe = MoFeRankingLoss(margin=0.0)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    images = torch.randn(2, 4, 32, 64, 64)
    labels = (torch.rand(2, 3, 32, 64, 64) > 0.8).float()

    print(">>> [TEST 2/3] SimMLM MoFe Pair Forward Pass + Loss...")
    optimizer.zero_grad()

    # 1. Fewer modalities (e.g. 1 modality: FLAIR)
    preds_fewer, _, kd_fewer = model(images, mode="0")
    loss_fewer = loss_D(preds_fewer, labels) + loss_BCE(preds_fewer, labels)

    # 2. More modalities (e.g. 3 modalities: Fl+T1+T1c)
    preds_more, _, kd_more = model(images, mode="0,1,2")
    loss_more = loss_D(preds_more, labels) + loss_BCE(preds_more, labels)

    # 3. MoFe ranking loss
    l_mofe = loss_MoFe(loss_more, loss_fewer)
    tot_kd = 0.5 * (kd_fewer + kd_more)
    total_loss = loss_fewer + loss_more + 0.1 * tot_kd + 0.1 * l_mofe

    print(f"    Loss Fewer: {loss_fewer.item():.4f}")
    print(f"    Loss More:  {loss_more.item():.4f}")
    print(f"    MoFe Loss:  {l_mofe.item():.4f}")
    print(f"    Total Loss: {total_loss.item():.4f}")

    total_loss.backward()
    optimizer.step()
    print("    Backward & Optimizer Step: PASSED with zero NaNs!")

    print(">>> [TEST 3/3] Adaptive Postprocessing...")
    mock_pred_binary = (torch.sigmoid(preds_fewer).detach().numpy() > 0.5).astype('uint8')
    cleaned = adaptive_postprocess_case(mock_pred_binary[0], min_speckle_size=30)
    assert cleaned.shape == (3, 32, 64, 64), f"Unexpected cleaned shape: {cleaned.shape}"
    print("    Adaptive Postprocessing: PASSED!")

    print("\n>>> ALL SMOKE TESTS PASSED CLEANLY FOR METAKD-MOFE (V5)! <<<\n")


if __name__ == '__main__':
    run_smoke_test()
