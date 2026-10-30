import os
import sys
import torch
import torch.nn as nn
import numpy as np

if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    nn.Module.cuda = lambda self, *args, **kwargs: self

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ALG2_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
REPO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

for p in [CURRENT_DIR, ALG2_DIR, REPO_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from v10_models.dual_net_v10 import DualNet_V10
from v10_losses.loss_v10 import (
    DecoupledUncertaintyWeightedKDLoss,
    SemanticPrototypeAlignmentLoss,
    TopologicalContainmentLoss,
)
from v10_utils.postprocess_v10 import adaptive_postprocess_case


def test_v10_components():
    print(">>> [TEST 1/4] Testing DualNet_V10 Forward Pass...")
    model = DualNet_V10(num_classes=3)
    model.eval()
    dummy_input = torch.randn(1, 4, 32, 32, 32)
    preds, mode_split, kd_loss, feats = model(dummy_input, mode='random', return_feat=True)
    assert preds.shape == (1, 3, 32, 32, 32), f"Unexpected preds shape {preds.shape}"
    assert feats.shape[0] == 1, f"Unexpected feature batch {feats.shape}"
    print(f"    Forward Pass OK! Preds: {preds.shape}, Feats: {feats.shape}, Mode: {mode_split}")

    print(">>> [TEST 2/4] Testing Decoupled Uncertainty Loss...")
    ukd_loss_fn = DecoupledUncertaintyWeightedKDLoss(tau_unc=0.5, min_wt_conf=0.70)
    teacher_preds = torch.randn(1, 3, 32, 32, 32)
    conf = ukd_loss_fn.compute_decoupled_confidence(teacher_preds)
    assert conf.shape == (1, 3, 32, 32, 32), f"Unexpected conf shape {conf.shape}"
    assert conf[:, 1:2].min() >= 0.70 - 1e-4, f"WT Confidence floor violated: {conf[:, 1:2].min()}"
    l_ukd, l_dice, l_bce, mean_c = ukd_loss_fn(preds, teacher_preds)
    assert l_ukd.item() >= 0.0
    print(f"    Decoupled Uncertainty Loss OK! Loss: {l_ukd.item():.4f}, Mean Conf: {mean_c.item():.4f}, WT Min Conf: {conf[:, 1:2].min():.4f}")

    print(">>> [TEST 3/4] Testing Semantic Prototype Alignment Loss & Topo Loss...")
    proto_loss_fn = SemanticPrototypeAlignmentLoss()
    topo_loss_fn = TopologicalContainmentLoss()
    dummy_labels = torch.randint(0, 2, (1, 3, 32, 32, 32)).float()
    teacher_feats = torch.randn_like(feats)
    l_proto, l_align, l_sep = proto_loss_fn(feats, teacher_feats, dummy_labels)
    l_topo = topo_loss_fn(preds)
    assert l_proto.item() >= 0.0
    assert l_topo.item() >= 0.0
    print(f"    Proto Loss: {l_proto.item():.4f} (align={l_align.item():.4f}, sep={l_sep.item():.4f}) | Topo Loss: {l_topo.item():.4f}")

    print(">>> [TEST 4/4] Testing Standard Post-processing...")
    mock_bin = np.random.randint(0, 2, (3, 16, 16, 16)).astype(np.uint8)
    pp_clean = adaptive_postprocess_case(mock_bin, min_speckle_size=5)
    # Check topological containment ET <= TC <= WT
    et = pp_clean[0]
    wt = pp_clean[1]
    tc = pp_clean[2]
    assert np.all(et <= tc), "Postprocessing violation: ET not subset of TC"
    assert np.all(tc <= wt), "Postprocessing violation: TC not subset of WT"
    print("    Standard Post-processing OK! Hierarchy ET subset TC subset WT strictly satisfied.")

    print("\n========================================================")
    print(">>> ALL V10 COMPONENT TESTS PASSED PERFECTLY!")
    print("========================================================")


if __name__ == '__main__':
    test_v10_components()
