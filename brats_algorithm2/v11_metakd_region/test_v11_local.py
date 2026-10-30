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

from v11_models.dual_net_v11 import DualNet_V11, KD_Weights_Region
from v11_losses.loss_v11 import RegionDecoupledUncertaintyKDLoss, DiceLoss4BraTS, BCELoss4BraTS


def test_v11_smoke():
    print("================================================================")
    print("SMOKE TEST: MetaKD-V11 Region-Disentangled Architecture & Loss")
    print("================================================================")
    
    # 1. Test KD_Weights_Region Module
    print("[1/5] Testing KD_Weights_Region Module...")
    kd_w = KD_Weights_Region(num_regions=3, num_modalities=4)
    norm_w = kd_w.get_normalized_weights()
    eff_w = kd_w.get_effective_modality_weights()
    assert norm_w.shape == (3, 4), f"Expected (3, 4), got {norm_w.shape}"
    assert eff_w.shape == (4,), f"Expected (4,), got {eff_w.shape}"
    assert torch.allclose(norm_w.sum(dim=1), torch.ones(3)), "Each region row must sum to 1.0"
    print(f"      Initial Region Weights shape: {norm_w.shape}")
    print(f"      Initial Effective Modality Weights: {eff_w.detach().numpy()}")
    print("      [PASSED] KD_Weights_Region verified.")

    # 2. Test DualNet_V11 Model Forward
    print("\n[2/5] Testing DualNet_V11 Forward Pass...")
    B, C, D, H, W = 1, 4, 32, 32, 32
    images = torch.randn(B, C, D, H, W)
    labels = (torch.rand(B, 3, D, H, W) > 0.8).float()

    model = DualNet_V11(num_classes=3)
    model.train()

    # Complete mode (Teacher)
    with torch.no_grad():
        preds_tea, mode_t, kd_loss_tea = model(images, mode="0,1,2,3")
    assert preds_tea.shape == (B, 3, D, H, W), f"Expected ({B}, 3, {D}, {H}, {W}), got {preds_tea.shape}"
    print(f"      Teacher forward: logits shape={preds_tea.shape}, mode={mode_t}")

    # Missing mode (Student: e.g., only FLAIR and T1ce)
    preds_stu, mode_s, kd_loss_stu = model(images, mode="0,2")
    assert preds_stu.shape == (B, 3, D, H, W)
    print(f"      Student forward (0,2 missing): logits shape={preds_stu.shape}, kd_loss={kd_loss_stu.item():.4f}")
    print("      [PASSED] DualNet_V11 forward verified.")

    # 3. Test Region-Decoupled Uncertainty KD Loss
    print("\n[3/5] Testing RegionDecoupledUncertaintyKDLoss...")
    loss_ukd_fn = RegionDecoupledUncertaintyKDLoss(tau_ET=0.5, tau_WT=1.0, tau_TC=0.5)
    loss_ukd, l_et, l_wt, l_tc, c_et, c_wt, c_tc = loss_ukd_fn(
        preds_stu, preds_tea,
        kd_weights_module=model.kd_weights,
        mode_split=mode_s
    )
    print(f"      UKD Total Loss: {loss_ukd.item():.4f}")
    print(f"      Sub-region Dice Losses: ET={l_et.item():.4f}, WT={l_wt.item():.4f}, TC={l_tc.item():.4f}")
    print(f"      Sub-region Confidences: ET={c_et.item():.3f}, WT={c_wt.item():.3f} (Relaxed), TC={c_tc.item():.3f}")
    assert 0.0 <= c_et.item() <= 1.0 and 0.0 <= c_wt.item() <= 1.0 and 0.0 <= c_tc.item() <= 1.0
    print("      [PASSED] RegionDecoupledUncertaintyKDLoss verified.")

    # 4. Test Backward Pass on Decoder & Prediction Head
    print("\n[4/5] Testing Backward Pass on Model Parameters...")
    model.zero_grad()
    loss_ukd.backward()

    dec_grads = [p.grad.norm().item() for p in model.shared_dec.parameters() if p.grad is not None and not torch.isnan(p.grad).any()]
    assert len(dec_grads) > 0 and any(g > 0 for g in dec_grads), "Gradients must flow to decoder parameters"
    print(f"      Active Decoder Layers with Gradients: {len(dec_grads)}/41")
    print("      [PASSED] Backward pass verified.")

    # 5. Test Meta-Update Gradient Flow on kd_weights (val=True)
    print("\n[5/5] Testing Bi-level Meta-Update Flow on kd_weights (val=True)...")
    model.zero_grad()
    val_out, _, _ = model(images, val=True, mode="0,1,2,3")
    loss_D = DiceLoss4BraTS()
    val_loss = loss_D(val_out, labels)
    
    model.kd_weights.kd_weights.grad = None
    val_loss.backward()
    
    assert model.kd_weights.kd_weights.grad is not None, "kd_weights must receive gradients during validation!"
    w_grad = model.kd_weights.kd_weights.grad
    print(f"      Meta Weights Gradient shape: {w_grad.shape}")
    print(f"      Meta Weights Gradient norm: {w_grad.norm().item():.6f}")
    assert w_grad.norm().item() > 0, "Meta weights gradient norm must be > 0"
    print("      [PASSED] Bi-level meta-update verified.")

    print("\n================================================================")
    print("ALL V11 TESTS COMPLETED SUCCESSFULLY! ZERO ERRORS!")
    print("================================================================")


if __name__ == "__main__":
    test_v11_smoke()
