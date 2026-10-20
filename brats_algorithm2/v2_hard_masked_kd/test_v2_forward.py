"""
Smoke test to verify forward/backward and zero-NaN safety of DualNet_Algorithm2.
"""
import sys
import os
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from v2_models.dual_net_v2 import DualNet_Algorithm2
from v2_losses.loss_v2 import RegionDecomposedLoss


def test_dry_run():
    print(">>> Running Algorithm 2 Dry-Run Smoke Test...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test dummy inputs: Batch=1, Modalities=4, Depth=32, Height=64, Width=64 (standard patch resolution)
    B, C, D, H, W = 1, 4, 32, 64, 64
    dummy_images = torch.randn(B, C, D, H, W, device=device)
    # Binary labels for 3 regions: ET, WT, TC
    dummy_labels = (torch.rand(B, 3, D, H, W, device=device) > 0.7).float()

    for use_shaspec in [True, False]:
        print(f"\n--- Testing with use_shaspec={use_shaspec} ---")
        model = DualNet_Algorithm2(
            norm_cfg='IN',
            activation_cfg='LeakyReLU',
            num_classes=3,
            weight_std=False,
            use_shaspec=use_shaspec
        ).to(device)

        loss_fn = RegionDecomposedLoss().to(device)

        # 1. Inner Loop Forward Pass (Train Mode with missing modalities)
        model.train()
        preds, mode_split, kd_loss, loss_ortho, loss_shared = model(
            dummy_images, labels=dummy_labels, val=False, mode='random'
        )
        print(f"Active modalities: {mode_split}")
        print(f"Preds shape: {preds.shape}")
        print(f"KD Loss: {kd_loss.item():.5f}, Ortho Loss: {loss_ortho.item():.5f}, Shared Loss: {loss_shared.item():.5f}")

        # Check for NaNs
        assert not torch.isnan(preds).any(), "NaN detected in predictions!"
        assert not torch.isnan(kd_loss), "NaN detected in KD loss!"
        assert not torch.isnan(loss_ortho), "NaN detected in Ortho loss!"
        assert not torch.isnan(loss_shared), "NaN detected in Shared loss!"

        # 2. Loss & Backward Pass
        seg_loss, (l_et, l_wt, l_tc) = loss_fn(preds, dummy_labels)
        total_loss = seg_loss + kd_loss + loss_ortho + loss_shared
        total_loss.backward()
        print(f"Total loss backward executed successfully! Loss = {total_loss.item():.4f}")

        # 3. Outer Loop Validation Forward (val=True)
        model.eval()
        with torch.no_grad():
            val_preds, _, _, _, _ = model(dummy_images, val=True, mode='0,1,2,3')
            print(f"Validation preds shape: {val_preds.shape}")
            assert not torch.isnan(val_preds).any(), "NaN detected in val predictions!"

        norm_w = model.region_weights.get_normalized_weights()
        print(f"Normalized Region Weights shape: {norm_w.shape}")
        print(f"Weights sum per region (should be 1.0): {norm_w.sum(dim=1).cpu().detach().numpy()}")

    print("\n[SUCCESS] ALL SMOKE TESTS PASSED WITH 100% SUCCESS!")


if __name__ == '__main__':
    test_dry_run()
