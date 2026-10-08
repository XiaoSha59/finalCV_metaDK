"""
AudioVision-MNIST Experiment Runner.
Reproduces Table 3a (Missing Audio) and Table 3b (Missing Visual) from MetaKD Paper (arXiv:2405.07155).
"""

import time
import torch
from audiovision.train import train_metakd_audiovision, evaluate_accuracy
from audiovision.dataset import get_audiovision_loaders
from audiovision.models import MetaKDAudioVision
import torch.nn as nn
import torch.optim as optim


def train_baseline(modality: str = "image_only", epochs: int = 50, batch_size: int = 32, lr: float = 0.001, device: str = "cpu", seed: int = 42):
    """
    Trains Lower Bound (Single Modality) or Upper Bound (Full Modality).
    """
    torch.manual_seed(seed)
    train_loader, val_loader, test_loader = get_audiovision_loaders(
        batch_size=batch_size, missing_modality="none", available_rate=1.0, seed=seed
    )

    model = MetaKDAudioVision(num_classes=10, feat_dim=128).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99, nesterov=True, weight_decay=1e-4)
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            images = batch['image'].to(device)
            audios = batch['audio'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            if modality == "image_only":
                logits, _ = model(images, None, eval_mode="image_only")
            elif modality == "audio_only":
                logits, _ = model(None, audios, eval_mode="audio_only")
            else:
                logits, _, _, _ = model(images, audios, eval_mode="full")

            loss = ce_loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        val_acc = evaluate_accuracy(model, val_loader, eval_mode=modality, device=device)
        test_acc = evaluate_accuracy(model, test_loader, eval_mode=modality, device=device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

    return best_test_acc


def run_all_experiments():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("[*] REPRODUCING METAKD ON AUDIOVISION-MNIST BENCHMARK")
    print(f"Hardware Device: {device.upper()}")
    print("=" * 70)

    rates = [0.05, 0.10, 0.15, 0.20]
    
    # --- Lower & Upper Bounds ---
    print("\n[1/3] Training Lower Bounds & Upper Bound Baselines...")
    lower_b_img = train_baseline(modality="image_only", epochs=40, device=device)
    lower_b_aud = train_baseline(modality="audio_only", epochs=40, device=device)
    upper_b_full = train_baseline(modality="full", epochs=40, device=device)

    print(f"  * Lower Bound (Image-only LowerB): {lower_b_img:.2f}%")
    print(f"  * Lower Bound (Audio-only LowerB): {lower_b_aud:.2f}%")
    print(f"  * Upper Bound (Full Modalities UpperB): {upper_b_full:.2f}%")

    # --- Table 3a: Missing Audio ---
    print("\n[2/3] Running Table 3a Experiments (Missing Audio during training, Image-only testing)...")
    tab3a_results = {}
    for rate in rates:
        start_t = time.time()
        acc, ckpt_path = train_metakd_audiovision(
            missing_modality="audio", available_rate=rate, epochs=100, batch_size=32, device=device, verbose=False
        )
        elapsed = time.time() - start_t
        tab3a_results[f"{int(rate*100)}%"] = acc
        print(f"  * Audio Rate {int(rate*100):2d}% -> Test Accuracy: {acc:.2f}% | Saved: {ckpt_path} (Time: {elapsed:.1f}s)")

    # --- Table 3b: Missing Visual ---
    print("\n[3/3] Running Table 3b Experiments (Missing Visual during training, Audio-only testing)...")
    tab3b_results = {}
    for rate in rates:
        start_t = time.time()
        acc, ckpt_path = train_metakd_audiovision(
            missing_modality="visual", available_rate=rate, epochs=100, batch_size=32, device=device, verbose=False
        )
        elapsed = time.time() - start_t
        tab3b_results[f"{int(rate*100)}%"] = acc
        print(f"  * Visual Rate {int(rate*100):2d}% -> Test Accuracy: {acc:.2f}% | Saved: {ckpt_path} (Time: {elapsed:.1f}s)")

    # --- Summary Tables ---
    print("\n" + "=" * 70)
    print("TABLE 3a: Performance on Missing Audio (Image-only Test)")
    print("=" * 70)
    print(f"| Model / Rate | 5% | 10% | 15% | 20% | UpperB |")
    print(f"|:---|:---:|:---:|:---:|:---:|:---:|")
    print(f"| LowerB (Image-only) | {lower_b_img:.2f}% | {lower_b_img:.2f}% | {lower_b_img:.2f}% | {lower_b_img:.2f}% | - |")
    print(f"| **MetaKD (Ours)** | **{tab3a_results['5%']:.2f}%** | **{tab3a_results['10%']:.2f}%** | **{tab3a_results['15%']:.2f}%** | **{tab3a_results['20%']:.2f}%** | **{upper_b_full:.2f}%** |")

    print("\n" + "=" * 70)
    print("TABLE 3b: Performance on Missing Visual (Audio-only Test)")
    print("=" * 70)
    print(f"| Model / Rate | 5% | 10% | 15% | 20% | UpperB |")
    print(f"|:---|:---:|:---:|:---:|:---:|:---:|")
    print(f"| LowerB (Audio-only) | {lower_b_aud:.2f}% | {lower_b_aud:.2f}% | {lower_b_aud:.2f}% | {lower_b_aud:.2f}% | - |")
    print(f"| **MetaKD (Ours)** | **{tab3b_results['5%']:.2f}%** | **{tab3b_results['10%']:.2f}%** | **{tab3b_results['15%']:.2f}%** | **{tab3b_results['20%']:.2f}%** | **{upper_b_full:.2f}%** |")
    print("=" * 70)


if __name__ == "__main__":
    run_all_experiments()
