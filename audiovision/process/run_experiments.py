"""
AudioVision-MNIST experiment runner.

Held-out split of the 1,500 pairs: utterance index 0-14 is the 450-file
test, and the remaining recordings are 900 train plus 150 meta-validation.
Table 3a tests images only. Table 3b tests audio only.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from audiovision.process.train import train_metakd_audiovision, evaluate_accuracy
from audiovision.data.dataset import get_audiovision_loaders
from audiovision.process.models import MetaKDAudioVision, LeNet5, SNetPluse

_AV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def train_baseline(
    modality: str = "image_only",
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
):
    """LowerB is a SMIL unimodal net. UpperB is the fusion net on both modalities."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, val_loader, test_loader = get_audiovision_loaders(
        batch_size=batch_size,
        missing_modality="none",
        available_rate=1.0,
        seed=seed,
    )

    if modality == "image_only":
        model = LeNet5().to(device)
        weight_decay = 1e-4
    elif modality == "audio_only":
        model = SNetPluse().to(device)
        weight_decay = 1e-4
    else:
        model = MetaKDAudioVision().to(device)
        weight_decay = 1e-2

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_state = None
    eval_mode = modality if modality in ("image_only", "audio_only") else "full"

    for _epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            audios = batch["audio"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            if isinstance(model, LeNet5):
                logits = model(images)
            elif isinstance(model, SNetPluse):
                logits = model(audios)
            else:
                logits, kd_loss = model(images, audios, eval_mode="full")
                # Upper bound sees every modality, so the KD term is not used.
                del kd_loss
            loss = ce_loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        val_acc = evaluate_accuracy(model, val_loader, eval_mode=eval_mode, device=device)
        test_acc = evaluate_accuracy(model, test_loader, eval_mode=eval_mode, device=device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    ckpt_dir = os.path.join(_AV_ROOT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    names = {"image_only": "lower_image", "audio_only": "lower_audio", "full": "upper_full"}
    ckpt_path = os.path.join(ckpt_dir, f"{names[modality]}.pth")
    torch.save(
        {"model_state_dict": best_state, "test_acc": best_test_acc, "modality": modality},
        ckpt_path,
    )
    return best_test_acc, best_state


def run_all_experiments():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("[*] AUDIOVISION-MNIST HELD-OUT SPLIT (900 TRAIN / 150 VAL / 450 TEST)")
    print(f"Hardware Device: {device.upper()}")
    print("=" * 70)

    rates = [0.05, 0.10, 0.15, 0.20]
    preview_train, preview_val, preview_test = get_audiovision_loaders(batch_size=32, seed=42)
    print(
        f"Split: train {len(preview_train.dataset)} / "
        f"val {len(preview_val.dataset)} / test {len(preview_test.dataset)}"
    )

    print("\n[1/3] Training Lower Bounds and Upper Bound (60 epochs)...")
    lower_b_img, image_state = train_baseline(modality="image_only", epochs=60, device=device)
    lower_b_aud, audio_state = train_baseline(modality="audio_only", epochs=60, device=device)
    upper_b_full, _full_state = train_baseline(modality="full", epochs=60, device=device)
    print(f"  * Lower Bound (LeNet-5, image only): {lower_b_img:.2f}%")
    print(f"  * Lower Bound (SNetPluse, audio only): {lower_b_aud:.2f}%")
    print(f"  * Upper Bound (both modalities): {upper_b_full:.2f}%")

    print("\n[2/3] Table 3a: missing audio, image-only test...")
    tab3a_results = {}
    for rate in rates:
        start_t = time.time()
        acc, ckpt_path = train_metakd_audiovision(
            missing_modality="audio",
            available_rate=rate,
            epochs=60,
            batch_size=32,
            device=device,
            verbose=False,
        )
        tab3a_results[f"{int(rate * 100)}%"] = acc
        print(
            f"  * Audio rate {int(rate * 100):2d}% -> Test accuracy: {acc:.2f}% | "
            f"Saved: {ckpt_path} (Time: {time.time() - start_t:.1f}s)"
        )

    print("\n[3/3] Table 3b: missing visual, audio-only test...")
    tab3b_results = {}
    for rate in rates:
        start_t = time.time()
        acc, ckpt_path = train_metakd_audiovision(
            missing_modality="visual",
            available_rate=rate,
            epochs=60,
            batch_size=32,
            device=device,
            verbose=False,
        )
        tab3b_results[f"{int(rate * 100)}%"] = acc
        print(
            f"  * Visual rate {int(rate * 100):2d}% -> Test accuracy: {acc:.2f}% | "
            f"Saved: {ckpt_path} (Time: {time.time() - start_t:.1f}s)"
        )

    print("\n" + "=" * 70)
    print("TABLE 3a: Missing audio, image-only test")
    print("=" * 70)
    print("| Model / Rate | 5% | 10% | 15% | 20% | UpperB |")
    print("|:---|:---:|:---:|:---:|:---:|:---:|")
    print(
        f"| LowerB (LeNet-5) | {lower_b_img:.2f}% | {lower_b_img:.2f}% | "
        f"{lower_b_img:.2f}% | {lower_b_img:.2f}% | - |"
    )
    print(
        f"| **MetaKD** | **{tab3a_results['5%']:.2f}%** | **{tab3a_results['10%']:.2f}%** | "
        f"**{tab3a_results['15%']:.2f}%** | **{tab3a_results['20%']:.2f}%** | "
        f"**{upper_b_full:.2f}%** |"
    )

    print("\n" + "=" * 70)
    print("TABLE 3b: Missing visual, audio-only test")
    print("=" * 70)
    print("| Model / Rate | 5% | 10% | 15% | 20% | UpperB |")
    print("|:---|:---:|:---:|:---:|:---:|:---:|")
    print(
        f"| LowerB (SNetPluse) | {lower_b_aud:.2f}% | {lower_b_aud:.2f}% | "
        f"{lower_b_aud:.2f}% | {lower_b_aud:.2f}% | - |"
    )
    print(
        f"| **MetaKD** | **{tab3b_results['5%']:.2f}%** | **{tab3b_results['10%']:.2f}%** | "
        f"**{tab3b_results['15%']:.2f}%** | **{tab3b_results['20%']:.2f}%** | "
        f"**{upper_b_full:.2f}%** |"
    )
    print("=" * 70)


if __name__ == "__main__":
    run_all_experiments()
