"""
Training Script for AudioVision-MNIST with Uncertainty KD (U-MetaKD)
===================================================================
Runs AudioVision missing modality training (Table 3 setting: 5%, 10%, 15%, 20%)
with Uncertainty-modulated knowledge distillation.

Key Upgrades vs Baseline audiovision/process/train.py:
1. Complete Teacher pass generates teacher logits and predictive entropy H(p_T).
2. Distillation loss weights each digit sample by u_k = exp(-H / tau).
3. Adds uncertainty-weighted soft logit distillation to sharpen student predictions.
"""

import os
import sys
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AV_ROOT = os.path.join(REPO_DIR, "audiovision")
for p in [REPO_DIR, AV_ROOT, os.path.join(AV_ROOT, "process")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from audiovision.data.dataset import get_audiovision_loaders
from audiovision.process.models import MetaKDAudioVision
from audiovision.process.train import evaluate_accuracy, _meta_update_weights
from improvements.audiovision_uncertainty.loss_audiovision_ukd import UncertaintyCrossModalKDLoss


def train_metakd_audiovision_ukd(
    missing_modality: str = "audio",
    available_rate: float = 0.10,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    kd_weight: float = 0.1,
    tau: float = 1.0,
    device: str = None,
    seed: int = 42,
    verbose: bool = True,
):
    if missing_modality in ("visual", "image"):
        missing_modality = "visual"
        target_eval_mode = "audio_only"
    else:
        missing_modality = "audio"
        target_eval_mode = "image_only"

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, test_loader = get_audiovision_loaders(
        batch_size=batch_size,
        missing_modality=missing_modality,
        available_rate=available_rate,
        seed=seed,
    )

    model = MetaKDAudioVision().to(device)
    network_params = [param for name, param in model.named_parameters() if "kd_weights" not in name]
    optimizer = optim.Adam(network_params, lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    kd_optim = optim.Adam(model.kd_weights.parameters(), lr=1e-2, weight_decay=1e-2)
    ce_loss_fn = nn.CrossEntropyLoss()
    ukd_loss_fn = UncertaintyCrossModalKDLoss(tau=tau)

    best_val_acc = 0.0
    best_test_acc = 0.0
    val_iter = itertools.cycle(val_loader)

    if verbose:
        print(f"\n--- Training AudioVision U-MetaKD: missing={missing_modality}, rate={available_rate*100:.0f}% (eval={target_eval_mode}) ---", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        tot_conf = 0.0
        n_steps = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            audios = batch["audio"].to(device)
            labels = batch["label"].to(device)
            visual_mask = batch["visual_mask"].to(device)
            audio_mask = batch["audio_mask"].to(device)

            # 1. Outer meta step on validation batch
            val_batch = next(val_iter)
            _meta_update_weights(
                model,
                val_batch["image"].to(device),
                val_batch["audio"].to(device),
                val_batch["label"].to(device),
                kd_optim,
                ce_loss_fn,
            )

            # 2. Complete teacher pass (both audio + vision available)
            with torch.no_grad():
                img_feat_t = model.img_enc(images)
                aud_feat_t = model.aud_enc(audios)
                teacher_logits = model.fuse(img_feat_t, aud_feat_t)

            # 3. Student forward pass with simulated missingness
            optimizer.zero_grad()
            student_logits, _ = model(
                images,
                audios,
                visual_mask=visual_mask,
                audio_mask=audio_mask,
                eval_mode="full",
            )
            img_feat_s, aud_feat_s = model.encode(images, audios)

            # 4. Uncertainty-modulated distillation
            weights = model.kd_weights()
            paired_mask = (visual_mask & audio_mask)
            ukd_loss, avg_conf = ukd_loss_fn(
                img_feat_s,
                aud_feat_s,
                w_v=weights[0],
                w_a=weights[1],
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                paired_mask=paired_mask,
            )

            step_loss = ce_loss_fn(student_logits, labels) + kd_weight * ukd_loss
            step_loss.backward()
            optimizer.step()

            total_loss += step_loss.item()
            tot_conf += avg_conf.item()
            n_steps += 1

        scheduler.step()
        val_acc = evaluate_accuracy(model, val_loader, eval_mode=target_eval_mode, device=device)
        test_acc = evaluate_accuracy(model, test_loader, eval_mode=target_eval_mode, device=device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            ckpt_dir = os.path.join(REPO_DIR, "improvements", "audiovision_uncertainty", "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": best_val_acc,
                    "test_acc": best_test_acc,
                    "missing_modality": missing_modality,
                    "available_rate": available_rate,
                },
                os.path.join(ckpt_dir, f"ukd_{missing_modality}_rate_{int(available_rate*100)}pct.pth")
            )

        if verbose and (epoch % 10 == 0 or epoch == 1):
            w = model.kd_weights().detach().cpu().numpy()
            print(f"  Epoch {epoch:02d}/{epochs} | loss {total_loss/max(1,n_steps):.4f} "
                  f"conf {tot_conf/max(1,n_steps):.2f} | "
                  f"Val {val_acc:5.2f}% (Best {best_val_acc:5.2f}%) | Test {test_acc:5.2f}% | "
                  f"IWV [V:{w[0]:.2f}, A:{w[1]:.2f}]", flush=True)

    if verbose:
        print(f"--> Finished. Best Val: {best_val_acc:.2f}%, Corresponding Test Acc: {best_test_acc:.2f}%", flush=True)

    return best_test_acc


def run_table3_experiments():
    import csv
    rates = [0.05, 0.10, 0.15, 0.20]
    print("=" * 60, flush=True)
    print("Running AudioVision-MNIST Uncertainty KD Table 3 Benchmark (CPU)", flush=True)
    print("=" * 60, flush=True)

    print("\n[Table 3a: Missing Audio -> Test Image Only]", flush=True)
    res_3a = {}
    for r in rates:
        res_3a[r] = train_metakd_audiovision_ukd(missing_modality="audio", available_rate=r, epochs=60, device="cpu")

    print("\n[Table 3b: Missing Visual -> Test Audio Only]", flush=True)
    res_3b = {}
    for r in rates:
        res_3b[r] = train_metakd_audiovision_ukd(missing_modality="visual", available_rate=r, epochs=60, device="cpu")

    print("\n" + "=" * 60, flush=True)
    print("Summary of AudioVision Table 3 (U-MetaKD / CPU)", flush=True)
    print("=" * 60, flush=True)
    print("Available Rate:       5%      10%     15%     20%", flush=True)
    print("Table 3a (Image): " + "  ".join(f"{res_3a[r]:6.2f}%" for r in rates), flush=True)
    print("Table 3b (Audio): " + "  ".join(f"{res_3b[r]:6.2f}%" for r in rates), flush=True)

    # Save to CSV
    os.makedirs(os.path.join(REPO_DIR, "logs"), exist_ok=True)
    csv_path = os.path.join(REPO_DIR, "logs", "audiovision_ukd_table3.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Setting", "5%", "10%", "15%", "20%"])
        writer.writerow(["Table 3a (Image Test)"] + [f"{res_3a[r]:.2f}" for r in rates])
        writer.writerow(["Table 3b (Audio Test)"] + [f"{res_3b[r]:.2f}" for r in rates])
    print(f"\n[Saved] Official CSV saved to: {csv_path}", flush=True)

    return res_3a, res_3b


if __name__ == "__main__":
    run_table3_experiments()

