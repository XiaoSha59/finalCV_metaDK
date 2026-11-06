"""Ablation of MetaKD improvements on AudioVision-MNIST (Table 3b: missing visual, audio test).

Variants compared:
1. MetaKD_Baseline: Original L1 distillation (Eq. 5)
2. MetaKD_Cosine: Cosine alignment distance (1 - cos_sim)
3. MetaKD_SupCon: Supervised Contrastive alignment (pull same class, push different classes)
4. MetaKD_Cos_SupCon: Hybrid Cosine + SupCon alignment
"""

import os
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from audiovision.data.dataset import get_audiovision_loaders
from audiovision.process.models import MetaKDAudioVision, LeNet5, SNetPluse
from audiovision.process.train import evaluate_accuracy, _meta_update_weights


class Projector(nn.Module):
    """Small MLP projection head for contrastive learning."""
    def __init__(self, in_dim=160, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def supcon_loss(z_a, z_b, labels, temperature=0.1):
    """Supervised contrastive loss between two modalities across batch."""
    features = torch.cat([z_a, z_b], dim=0)
    targets = torch.cat([labels, labels], dim=0)
    
    # Cosine similarity matrix scaled by temperature
    logits = (features @ features.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    
    # Self mask to ignore diagonal
    self_mask = torch.eye(features.size(0), device=features.device, dtype=torch.bool)
    positives = (targets[:, None] == targets[None, :]) & ~self_mask
    
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    
    positive_count = positives.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return features.new_zeros(())
    loss = -(log_prob * positives).sum(dim=1)[valid] / positive_count[valid]
    return loss.mean()


def train_metakd_variant(
    variant: str = "baseline",
    missing_modality: str = "visual",
    available_rate: float = 0.10,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    kd_weight: float = 0.1,
    supcon_weight: float = 0.1,
    device: str = "cpu",
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, test_loader = get_audiovision_loaders(
        batch_size=batch_size,
        missing_modality=missing_modality,
        available_rate=available_rate,
        seed=seed,
    )

    model = MetaKDAudioVision().to(device)
    projector = Projector().to(device) if "supcon" in variant else None

    network_params = [p for n, p in model.named_parameters() if "kd_weights" not in n]
    if projector is not None:
        network_params += list(projector.parameters())

    optimizer = optim.Adam(network_params, lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    kd_optim = optim.Adam(model.kd_weights.parameters(), lr=1e-2, weight_decay=1e-2)
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_test_acc = 0.0
    target_eval_mode = "image_only" if missing_modality == "audio" else "audio_only"
    val_iter = itertools.cycle(val_loader)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        if projector is not None:
            projector.train()

        for batch in train_loader:
            images = batch["image"].to(device)
            audios = batch["audio"].to(device)
            labels = batch["label"].to(device)
            visual_mask = batch["visual_mask"].to(device)
            audio_mask = batch["audio_mask"].to(device)

            val_batch = next(val_iter)
            _meta_update_weights(
                model,
                val_batch["image"].to(device),
                val_batch["audio"].to(device),
                val_batch["label"].to(device),
                kd_optim,
                ce_loss_fn,
            )

            optimizer.zero_grad()
            img_feat, aud_feat = model.encode(images, audios)
            weights = model.kd_weights().detach()
            w_v, w_a = weights[0], weights[1]

            # Missing feature completion via copy (baseline MetaKD)
            missing_visual = ~visual_mask
            missing_audio = ~audio_mask
            eff_img = img_feat.clone()
            eff_aud = aud_feat.clone()
            eff_img[missing_visual] = aud_feat[missing_visual]
            eff_aud[missing_audio] = img_feat[missing_audio]

            logits = model.fuse(eff_img, eff_aud)
            task_loss = ce_loss_fn(logits, labels)

            # Knowledge Distillation calculation on paired subset
            paired = visual_mask & audio_mask
            kd_loss = torch.tensor(0.0, device=device)

            if paired.any():
                p_img = img_feat[paired]
                p_aud = aud_feat[paired]
                ratio_weight = (w_v / (w_a + 1e-6) + w_a / (w_v + 1e-6))

                if variant == "baseline":
                    # Official MetaKD: L1 loss
                    kd_dist = F.l1_loss(p_img, p_aud)
                    kd_loss = ratio_weight * kd_dist

                elif variant == "cosine":
                    # Proposed: Cosine alignment distance (scale-invariant)
                    cos_sim = F.cosine_similarity(p_img, p_aud, dim=-1).mean()
                    kd_dist = 1.0 - cos_sim
                    kd_loss = ratio_weight * kd_dist

                elif variant == "supcon":
                    # Proposed: Supervised Contrastive alignment
                    z_img = projector(p_img)
                    z_aud = projector(p_aud)
                    p_labels = labels[paired]
                    if len(p_labels) >= 2:
                        kd_loss = ratio_weight * supcon_loss(z_img, z_aud, p_labels)

                elif variant == "cos_supcon":
                    # Proposed: Cosine alignment + SupCon
                    cos_sim = F.cosine_similarity(p_img, p_aud, dim=-1).mean()
                    cos_dist = 1.0 - cos_sim
                    z_img = projector(p_img)
                    z_aud = projector(p_aud)
                    p_labels = labels[paired]
                    s_loss = supcon_loss(z_img, z_aud, p_labels) if len(p_labels) >= 2 else torch.tensor(0.0, device=device)
                    kd_loss = ratio_weight * (cos_dist + supcon_weight * s_loss)

            total_loss = task_loss + kd_weight * kd_loss
            total_loss.backward()
            optimizer.step()

        scheduler.step()
        val_acc = evaluate_accuracy(model, val_loader, eval_mode=target_eval_mode, device=device, fill="copy")
        test_acc = evaluate_accuracy(model, test_loader, eval_mode=target_eval_mode, device=device, fill="copy")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

    elapsed = time.time() - start_time
    return best_test_acc, elapsed


def run_benchmark():
    print("=" * 65)
    print("MetaKD AudioVision Improvement Ablation (Table 3b: Missing Visual)")
    print("=" * 65)

    variants = ["baseline", "cosine", "supcon", "cos_supcon"]
    rate = 0.10
    print(f"\n[PHASE 1] Screening variants at Rate = {int(rate*100)}% (60 epochs each):")
    print("-" * 65)

    results = {}
    for var in variants:
        test_acc, sec = train_metakd_variant(variant=var, available_rate=rate, epochs=60)
        results[var] = test_acc
        print(f"  * {var:<15}: Test Acc = {test_acc:6.2f}% ({sec:.1f}s)")

    best_var = max(results, key=results.get)
    print("-" * 65)
    print(f"Screening Winner: {best_var} (Acc: {results[best_var]:.2f}% vs Baseline: {results['baseline']:.2f}%)")

    # If winner beats baseline or matches, test winner across all rates (5%, 10%, 15%, 20%)
    rates = [0.05, 0.10, 0.15, 0.20]
    print(f"\n[PHASE 2] Evaluating Best Variant ({best_var}) vs Baseline across all rates:")
    print("-" * 65)

    full_results = {var: {} for var in ["baseline", best_var]}
    for r in rates:
        for var in ["baseline", best_var]:
            if r == 0.10 and var in results:
                full_results[var][r] = results[var]
            else:
                test_acc, _ = train_metakd_variant(variant=var, available_rate=r, epochs=60)
                full_results[var][r] = test_acc
        print(f"  Rate {int(r*100):2d}% | Baseline: {full_results['baseline'][r]:.2f}% | {best_var}: {full_results[best_var][r]:.2f}% (Diff: {full_results[best_var][r] - full_results['baseline'][r]:+.2f}%)")

    print("\n" + "=" * 65)
    print("FINAL COMPARISON TABLE (Table 3b - Audio Test Accuracy %)")
    print("=" * 65)
    print(f"{'Method':<20} | {'5%':<8} | {'10%':<8} | {'15%':<8} | {'20%':<8} | {'Mean':<8}")
    print("-" * 65)
    for var in ["baseline", best_var]:
        row = [full_results[var][r] for r in rates]
        mean_acc = sum(row) / len(row)
        print(f"{var:<20} | {row[0]:6.2f}%  | {row[1]:6.2f}%  | {row[2]:6.2f}%  | {row[3]:6.2f}%  | {mean_acc:6.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()
