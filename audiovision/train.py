import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .dataset import get_audiovision_loaders
from .models import MetaKDAudioVision


def evaluate_accuracy(model: nn.Module, loader, eval_mode: str = "image_only", device: str = "cpu") -> float:
    """
    Computes top-1 classification accuracy on a DataLoader given the evaluation mode.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            audios = batch['audio'].to(device)
            labels = batch['label'].to(device)

            if eval_mode == "image_only":
                logits, _ = model(images=images, audios=None, eval_mode="image_only")
            elif eval_mode == "audio_only":
                logits, _ = model(images=None, audios=audios, eval_mode="audio_only")
            else:
                logits, _, _, _ = model(images=images, audios=audios, eval_mode="full")

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return (correct / total) * 100.0 if total > 0 else 0.0


def train_metakd_audiovision(
    missing_modality: str = "audio",  # "audio" (Tab 3a) or "visual" (Tab 3b)
    available_rate: float = 0.10,     # 0.05, 0.10, 0.15, 0.20, 1.0
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 0.001,
    kd_weight: float = 0.1,
    device: str = None,
    seed: int = 42,
    verbose: bool = False
):
    """
    Trains MetaKD on AudioVision-MNIST with Bi-level Optimization.
    Lower-level (SGD): Model parameters trained on Meta-Train.
    Upper-level (Adam): KD weight matrix optimized on Meta-Val.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, test_loader = get_audiovision_loaders(
        batch_size=batch_size,
        missing_modality=missing_modality,
        available_rate=available_rate,
        seed=seed
    )

    model = MetaKDAudioVision(num_classes=10, feat_dim=128).to(device)

    # Lower-level optimizer (SGD + Nesterov momentum 0.99 for network parameters)
    network_params = [p for n, p in model.named_parameters() if 'kd_weights' not in n]
    optimizer = optim.SGD(network_params, lr=lr, momentum=0.99, nesterov=True, weight_decay=1e-4)

    # Upper-level optimizer (Adam for KD weights)
    kd_optim = optim.Adam(model.kd_weights.parameters(), lr=1e-2, weight_decay=5e-5)
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_model_state = None
    target_eval_mode = "image_only" if missing_modality == "audio" else "audio_only"

    # Directory to save checkpoints
    ckpt_dir = "audiovision/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_filename = f"metakd_{missing_modality}_rate_{int(available_rate*100)}pct.pth"
    ckpt_path = os.path.join(ckpt_dir, ckpt_filename)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        # Step 1: Lower-level training on Meta-Train
        for batch in train_loader:
            images = batch['image'].to(device)
            audios = batch['audio'].to(device)
            labels = batch['label'].to(device)
            v_mask = batch['visual_mask'].to(device)
            a_mask = batch['audio_mask'].to(device)

            optimizer.zero_grad()

            logits, img_logits, aud_logits, kd_loss = model(
                images, audios, visual_mask=v_mask, audio_mask=a_mask, eval_mode="full"
            )

            loss_fused = ce_loss_fn(logits, labels)
            loss_img = ce_loss_fn(img_logits[v_mask], labels[v_mask]) if v_mask.sum() > 0 else 0.0
            loss_aud = ce_loss_fn(aud_logits[a_mask], labels[a_mask]) if a_mask.sum() > 0 else 0.0

            loss = loss_fused + 0.5 * (loss_img + loss_aud) + kd_weight * kd_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Step 2: Upper-level Meta update on Meta-Val
        model.train()
        for val_batch in val_loader:
            val_images = val_batch['image'].to(device)
            val_audios = val_batch['audio'].to(device)
            val_labels = val_batch['label'].to(device)

            kd_optim.zero_grad()
            val_logits, _, _, _ = model(val_images, val_audios, eval_mode="full")
            meta_val_loss = ce_loss_fn(val_logits, val_labels)
            meta_val_loss.backward()
            kd_optim.step()

        # Validation & Testing
        val_acc = evaluate_accuracy(model, val_loader, eval_mode=target_eval_mode, device=device)
        test_acc = evaluate_accuracy(model, test_loader, eval_mode=target_eval_mode, device=device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            # Save best checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'kd_optim_state_dict': kd_optim.state_dict(),
                'kd_weights': model.kd_weights().detach().cpu().numpy(),
                'val_acc': best_val_acc,
                'test_acc': best_test_acc,
                'missing_modality': missing_modality,
                'available_rate': available_rate
            }, ckpt_path)

        if verbose and (epoch % 10 == 0 or epoch == epochs):
            weights = model.kd_weights().detach().cpu().numpy()
            print(f"Epoch [{epoch:03d}/{epochs:03d}] Loss: {total_loss/len(train_loader):.4f} | "
                  f"Val Acc ({target_eval_mode}): {val_acc:.2f}% | Best Test Acc: {best_test_acc:.2f}% | "
                  f"KD Weights: [V->A: {weights[0,1]:.3f}, A->V: {weights[1,0]:.3f}]")

    return best_test_acc, ckpt_path
