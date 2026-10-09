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

    # Lower-level optimizer (Adam lr=1e-3, weight_decay=1e-2, decayed by 10% every 20 epochs)
    network_params = [p for n, p in model.named_parameters() if 'kd_weights' not in n]
    optimizer = optim.Adam(network_params, lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    # Upper-level optimizer (Adam lr=1e-2, weight_decay=1e-2 for KD importance weights)
    kd_optim = optim.Adam(model.kd_weights.parameters(), lr=1e-2, weight_decay=1e-2)
    ce_loss_fn = nn.CrossEntropyLoss()

    import itertools

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_model_state = None
    target_eval_mode = "image_only" if missing_modality == "audio" else "audio_only"

    # Directory to save checkpoints
    ckpt_dir = "audiovision/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_filename = f"metakd_{missing_modality}_rate_{int(available_rate*100)}pct.pth"
    ckpt_path = os.path.join(ckpt_dir, ckpt_filename)

    val_iter = itertools.cycle(val_loader)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        current_lr = scheduler.get_last_lr()[0]

        for batch in train_loader:
            images = batch['image'].to(device)
            audios = batch['audio'].to(device)
            labels = batch['label'].to(device)
            v_mask = batch['visual_mask'].to(device)
            a_mask = batch['audio_mask'].to(device)

            val_batch = next(val_iter)
            val_images = val_batch['image'].to(device)
            val_audios = val_batch['audio'].to(device)
            val_labels = val_batch['label'].to(device)

            # -------------------------------------------------------------
            # Step 1: Bi-level Upper-Level Meta Update on Meta-Val (kd_weights)
            # -------------------------------------------------------------
            logits, img_logits, aud_logits, kd_loss = model(
                images, audios, visual_mask=v_mask, audio_mask=a_mask, eval_mode="full"
            )
            loss_fused = ce_loss_fn(logits, labels)
            loss_img = ce_loss_fn(img_logits[v_mask], labels[v_mask]) if v_mask.sum() > 0 else 0.0
            loss_aud = ce_loss_fn(aud_logits[a_mask], labels[a_mask]) if a_mask.sum() > 0 else 0.0
            train_loss = loss_fused + 0.5 * (loss_img + loss_aud) + kd_weight * kd_loss

            net_params = {k: v for k, v in model.named_parameters() if 'kd_weights' not in k and v.requires_grad}
            grads = torch.autograd.grad(train_loss, net_params.values(), create_graph=True, allow_unused=True)

            virtual_params = {k: (p - current_lr * g if g is not None else p) for (k, p), g in zip(net_params.items(), grads)}
            all_virtual_params = {**dict(model.named_parameters()), **virtual_params}

            # Evaluate on meta-validation batch with virtual parameters
            if target_eval_mode == "image_only":
                val_logits, _ = torch.func.functional_call(model, all_virtual_params, (val_images, None), {'eval_mode': 'image_only'})
            else:
                val_logits, _ = torch.func.functional_call(model, all_virtual_params, (None, val_audios), {'eval_mode': 'audio_only'})

            meta_val_loss = ce_loss_fn(val_logits, val_labels)
            meta_grads = torch.autograd.grad(meta_val_loss, model.kd_weights.weights, allow_unused=True)

            if meta_grads[0] is not None:
                kd_optim.zero_grad()
                model.kd_weights.weights.grad = meta_grads[0]
                kd_optim.step()

            # -------------------------------------------------------------
            # Step 2: Lower-Level Optimization on Meta-Train (Network Parameters)
            # -------------------------------------------------------------
            optimizer.zero_grad()
            logits, img_logits, aud_logits, kd_loss = model(
                images, audios, visual_mask=v_mask, audio_mask=a_mask, eval_mode="full"
            )
            loss_fused = ce_loss_fn(logits, labels)
            loss_img = ce_loss_fn(img_logits[v_mask], labels[v_mask]) if v_mask.sum() > 0 else 0.0
            loss_aud = ce_loss_fn(aud_logits[a_mask], labels[a_mask]) if a_mask.sum() > 0 else 0.0
            step_loss = loss_fused + 0.5 * (loss_img + loss_aud) + kd_weight * kd_loss

            step_loss.backward()
            optimizer.step()
            total_loss += step_loss.item()

        scheduler.step()

        # Validation & Testing
        val_acc = evaluate_accuracy(model, val_loader, eval_mode=target_eval_mode, device=device)
        test_acc = evaluate_accuracy(model, test_loader, eval_mode=target_eval_mode, device=device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
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
