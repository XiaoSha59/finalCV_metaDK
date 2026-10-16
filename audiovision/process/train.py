import os
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..data.dataset import get_audiovision_loaders
from .models import MetaKDAudioVision, LeNet5, SNetPluse

_AV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def evaluate_accuracy(model: nn.Module, loader, eval_mode: str = "image_only", device: str = "cpu") -> float:
    """Top-1 accuracy. LeNet5 and SNetPluse are the unimodal lower bounds."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            audios = batch["audio"].to(device)
            labels = batch["label"].to(device)

            if isinstance(model, LeNet5):
                logits = model(images)
            elif isinstance(model, SNetPluse):
                logits = model(audios)
            elif eval_mode == "image_only":
                logits, _ = model(images=images, eval_mode="image_only")
            elif eval_mode == "audio_only":
                logits, _ = model(audios=audios, eval_mode="audio_only")
            else:
                logits, _ = model(images=images, audios=audios, eval_mode="full")

            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

    return (correct / total) * 100.0 if total > 0 else 0.0


def _meta_update_weights(model, images, audios, labels, kd_optim, ce_loss_fn):
    """Outer step of Eq. (4): fit the importance weights on a validation batch.

    Encoder features are detached and computed in eval mode so this step does
    not update batch-norm statistics or the network weights.
    """
    training = model.training
    model.eval()
    with torch.no_grad():
        img_feat, aud_feat = model.encode(images, audios)
    model.train(training)

    weights = model.kd_weights()
    logits = model.fuse(weights[0] * img_feat, weights[1] * aud_feat)
    meta_loss = ce_loss_fn(logits, labels)
    weight_grad = torch.autograd.grad(meta_loss, model.kd_weights.weights)[0]
    kd_optim.zero_grad()
    model.kd_weights.weights.grad = weight_grad
    kd_optim.step()


def _encoder_weights(state_dict):
    prefix = "encoder."
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}


def train_metakd_audiovision(
    missing_modality: str = "audio",
    available_rate: float = 0.10,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    kd_weight: float = 0.1,
    device: str = None,
    seed: int = 42,
    verbose: bool = False,
    image_encoder_state=None,
    audio_encoder_state=None,
    loaders=None,
    ckpt_name=None,
):
    """Train MetaKD. Network: Adam 1e-3, decay 10% every 20 epochs. IWV: Adam 1e-2."""
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    if loaders is None:
        train_loader, val_loader, test_loader = get_audiovision_loaders(
            batch_size=batch_size,
            missing_modality=missing_modality,
            available_rate=available_rate,
            seed=seed,
        )
    else:
        train_loader, val_loader, test_loader = loaders

    model = MetaKDAudioVision().to(device)
    if image_encoder_state is not None:
        model.img_enc.load_state_dict(_encoder_weights(image_encoder_state))
    if audio_encoder_state is not None:
        model.aud_enc.load_state_dict(_encoder_weights(audio_encoder_state))
    network_params = [param for name, param in model.named_parameters() if "kd_weights" not in name]
    optimizer = optim.Adam(network_params, lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    kd_optim = optim.Adam(model.kd_weights.parameters(), lr=1e-2, weight_decay=1e-2)
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_test_acc = 0.0
    target_eval_mode = "image_only" if missing_modality == "audio" else "audio_only"

    ckpt_dir = os.path.join(_AV_ROOT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    filename = ckpt_name or f"metakd_{missing_modality}_rate_{int(available_rate * 100)}pct.pth"
    ckpt_path = os.path.join(ckpt_dir, filename)
    val_iter = itertools.cycle(val_loader)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

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
            logits, kd_loss = model(
                images,
                audios,
                visual_mask=visual_mask,
                audio_mask=audio_mask,
                eval_mode="full",
            )
            step_loss = ce_loss_fn(logits, labels) + kd_weight * kd_loss
            step_loss.backward()
            optimizer.step()
            total_loss += step_loss.item()

        scheduler.step()
        val_acc = evaluate_accuracy(model, val_loader, eval_mode=target_eval_mode, device=device)
        test_acc = evaluate_accuracy(model, test_loader, eval_mode=target_eval_mode, device=device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "kd_optim_state_dict": kd_optim.state_dict(),
                    "kd_weights": model.kd_weights().detach().cpu().numpy(),
                    "val_acc": best_val_acc,
                    "test_acc": best_test_acc,
                    "missing_modality": missing_modality,
                    "available_rate": available_rate,
                },
                ckpt_path,
            )

        if verbose and (epoch % 10 == 0 or epoch == epochs):
            weights = model.kd_weights().detach().cpu().numpy()
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] Loss: {total_loss / len(train_loader):.4f} | "
                f"Val Acc ({target_eval_mode}): {val_acc:.2f}% | Best Test Acc: {best_test_acc:.2f}% | "
                f"IWV Weights: [Visual: {weights[0]:.3f}, Audio: {weights[1]:.3f}]"
            )

    return best_test_acc, ckpt_path
