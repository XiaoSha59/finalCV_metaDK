"""
Bi-Level Meta-Learning Training Script for MetaKD on ADNI Dataset.
Optimizes both:
1. Inner Task Parameters (Encoders + Classifier) with Task Loss + Modality-Weighted KD Loss.
   Includes random modality dropout during training (MetaKD Supplementary Section 8).
2. Outer Meta Parameters (IWV kd_weights) with Validation Meta Loss.
Runs efficiently on CPU or GPU.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from .models import MetaKD_ADNI
from .dataset import get_adni_dataloaders


def get_arguments():
    parser = argparse.ArgumentParser(description="Train MetaKD on ADNI Dataset.")
    parser.add_argument("--npz_path", type=str, default="adni/data_processed/adni_processed.npz")
    parser.add_argument("--snapshot_dir", type=str, default="adni/checkpoints")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--kd_lr", type=float, default=1e-2)
    parser.add_argument("--kd_weight_decay", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=0.1, help="KD loss trade-off factor")
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def evaluate(model, data_loader, criterion, device):
    """Evaluates model on validation or test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for (x_c, x_g, x_b, x_i), masks, labels in data_loader:
            x_list = [x_c.to(device), x_g.to(device), x_b.to(device), x_i.to(device)]
            masks = masks.to(device)
            labels = labels.to(device)

            logits, _ = model(x_list, masks, mode="eval")
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    acc = accuracy_score(all_targets, all_preds) * 100.0
    f1 = f1_score(all_targets, all_preds, average='macro') * 100.0
    bal_acc = balanced_accuracy_score(all_targets, all_preds) * 100.0
    return avg_loss, acc, f1, bal_acc


def main():
    args = get_arguments()
    os.makedirs(args.snapshot_dir, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device for training: {device}")

    train_loader, val_loader, test_loader, dims = get_adni_dataloaders(
        npz_path=args.npz_path, batch_size=args.batch_size
    )
    print(f"Loaded ADNI dataset dims: C={dims[0]}, G={dims[1]}, B={dims[2]}, I={dims[3]}")
    print(f"Train samples: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    model = MetaKD_ADNI(dims=dims, embed_dim=args.embed_dim, num_classes=3, alpha=args.alpha).to(device)

    # Optimizers (Adam with weight decay 1e-2 matching Paper Page 13)
    task_params = [p for n, p in model.named_parameters() if 'kd_weights' not in n]
    task_optimizer = optim.Adam(task_params, lr=args.lr, weight_decay=args.weight_decay)
    task_scheduler = optim.lr_scheduler.StepLR(task_optimizer, step_size=20, gamma=0.9)
    kd_optimizer = optim.Adam(model.kd_weights.parameters(), lr=args.kd_lr, weight_decay=args.kd_weight_decay)

    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_checkpoint_path = os.path.join(args.snapshot_dir, "best_adni_model.pth")

    print("\nStarting MetaKD Bi-level Training with Modality Dropout...")
    val_iter = iter(val_loader)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_kd_total = 0.0

        for (x_c, x_g, x_b, x_i), masks, labels in train_loader:
            x_list = [x_c.to(device), x_g.to(device), x_b.to(device), x_i.to(device)]
            masks = masks.to(device)
            labels = labels.to(device)

            # ---------------------------------------------------------
            # Inner Loop: Modality Dropout + Update Encoders & Classifier
            # ---------------------------------------------------------
            train_masks = masks.clone()
            # Random Modality Dropout (MetaKD Supp Sec 8)
            for b in range(train_masks.size(0)):
                if torch.rand(1).item() < 0.5:
                    avail_idx = torch.where(train_masks[b])[0]
                    if len(avail_idx) > 1:
                        num_keep = torch.randint(1, len(avail_idx), (1,)).item()
                        perm = torch.randperm(len(avail_idx))
                        keep_idx = avail_idx[perm[:num_keep]]
                        new_mask = torch.zeros(4, dtype=torch.bool, device=device)
                        new_mask[keep_idx] = True
                        train_masks[b] = new_mask

            task_optimizer.zero_grad()
            logits, kd_loss = model(x_list, train_masks, val=False)
            task_loss = criterion(logits, labels)
            total_loss = task_loss + args.alpha * kd_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(task_params, max_norm=5.0)
            task_optimizer.step()

            train_loss_total += task_loss.item() * len(labels)
            train_kd_total += kd_loss.item() * len(labels)

            # ---------------------------------------------------------
            # Outer Loop: Meta-update IWV (kd_weights) on Validation Data
            # ---------------------------------------------------------
            try:
                (vx_c, vx_g, vx_b, vx_i), v_masks, v_labels = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                (vx_c, vx_g, vx_b, vx_i), v_masks, v_labels = next(val_iter)

            vx_list = [vx_c.to(device), vx_g.to(device), vx_b.to(device), vx_i.to(device)]
            v_masks = v_masks.to(device)
            v_labels = v_labels.to(device)

            kd_optimizer.zero_grad()
            v_logits, _ = model(vx_list, v_masks, mode="meta_val")
            meta_loss = criterion(v_logits, v_labels)
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.kd_weights.parameters(), max_norm=5.0)
            kd_optimizer.step()

        # Epoch Metrics
        train_loss_avg = train_loss_total / len(train_loader.dataset)
        val_loss, val_acc, val_f1, val_bal_acc = evaluate(model, val_loader, criterion, device)
        curr_iwv = model.kd_weights.get_weights().detach().cpu().numpy()

        print(f"Epoch [{epoch:02d}/{args.epochs}] | Train Loss: {train_loss_avg:.4f} | "
              f"Val Acc: {val_acc:.2f}%, F1: {val_f1:.2f}%, BalAcc: {val_bal_acc:.2f}% | "
              f"IWV [C,G,B,I]: [{curr_iwv[0]:.3f}, {curr_iwv[1]:.3f}, {curr_iwv[2]:.3f}, {curr_iwv[3]:.3f}]")

        task_scheduler.step()

        # Save Best Checkpoint based on Validation Macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'model_state': model.state_dict(),
                'kd_weights': model.kd_weights.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'dims': dims
            }, best_checkpoint_path)
            print(f"  >>> New Best Model saved with Val F1: {val_f1:.2f}% <<<")

    print(f"\nTraining completed! Best Validation F1: {best_val_f1:.2f}%. Checkpoint saved at {best_checkpoint_path}")


if __name__ == '__main__':
    main()
