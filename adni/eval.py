"""
Comprehensive Evaluation of MetaKD across all 15 Modality Combinations on ADNI Test Set.
Reproduces Table 2 in the MetaKD paper (arXiv:2405.07155) with side-by-side comparison:
Reports: Paper Acc vs Our Acc, Paper Macro F1 vs Our Macro F1.
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from .models import MetaKD_ADNI
from .dataset import ADNIDataset


# Reference results from MetaKD Paper Table 2 (arXiv:2405.07155)
PAPER_TABLE_2 = {
    'C':       {'paper_acc': 59.65, 'paper_f1': 52.54, 'pattern': '[X . . .]', 'desc': 'Clinical only'},
    'G':       {'paper_acc': 77.32, 'paper_f1': 74.07, 'pattern': '[. X . .]', 'desc': 'Genomic only'},
    'B':       {'paper_acc': 55.22, 'paper_f1': 48.74, 'pattern': '[. . X .]', 'desc': 'Biospecimen only'},
    'I':       {'paper_acc': 63.30, 'paper_f1': 57.75, 'pattern': '[. . . X]', 'desc': 'Imaging only'},
    'C,G':     {'paper_acc': 78.41, 'paper_f1': 75.31, 'pattern': '[X X . .]', 'desc': 'Clinical + Genomic'},
    'C,B':     {'paper_acc': 64.71, 'paper_f1': 58.62, 'pattern': '[X . X .]', 'desc': 'Clinical + Biospecimen'},
    'C,I':     {'paper_acc': 66.82, 'paper_f1': 61.43, 'pattern': '[X . . X]', 'desc': 'Clinical + Imaging'},
    'G,B':     {'paper_acc': 77.89, 'paper_f1': 74.92, 'pattern': '[. X X .]', 'desc': 'Genomic + Biospecimen'},
    'G,I':     {'paper_acc': 78.14, 'paper_f1': 75.10, 'pattern': '[. X . X]', 'desc': 'Genomic + Imaging'},
    'B,I':     {'paper_acc': 67.45, 'paper_f1': 62.18, 'pattern': '[. . X X]', 'desc': 'Biospecimen + Imaging'},
    'C,G,B':   {'paper_acc': 79.12, 'paper_f1': 76.24, 'pattern': '[X X X .]', 'desc': 'Clinical + Genomic + Biospecimen'},
    'C,G,I':   {'paper_acc': 79.48, 'paper_f1': 76.59, 'pattern': '[X X . X]', 'desc': 'Clinical + Genomic + Imaging'},
    'C,B,I':   {'paper_acc': 68.91, 'paper_f1': 63.85, 'pattern': '[X . X X]', 'desc': 'Clinical + Biospecimen + Imaging'},
    'G,B,I':   {'paper_acc': 78.95, 'paper_f1': 76.01, 'pattern': '[. X X X]', 'desc': 'Genomic + Biospecimen + Imaging'},
    'C,G,B,I': {'paper_acc': 80.25, 'paper_f1': 77.40, 'pattern': '[X X X X]', 'desc': 'Full Modalities (All 4)'},
}


def evaluate_combination(model, npz_path, active_modalities, device):
    """Evaluates test set under a specific active modality combination."""
    test_ds = ADNIDataset(npz_path=npz_path, split="test", active_modalities=active_modalities, filter_available=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for (x_c, x_g, x_b, x_i), masks, labels in test_loader:
            x_list = [x_c.to(device), x_g.to(device), x_b.to(device), x_i.to(device)]
            masks = masks.to(device)

            logits, _ = model(x_list, masks, mode="eval")
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    acc = accuracy_score(all_targets, all_preds) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average='macro') * 100.0
    bal_acc = balanced_accuracy_score(all_targets, all_preds) * 100.0
    n_samples = len(all_targets)

    return acc, macro_f1, bal_acc, n_samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate MetaKD across all 15 modality combinations on ADNI.")
    parser.add_argument("--npz_path", type=str, default="adni/data_processed/adni_processed.npz")
    parser.add_argument("--checkpoint", type=str, default="adni/checkpoints/best_adni_model.pth")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device for evaluation: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}. Please run train.py first.")

    print(f"Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    dims = ckpt['dims']

    model = MetaKD_ADNI(dims=dims, embed_dim=64, num_classes=3).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    print("\n" + "=" * 110)
    print(f"{'Modality':<10} | {'Pattern':<10} | {'Paper Acc':<10} | {'Our Acc':<10} | {'Acc Diff':<10} | {'Paper F1':<10} | {'Our F1':<10} | {'F1 Diff':<10} | {'N Test'}")
    print("=" * 110)

    results = []

    for mode_str, meta in PAPER_TABLE_2.items():
        acc, f1, bal_acc, n_samples = evaluate_combination(model, args.npz_path, mode_str, device)
        p_acc = meta['paper_acc']
        p_f1 = meta['paper_f1']
        diff_acc = acc - p_acc
        diff_f1 = f1 - p_f1

        results.append((acc, f1, p_acc, p_f1))
        diff_acc_str = f"{diff_acc:+.2f}%"
        diff_f1_str = f"{diff_f1:+.2f}%"

        print(f"{mode_str:<10} | {meta['pattern']:<10} | {p_acc:<10.2f} | {acc:<10.2f} | {diff_acc_str:<10} | {p_f1:<10.2f} | {f1:<10.2f} | {diff_f1_str:<10} | {n_samples}")

    avg_our_acc = np.mean([r[0] for r in results])
    avg_our_f1 = np.mean([r[1] for r in results])
    avg_p_acc = np.mean([r[2] for r in results])
    avg_p_f1 = np.mean([r[3] for r in results])

    diff_avg_acc = avg_our_acc - avg_p_acc
    diff_avg_f1 = avg_our_f1 - avg_p_f1

    print("-" * 110)
    print(f"{'Average':<10} | {'[15 combos]':<10} | {avg_p_acc:<10.2f} | {avg_our_acc:<10.2f} | {diff_avg_acc:+.2f}%{'':<3} | {avg_p_f1:<10.2f} | {avg_our_f1:<10.2f} | {diff_avg_f1:+.2f}%{'':<3} |")
    print("=" * 110 + "\n")


if __name__ == '__main__':
    main()
