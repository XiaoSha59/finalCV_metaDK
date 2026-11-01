"""
Training Pipeline for ADNI with Class-Decoupled Uncertainty KD (U-MetaKD)
========================================================================
Runs ADNI C/B/I multimodal classification under Table 2 setting with
Uncertainty-modulated knowledge distillation.

Key Upgrades vs Baseline adni/process/run_pipeline.py:
1. Calculates sample-level predictive entropy from observed modalities.
2. Modulates pairwise L1 feature distillation with class-decoupled confidence (protecting MCI).
3. Evaluates 7 modality scenarios (C, B, I, C+B, C+I, B+I, C+B+I) with uncertainty modulation.
"""

import argparse
import copy
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ADNI_ROOT = os.path.join(REPO_DIR, "adni")
for p in [REPO_DIR, ADNI_ROOT, os.path.join(ADNI_ROOT, "process")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from adni.process.run_pipeline import (
    get_arguments,
    load_protocol,
    scenarios_for,
    active_mods,
    apply_smote,
    meta_step,
    random_drop_modalities,
    predict,
    metrics,
    table2_mean_acc,
    PAPER_BENCHMARKS
)
from adni.process.models.metakd_model import MetaKDADNIModel
from improvements.adni_uncertainty.loss_adni_ukd import ClassDecoupledUncertaintyLossADNI


def run_seed_ukd(seed, data, args, device):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    xs_all, obs_all, y_all = data["train"]

    if args.meta_val_source == "train":
        keep = obs_all.any(1)
        idx = np.where(keep)[0]
        idx_t, idx_v = train_test_split(idx, test_size=args.meta_val_frac, stratify=y_all[idx], random_state=seed)
        meta_part = ([x[idx_v] for x in xs_all], obs_all[idx_v], y_all[idx_v])
        train_part = ([x[idx_t] for x in xs_all], obs_all[idx_t], y_all[idx_t])
        print(f"  [seed {seed}] D_t {len(idx_t)} / D_v {len(idx_v)} carved from train (D_v complete rows {int(obs_all[idx_v].all(1).sum())})", flush=True)
    else:
        meta_part, train_part = data["val"], data["train"]

    xs_tr, obs_tr, y_tr = apply_smote(
        *train_part, rng=rng,
        neighbour_mods=None if (args.drop_g or not args.smote_no_g) else [0, 2, 3]
    )
    scenarios = scenarios_for(args.drop_g)
    mods = active_mods(args.drop_g)

    def loader(xs, obs, y, shuffle, drop_last=False):
        ds = TensorDataset(*[torch.from_numpy(x) for x in xs], torch.from_numpy(y), torch.from_numpy(obs))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, drop_last=drop_last)

    def forever(dl):
        while True:
            yield from dl

    train_loader = loader(xs_tr, obs_tr, y_tr, True, drop_last=True)
    val_loader = loader(*data["val"], False)
    test_loader = loader(*data["test"], False)
    meta_iter = forever(loader(*meta_part, True))

    model = MetaKDADNIModel(
        [x.shape[1] for x in xs_tr], args.latent_dim, num_patches=args.num_patches,
        shared_encoder=args.shared_encoder
    ).to(device)

    task_params = [p for n, p in model.named_parameters() if n != "raw_weights"]
    optimizer = optim.Adam(task_params, lr=args.lr, weight_decay=args.weight_decay)
    meta_optimizer = optim.Adam([model.raw_weights], lr=args.meta_lr, weight_decay=args.meta_weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    criterion = nn.CrossEntropyLoss()
    ukd_loss_fn = ClassDecoupledUncertaintyLossADNI(
        tau_cn=0.5, tau_mci=0.8, tau_ad=0.5, min_mci_conf=0.5
    )

    n_epochs = args.epochs
    if args.total_iters > 0:
        n_epochs = -(-args.total_iters // len(train_loader))

    best_acc, best_state, last_state, it = -1.0, None, None, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        tot_task, tot_kd, tot_conf, n_batches = 0.0, 0.0, 0.0, 0
        for *xs, y, obs in train_loader:
            if args.total_iters > 0 and it >= args.total_iters:
                break
            if it % args.meta_every == 0:
                *vxs, vy, vobs = next(meta_iter)
                vxs = [x.to(device) for x in vxs]
                meta_step(model, vxs, vy.to(device), vobs.to(device), meta_optimizer, criterion)

            xs = [x.to(device) for x in xs]
            y, obs = y.to(device), obs.to(device)
            active = obs if args.no_modality_drop else random_drop_modalities(obs)
            logits, feats, active = model(xs, active)
            rows = active.any(1)
            if not rows.any():
                continue

            l_task = criterion(logits[rows], y[rows])

            # Teacher forward pass with observed modalities for predictive entropy
            with torch.no_grad():
                teacher_logits, _, _ = model(xs, obs)

            iwv = model.get_importance_weights()
            l_kd, avg_conf = ukd_loss_fn(feats, active, iwv_weights=iwv, teacher_logits=teacher_logits)

            optimizer.zero_grad()
            (l_task + args.alpha * l_kd).backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(task_params, max_norm=args.grad_clip)
            optimizer.step()

            tot_task += l_task.item()
            tot_kd += l_kd.item()
            tot_conf += avg_conf.item()
            n_batches += 1
            it += 1

        scheduler.step()
        last_state = copy.deepcopy(model.state_dict())
        val = metrics(*predict(model, val_loader, device))
        val_t2 = table2_mean_acc(model, val_loader, device, scenarios)

        if val_t2 > best_acc:
            best_acc, best_state = val_t2, copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            w = model.get_importance_weights().detach().cpu().numpy()
            print(f"  [seed {seed}] Epoch {epoch:02d}/{n_epochs} | "
                  f"task {tot_task / max(1, n_batches):.4f} kd {tot_kd / max(1, n_batches):.4f} "
                  f"conf {tot_conf / max(1, n_batches):.2f} | "
                  f"Val Acc {val['acc']:.2f}% Table2 {val_t2:.2f}% | "
                  f"IWV " + " ".join(f"{m}:{w[i]:.3f}" for i, m in enumerate(mods)), flush=True)

    ckpt_dir = os.path.join(REPO_DIR, "improvements", "adni_uncertainty", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    chosen_state = last_state if args.last_checkpoint else best_state
    torch.save(chosen_state, os.path.join(ckpt_dir, f"ukd_adni_seed{seed}.pth"))
    model.load_state_dict(chosen_state)

    return {name: metrics(*predict(model, test_loader, device, scenario)) for name, scenario in scenarios}


def main():
    args = get_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print(f"ADNI Uncertainty KD Pipeline (CDU-KD) | device={device}", flush=True)
    print("=" * 80, flush=True)

    subjects, data, idx_train, idx_val, idx_test, h5ad = load_protocol(args)
    print(f"Split: train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}", flush=True)

    if args.dry_run:
        print("[Dry Run] Data loaded and verified successfully!", flush=True)
        return

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    per_seed = []
    for seed in seeds:
        print(f"\n--- [Seed {seed}] Training ADNI U-MetaKD ---", flush=True)
        per_seed.append(run_seed_ukd(seed, data, args, device))

    scenarios = scenarios_for(args.drop_g)
    rows = []
    for name, _ in scenarios:
        row = {"Modality": name}
        for key in ("acc", "f1", "bacc", "mf1"):
            vals = np.array([r[name][key] for r in per_seed])
            row[f"Test_{key}"], row[f"Std_{key}"] = vals.mean(), vals.std()
            row[f"Paper_{key}"] = PAPER_BENCHMARKS[name][key]
            row[f"Diff_{key}"] = row[f"Test_{key}"] - row[f"Paper_{key}"]
        rows.append(row)
    res = pd.DataFrame(rows)

    print("\n" + "=" * 100, flush=True)
    print(f"ADNI TABLE 2 (U-MetaKD, test {len(idx_test)} subjects, mean over seeds {seeds})", flush=True)
    print("=" * 100, flush=True)
    header = f"{'Modality':<9}" + "".join(f" | {lbl:<22}" for lbl in ("Accuracy", "Macro-F1", "Balanced Acc", "Micro-F1"))
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in rows:
        print(f"{r['Modality']:<9} | {r['Test_acc']:6.2f}% +- {r['Std_acc']:4.2f}%   | "
              f"{r['Test_f1']:6.2f}% +- {r['Std_f1']:4.2f}%   | "
              f"{r['Test_bacc']:6.2f}% +- {r['Std_bacc']:4.2f}%   | "
              f"{r['Test_mf1']:6.2f}% +- {r['Std_mf1']:4.2f}%", flush=True)

    os.makedirs(os.path.join(REPO_DIR, "logs"), exist_ok=True)
    out_csv = os.path.join(REPO_DIR, "logs", "adni_table2_ukd.csv")
    res.to_csv(out_csv, index=False)
    print(f"\n[Saved] Official CSV saved to: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
