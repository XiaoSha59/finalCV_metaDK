"""MetaKD Table 2 on the official UNITES-Lab/Flex-MoE ADNI protocol.

Data: Flex-MoE C/B/I loaders (UCSFFSX7). Genomic is omitted by default (--keep_g to restore).
Training: Algorithm 1 / Eq. 2 / Eq. 5, SMOTE k=5, non-dedicated dropout of
0 to 3 modalities (supplement Sec. 8 / BraTS), AudioVision classification
optimizer (Adam 1e-3, wd 1e-2, StepLR 20 x 0.9; IWV Adam 1e-2).
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

_PROCESS_DIR = os.path.dirname(os.path.abspath(__file__))
ADNI_ROOT = os.path.dirname(_PROCESS_DIR)
REPO_ROOT = os.path.dirname(ADNI_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from adni.process.models.metakd_model import MetaKDADNIModel
from adni.data.preprocess.flexmoe_tables import load_biospecimen, load_clinical, load_image, load_labels

MODS = ("C", "G", "B", "I")
PAPER_BENCHMARKS = {
    "C":       {"acc": 59.65, "f1": 43.25, "bacc": 56.24, "mf1": 59.29},
    "G":       {"acc": 77.32, "f1": 47.32, "bacc": 71.89, "mf1": 77.01},
    "B":       {"acc": 56.23, "f1": 40.57, "bacc": 53.46, "mf1": 55.98},
    "I":       {"acc": 55.32, "f1": 39.05, "bacc": 51.89, "mf1": 55.14},
    "C,G":     {"acc": 61.35, "f1": 42.36, "bacc": 57.92, "mf1": 61.08},
    "C,B":     {"acc": 63.24, "f1": 45.87, "bacc": 59.78, "mf1": 62.95},
    "C,I":     {"acc": 62.75, "f1": 41.23, "bacc": 59.42, "mf1": 62.51},
    "G,B":     {"acc": 63.76, "f1": 38.63, "bacc": 60.34, "mf1": 63.52},
    "G,I":     {"acc": 64.87, "f1": 44.21, "bacc": 61.78, "mf1": 64.65},
    "B,I":     {"acc": 57.56, "f1": 39.54, "bacc": 54.67, "mf1": 57.32},
    "C,G,B":   {"acc": 60.35, "f1": 42.11, "bacc": 57.23, "mf1": 60.12},
    "C,G,I":   {"acc": 65.04, "f1": 50.43, "bacc": 62.15, "mf1": 64.82},
    "C,B,I":   {"acc": 63.06, "f1": 42.65, "bacc": 59.89, "mf1": 62.84},
    "G,B,I":   {"acc": 64.12, "f1": 55.09, "bacc": 61.78, "mf1": 63.89},
    "C,G,B,I": {"acc": 67.87, "f1": 57.23, "bacc": 65.34, "mf1": 67.65},
}
SCENARIOS = [(name, tuple(m in name.split(",") for m in MODS)) for name in PAPER_BENCHMARKS]


def active_mods(drop_g):
    return ("C", "B", "I") if drop_g else MODS


def scenarios_for(drop_g):
    mods = active_mods(drop_g)
    names = [n for n in PAPER_BENCHMARKS if all(m in mods for m in n.split(","))]
    return [(n, tuple(m in n.split(",") for m in mods)) for n in names]


def get_arguments():
    parser = argparse.ArgumentParser(description="MetaKD Table 2 / Flex-MoE ADNI protocol")
    parser.add_argument("--data_root", type=str, default=os.path.join(ADNI_ROOT, "data", "raw"))
    parser.add_argument("--processed_root", type=str, default=os.path.join(ADNI_ROOT, "data", "processed"))
    parser.add_argument("--split_json", type=str, default=os.path.join(ADNI_ROOT, "data", "PTID_splits.json"))
    parser.add_argument("--genomic_h5ad", type=str, default="")
    parser.add_argument("--tools_dir", type=str, default="")
    parser.add_argument("--allow_generated_split", action="store_true")
    parser.add_argument("--allow_raw_genomic_fallback", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Load protocol data and exit")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--total_iters", type=int, default=0,
                        help="Stop after this many training iterations (overrides --epochs); "
                             "MetaKD Fig. 9 shows ADNI IWV over 120k iterations")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="MetaKD classification Adam lr (AudioVision / supplement)")
    parser.add_argument("--class_weight", action="store_true", default=False,
                        help="Not in MetaKD/Flex-MoE ADNI; off by default")
    parser.add_argument("--no_class_weight", action="store_false", dest="class_weight")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--meta_lr", type=float, default=1e-2)
    parser.add_argument("--meta_weight_decay", type=float, default=1e-2)
    parser.add_argument("--meta_every", type=int, default=1,
                        help="Outer IWV step every N train batches (AudioVision: 1)")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--kd_reduction", choices=("sum", "mean"), default="mean",
                        help="sum = Eq. 5 L1 norm over the feature vector (= mean with alpha x latent_dim); "
                             "with alpha=0.1 it collapses every scenario to the majority class on ADNI")
    parser.add_argument("--meta_val_source", choices=("train", "val"), default="train",
                        help="train: D_v carved out of the training split, disjoint from D_t (Eq. 4); "
                             "val: old behaviour, reuse the Flex-MoE val split")
    parser.add_argument("--meta_val_frac", type=float, default=0.1,
                        help="Fraction of the training split held out as D_v (paper gives no ratio)")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--num_patches", type=int, default=16)
    parser.add_argument("--shared_encoder", action="store_true",
                        help="Per-modality input stem + one trunk shared by all modalities (our own design).")
    parser.add_argument("--last_checkpoint", action="store_true",
                        help="Evaluate the final-epoch weights instead of the best-val checkpoint "
                             "(paper BraTS: 'without model selection').")
    parser.add_argument("--kd_detach", action="store_true",
                        help="Stop-gradient on the teacher f_i in Eq. 5 (not in the paper)")
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help="Clip task-parameter grad norm (0 = off; not in the paper)")
    parser.add_argument("--no_modality_drop", action="store_true",
                        help="Train on the naturally observed modalities only (ADNI is already missing); "
                             "random 0-3 dropping is the BraTS non-dedicated setting")
    parser.add_argument("--center_inputs", action="store_true",
                        help="Subtract per-column train mean (observed rows) before encoders. "
                             "Not in Flex-MoE; fixes C-only collapse from min-max [-1,1] sparse inputs.")
    parser.add_argument("--g_pca", type=int, default=0,
                        help="Project G onto K whitened PCA components fit on train-observed rows (0 = off; not in the paper)")
    parser.add_argument("--smote_no_g", action="store_true",
                        help="SMOTE neighbours from C/B/I only; G is ~98%% of the joint variance (not in the paper)")
    parser.add_argument("--drop_g", dest="drop_g", action="store_true", default=True,
                        help="Train and evaluate C/B/I only (default; same frozen split)")
    parser.add_argument("--keep_g", dest="drop_g", action="store_false",
                        help="Keep genomic. Local 4-mod experiments only; not the GitHub default.")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--output_csv", type=str, default="logs/adni_table2_results.csv")
    return parser.parse_args()


def make_split(args, subjects, observed):
    path = args.split_json or os.path.join(args.processed_root, "PTID_splits.json")
    id_to_idx = {ptid: i for i, ptid in enumerate(subjects["PTID"])}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            split = json.load(handle)
        # sorted: set() order follows Python's per-process string hash, which breaks seed reproducibility
        idx_train = np.array(sorted(id_to_idx[x] for x in set(split["training"]) if x in id_to_idx), dtype=int)
        idx_val = np.array(sorted(id_to_idx[x] for x in set(split["validation"]) if x in id_to_idx), dtype=int)
        idx_test = np.array(sorted(id_to_idx[x] for x in set(split["testing"]) if x in id_to_idx), dtype=int)
        source = os.path.abspath(path)
    else:
        if not args.allow_generated_split:
            raise FileNotFoundError(
                f"{path} is unpublished in UNITES-Lab/Flex-MoE. Place the official file "
                "or pass --allow_generated_split for a protocol-correct but non-identical split."
            )
        y = subjects["label"].to_numpy()
        complete = np.where(observed.all(1))[0]
        n_hold = int(round(0.15 * len(subjects)))
        if len(complete) < 2 * n_hold + 3:
            raise ValueError(f"Four-modality intersection is too small for 15/15 hold-out: {len(complete)}")
        rest, idx_test = train_test_split(complete, test_size=n_hold, random_state=42, stratify=y[complete])
        _, idx_val = train_test_split(rest, test_size=n_hold, random_state=42, stratify=y[rest])
        held = np.zeros(len(subjects), dtype=bool)
        held[idx_val] = held[idx_test] = True
        idx_train = np.where(~held & observed.any(1))[0]
        source = "generated(random_state=42; Flex-MoE 70/15/15 intersection protocol)"
        os.makedirs("logs", exist_ok=True)
        with open("logs/adni_generated_PTID_splits.json", "w", encoding="utf-8") as handle:
            json.dump({
                "training": subjects.iloc[idx_train]["PTID"].tolist(),
                "validation": subjects.iloc[idx_val]["PTID"].tolist(),
                "testing": subjects.iloc[idx_test]["PTID"].tolist(),
            }, handle, indent=2)
        print("WARNING: official PTID_splits.json is missing; Table 2 numbers will not be subject-identical.")
    if not observed[idx_val].all() or not observed[idx_test].all():
        raise ValueError("Val/test must lie in the intersection of the active modalities")
    print(f"Split source: {source}")
    return idx_train, idx_val, idx_test


def load_protocol(args):
    if not os.path.isdir(args.data_root):
        legacy = os.path.join(REPO_ROOT, "adni_raw_flexmoe")
        if os.path.isdir(legacy):
            args.data_root = legacy
    subjects = load_labels(args.data_root)
    ptids = subjects["PTID"]
    print(f"Labels: {len(subjects)}  CN={(subjects.label == 0).sum()}  "
          f"MCI={(subjects.label == 1).sum()}  AD={(subjects.label == 2).sum()}")

    clinical, obs_c = load_clinical(args.data_root, ptids)
    biospecimen, obs_b = load_biospecimen(args.data_root, ptids)
    image, obs_i = load_image(args.data_root, ptids)
    if args.drop_g:
        observed = np.stack([obs_c, obs_b, obs_i], axis=1)
        arrays = [clinical, biospecimen, image]
        mods = active_mods(True)
        h5ad = ""
        print("Observed (G omitted): " + ", ".join(f"{m}={int(observed[:, i].sum())}" for i, m in enumerate(mods))
              + f" | all three={int(observed.all(1).sum())}")
    else:
        from adni.data.preprocess.genomic_preprocess import h5ad_to_matrix, load_or_build_genomic
        h5ad = args.genomic_h5ad or os.path.join(args.processed_root, "genomic", "genomic_merged.h5ad")
        h5ad = load_or_build_genomic(args.data_root, h5ad, args.tools_dir, args.allow_raw_genomic_fallback)
        genomic, obs_g = h5ad_to_matrix(h5ad, ptids)
        observed = np.stack([obs_c, obs_g, obs_b, obs_i], axis=1)
        arrays = [clinical, genomic, biospecimen, image]
        mods = MODS
        print("Observed: " + ", ".join(f"{m}={int(observed[:, i].sum())}" for i, m in enumerate(mods))
              + f" | all four={int(observed.all(1).sum())}")
    for name, arr in zip(mods, arrays):
        print(f"  {name}: {arr.shape[1]} features")

    idx_train, idx_val, idx_test = make_split(args, subjects, observed)
    data = {
        "train": ([x[idx_train] for x in arrays], observed[idx_train], subjects.iloc[idx_train]["label"].to_numpy(np.int64)),
        "val": ([x[idx_val] for x in arrays], observed[idx_val], subjects.iloc[idx_val]["label"].to_numpy(np.int64)),
        "test": ([x[idx_test] for x in arrays], observed[idx_test], subjects.iloc[idx_test]["label"].to_numpy(np.int64)),
    }
    if args.center_inputs:
        obs_tr = data["train"][1]
        for m, name in enumerate(mods):
            mu = data["train"][0][m][obs_tr[:, m]].mean(0, keepdims=True).astype(np.float32)
            for part in ("train", "val", "test"):
                data[part][0][m] -= mu
            print(f"  centered {name} with train-observed mean (n={int(obs_tr[:, m].sum())})")
    if args.g_pca > 0:
        if args.drop_g:
            raise ValueError("--g_pca requires --keep_g")
        from sklearn.decomposition import PCA
        obs_tr = data["train"][1][:, 1]
        pca = PCA(n_components=args.g_pca, whiten=True, random_state=0).fit(data["train"][0][1][obs_tr])
        for part in ("train", "val", "test"):
            xs, obs, _ = data[part]
            xs[1] = (pca.transform(xs[1]) * obs[:, 1:2]).astype(np.float32)
        print(f"  G -> {args.g_pca} PCA comps (fit n={int(obs_tr.sum())}, "
              f"explained {100 * pca.explained_variance_ratio_.sum():.1f}%)")
    return subjects, data, idx_train, idx_val, idx_test, h5ad


def apply_smote(xs, observed, y, k_neighbors=5, rng=None, neighbour_mods=None):
    """MetaKD supplement: SMOTE k=5. Only interpolate complete rows so missingness is not invented."""
    complete_idx = np.where(observed.all(1))[0]
    y_complete = y[complete_idx]
    classes, _ = np.unique(y_complete, return_counts=True)
    if len(classes) < 2:
        print("  [SMOTE k=5] skipped: fewer than two classes in complete training rows")
        return xs, observed, y
    all_classes, all_counts = np.unique(y, return_counts=True)
    count_by_class = dict(zip(all_classes, all_counts))
    target = all_counts.max()
    mods = range(len(xs)) if neighbour_mods is None else neighbour_mods
    joint = np.hstack([xs[m][complete_idx] for m in mods])
    out_x = [[x] for x in xs]
    out_obs, out_y = [observed], [y]
    for cls in classes:
        n_gen = target - count_by_class[cls]
        cls_local = np.where(y_complete == cls)[0]
        cls_idx = complete_idx[cls_local]
        k = min(k_neighbors, len(cls_idx) - 1)
        if n_gen <= 0 or k < 1:
            continue
        _, nbrs = NearestNeighbors(n_neighbors=k + 1).fit(joint[cls_local]).kneighbors(joint[cls_local])
        base = rng.randint(len(cls_idx), size=n_gen)
        nb = nbrs[base, rng.randint(1, k + 1, size=n_gen)]
        lam = rng.uniform(0.0, 1.0, size=(n_gen, 1)).astype(np.float32)
        for m, x in enumerate(xs):
            xc = x[cls_idx]
            out_x[m].append(xc[base] + lam * (xc[nb] - xc[base]))
        out_obs.append(np.ones((n_gen, observed.shape[1]), dtype=bool))
        out_y.append(np.full(n_gen, cls, dtype=np.int64))
    xs = [np.vstack(v).astype(np.float32) for v in out_x]
    y_out = np.concatenate(out_y)
    print(f"  [SMOTE k=5] {len(y)} -> {len(y_out)}  counts={np.bincount(y_out)}")
    return xs, np.vstack(out_obs), y_out


def random_drop_modalities(obs):
    """Non-dedicated training: drop 0 to n-1 of the observed modalities.

    Supplement Sec. 8 (BraTS) and ADNI Table 2 both evaluate all 15 subsets.
    Never turns on a modality that was truly missing in `obs`.
    """
    active = obs.clone()
    for b in range(obs.size(0)):
        present = obs[b].nonzero(as_tuple=True)[0]
        n = present.numel()
        if n <= 1:
            continue
        n_keep = int(torch.randint(1, n + 1, (1,), device=obs.device))
        perm = present[torch.randperm(n, device=obs.device)]
        keep = torch.zeros(obs.size(1), dtype=torch.bool, device=obs.device)
        keep[perm[:n_keep]] = True
        active[b] = keep
    return active


def predict(model, loader, device, scenario=None):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for *xs, y, obs in loader:
            xs = [x.to(device) for x in xs]
            n_mod = obs.size(1)
            active = obs.to(device) if scenario is None else torch.tensor(scenario, device=device).expand(len(y), n_mod)
            logits, _, _ = model(xs, active)
            preds.append(logits.argmax(-1).cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(targets), np.concatenate(preds)


def metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred) * 100,
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "bacc": balanced_accuracy_score(y_true, y_pred) * 100,
        "mf1": f1_score(y_true, y_pred, average="micro", zero_division=0) * 100,
    }


def table2_mean_acc(model, loader, device, scenarios):
    accs = [metrics(*predict(model, loader, device, scenario))["acc"] for _, scenario in scenarios]
    return float(np.mean(accs))


def meta_step(model, xs, y, obs, meta_optimizer, criterion):
    """Algorithm 1, stage "Meta-validation & Meta-updating".

    w* = argmin_w sum_{D_v} l_meta(y, f_zeta(concat_i w_i x f_i)) with the current
    theta, zeta (no stars in Algorithm 1), then normalised by Eq. (6) (softmax in
    get_importance_weights). Only w is updated here.
    """
    training = model.training
    model.eval()
    with torch.no_grad():
        stack, _ = model.encode_stack(xs, obs)
    logits = model.fuse(stack, use_iwv=True)
    meta_optimizer.zero_grad()
    criterion(logits, y).backward()
    meta_optimizer.step()
    model.train(training)


def run_seed(seed, data, args, device):
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
        print(f"  [seed {seed}] D_t {len(idx_t)} / D_v {len(idx_v)} carved from train "
              f"(D_v complete rows {int(obs_all[idx_v].all(1).sum())})")
    else:
        meta_part, train_part = data["val"], data["train"]
    xs_tr, obs_tr, y_tr = apply_smote(*train_part, rng=rng,
                                      neighbour_mods=None if (args.drop_g or not args.smote_no_g)
                                      else [0, 2, 3])
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

    model = MetaKDADNIModel([x.shape[1] for x in xs_tr], args.latent_dim, num_patches=args.num_patches,
                           shared_encoder=args.shared_encoder).to(device)
    print(
        f"  [seed {seed}] encoders {[type(enc).__name__ for enc in model.encoders]} "
        f"dims {[x.shape[1] for x in xs_tr]} lr={args.lr} dropout=0-3",
        flush=True,
    )
    task_params = [p for n, p in model.named_parameters() if n != "raw_weights"]
    optimizer = optim.Adam(task_params, lr=args.lr, weight_decay=args.weight_decay)
    meta_optimizer = optim.Adam([model.raw_weights], lr=args.meta_lr, weight_decay=args.meta_weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    if args.class_weight:
        counts = np.bincount(y_tr, minlength=3).astype(np.float64)
        weights = counts.sum() / (len(counts) * np.maximum(counts, 1.0))
        print(f"  [seed {seed}] class weights {weights} from counts {counts.astype(int)}")
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    else:
        criterion = nn.CrossEntropyLoss()

    n_epochs = args.epochs
    if args.total_iters > 0:
        n_epochs = -(-args.total_iters // len(train_loader))
        print(f"  [seed {seed}] {args.total_iters} iterations = {n_epochs} epochs x {len(train_loader)} batches")
    best_acc, best_state, last_state, it = -1.0, None, None, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        tot_task, tot_kd, n_batches = 0.0, 0.0, 0
        for *xs, y, obs in train_loader:
            if args.total_iters > 0 and it >= args.total_iters:
                break
            # Algorithm 1: meta-update w* first, then N = meta_every meta-training iterations with w* fixed.
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
            l_kd = model.compute_pairwise_kd_loss(feats, active, reduction=args.kd_reduction,
                                                  detach_teacher=args.kd_detach)
            optimizer.zero_grad()
            (l_task + args.alpha * l_kd).backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(task_params, max_norm=args.grad_clip)
            optimizer.step()
            tot_task += l_task.item()
            tot_kd += l_kd.item()
            n_batches += 1
            it += 1
        scheduler.step()
        last_state = copy.deepcopy(model.state_dict())
        val = metrics(*predict(model, val_loader, device))
        val_t2 = table2_mean_acc(model, val_loader, device, scenarios)
        if val_t2 > best_acc:
            best_acc, best_state = val_t2, copy.deepcopy(model.state_dict())
        if epoch % 5 == 0 or epoch == 1:
            w = model.get_importance_weights().detach().cpu().numpy()
            print(f"  [seed {seed}] Epoch {epoch:02d}/{n_epochs} it {it} | "
                  f"task {tot_task / max(1, n_batches):.4f} kd {tot_kd / max(1, n_batches):.4f} | "
                  f"Val Acc {val['acc']:.2f} Table2 {val_t2:.2f} F1 {val['f1']:.2f} | "
                  f"IWV " + " ".join(f"{m}:{w[i]:.3f}" for i, m in enumerate(mods)), flush=True)

    ckpt_dir = os.path.join(ADNI_ROOT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(last_state, os.path.join(ckpt_dir, f"last_adni_model_seed{seed}.pth"))
    torch.save(best_state, os.path.join(ckpt_dir, f"best_adni_model_seed{seed}.pth"))
    # Flex-MoE official ADNI run keeps the best validation-accuracy checkpoint.
    if args.last_checkpoint:
        model.load_state_dict(last_state)
        print(f"  [seed {seed}] evaluating LAST epoch weights (no model selection); "
              f"best Table-2 val Accuracy was {best_acc:.2f}")
    else:
        model.load_state_dict(best_state)
        print(f"  [seed {seed}] best Table-2 val Accuracy {best_acc:.2f}")
    return {name: metrics(*predict(model, test_loader, device, scenario)) for name, scenario in scenarios}


def main():
    args = get_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"ADNI METAKD TABLE 2 | Flex-MoE official data protocol | device={device}")
    print("=" * 80)

    subjects, data, idx_train, idx_val, idx_test, h5ad = load_protocol(args)
    print(f"Split: train {len(idx_train)} (incomplete OK), val {len(idx_val)}, "
          f"test {len(idx_test)} (all four modalities)")
    os.makedirs("logs", exist_ok=True)
    with open("logs/adni_protocol_manifest.json", "w", encoding="utf-8") as handle:
        json.dump({
            "flexmoe_repo": "https://github.com/UNITES-Lab/Flex-MoE",
            "flexmoe_commit": "21688077488e58c1a634ca7122cffdb28cae3bc9",
            "data_root": os.path.abspath(args.data_root),
            "genomic_h5ad": os.path.abspath(h5ad) if h5ad else "",
            "allow_generated_split": args.allow_generated_split,
            "allow_raw_genomic_fallback": args.allow_raw_genomic_fallback,
            "split_sizes": {"train": int(len(idx_train)), "val": int(len(idx_val)), "test": int(len(idx_test))},
            "hyperparameters": {
                "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                "class_weight": args.class_weight,
                "weight_decay": args.weight_decay, "meta_lr": args.meta_lr,
                "meta_every": args.meta_every, "alpha": args.alpha,
                "latent_dim": args.latent_dim, "num_patches": args.num_patches,
                "modality_dropout": "none" if args.no_modality_drop else "drop_0_to_n_minus_1_observed",
                "center_inputs": args.center_inputs,
                "g_pca": args.g_pca,
                "smote_no_g": args.smote_no_g,
                "drop_g": args.drop_g,
                "shared_encoder": args.shared_encoder,
                "kd_reduction": args.kd_reduction,
                "meta_val_source": args.meta_val_source,
                "meta_val_frac": args.meta_val_frac,
                "last_checkpoint": args.last_checkpoint,
            },
        }, handle, indent=2)
    if args.dry_run:
        print("Dry run only; not training.")
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    per_seed = []
    for seed in seeds:
        print("\n" + "=" * 80 + f"\nTRAINING METAKD, seed {seed}\n" + "=" * 80)
        per_seed.append(run_seed(seed, data, args, device))

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

    print("\n" + "=" * 112)
    print(f"ADNI TABLE 2 (test {len(idx_test)} full-modality subjects, mean over seeds {seeds})")
    print("=" * 112)
    header = f"{'Modality':<9}" + "".join(f" | {lbl:<25}" for lbl in ("Accuracy", "Macro-F1", "Balanced Acc", "Micro-F1"))
    print(header)
    print(f"{'':<9}" + " | Ours+-std    Paper  Diff  " * 4)
    print("-" * len(header))
    for r in rows + [dict({"Modality": "AVERAGE"}, **res.drop(columns="Modality").mean().to_dict())]:
        line = f"{r['Modality']:<9}"
        for key in ("acc", "f1", "bacc", "mf1"):
            line += f" | {r[f'Test_{key}']:5.2f}+-{r[f'Std_{key}']:<5.2f} {r[f'Paper_{key}']:5.2f} {r[f'Diff_{key}']:+6.2f}"
        print(line)
    print("=" * 112)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    res.to_csv(args.output_csv, index=False)
    pd.DataFrame([dict(seed=s, Modality=n, **per_seed[i][n]) for i, s in enumerate(seeds) for n, _ in scenarios]) \
        .to_csv(os.path.splitext(args.output_csv)[0] + "_per_seed.csv", index=False)
    print(f"Saved {args.output_csv}")


if __name__ == "__main__":
    main()
