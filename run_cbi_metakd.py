import argparse
import itertools
import json
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Modalities: C, B, I (OMITTING G)
MODS = ["clinical", "biospecimen", "imaging"]
SHORT = {"clinical": "C", "biospecimen": "B", "imaging": "I"}

# Reference Paper Table 2 numbers for non-G combinations
PAPER_TABLE2 = {
    "C": (59.65, 43.25, 56.24),
    "B": (56.23, 40.57, 53.46),
    "I": (55.32, 39.05, 51.89),
    "C,B": (63.24, 45.87, 59.78),
    "C,I": (62.75, 41.23, 59.42),
    "B,I": (57.56, 39.54, 54.67),
    "C,B,I": (63.06, 42.65, 59.89),
}

# ----------------------------------------------------------------------------
# Model: MetaKD with Bi-level IWV (C, B, I)
# ----------------------------------------------------------------------------
class MetaKDNet3(nn.Module):
    def __init__(self, in_dims, hidden=128, n_classes=3, dropout=0.2):
        super().__init__()
        self.N = len(in_dims)
        self.hidden = hidden
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU())
            for d in in_dims
        ])
        self.encoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.N * hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
        # Importance weight vector (w_C, w_B, w_I)
        self.iwv = nn.Parameter(0.01 * torch.randn(self.N))

    def net_parameters(self):
        return [p for n, p in self.named_parameters() if n != "iwv"]

    def weights(self):
        return F.softmax(self.iwv, dim=0)

    def encode(self, xs, mask):
        f = torch.stack([self.encoder(self.proj[i](xs[i])) for i in range(self.N)], dim=1)
        m = mask.float().unsqueeze(-1)
        mean = (f * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1.0)
        f_full = m * f + (1 - m) * mean
        return f, f_full

    def decode(self, f_full, w=None):
        if w is not None:
            f_full = f_full * w.view(1, -1, 1)
        return self.decoder(f_full.flatten(1))


def kd_loss(f, mask, w, p=1, mode="ratio", normalize=True, detach=True):
    B, N, H = f.shape
    teacher = f.detach() if detach else f
    diff = teacher.unsqueeze(2) - f.unsqueeze(1)
    d = diff.abs().sum(-1) if p == 1 else diff.norm(p=p, dim=-1)
    if normalize:
        d = d / H
    mf = mask.float()
    pair = mf.unsqueeze(2) * mf.unsqueeze(1)
    pair = pair * (1 - torch.eye(N, device=f.device)).unsqueeze(0)
    if mode == "product":
        coef = w.view(N, 1) * w.view(1, N)
    else:
        coef = w.view(N, 1) / w.view(1, N).clamp(min=1e-8)
    return (pair * coef.unsqueeze(0) * d).sum((1, 2)).mean()


def random_drop(avail):
    B, N = avail.shape
    n_avail = avail.sum(1)
    k = torch.floor(torch.rand(B, device=avail.device) * n_avail).long()
    scores = torch.rand(B, N, device=avail.device)
    scores[~avail] = 2.0
    ranks = scores.argsort(1).argsort(1)
    drop = (ranks < k.unsqueeze(1)) & avail
    return avail & ~drop


def apply_mask(xs, mask):
    return [x * mask[:, i:i + 1].float() for i, x in enumerate(xs)]


# ----------------------------------------------------------------------------
# Data loading (adni.npz without genomic)
# ----------------------------------------------------------------------------
def load_data(npz_path, seed=42, meta_val_ratio=0.2, center=True):
    raw = dict(np.load(npz_path, allow_pickle=True))
    X = [np.nan_to_num(raw[m].astype(np.float32)) for m in MODS]
    
    # Original mask has columns: [C, G, B, I] -> we take [0, 2, 3] for [C, B, I]
    orig_mask = raw["mask"].astype(bool)
    mask = orig_mask[:, [0, 2, 3]]
    y = raw["y"].astype(np.int64)

    keep = mask.any(1)
    X = [x[keep] for x in X]
    mask = mask[keep]
    y = y[keep]

    if "idx_train" in raw:
        remap = -np.ones(len(keep), dtype=np.int64)
        remap[keep] = np.arange(int(keep.sum()))
        tr, va, te = (remap[raw[k]] for k in ("idx_train", "idx_val", "idx_test"))
        assert (tr >= 0).all() and (va >= 0).all() and (te >= 0).all()
    else:

        tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
        va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=seed)

    dt, dv = train_test_split(tr, test_size=meta_val_ratio, stratify=y[tr], random_state=seed)

    # MinMax [-1, 1] fit only on training rows
    for i in range(len(MODS)):
        rows = tr[mask[tr, i]]
        lo, hi = X[i][rows].min(0), X[i][rows].max(0)
        X[i] = np.clip(2 * (X[i] - lo) / np.maximum(hi - lo, 1e-8) - 1, -1, 1)
        if center:
            X[i] = X[i] - X[i][rows].mean(0, keepdims=True)
        X[i] = X[i] * mask[:, i:i + 1]

    def pack(ix):
        return dict(
            X=[torch.tensor(x[ix]) for x in X],
            mask=torch.tensor(mask[ix]),
            y=torch.tensor(y[ix])
        )

    return pack(dt), pack(dv), pack(va), pack(te), [x.shape[1] for x in X]


def make_loader(split, bs, shuffle):
    ds = TensorDataset(*split["X"], split["mask"], split["y"])
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=shuffle and len(ds) > bs)


def cycle(loader):
    while True:
        for b in loader:
            yield b


# ----------------------------------------------------------------------------
# Evaluation on all 7 modality combinations
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict(model, X, mask_in, device, bs=512):
    model.eval()
    out = []
    for s in range(0, len(mask_in), bs):
        xs = [x[s:s + bs].to(device) for x in X]
        m = mask_in[s:s + bs].to(device)
        _, ff = model.encode(apply_mask(xs, m), m)
        out.append(model.decode(ff).argmax(1).cpu())
    return torch.cat(out).numpy()


def evaluate(model, split, device):
    results = {}
    for r in range(1, len(MODS) + 1):
        for combo in itertools.combinations(range(len(MODS)), r):
            sel = split["mask"][:, list(combo)].all(1)
            if sel.sum() == 0:
                continue
            X = [x[sel] for x in split["X"]]
            m = torch.zeros(int(sel.sum()), len(MODS), dtype=torch.bool)
            m[:, list(combo)] = True
            pred = predict(model, X, m, device)
            yt = split["y"][sel].numpy()
            tag = ",".join(SHORT[MODS[i]] for i in combo)
            results[tag] = dict(
                n=int(sel.sum()),
                acc=100 * accuracy_score(yt, pred),
                f1=100 * f1_score(yt, pred, average="macro", zero_division=0),
                bal_acc=100 * balanced_accuracy_score(yt, pred),
            )
    avg = {k: float(np.mean([v[k] for v in results.values()])) for k in ["acc", "f1", "bal_acc"]}
    return results, avg


# ----------------------------------------------------------------------------
# Train Run
# ----------------------------------------------------------------------------
def run_experiment(npz_path, seed=42, epochs=30, bs=64, lr=1e-3, meta_lr=1e-2, alpha=0.05):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Dt, Dv, VAL, TEST, in_dims = load_data(npz_path, seed=seed, center=True)

    print(f"=== METAKD TRAINING WITHOUT G (Modality C, B, I) | Seed: {seed} ===")
    print(f"Dt={len(Dt['y'])}, Dv={len(Dv['y'])}, Val={len(VAL['y'])}, Test={len(TEST['y'])}")
    print(f"Input dims: C={in_dims[0]}, B={in_dims[1]}, I={in_dims[2]}")

    model = MetaKDNet3(in_dims, hidden=128, n_classes=3, dropout=0.2).to(device)
    opt = torch.optim.Adam(model.net_parameters(), lr=lr, weight_decay=1e-2)
    meta_opt = torch.optim.Adam([model.iwv], lr=meta_lr, weight_decay=1e-2)

    cnt = torch.bincount(Dt["y"], minlength=3).float()
    cw = (cnt.sum() / (3.0 * cnt.clamp(min=1))).to(device)

    train_loader = make_loader(Dt, bs, True)
    meta_iter = cycle(make_loader(Dv, bs, True))
    N = len(in_dims)

    best_val_acc, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_kd = 0.0, 0.0
        for batch in train_loader:
            xs = [b.to(device) for b in batch[:N]]
            avail, y = batch[N].to(device), batch[N + 1].to(device)

            m_in = random_drop(avail)
            f, ff = model.encode(apply_mask(xs, m_in), m_in)
            w = model.weights().detach()

            l_task = F.cross_entropy(model.decode(ff), y, weight=cw)
            l_kd = kd_loss(f, m_in, w, p=1, mode="ratio", normalize=True, detach=True)
            loss = l_task + alpha * l_kd

            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += l_task.item(); total_kd += l_kd.item()

            # Meta-update
            mb = next(meta_iter)
            xv = [b.to(device) for b in mb[:N]]
            mv, yv = random_drop(mb[N].to(device)), mb[N + 1].to(device)

            model.eval()
            with torch.no_grad():
                _, ffv = model.encode(apply_mask(xv, mv), mv)
            l_meta = F.cross_entropy(model.decode(ffv, model.weights()), yv, weight=cw)

            meta_opt.zero_grad()
            model.iwv.grad = torch.autograd.grad(l_meta, model.iwv)[0]
            meta_opt.step()

        _, val_avg = evaluate(model, VAL, device)
        w_vals = model.weights().tolist()
        w_str = f"C={w_vals[0]:.3f}, B={w_vals[1]:.3f}, I={w_vals[2]:.3f}"
        print(f"Epoch {epoch:02d}/{epochs} | Val Acc: {val_avg['acc']:.2f}% | Val F1: {val_avg['f1']:.2f}% | IWV: [{w_str}]")

        if val_avg["acc"] > best_val_acc:
            best_val_acc = val_avg["acc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_res, test_avg = evaluate(model, TEST, device)
    final_w = model.weights().tolist()

    return test_res, test_avg, final_w


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetaKD on ADNI with C, B, I (omitting G)")
    parser.add_argument("--data", type=str, default="kaggle_adni_run/adni.npz", help="Path to adni.npz")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs per seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999], help="Random seeds")
    args = parser.parse_args()

    npz = args.data
    if not os.path.isabs(npz) and not os.path.exists(npz):
        # Fallback to repo root lookup
        repo_root = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.join(repo_root, npz)
        if os.path.exists(alt):
            npz = alt

    seeds = args.seeds
    all_runs = []

    print("\n" + "=" * 90)
    print("CHẠY THỰC NGHIỆM METAKD KHÔNG CÓ GENOMIC (CHỈ DÙNG CLINICAL, BIOSPECIMEN, IMAGING)")
    print(f"Dữ liệu: {npz} | Số epochs: {args.epochs} | Seeds: {seeds}")
    print("=" * 90)

    for s in seeds:
        res, avg, weights = run_experiment(npz, seed=s, epochs=args.epochs)
        all_runs.append((res, avg, weights))


    print("\n" + "=" * 95)
    print(f"KẾT QUẢ TEST ĐỐI CHIẾU TRỰC TIẾP VỚI BÀI BÁO (TRUNG BÌNH QUA {len(seeds)} SEEDS)")
    print("=" * 95)
    print(f"{'Tổ hợp':<10} | {'Ours Acc (Mean+-Std)':>22} | {'Paper Acc':>10} | {'Acc Diff':>10} | {'Ours F1':>10} | {'Paper F1':>10} | {'F1 Diff':>10}")
    print("-" * 95)

    combos = ["C", "B", "I", "C,B", "C,I", "B,I", "C,B,I"]
    for c in combos:
        accs = [run[0][c]["acc"] for run in all_runs]
        f1s = [run[0][c]["f1"] for run in all_runs]
        m_acc, s_acc = np.mean(accs), np.std(accs)
        m_f1 = np.mean(f1s)
        p_acc, p_f1, _ = PAPER_TABLE2[c]
        d_acc = m_acc - p_acc
        d_f1 = m_f1 - p_f1
        d_acc_str = f"+{d_acc:.2f}%" if d_acc >= 0 else f"{d_acc:.2f}%"
        d_f1_str = f"+{d_f1:.2f}%" if d_f1 >= 0 else f"{d_f1:.2f}%"

        print(f"{c:<10} | {m_acc:>6.2f}% +- {s_acc:4.2f}%       | {p_acc:>9.2f}% | {d_acc_str:>10} | {m_f1:>9.2f}% | {p_f1:>9.2f}% | {d_f1_str:>10}")

    mean_acc = np.mean([[run[0][c]["acc"] for c in combos] for run in all_runs])
    mean_f1 = np.mean([[run[0][c]["f1"] for c in combos] for run in all_runs])
    p_mean_acc = np.mean([PAPER_TABLE2[c][0] for c in combos])
    p_mean_f1 = np.mean([PAPER_TABLE2[c][1] for c in combos])

    print("-" * 95)
    print(f"{'TRUNG BÌNH':<10} | {mean_acc:>6.2f}%                  | {p_mean_acc:>9.2f}% | {mean_acc - p_mean_acc:>+9.2f}% | {mean_f1:>9.2f}% | {p_mean_f1:>9.2f}% | {mean_f1 - p_mean_f1:>+9.2f}%")
    print("=" * 95)
