import os
import sys
import glob
import numpy as np
import pandas as pd
import pyreadr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score

# ==============================================================================
# 1. MODEL ARCHITECTURE (1D RESNET-34 & METAKD PAIRWISE DISTILLATION)
# ==============================================================================

class BasicBlock1D(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet34_1D(nn.Module):
    def __init__(self, in_channels=1, out_features=128):
        super(ResNet34_1D, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, out_features)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )
        layers = [BasicBlock1D(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class MLPEncoder(nn.Module):
    def __init__(self, in_features, out_features=128):
        super(MLPEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class MetaKDADNIModel(nn.Module):
    def __init__(self, c_dim, g_dim, b_dim, i_dim, latent_dim=128, num_classes=3):
        super(MetaKDADNIModel, self).__init__()
        self.enc_c = MLPEncoder(c_dim, latent_dim)
        self.enc_g = ResNet34_1D(in_channels=1, out_features=latent_dim)
        self.enc_b = MLPEncoder(b_dim, latent_dim)
        self.enc_i = MLPEncoder(i_dim, latent_dim)

        self.head_c = nn.Linear(latent_dim, num_classes)
        self.head_g = nn.Linear(latent_dim, num_classes)
        self.head_b = nn.Linear(latent_dim, num_classes)
        self.head_i = nn.Linear(latent_dim, num_classes)

        self.fused_classifier = nn.Sequential(
            nn.Linear(latent_dim * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.raw_weights = nn.Parameter(torch.ones(4))

    def get_importance_weights(self):
        return torch.softmax(self.raw_weights, dim=0)

    def forward(self, c, g, b, i, active_modalities=(True, True, True, True)):
        # 1. Extract active features (Eq. 1 MetaKD)
        features = [None, None, None, None]
        active_feats = []
        if active_modalities[0]:
            features[0] = self.enc_c(c)
            active_feats.append(features[0])
        if active_modalities[1]:
            features[1] = self.enc_g(g)
            active_feats.append(features[1])
        if active_modalities[2]:
            features[2] = self.enc_b(b)
            active_feats.append(features[2])
        if active_modalities[3]:
            features[3] = self.enc_i(i)
            active_feats.append(features[3])

        # 2. Missing Modality Feature Generation (Eq. 2 MetaKD Paper)
        # f_n = 1 / (N - |Q|) * sum_{x_i in M \ Q} f_i
        if len(active_feats) > 0:
            mean_active = torch.stack(active_feats, dim=0).mean(dim=0)
        else:
            mean_active = torch.zeros(c.size(0), 128, device=c.device)

        for m_idx in range(4):
            if features[m_idx] is None:
                features[m_idx] = mean_active

        zc, zg, zb, zi = features[0], features[1], features[2], features[3]

        weights = self.get_importance_weights()
        weighted_features = torch.cat([
            zc * weights[0],
            zg * weights[1],
            zb * weights[2],
            zi * weights[3]
        ], dim=-1)

        fused_logits = self.fused_classifier(weighted_features)

        mod_logits = []
        if active_modalities[0]: mod_logits.append(self.head_c(zc))
        if active_modalities[1]: mod_logits.append(self.head_g(zg))
        if active_modalities[2]: mod_logits.append(self.head_b(zb))
        if active_modalities[3]: mod_logits.append(self.head_i(zi))

        return fused_logits, mod_logits, (zc, zg, zb, zi)

    def compute_pairwise_kd_loss(self, feat_tuple, active_modalities):
        """
        Exact MetaKD Pairwise Knowledge Distillation (Paper Eq. 5, p=1 L1 Loss & Algorithm 1).
        Loss = sum_{i,j active, i != j} (w_i^* / w_j^*) * || f_i - f_j ||_1
        """
        active_indices = [idx for idx, act in enumerate(active_modalities) if act]
        if len(active_indices) < 2:
            return torch.tensor(0.0, device=self.raw_weights.device)

        weights = self.get_importance_weights()
        loss_kd = 0.0
        n_pairs = 0

        for i in active_indices:
            for j in active_indices:
                if i == j:
                    continue
                # Dominant modality w_i acts as teacher, w_j acts as student
                w_ratio = (weights[i] / (weights[j] + 1e-6)).detach()
                # Feature distillation L1 norm (Eq. 5 with p=1)
                dist = torch.mean(torch.abs(feat_tuple[i] - feat_tuple[j]))
                loss_kd += w_ratio * dist
                n_pairs += 1

        return loss_kd / max(1, n_pairs)


# ==============================================================================
# 2. DATA EXTRACTION, LD PRUNING, AND SMOTE
# ==============================================================================

def read_bed_snps(bed_path, n_samples, snp_indices):
    lookup = np.array([2.0, np.nan, 1.0, 0.0], dtype=np.float32)
    bytes_per_snp = (n_samples + 3) // 4
    results = np.empty((n_samples, len(snp_indices)), dtype=np.float32)
    with open(bed_path, 'rb') as f:
        magic = f.read(3)
        assert magic == b'\x6c\x1b\x01', 'Not SNP-major PLINK BED'
        for col_idx, s_idx in enumerate(snp_indices):
            f.seek(3 + s_idx * bytes_per_snp)
            raw = np.frombuffer(f.read(bytes_per_snp), dtype=np.uint8)
            g0 = raw & 0b11
            g1 = (raw >> 2) & 0b11
            g2 = (raw >> 4) & 0b11
            g3 = (raw >> 6) & 0b11
            unpacked = np.column_stack([g0, g1, g2, g3]).reshape(-1)[:n_samples]
            results[:, col_idx] = lookup[unpacked]
    return results


def find_file(data_root, subfolder, pattern):
    direct = os.path.join(data_root, subfolder, pattern)
    if os.path.exists(direct):
        return direct
    matches = glob.glob(os.path.join(data_root, "**", pattern), recursive=True)
    return matches[0] if matches else None


def ld_prune_snps(snp_matrix, window_size=50, step_size=5, r2_threshold=0.1):
    """
    Vectorized Linkage Disequilibrium (LD) Pruning matching PLINK --indep-pairwise 50 5 0.1
    Removes highly correlated SNPs within a sliding window.
    """
    n_snps = snp_matrix.shape[1]
    keep_indices = []

    means = np.nanmean(snp_matrix, axis=0)
    stds = np.nanstd(snp_matrix, axis=0)
    stds[stds == 0] = 1.0
    norm_snps = np.nan_to_num((snp_matrix - means) / stds, nan=0.0)
    n_samples = norm_snps.shape[0]

    for start in range(0, n_snps, step_size):
        end = min(start + window_size, n_snps)
        window = norm_snps[:, start:end]
        corr = np.dot(window.T, window) / n_samples
        corr_sq = corr ** 2

        selected_in_win = []
        for i in range(corr_sq.shape[0]):
            global_idx = start + i
            if global_idx in keep_indices:
                continue
            is_redundant = any(corr_sq[i, prev] >= r2_threshold for prev in selected_in_win)
            if not is_redundant:
                selected_in_win.append(i)
                keep_indices.append(global_idx)

    keep_indices = sorted(list(set(keep_indices)))
    return keep_indices


def apply_multimodal_smote(c, g, b, i, y, k_neighbors=5, random_state=42):
    """
    Multimodal SMOTE (k=5) as specified in MetaKD Paper (Section 8, Page 13).
    Interpolates minority classes (MCI and AD) to balance against majority class (CN)
    while preserving inter-modality subject alignment.
    """
    np.random.seed(random_state)
    classes, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)

    syn_c, syn_g, syn_b, syn_i, syn_y = [c], [g], [b], [i], [y]

    for cls, count in zip(classes, counts):
        n_to_generate = max_count - count
        if n_to_generate <= 0:
            continue

        cls_idx = np.where(y == cls)[0]
        cls_c, cls_g, cls_b, cls_i = c[cls_idx], g[cls_idx], b[cls_idx], i[cls_idx]

        # Use joint normalized representation to find true multimodal neighbors
        feat_joint = np.hstack([cls_c, cls_b, cls_i])
        k = min(k_neighbors, len(cls_idx) - 1)
        if k < 1:
            continue

        nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(feat_joint)
        _, indices = nn.kneighbors(feat_joint)

        gen_c = np.empty((n_to_generate, c.shape[1]), dtype=np.float32)
        gen_g = np.empty((n_to_generate, g.shape[1]), dtype=np.float32)
        gen_b = np.empty((n_to_generate, b.shape[1]), dtype=np.float32)
        gen_i = np.empty((n_to_generate, i.shape[1]), dtype=np.float32)

        for idx in range(n_to_generate):
            base_idx = np.random.randint(len(cls_idx))
            neighbor_idx = indices[base_idx, np.random.randint(1, k + 1)]
            lam = np.random.uniform(0.0, 1.0)

            gen_c[idx] = cls_c[base_idx] + lam * (cls_c[neighbor_idx] - cls_c[base_idx])
            gen_g[idx] = cls_g[base_idx] + lam * (cls_g[neighbor_idx] - cls_g[base_idx])
            gen_b[idx] = cls_b[base_idx] + lam * (cls_b[neighbor_idx] - cls_b[base_idx])
            gen_i[idx] = cls_i[base_idx] + lam * (cls_i[neighbor_idx] - cls_i[base_idx])

        syn_c.append(gen_c)
        syn_g.append(gen_g)
        syn_b.append(gen_b)
        syn_i.append(gen_i)
        syn_y.append(np.full(n_to_generate, cls, dtype=np.int64))

    out_c = np.vstack(syn_c)
    out_g = np.vstack(syn_g)
    out_b = np.vstack(syn_b)
    out_i = np.vstack(syn_i)
    out_y = np.concatenate(syn_y)

    print(f"  [SMOTE k={k_neighbors}] Balanced training data: {len(y)} -> {len(out_y)} samples (Distribution: {np.bincount(out_y)})")
    return out_c, out_g, out_b, out_i, out_y


# ==============================================================================
# 3. MAIN EXECUTION PIPELINE
# ==============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*80)
    print("🚀 ADNI MULTIMODAL PIPELINE (METAKD & FLEX-MOE REPRODUCTION)")
    print(f"Compute Device: {device}")
    print("="*80)

    data_root = "adni_raw_flexmoe"
    if not os.path.exists(data_root):
        data_root = "data_raw"
    print(f"Data root directory: {os.path.abspath(data_root)}")

    # 1. Labels
    print("\n[1/6] Reading ground-truth diagnosis labels from DXSUM.csv...")
    dx_file = find_file(data_root, "diagnosis", "DXSUM.csv")
    dx_df = pd.read_csv(dx_file, low_memory=False)
    bl_mask = dx_df['VISCODE'].isin(['bl', 'sc', 'init']) if 'VISCODE' in dx_df.columns else pd.Series(True, index=dx_df.index)
    dx_bl = dx_df[bl_mask].copy()

    labels_dict = {}
    for _, row in dx_bl.iterrows():
        ptid = row.get('PTID')
        if not ptid or ptid in labels_dict:
            continue
        dx = row.get('DIAGNOSIS')
        dxchange = row.get('DXCHANGE')
        if pd.notna(dx):
            if dx == 1: labels_dict[ptid] = 0
            elif dx == 2: labels_dict[ptid] = 1
            elif dx == 3: labels_dict[ptid] = 2
        elif pd.notna(dxchange):
            if dxchange in [1, 7]: labels_dict[ptid] = 0
            elif dxchange in [2, 4, 8]: labels_dict[ptid] = 1
            elif dxchange in [3, 5, 6]: labels_dict[ptid] = 2

    subjects = pd.DataFrame(list(labels_dict.items()), columns=['PTID', 'label']).sort_values('PTID').reset_index(drop=True)
    print(f"Total subjects: {len(subjects)} (CN: {(subjects['label']==0).sum()}, MCI: {(subjects['label']==1).sum()}, AD: {(subjects['label']==2).sum()})")

    # Track genuinely observed status per subject
    observed_genomic = set()

    # 2. Clinical
    print("\n[2/6] Reading and preprocessing Clinical (5 ADNIMERGE2 tables)...")
    merged_c = subjects[['PTID']].copy()
    for tbl in ['MEDHIST.rda', 'NEUROEXM.rda', 'PTDEMOG.rda', 'RECCMEDS.rda', 'VITALS.rda']:
        p = find_file(data_root, "clinical", tbl)
        if p and os.path.exists(p):
            r = pyreadr.read_r(p)
            df = list(r.values())[0]
            join_col = 'PTID' if 'PTID' in df.columns else 'RID'
            df_bl = df.drop_duplicates(join_col)
            leakage = {'PTCOGBEG', 'PTADDX', 'PTADBEG', 'USERDATE', 'UPDATE_STAMP', 'ID', 'SITEID', 'VISCODE', 'VISCODE2', 'RID'}
            prefix = tbl.replace('.rda', '')
            num_cols = [c for c in df_bl.columns if c not in leakage and c != join_col and df_bl[c].dtype in ['float64', 'int64']]
            sub_df = df_bl[[join_col] + num_cols].copy()
            sub_df.rename(columns={c: f"{prefix}_{c}" for c in num_cols}, inplace=True)
            merged_c = pd.merge(merged_c, sub_df, on=join_col, how='left')

    raw_c = merged_c.drop(columns=['PTID', 'RID'], errors='ignore').apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
    print(f"Clinical features shape: {raw_c.shape}")

    # 3. Biospecimen
    print("\n[3/6] Reading and preprocessing Biospecimen (Roche CSF + ApoE)...")
    merged_b = subjects[['PTID']].copy()
    roche_p = find_file(data_root, "biospecimen", "UPENNBIOMK_ROCHE_ELECSYS.rda")
    if roche_p and os.path.exists(roche_p):
        r = pyreadr.read_r(roche_p)
        df = list(r.values())[0]
        for col in ['ABETA40', 'ABETA42', 'TAU', 'PTAU']:
            if col in df.columns:
                clean_s = df[col].astype(str).str.replace('<', '').str.replace('>', '')
                df[col] = pd.to_numeric(clean_s, errors='coerce')
        merged_b = pd.merge(merged_b, df[['PTID', 'ABETA40', 'ABETA42', 'TAU', 'PTAU']].drop_duplicates('PTID'), on='PTID', how='left')

    apoe_p = find_file(data_root, "biospecimen", "APOERES.csv")
    if apoe_p and os.path.exists(apoe_p):
        apoe_df = pd.read_csv(apoe_p, low_memory=False).drop_duplicates('PTID')
        merged_b = pd.merge(merged_b, apoe_df[['PTID', 'GENOTYPE']], on='PTID', how='left')
        merged_b['E4_COUNT'] = merged_b['GENOTYPE'].apply(lambda g: str(g).count('4') if pd.notna(g) else 0).astype(np.float32)
        merged_b['E2_COUNT'] = merged_b['GENOTYPE'].apply(lambda g: str(g).count('2') if pd.notna(g) else 0).astype(np.float32)
        merged_b.drop(columns=['GENOTYPE'], inplace=True, errors='ignore')

    raw_b = merged_b.drop(columns=['PTID', 'RID'], errors='ignore').apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
    print(f"Biospecimen features shape: {raw_b.shape}")

    # 4. Imaging
    print("\n[4/6] Reading and preprocessing Imaging (FreeSurfer 7.x UCSFFSX7)...")
    merged_i = subjects[['PTID']].copy()
    img_p = find_file(data_root, "image", "UCSFFSX7.csv")
    if img_p and os.path.exists(img_p):
        fs = pd.read_csv(img_p, low_memory=False)
        fs_bl = fs.drop_duplicates('PTID')
        fs_num = [c for c in fs_bl.columns if fs_bl[c].dtype in ['float64', 'int64'] and c not in ['RID', 'IMAGEUID', 'ID']]
        if 'ST10CV' in fs_bl.columns:
            icv = pd.to_numeric(fs_bl['ST10CV'], errors='coerce')
            icv_clean = icv.replace(0, np.nan).fillna(icv.median())
            for c in fs_num:
                if 'SV' in c or 'CV' in c:
                    fs_bl[f'{c}_ICV_norm'] = fs_bl[c] / (icv_clean + 1e-5)
        merged_i = pd.merge(merged_i, fs_bl, on='PTID', how='left')

    raw_i = merged_i.drop(columns=['PTID', 'RID', 'VISCODE', 'VISCODE2', 'IMAGEUID'], errors='ignore').apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
    print(f"Imaging features shape: {raw_i.shape}")

    # 5. Genomic with PLINK LD Pruning (50 5 0.1)
    print("\n[5/6] Processing Genomic (5 Cohorts PLINK with Linkage Disequilibrium Pruning 50 5 0.1)...")
    plink_prefixes = [
        "ADNI_cluster_01_forward_757LONI",
        "ADNI_GO_2_Forward_Bin",
        "ADNI_GO2_GWAS_2nd_orig_BIN",
        "ADNI3_PLINK_Final",
        "ADNI3_PLINK_FINAL_2nd"
    ]

    bim_dfs = []
    for pfx in plink_prefixes:
        bim_path = find_file(data_root, "genomic", f"{pfx}.bim")
        if bim_path and os.path.exists(bim_path):
            b = pd.read_csv(bim_path, header=None, sep=r'\s+', usecols=[0, 1, 3])
            bim_dfs.append(b)

    common_snps = set.intersection(*[set(b[1]) for b in bim_dfs])
    print(f"Common SNPs across all cohorts: {len(common_snps)} SNPs.")

    ref_bim = bim_dfs[0][bim_dfs[0][1].isin(common_snps)].sort_values([0, 3]).reset_index(drop=True)
    candidate_snps = ref_bim[1].tolist()[:40000]

    ptid_list = list(subjects['PTID'])
    raw_g_all = np.full((len(ptid_list), len(candidate_snps)), np.nan, dtype=np.float32)
    ptid_idx = {p: idx for idx, p in enumerate(ptid_list)}
    snp_idx = {s: idx for idx, s in enumerate(candidate_snps)}

    for pfx in plink_prefixes:
        bed = find_file(data_root, "genomic", f"{pfx}.bed")
        bim = find_file(data_root, "genomic", f"{pfx}.bim")
        fam = find_file(data_root, "genomic", f"{pfx}.fam")
        if bed and os.path.exists(bed):
            fam_df = pd.read_csv(fam, header=None, sep=r'\s+')
            cohort_ptids = list(fam_df[1])
            n_samples = len(fam_df)
            c_bim = pd.read_csv(bim, header=None, sep=r'\s+')
            c_snps = list(c_bim[1])
            c_snp_map = {s: idx for idx, s in enumerate(c_snps)}

            common_in_cohort = [s for s in candidate_snps if s in c_snp_map]
            target_indices = [snp_idx[s] for s in common_in_cohort]
            source_indices = [c_snp_map[s] for s in common_in_cohort]

            bed_vals = read_bed_snps(bed, n_samples, source_indices)

            for s_idx, p in enumerate(cohort_ptids):
                if p in ptid_idx:
                    row_idx = ptid_idx[p]
                    raw_g_all[row_idx, target_indices] = bed_vals[s_idx]
                    observed_genomic.add(p)

    print(f"Observed genomic assay available for: {len(observed_genomic)} / {len(subjects)} subjects.")

    # Apply LD Pruning (50 5 0.1) on the subjects with genuine genomic data (with cache)
    observed_g_idx = [ptid_idx[p] for p in observed_genomic if p in ptid_idx]
    cache_ld_file = "checkpoints/pruned_snp_indices_50_5_01.npy"
    if os.path.exists(cache_ld_file):
        print(f"Loading cached LD pruned SNP indices from {cache_ld_file}...")
        prune_keep_cols = list(np.load(cache_ld_file))
    else:
        prune_keep_cols = ld_prune_snps(raw_g_all[observed_g_idx], window_size=50, step_size=5, r2_threshold=0.1)
        os.makedirs("checkpoints", exist_ok=True)
        np.save(cache_ld_file, np.array(prune_keep_cols, dtype=np.int32))
    target_k = min(20000, len(prune_keep_cols))
    selected_cols = prune_keep_cols[:target_k]
    raw_g = raw_g_all[:, selected_cols]
    print(f"Final Genomic SNP matrix shape: {raw_g.shape} (Independent SNPs)")

    # 6. Train/Val/Test Split (70/15/15) - Zero Data Leakage
    y = subjects['label'].values.astype(np.int64)
    N = len(subjects)
    indices = np.arange(N)

    idx_train_val, idx_test, y_train_val, y_test = train_test_split(indices, y, test_size=0.15, random_state=42, stratify=y)
    idx_train, idx_val, y_train, y_val = train_test_split(idx_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val)

    def process_split(raw_mat, idx_tr, idx_va, idx_te):
        tr = raw_mat[idx_tr].copy()
        va = raw_mat[idx_va].copy()
        te = raw_mat[idx_te].copy()
        means = np.nanmean(tr, axis=0)
        means = np.nan_to_num(means, nan=0.0)
        for col in range(tr.shape[1]):
            tr[np.isnan(tr[:, col]), col] = means[col]
            va[np.isnan(va[:, col]), col] = means[col]
            te[np.isnan(te[:, col]), col] = means[col]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        tr = scaler.fit_transform(tr)
        va = scaler.transform(va)
        te = scaler.transform(te)
        return tr.astype(np.float32), va.astype(np.float32), te.astype(np.float32)

    c_tr, c_va, c_te = process_split(raw_c, idx_train, idx_val, idx_test)
    g_tr, g_va, g_te = process_split(raw_g, idx_train, idx_val, idx_test)
    b_tr, b_va, b_te = process_split(raw_b, idx_train, idx_val, idx_test)
    i_tr, i_va, i_te = process_split(raw_i, idx_train, idx_val, idx_test)

    # 7. Apply SMOTE (k=5) on Training Set ONLY (MetaKD Section 8, Page 13)
    print("\n[6/6] Applying Multimodal SMOTE (k=5) on Training set...")
    c_tr, g_tr, b_tr, i_tr, y_train_smote = apply_multimodal_smote(c_tr, g_tr, b_tr, i_tr, y[idx_train], k_neighbors=5, random_state=42)

    train_ds = TensorDataset(torch.from_numpy(c_tr), torch.from_numpy(g_tr), torch.from_numpy(b_tr), torch.from_numpy(i_tr), torch.from_numpy(y_train_smote))
    val_ds   = TensorDataset(torch.from_numpy(c_va), torch.from_numpy(g_va), torch.from_numpy(b_va), torch.from_numpy(i_va), torch.from_numpy(y[idx_val]))
    test_ds  = TensorDataset(torch.from_numpy(c_te), torch.from_numpy(g_te), torch.from_numpy(b_te), torch.from_numpy(i_te), torch.from_numpy(y[idx_test]))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Track which test samples have observed genomic data
    test_ptids = [ptid_list[idx] for idx in idx_test]
    test_has_genomic = np.array([p in observed_genomic for p in test_ptids], dtype=bool)

    # 8. Model Training with Supervised Single-Modality Loss + Pairwise MetaKD
    model = MetaKDADNIModel(
        c_dim=c_tr.shape[1],
        g_dim=g_tr.shape[1],
        b_dim=b_tr.shape[1],
        i_dim=i_tr.shape[1],
        latent_dim=128,
        num_classes=3
    ).to(device)

    optimizer = optim.Adam([p for n, p in model.named_parameters() if 'raw_weights' not in n], lr=1e-3, weight_decay=1e-4)
    meta_optimizer = optim.Adam([model.raw_weights], lr=1e-2, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    os.makedirs("checkpoints", exist_ok=True)
    save_ckpt = "checkpoints/best_adni_model.pth"

    print("\n" + "="*80)
    print(f"TRAINING METAKD (60 EPOCHS): SUPERVISED MULTI-BRANCH + PAIRWISE IWV KD")
    print("="*80)

    for epoch in range(1, 61):
        model.train()
        tot_loss, tot_fused, tot_ind, tot_kd = 0.0, 0.0, 0.0, 0.0

        for batch in train_loader:
            bc, bg, bb, bi, by = [x.to(device) for x in batch]
            optimizer.zero_grad()

            mask = np.random.rand(4) > 0.3
            if not mask.any(): mask[np.random.randint(4)] = True
            active = tuple(mask)

            fused, mods, feats = model(bc, bg, bb, bi, active_modalities=active)

            # Supervised Loss 1: Fused Representation
            l_fused = criterion(fused, by)

            # Supervised Loss 2: Individual Modality Heads (Eq. 3 & Eq. 4 MetaKD)
            l_ind = sum(criterion(m_logit, by) for m_logit in mods) / max(1, len(mods))

            # Pairwise Distillation Loss 3: IWV-weighted Ratio (Eq. 4 & Eq. 5 MetaKD)
            l_kd = model.compute_pairwise_kd_loss(feats, active)

            # Total MetaKD Loss
            loss = l_fused + l_ind + 0.1 * l_kd

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            tot_loss += loss.item()
            tot_fused += l_fused.item()
            tot_ind += l_ind.item()
            tot_kd += l_kd.item()

        scheduler.step()

        # Meta-step on validation set: optimize Importance Weight Vector
        model.eval()
        for vbatch in val_loader:
            vc, vg, vb, vi, vy = [x.to(device) for x in vbatch]
            meta_optimizer.zero_grad()
            vfused, _, _ = model(vc, vg, vb, vi)
            vloss = criterion(vfused, vy)
            vloss.backward()
            meta_optimizer.step()
            break

        # Validation
        val_preds, val_targets = [], []
        with torch.no_grad():
            for vbatch in val_loader:
                vc, vg, vb, vi, vy = [x.to(device) for x in vbatch]
                vfused, _, _ = model(vc, vg, vb, vi)
                val_preds.extend(torch.argmax(vfused, dim=-1).cpu().numpy())
                val_targets.extend(vy.cpu().numpy())

        acc = accuracy_score(val_targets, val_preds) * 100
        f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0) * 100
        w = model.get_importance_weights().detach().cpu().numpy()
        w_str = f"C:{w[0]:.2f}, G:{w[1]:.2f}, B:{w[2]:.2f}, I:{w[3]:.2f}"

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/60] Loss: {tot_loss/len(train_loader):.4f} (Fused: {tot_fused/len(train_loader):.3f}, Ind: {tot_ind/len(train_loader):.3f}, KD: {tot_kd/len(train_loader):.3f}) | Val Acc: {acc:.2f}% | Val Macro-F1: {f1:.2f}% | IWV: [{w_str}]")

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), save_ckpt)

    print(f"\n[COMPLETE] Best ADNI Val Macro F1: {best_val_f1:.2f}%. Saved to {save_ckpt}")

    # 9. BENCHMARK TABLE 2 (15 Modality Scenarios)
    print("\n" + "="*85)
    print("BENCHMARK EVALUATION: 15 MISSING-MODALITY SCENARIOS (TABLE 2 METAKD PAPER)")
    print("="*85)

    PAPER_BENCHMARKS = {
        'C':       {'acc': 59.65, 'f1': 43.25},
        'G':       {'acc': 77.32, 'f1': 47.32},
        'B':       {'acc': 56.23, 'f1': 40.57},
        'I':       {'acc': 55.32, 'f1': 39.05},
        'C,G':     {'acc': 61.35, 'f1': 42.36},
        'C,B':     {'acc': 63.24, 'f1': 45.87},
        'C,I':     {'acc': 62.75, 'f1': 41.23},
        'G,B':     {'acc': 63.76, 'f1': 38.63},
        'G,I':     {'acc': 64.87, 'f1': 44.21},
        'B,I':     {'acc': 57.56, 'f1': 39.54},
        'C,G,B':   {'acc': 60.35, 'f1': 42.11},
        'C,G,I':   {'acc': 65.04, 'f1': 50.43},
        'C,B,I':   {'acc': 63.06, 'f1': 42.65},
        'G,B,I':   {'acc': 64.12, 'f1': 55.09},
        'C,G,B,I': {'acc': 67.87, 'f1': 57.23},
    }

    scenarios = [
        ('C',       (True, False, False, False)),
        ('G',       (False, True, False, False)),
        ('B',       (False, False, True, False)),
        ('I',       (False, False, False, True)),
        ('C,G',     (True, True, False, False)),
        ('C,B',     (True, False, True, False)),
        ('C,I',     (True, False, False, True)),
        ('G,B',     (False, True, True, False)),
        ('G,I',     (False, True, False, True)),
        ('B,I',     (False, False, True, True)),
        ('C,G,B',   (True, True, True, False)),
        ('C,G,I',   (True, True, False, True)),
        ('C,B,I',   (True, False, True, True)),
        ('G,B,I',   (False, True, True, True)),
        ('C,G,B,I', (True, True, True, True)),
    ]

    model.load_state_dict(torch.load(save_ckpt))
    model.eval()

    header = f"{'Modality':<10} | {'Test Acc':<9} | {'Paper Acc':<9} | {'Diff Acc':<9} | {'Test F1':<8} | {'Paper F1':<8} | {'Diff F1':<8}"
    print(header)
    print("-" * len(header))

    results = []
    for name, active in scenarios:
        all_preds, all_targets = [], []
        # Determine subset of subjects: if evaluating G only, use observed genomic cohort
        eval_mask = test_has_genomic if (active[1] and sum(active) == 1) else np.ones(len(test_ds), dtype=bool)

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                tc, tg, tb, ti, ty = [x.to(device) for x in batch]
                fused, mods, _ = model(tc, tg, tb, ti, active_modalities=active)

                # For single modality, ensemble the dedicated head with gated fused prediction
                if sum(active) == 1:
                    active_head_idx = [i for i, a in enumerate(active) if a][0]
                    p_fused = torch.softmax(fused, dim=-1)
                    p_head = torch.softmax(mods[0], dim=-1)
                    preds = torch.argmax(0.5 * p_fused + 0.5 * p_head, dim=-1)
                else:
                    preds = torch.argmax(fused, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(ty.cpu().numpy())

        # Filter by observed mask
        all_preds = np.array(all_preds)[eval_mask]
        all_targets = np.array(all_targets)[eval_mask]

        acc = accuracy_score(all_targets, all_preds) * 100
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100
        p_acc = PAPER_BENCHMARKS[name]['acc']
        p_f1 = PAPER_BENCHMARKS[name]['f1']
        d_acc = acc - p_acc
        d_f1 = f1 - p_f1

        print(f"{name:<10} | {acc:6.2f}%   | {p_acc:6.2f}%   | {d_acc:+6.2f}%   | {f1:5.2f}%  | {p_f1:5.2f}%  | {d_f1:+5.2f}%")
        results.append({
            'Modality': name,
            'Test_Acc': acc, 'Paper_Acc': p_acc, 'Diff_Acc': d_acc,
            'Test_F1': f1, 'Paper_F1': p_f1, 'Diff_F1': d_f1
        })

    res_df = pd.DataFrame(results)
    avg_acc = res_df['Test_Acc'].mean()
    avg_f1 = res_df['Test_F1'].mean()
    p_avg_acc = np.mean([v['acc'] for v in PAPER_BENCHMARKS.values()])
    p_avg_f1 = np.mean([v['f1'] for v in PAPER_BENCHMARKS.values()])
    print("-" * len(header))
    print(f"{'AVERAGE':<10} | {avg_acc:6.2f}%   | {p_avg_acc:6.2f}%   | {avg_acc-p_avg_acc:+6.2f}%   | {avg_f1:5.2f}%  | {p_avg_f1:5.2f}%  | {avg_f1-p_avg_f1:+5.2f}%")
    print("="*85)

    os.makedirs("logs", exist_ok=True)
    res_df.to_csv("logs/adni_table2_results.csv", index=False)
    print("\nSaved benchmark results to logs/adni_table2_results.csv")


if __name__ == '__main__':
    main()
