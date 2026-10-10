"""
MetaKD Architecture for ADNI Multi-Modal Multi-Class Classification.
Implements:
- 4 MLP Modality Encoders: Clinical (C), Genomic (G), Biospecimen (B), Imaging (I).
- Feature Imputation for Missing Modalities (Eq. 2 in MetaKD Paper).
- Modality Importance Weight Vector (IWV) with Softmax Normalization (Eq. 6).
- Pairwise Modality-Weighted Knowledge Distillation with L1 Loss (Eq. 4, 5).
- 3-Class AD Progression Classifier: CTL (0), MCI (1), AD (2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularEncoder(nn.Module):
    """Encodes 1D tabular modality features into a shared 64-dimensional latent embedding."""
    def __init__(self, in_features, hidden_dim=128, out_features=64, dropout_rate=0.2):
        super(TabularEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.net(x)


class KD_Weights(nn.Module):
    """
    Modality Importance Weight Vector (IWV) module.
    4 weights representing [w_C, w_G, w_B, w_I], normalized via Softmax (Eq. 6).
    """
    def __init__(self, num_modalities=4):
        super(KD_Weights, self).__init__()
        self.weights = nn.Parameter(torch.rand(num_modalities))

    def get_weights(self):
        return F.softmax(self.weights, dim=0)

    def get_ratio(self, i, j):
        w = self.get_weights()
        return w[i] / (w[j] + 1e-7)


class MetaKD_ADNI(nn.Module):
    """
    MetaKD Multi-Modal Classifier for ADNI.
    """
    def __init__(self, dims=[62, 64, 9, 325], embed_dim=64, num_classes=3, alpha=0.1):
        super(MetaKD_ADNI, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.alpha = alpha  # KD loss trade-off factor

        # 4 Modality Encoders: [C, G, B, I]
        self.encoder_c = TabularEncoder(dims[0], hidden_dim=128, out_features=embed_dim)
        self.encoder_g = TabularEncoder(dims[1], hidden_dim=128, out_features=embed_dim)
        self.encoder_b = TabularEncoder(dims[2], hidden_dim=64, out_features=embed_dim)
        self.encoder_i = TabularEncoder(dims[3], hidden_dim=256, out_features=embed_dim)

        self.encoders = nn.ModuleList([self.encoder_c, self.encoder_g, self.encoder_b, self.encoder_i])

        # Modality Importance Weight Vector
        self.kd_weights = KD_Weights(num_modalities=4)

        # Classifier Head (2 FC layers with Dropout, Eq. 3)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        self.l1_loss = nn.L1Loss(reduction='none')

    def extract_features(self, x_list):
        """Extract embeddings from available modalities."""
        return [enc(x) for enc, x in zip(self.encoders, x_list)]

    def impute_missing_features(self, feats, masks):
        """
        Missing Modality Feature Generation: Mean imputation from available modalities (Eq. 2).
        Vectorized and non-inplace for stable autograd.
        feats: list of 4 tensors, each (B, embed_dim)
        masks: tensor of shape (B, 4) with bool values indicating availability
        """
        # stacked: shape (4, B, embed_dim)
        stacked = torch.stack(feats, dim=0)
        # mask_weights: shape (4, B, 1)
        mask_weights = masks.t().unsqueeze(-1).float()
        avail_counts = mask_weights.sum(dim=0).clamp(min=1.0)  # (B, 1)
        mean_feats = (stacked * mask_weights).sum(dim=0) / avail_counts  # (B, embed_dim)

        imputed_feats = []
        for i in range(4):
            m_i = masks[:, i].unsqueeze(-1)  # (B, 1) bool
            imp_f = torch.where(m_i, feats[i], mean_feats)
            imputed_feats.append(imp_f)

        return imputed_feats

    def compute_kd_loss(self, feats, masks):
        """
        Pairwise Modality-Weighted Knowledge Distillation Loss (Eq. 4, 5).
        L_kd = (w_i / w_j) * ||f_i - f_j||_1
        """
        B = feats[0].size(0)
        kd_loss = torch.tensor(0.0, device=feats[0].device)
        pair_count = 0

        for i in range(4):  # Teacher modality
            for j in range(4):  # Student modality
                if i == j:
                    continue
                # Both modalities must be present in ground truth for valid distillation
                valid_pair_mask = masks[:, i] & masks[:, j]
                if not valid_pair_mask.any():
                    continue

                diff = self.l1_loss(feats[i].detach(), feats[j])
                sample_loss = diff.mean(dim=-1)  # (B,)
                sample_loss = sample_loss[valid_pair_mask].mean()

                ratio = self.kd_weights.get_ratio(i, j)
                kd_loss = kd_loss + ratio.detach() * sample_loss
                pair_count += 1

        if pair_count > 0:
            kd_loss = kd_loss / pair_count
        return kd_loss

    def forward(self, x_list, masks, val=False, mode=None):
        """
        x_list: list of [x_c, x_g, x_b, x_i]
        masks: tensor of shape (B, 4) indicating presence of each modality
        mode: 'train', 'meta_val', or 'eval'. If not set, derives from val.
        """
        if mode is None:
            mode = "eval" if val else "train"

        # 1. Extract features
        feats = self.extract_features(x_list)

        # 2. Compute KD loss on available teacher-student pairs (before imputation)
        kd_loss = self.compute_kd_loss(feats, masks) if mode == "train" else torch.tensor(0.0, device=feats[0].device)

        # 3. Impute missing features (Eq. 2)
        feats = self.impute_missing_features(feats, masks)

        # 4. Feature fusion (Eq. 3)
        if mode == "meta_val":
            # Outer loop meta-objective to optimize IWV weights
            # Scale by num_modalities (4.0) so mean scale remains 1.0
            w = self.kd_weights.get_weights()
            fused_feats = torch.cat([4.0 * w[i] * feats[i] for i in range(4)], dim=-1)
        else:
            # Consistent unattenuated fusion for inner-loop training and test evaluation
            fused_feats = torch.cat(feats, dim=-1)

        # 5. Classifier prediction (Eq. 3)
        logits = self.classifier(fused_feats)
        return logits, kd_loss
