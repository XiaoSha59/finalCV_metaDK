"""MetaKD ADNI model: Flex-MoE PatchEmbeddings + Algorithm 1 / Eq. 2 / Eq. 5.

Each modality still emits one vector f_i (Eq. 1). Missing slots are filled
with the mean of observed features (Eq. 2). Fusion is concat + decoder (Eq. 3).

Do not mean-pool the 16 Flex-MoE tokens. Genomic patches are ~24k SNPs each;
a uniform average of those Linear projections is nearly constant across
subjects and collapses every G-only (and C-only) eval to the majority class.
Short tables (C/B/I) use an MLP encoder. Genomic keeps the 16-token sequence
and a 1-layer Transformer (4 heads, Flex-MoE Table 6) with a CLS token as f_i.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Tables shorter than this use MLP; genomic (~387k SNPs) uses patch+Transformer.
MLP_DIM_LIMIT = 8192


class PatchEmbeddings(nn.Module):
    """UNITES-Lab Flex-MoE tokenizer (models.py)."""

    def __init__(self, feature_size, num_patches, embed_dim):
        super().__init__()
        patch_size = math.ceil(feature_size / num_patches)
        self.pad_size = num_patches * patch_size - feature_size
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        x = F.pad(x, (0, self.pad_size)).view(x.shape[0], self.num_patches, self.patch_size)
        return self.projection(x)


class MLPEncoder(nn.Module):
    """MetaKD f_i for Clinical / Biospecimen / Image."""

    def __init__(self, feature_size, latent_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_size, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class PatchEncoder(nn.Module):
    """Flex-MoE 16-token sequence -> one MetaKD feature vector f_i via CLS."""

    def __init__(self, feature_size, latent_dim=128, num_patches=16, num_heads=4):
        super().__init__()
        self.patch = PatchEmbeddings(feature_size, num_patches, latent_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.pos = nn.Parameter(torch.zeros(1, num_patches + 1, latent_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        tokens = self.patch(x)
        cls = self.cls.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos
        return self.norm(self.transformer(tokens)[:, 0])


def make_encoder(feature_size, latent_dim=128, num_patches=16):
    if feature_size > MLP_DIM_LIMIT:
        return PatchEncoder(feature_size, latent_dim, num_patches)
    return MLPEncoder(feature_size, latent_dim)


class SharedTrunkEncoder(nn.Module):
    """Per-modality input stem -> trunk whose weights are shared by all modalities.

    MetaKD says the encoder parameters theta are shared among modalities but
    not how inputs of different width are handled. This is our own design:
    each modality has its own stem that maps its raw width to a common
    `stem_dim` (Linear for C/B/I, Flex-MoE patches + Transformer for G),
    then the same trunk module (one set of weights) produces f_i.
    """

    def __init__(self, stem, trunk):
        super().__init__()
        self.stem = stem
        self.trunk = trunk  # same module object for every modality -> shared parameters

    def forward(self, x):
        return self.trunk(self.stem(x))


def make_shared_encoders(dims, latent_dim=128, num_patches=16, stem_dim=256):
    trunk = nn.Sequential(
        nn.Linear(stem_dim, stem_dim),
        nn.LayerNorm(stem_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(stem_dim, latent_dim),
        nn.LayerNorm(latent_dim),
    )
    encoders = []
    for dim in dims:
        if dim > MLP_DIM_LIMIT:
            stem = PatchEncoder(dim, stem_dim, num_patches)  # outputs stem_dim, LayerNorm'd
        else:
            stem = nn.Sequential(nn.Linear(dim, stem_dim), nn.LayerNorm(stem_dim), nn.ReLU(inplace=True))
        encoders.append(SharedTrunkEncoder(stem, trunk))
    return encoders


class MetaKDADNIModel(nn.Module):
    def __init__(self, dims, latent_dim=128, num_classes=3, num_patches=16, shared_encoder=False):
        super().__init__()
        self.latent_dim = latent_dim
        if shared_encoder:
            self.encoders = nn.ModuleList(make_shared_encoders(dims, latent_dim, num_patches))
        else:
            self.encoders = nn.ModuleList(
                [make_encoder(dim, latent_dim, num_patches) for dim in dims]
            )
        self.n_mod = len(dims)
        # AudioVision classification decoder: 2 FC + dropout after concat (Eq. 3).
        self.fused_classifier = nn.Sequential(
            nn.Linear(latent_dim * self.n_mod, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(64, num_classes),
        )
        # supp. Sec. 8: "Network parameters and MetaKD weights are randomly initialized"
        self.raw_weights = nn.Parameter(torch.rand(self.n_mod))

    def get_importance_weights(self):
        return torch.softmax(self.raw_weights, dim=0)

    def encode_stack(self, xs, active):
        """Eq. (1)–(2): per-modality f_i, missing slots filled by the mean of available ones."""
        active = active.clone()
        n = active.size(0)
        feats = []
        for m in range(self.n_mod):
            f = xs[m].new_zeros(n, self.latent_dim)
            rows = active[:, m].nonzero(as_tuple=True)[0]
            if rows.numel() > 0:
                f = f.index_copy(0, rows, self.encoders[m](xs[m][rows]))
            feats.append(f)
        stack = torch.stack(feats, dim=1)
        act_f = active.unsqueeze(-1).float()
        mean_active = (stack * act_f).sum(1) / act_f.sum(1).clamp(min=1.0)
        stack = torch.where(act_f.bool(), stack, mean_active.unsqueeze(1))
        return stack, active

    def fuse(self, stack, use_iwv=False):
        if use_iwv:
            stack = stack * self.get_importance_weights().view(1, self.n_mod, 1)
        return self.fused_classifier(stack.reshape(stack.size(0), -1))

    def forward(self, xs, active, use_iwv=False):
        stack, active = self.encode_stack(xs, active)
        logits = self.fuse(stack, use_iwv=use_iwv)
        return logits, stack, active

    def compute_pairwise_kd_loss(self, feats, active, reduction="mean", detach_teacher=False):
        """Eq. 4-5: sum over i,j of (w*_i / w*_j) * delta_i * delta_j * ||f_i - f_j||_1.

        Supp. Sec. 8: "we chose p = 1 to form an L1 loss" -> reduction="mean" (L1 loss);
        reduction="sum" is the literal vector norm of Eq. 5. Gradients reach both f_i
        and f_j (Eq. 5 has no stop-gradient). w* is a constant in the inner problem.
        """
        weights = self.get_importance_weights().detach()
        loss = feats.new_zeros(())
        for i in range(self.n_mod):
            for j in range(self.n_mod):
                if i == j:
                    continue
                rows = active[:, i] & active[:, j]
                if rows.sum() == 0:
                    continue
                teacher = feats[rows, i].detach() if detach_teacher else feats[rows, i]
                diff = torch.abs(teacher - feats[rows, j])
                dist = diff.sum(-1).mean() if reduction == "sum" else diff.mean()
                loss = loss + (weights[i] / weights[j]) * dist
        return loss
