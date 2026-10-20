import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BRATS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "brats"))
for p in [PARENT_DIR, BRATS_DIR, os.path.join(BRATS_DIR, "process")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from models.dual_net import U_Res3D_enc, U_Res3D_dec
except ImportError:
    from process.models.dual_net import U_Res3D_enc, U_Res3D_dec


class KD_Weights_RegionAware(nn.Module):
    """
    Region-Aware Modality Importance Weight (IWV) Module (inspired by RFNet).
    Maintains 3 distinct weight vectors in R^(3 x 4) for:
      - Row 0: Enhancing Tumor (ET)
      - Row 1: Tumor Core (TC)
      - Row 2: Whole Tumor (WT)
    Across 4 modalities: [0: FLAIR, 1: T1, 2: T1ce, 3: T2].
    """
    def __init__(self, num_regions=3, num_modalities=4):
        super(KD_Weights_RegionAware, self).__init__()
        self.kd_weights = nn.Parameter(torch.rand(num_regions, num_modalities), requires_grad=True)

    def get_normalized_weights(self):
        """Returns softmax-normalized weights per region: shape (3, 4)."""
        return F.softmax(self.kd_weights, dim=1)

    def get_region_weights(self, region_idx):
        """Returns softmax-normalized weight vector for specific region: shape (4,)."""
        norm_w = F.softmax(self.kd_weights, dim=1)
        return norm_w[region_idx]

    def forward(self, region_idx, mod_i, mod_j):
        """Pairwise relative weight ratio for region_idx between modality i and modality j."""
        norm_w = F.softmax(self.kd_weights, dim=1)[region_idx]
        ratio = norm_w[mod_i] / (norm_w[mod_j] + 1e-7)
        return torch.clamp(ratio, max=10.0)


class NonInvasiveShaSpecHead(nn.Module):
    """
    Non-Invasive Auxiliary Shared-Specific Disentanglement Head.
    Operates in parallel to the main forward segmentation path without altering
    or bottlenecking the features entering the Phase 1 pre-trained decoder.
      - shared_proj: Extracts shared invariant representation r (128 channels)
      - specific_proj: Extracts modality-specific representation s (128 channels)
    """
    def __init__(self, in_channels=256, mid_channels=128):
        super(NonInvasiveShaSpecHead, self).__init__()
        self.shared_proj = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(mid_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.specific_proj = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(mid_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        r = self.shared_proj(x)
        s = self.specific_proj(x)
        return r, s


class DualNet_V3(nn.Module):
    """
    DualNet-V3: Non-Invasive ShaSpec + Soft Region-Aware KD
    Key Design Principles:
      1. Main Forward Flow is 100% Non-Invasive:
         Directly feeds pre-trained Phase 1 features with author's robust Mean-Imputation
         into shared_dec. ZERO feature distribution shift, zero bottleneck.
      2. Auxiliary ShaSpec Disentanglement:
         Disentangles Shared (r) vs Specific (s). Enforces r_m _|_ s_m (orthogonality)
         and r_i ~ r_j (shared alignment). KD is performed strictly on Shared features (r),
         preventing the Student from being forced to hallucinate missing specific features.
      3. Soft Region-Aware KD Loss:
         Background voxels receive weight 1.0 (actively teaching Student to avoid false positives),
         while tumor sub-regions receive up to 3.0x gradient emphasis.
      4. Safe Loss Scaling:
         Properly normalized by total spatial weight and feature channels, avoiding loss explosion.
    """
    def __init__(self, norm_cfg='IN', activation_cfg='LeakyReLU', num_classes=3,
                 weight_std=True, use_shaspec=True, tumor_boost=2.0):
        super(DualNet_V3, self).__init__()
        self.num_classes = num_classes
        self.use_shaspec = use_shaspec
        self.tumor_boost = tumor_boost

        self.shared_enc = U_Res3D_enc(norm_cfg, activation_cfg, weight_std)
        self.shared_dec = U_Res3D_dec(norm_cfg, activation_cfg, num_classes=num_classes, weight_std=weight_std)

        # Region-aware importance weights W in R^(3 x 4)
        self.region_weights = KD_Weights_RegionAware(num_regions=3, num_modalities=4)

        # Non-invasive ShaSpec Auxiliary Head (Only active when use_shaspec=True)
        if self.use_shaspec:
            self.shaspec = NonInvasiveShaSpecHead(in_channels=256, mid_channels=128)

    def forward(self, images, labels=None, val=False, mode='0,1,2,3'):
        """
        Args:
            images: Tensor of shape (N, 4, 1, D, H, W)
            labels: Optional Tensor of shape (N, 3, D, H, W) -> [ET, WT, TC]
            val: Boolean flag for validation/inference
            mode: Available modality combination string e.g. '0,1,2,3' or 'random'
        """
        # Support both 5D (N, C, D, H, W) from BraTS dataset and 6D (N, C, 1, D, H, W)
        if images.dim() == 5:
            N, C, d, w, h = images.shape
            Ch = 1
            _images = images.clone().unsqueeze(2)  # (N, C, 1, d, w, h)
        else:
            N, C, Ch, d, w, h = images.shape
            _images = images.clone()

        if mode == 'random':
            k = random.randint(1, 4)
            mode_split = sorted(random.sample(range(4), k))
        elif isinstance(mode, str):
            mode_split = sorted([int(m) for m in mode.split(',') if m.strip()])
        else:
            mode_split = [0, 1, 2, 3]

        # Missing modality zeroing on input image level (as in original MetaKD)
        for m in [0, 1, 2, 3]:
            if m not in mode_split:
                _images[:, m, :, :, :, :] = 0

        images_reshaped = _images.view(N * C, Ch, d, w, h)

        # Shared Encoder Feature Extraction: (4N, 256, d', w', h')
        shared_ft, shared_gft = self.shared_enc(images_reshaped)
        shared_layers = self.shared_enc.backbone.get_layers()
        _shared_layers = [
            i_layer.view(N, C * i_layer.shape[1], i_layer.shape[2], i_layer.shape[3], i_layer.shape[4])
            for i_layer in shared_layers
        ]

        # Extract features per modality stream: each is (N, 256, d', w', h')
        ft_streams = [shared_ft[m:N * C + 1:C] for m in range(4)]

        # ----------------------------------------------------------------------
        # 1. MAIN FORWARD FLOW (100% NON-INVASIVE MEAN-IMPUTATION)
        # Guarantees zero disruption to Phase 1 pre-trained representation energy
        # ----------------------------------------------------------------------
        if len(mode_split) > 0:
            fill_fts = sum(ft_streams[m] for m in mode_split) / len(mode_split)
        else:
            fill_fts = torch.zeros_like(ft_streams[0])

        stream_list = []
        for m in range(4):
            if m in mode_split:
                stream_list.append(ft_streams[m])
            else:
                stream_list.append(fill_fts)
        fused_5d = torch.stack(stream_list, dim=1)  # (N, 4, 256, d', w', h')

        # Outer-loop meta-weight guidance during validation
        if val:
            norm_w = self.region_weights.get_normalized_weights()  # shape (3, 4)
            w_mean = norm_w.mean(dim=0).view(1, 4, 1, 1, 1, 1)
            fused_5d = fused_5d * w_mean

        out_ft = fused_5d.view(N, 4 * fused_5d.shape[2], fused_5d.shape[3], fused_5d.shape[4], fused_5d.shape[5])

        # ----------------------------------------------------------------------
        # 2. AUXILIARY SHASPEC DISENTANGLEMENT (Orthogonal & Shared Alignment)
        # ----------------------------------------------------------------------
        loss_ortho = torch.tensor(0.0, device=images.device)
        loss_shared = torch.tensor(0.0, device=images.device)

        r_dict = {}
        s_dict = {}
        if self.use_shaspec:
            for m in mode_split:
                r_m, s_m = self.shaspec(ft_streams[m])
                r_dict[m] = r_m
                s_dict[m] = s_m

            if not val and len(mode_split) > 0:
                # Orthogonal Loss: cos(r_m, s_m) -> 0
                for m in mode_split:
                    r_flat = r_dict[m].reshape(N, -1)
                    s_flat = s_dict[m].reshape(N, -1)
                    dot = torch.sum(r_flat * s_flat, dim=1)
                    norm_r = torch.norm(r_flat, p=2, dim=1) + 1e-7
                    norm_s = torch.norm(s_flat, p=2, dim=1) + 1e-7
                    cos_sim = torch.clamp(dot / (norm_r * norm_s), -1.0, 1.0)
                    loss_ortho = loss_ortho + torch.abs(cos_sim).mean()
                loss_ortho = loss_ortho / len(mode_split)

                # Shared Consistency Loss: pairwise MSE between available r_i and r_j
                if len(mode_split) > 1:
                    pair_c = 0
                    for i_idx, i in enumerate(mode_split):
                        for j in mode_split[i_idx + 1:]:
                            loss_shared = loss_shared + F.mse_loss(r_dict[i], r_dict[j])
                            pair_c += 1
                    if pair_c > 0:
                        loss_shared = loss_shared / pair_c

        # ----------------------------------------------------------------------
        # 3. SOFT REGION-AWARE KNOWLEDGE DISTILLATION
        # Background gets weight 1.0 (no dropout/loss of normal brain anatomy)
        # Tumor regions get soft boost (+ tumor_boost * mask)
        # ----------------------------------------------------------------------
        totKDLoss = torch.tensor(0.0, device=images.device)
        if not val and len(mode_split) > 1:
            norm_w = self.region_weights.get_normalized_weights()  # shape (3, 4)

            # Prepare soft region masks
            # labels: [N, 3, D, H, W] -> 0: ET, 1: WT, 2: TC
            has_masks = labels is not None
            if has_masks:
                ft_size = ft_streams[0].shape[2:]  # (d', w', h')
                mask_ET = F.interpolate(labels[:, 0:1].float(), size=ft_size, mode='nearest')
                mask_WT = F.interpolate(labels[:, 1:2].float(), size=ft_size, mode='nearest')
                mask_TC = F.interpolate(labels[:, 2:3].float(), size=ft_size, mode='nearest')
                region_masks = [mask_ET, mask_TC, mask_WT]  # 0: ET, 1: TC, 2: WT
            else:
                region_masks = [None, None, None]

            pair_count = 0
            for r_idx in range(3):
                w_r = norm_w[r_idx]
                r_mask = region_masks[r_idx]

                for i in mode_split:      # Teacher modality
                    for j in mode_split:  # Student modality
                        if i == j:
                            continue
                        ratio_w = self.region_weights(r_idx, i, j)

                        # Distill on Shared representation r if ShaSpec is active, else raw feature
                        if self.use_shaspec:
                            t_feat = r_dict[i].detach()
                            s_feat = r_dict[j]
                        else:
                            t_feat = ft_streams[i].detach()
                            s_feat = ft_streams[j]

                        diff = torch.abs(t_feat - s_feat)

                        if r_mask is not None:
                            # Soft Region Weighting: Background=1.0, Tumor=1.0 + tumor_boost
                            soft_weight = 1.0 + self.tumor_boost * r_mask
                            loss_pair = (diff * soft_weight).sum() / (soft_weight.sum() * diff.shape[1] + 1e-7)
                        else:
                            loss_pair = diff.mean()

                        totKDLoss = totKDLoss + ratio_w.detach() * loss_pair
                        pair_count += 1

            if pair_count > 0:
                totKDLoss = totKDLoss / pair_count

        # Main Decoder Segmentation Prediction
        logits = self.shared_dec(out_ft, _shared_layers)
        return logits[0], mode_split, totKDLoss, loss_ortho, loss_shared
