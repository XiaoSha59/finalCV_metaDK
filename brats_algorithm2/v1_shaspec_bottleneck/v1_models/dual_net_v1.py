import sys
import os

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BRATS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'brats'))
for p in [PARENT_DIR, BRATS_DIR, os.path.join(BRATS_DIR, 'process')]:
    if p not in sys.path:
        sys.path.insert(0, p)
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BRATS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "brats"))
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
        # Initialized with uniform random weights, requires gradient for meta-update
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
        return norm_w[mod_i] / (norm_w[mod_j] + 1e-7)


class ShaSpecHead(nn.Module):
    """
    Shared-Specific Feature Disentanglement Head (inspired by ShaSpec, CVPR 2023).
    Projects incoming feature map (in_channels=256) into:
      1. Shared representation r (in_channels // 2 = 128)
      2. Specific representation s (in_channels // 2 = 128)
    """
    def __init__(self, in_channels=256):
        super(ShaSpecHead, self).__init__()
        mid_channels = in_channels // 2
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
        # Reconstruct back to original channel dimension for seamless decoder compatibility
        self.recon_proj = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        r = self.shared_proj(x)
        s = self.specific_proj(x)
        return r, s

    def reconstruct(self, r, s):
        concat = torch.cat([r, s], dim=1)
        return self.recon_proj(concat)


class DualNet_Algorithm2(nn.Module):
    """
    DualNet-v2 Architecture implementing Algorithm 2:
      1. Region-aware importance weights (w_ET, w_TC, w_WT) from RFNet
      2. Shared-Specific feature decomposition from ShaSpec
      3. Selective zero-imputation for missing modality specific features from mmFormer
      4. Safe Region-Masked Knowledge Distillation loss (Zero-NaN safe)
    """
    def __init__(self, norm_cfg='IN', activation_cfg='LeakyReLU', num_classes=3,
                 weight_std=True, use_shaspec=True, impute_mode='mean'):
        super(DualNet_Algorithm2, self).__init__()
        self.use_shaspec = use_shaspec
        self.num_classes = num_classes
        self.impute_mode = impute_mode

        # Reusable standard backbones from original codebase
        self.shared_enc = U_Res3D_enc(norm_cfg, activation_cfg, num_classes, weight_std)
        self.shared_dec = U_Res3D_dec(norm_cfg, activation_cfg, num_classes, weight_std)

        # Region-Aware Modality Importance Weights in R^(3 x 4)
        self.region_weights = KD_Weights_RegionAware(num_regions=3, num_modalities=4)

        # ShaSpec Disentanglement Head (Channels: 256 -> 128 shared, 128 specific)
        if self.use_shaspec:
            self.shaspec = ShaSpecHead(in_channels=256)

        self.kd_loss = nn.L1Loss(reduction='none')

    def forward(self, images, labels=None, val=False, mode='0,1,2,3', impute_mode=None):
        """
        Forward pass with safe Region-Masked KD and configurable Imputation.
        impute_mode: 'mean' (Shared-Mean Imputation) or 'zero' (Zero-Imputation)
        """
        if impute_mode is None:
            impute_mode = self.impute_mode

        N, C, D, W, H = images.shape

        if mode == 'random':
            all_modes = [0, 1, 2, 3]
            mode_num = random.randint(1, 4)
            mode_split = random.sample(all_modes, mode_num)
        else:
            mode_split = list(map(int, mode.split(',')))

        # Mask missing modalities at input level
        _images = images.clone()
        for m in [0, 1, 2, 3]:
            if m not in mode_split:
                _images[:, m, :, :, :] = 0.0

        # Encode each modality through the shared encoder
        x = _images.view(N * C, 1, D, W, H)
        shared_ft, shared_gft = self.shared_enc(x)
        shared_layers = self.shared_enc.backbone.get_layers()
        _shared_layers = [
            i_layer.view(N, C * i_layer.shape[1], i_layer.shape[2], i_layer.shape[3], i_layer.shape[4])
            for i_layer in shared_layers
        ]

        # Extract features per modality stream: each is (N, 256, d, w, h)
        ft_streams = [shared_ft[m:N * C + 1:C] for m in range(4)]
        fused_ft = shared_ft.clone()

        loss_ortho = torch.tensor(0.0, device=images.device)
        loss_shared = torch.tensor(0.0, device=images.device)

        if self.use_shaspec:
            # 1. Decompose available modalities into shared (r) and specific (s)
            r_dict = {}
            s_dict = {}
            for m in mode_split:
                r_m, s_m = self.shaspec(ft_streams[m])
                r_dict[m] = r_m
                s_dict[m] = s_m

            # 2. Compute aggregate representations from available modalities
            if len(mode_split) > 0:
                r_mean = sum(r_dict[m] for m in mode_split) / len(mode_split)
                s_mean = sum(s_dict[m] for m in mode_split) / len(mode_split)
            else:
                r_mean = torch.zeros((N, 128, ft_streams[0].shape[2], ft_streams[0].shape[3], ft_streams[0].shape[4]),
                                     device=images.device, dtype=ft_streams[0].dtype)
                s_mean = torch.zeros_like(r_mean)

            # Compute ShaSpec losses on available modalities (during training)
            if not val and len(mode_split) > 0:
                # 1. Orthogonal Loss: cos(r_m, s_m) -> 0
                for m in mode_split:
                    r_flat = r_dict[m].reshape(N, -1)
                    s_flat = s_dict[m].reshape(N, -1)
                    dot = torch.sum(r_flat * s_flat, dim=1)
                    norm_r = torch.norm(r_flat, p=2, dim=1) + 1e-7
                    norm_s = torch.norm(s_flat, p=2, dim=1) + 1e-7
                    cos_sim = torch.clamp(dot / (norm_r * norm_s), -1.0, 1.0)
                    loss_ortho += torch.abs(cos_sim).mean()
                loss_ortho = loss_ortho / len(mode_split)

                # 2. Shared Consistency Loss: pairwise distance between available r_i and r_j
                if len(mode_split) > 1:
                    pair_count = 0
                    for i_idx, i in enumerate(mode_split):
                        for j in mode_split[i_idx + 1:]:
                            loss_shared += F.mse_loss(r_dict[i], r_dict[j])
                            pair_count += 1
                    if pair_count > 0:
                        loss_shared = loss_shared / pair_count

            # 3. Reconstruct fused features per modality stream:
            # - For available modalities: r = r_mean, s = s_dict[m]
            # - For missing modalities:
            #   * If impute_mode == 'mean': r = r_mean, s = s_mean (Shared-Mean Imputation)
            #   * If impute_mode == 'zero': r = r_mean, s = 0 (Zero-Imputation)
            reconstructed_streams = []
            for m in range(4):
                if m in mode_split:
                    s_cur = s_dict[m]
                else:
                    if impute_mode == 'mean':
                        s_cur = s_mean
                    else:
                        s_cur = torch.zeros_like(s_mean)
                f_rec = self.shaspec.reconstruct(r_mean, s_cur)
                reconstructed_streams.append(f_rec)

            fused_5d = torch.stack(reconstructed_streams, dim=1)  # (N, 4, 256, d, w, h)
        else:
            # Baseline feature handling
            if impute_mode == 'mean' and len(mode_split) > 0:
                fill_fts = sum(ft_streams[m] for m in mode_split) / len(mode_split)
            else:
                fill_fts = torch.zeros_like(ft_streams[0])

            stream_list = []
            for m in range(4):
                if m in mode_split:
                    stream_list.append(ft_streams[m])
                else:
                    stream_list.append(fill_fts)
            fused_5d = torch.stack(stream_list, dim=1)  # (N, 4, 256, d, w, h)

        # Region-weighted modulation during validation (Outer-loop meta guidance)
        if val:
            norm_w = self.region_weights.get_normalized_weights()  # shape (3, 4)
            w_mean = norm_w.mean(dim=0).view(1, 4, 1, 1, 1, 1)
            fused_5d = fused_5d * w_mean

        out_ft = fused_5d.view(N, 4 * fused_5d.shape[2], fused_5d.shape[3], fused_5d.shape[4], fused_5d.shape[5])

        # Region-Masked Knowledge Distillation (Inner loop training)
        totKDLoss = torch.tensor(0.0, device=images.device)
        if not val and len(mode_split) > 1:
            norm_w = self.region_weights.get_normalized_weights()  # shape (3, 4)

            # Prepare spatial masks if labels are provided
            # labels: [N, 3, D, H, W] -> 0: ET, 1: WT, 2: TC
            has_masks = labels is not None
            if has_masks:
                ft_size = ft_streams[0].shape[2:]  # (d, w, h)
                mask_ET = F.interpolate(labels[:, 0:1].float(), size=ft_size, mode='nearest')
                mask_WT = F.interpolate(labels[:, 1:2].float(), size=ft_size, mode='nearest')
                mask_TC = F.interpolate(labels[:, 2:3].float(), size=ft_size, mode='nearest')
                region_masks = [mask_ET, mask_TC, mask_WT]  # Index 0: ET, 1: TC, 2: WT
            else:
                region_masks = [None, None, None]

            # Compute masked KD per sub-region
            pair_count = 0
            for r_idx in range(3):
                w_r = norm_w[r_idx]
                r_mask = region_masks[r_idx]

                for i in mode_split:  # Teacher modality
                    for j in mode_split:  # Student modality
                        if i == j:
                            continue
                        ratio_w = w_r[i] / (w_r[j] + 1e-7)

                        if self.use_shaspec:
                            t_feat = r_dict[i].detach()
                            s_feat = r_dict[j]
                        else:
                            t_feat = ft_streams[i].detach()
                            s_feat = ft_streams[j]

                        diff = torch.abs(t_feat - s_feat)

                        if r_mask is not None:
                            mask_sum = r_mask.sum()
                            if mask_sum > 0:
                                # Properly normalized over spatial voxels AND feature channels
                                loss_pair = (diff * r_mask).sum() / (mask_sum * diff.shape[1] + 1e-7)
                            else:
                                loss_pair = torch.tensor(0.0, device=images.device)
                        else:
                            loss_pair = diff.mean()

                        totKDLoss = totKDLoss + ratio_w.detach() * loss_pair
                        pair_count += 1

            if pair_count > 0:
                totKDLoss = totKDLoss / pair_count

        logits = self.shared_dec(out_ft, _shared_layers)
        return logits[0], mode_split, totKDLoss, loss_ortho, loss_shared
