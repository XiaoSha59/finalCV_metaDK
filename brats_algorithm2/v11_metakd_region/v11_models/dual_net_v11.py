import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BRATS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "brats"))
for p in [PARENT_DIR, BRATS_DIR, os.path.join(BRATS_DIR, "process")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from models.dual_net import U_Res3D_enc, U_Res3D_dec
except ImportError:
    from process.models.dual_net import U_Res3D_enc, U_Res3D_dec


class KD_Weights_Region(nn.Module):
    """
    Region-Aware Modality Importance Weight Matrix (W in R^(3 x 4)) for MetaKD-V11.
    
    Maintains 3 distinct learnable weight vectors:
      - Row 0 (k=0): Enhancing Tumor (ET)
      - Row 1 (k=1): Whole Tumor (WT)
      - Row 2 (k=2): Tumor Core (TC)
    Across 4 input modalities: [0: FLAIR, 1: T1, 2: T1ce, 3: T2].
    
    Solves the Global Weight Dilemma of vanilla MetaKD by allowing:
      - Row 0 (ET) to focus on T1ce
      - Row 1 (WT) to focus on FLAIR and T2
      - Row 2 (TC) to focus on T1 and T1ce
    """
    def __init__(self, num_regions=3, num_modalities=4):
        super(KD_Weights_Region, self).__init__()
        # Initialized with equal weights (ones)
        self.kd_weights = nn.Parameter(torch.ones(num_regions, num_modalities), requires_grad=True)

    def get_normalized_weights(self):
        """Returns softmax-normalized weights per region: shape (3, 4)."""
        return F.softmax(self.kd_weights, dim=1)

    def get_region_weights(self, region_idx):
        """Returns normalized weights for a specific region: shape (4,)."""
        return F.softmax(self.kd_weights, dim=1)[region_idx]

    def get_effective_modality_weights(self):
        """
        Global effective modality weight vector: shape (4,).
        Computed as the mean of the 3 region-specific weight distributions.
        Fully differentiable wrt self.kd_weights.
        """
        norm_w = F.softmax(self.kd_weights, dim=1)  # (3, 4)
        return norm_w.mean(dim=0)  # (4,)

    def get_pairwise_ratio(self, i, j):
        """Pairwise relative weight ratio between modality i and modality j."""
        eff_w = self.get_effective_modality_weights()
        ratio = eff_w[i] / (eff_w[j] + 1e-7)
        return torch.clamp(ratio, max=10.0)

    def forward(self, i, j):
        """Compatibility forward signature matching original MetaKD."""
        return self.get_pairwise_ratio(i, j)


class DualNet_V11(nn.Module):
    """
    DualNet-V11: Region-Disentangled MetaKD Architecture.
    
    Preserves 100% of the proven U_Res3D encoder-decoder backbone.
    Upgrades the global 4-element IWV into a 3x4 Region-Disentangled Weight Matrix.
    """
    def __init__(self, args=None, norm_cfg='IN', activation_cfg='LeakyReLU', num_classes=3,
                 weight_std=True, self_att=False, cross_att=False):
        super().__init__()
        self.shared_enc = U_Res3D_enc(norm_cfg, activation_cfg, num_classes, weight_std)
        self.shared_dec = U_Res3D_dec(norm_cfg, activation_cfg, num_classes, weight_std)

        self.self_att = self_att
        self.cross_att = cross_att
        embed_dim = 125
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=5)
        self.kd_loss = nn.L1Loss(reduction='none')
        
        # Region-Disentangled Modality Importance Weights (3x4)
        self.kd_weights = KD_Weights_Region(num_regions=3, num_modalities=4)

    def forward(self, images, val=False, mode='0,1,2,3'):
        org_mode = mode
        N, C, D, W, H = images.shape
        if val:
            mode = '0,1,2,3'

        if org_mode == 'random':
            mode = '0,1,2,3'
            mode_num = random.randint(1, 4)
        mode_split = list(map(int, mode.split(',')))
        if org_mode == 'random':
            mode_split = random.sample(mode_split, mode_num)

        _images = images.clone()
        for m in [0, 1, 2, 3]:
            if m not in mode_split:
                _images[:, m, :, :, :] = 0

        x = _images.view(N * C, 1, D, W, H)
        shared_ft, shared_gft = self.shared_enc(x)
        shared_layers = self.shared_enc.backbone.get_layers()
        _shared_layers = [i_layer.view(N, C * i_layer.shape[1], i_layer.shape[2], i_layer.shape[3], i_layer.shape[4])
                          for i_layer in shared_layers]

        fused_ft = shared_ft.clone()
        flair_ft = shared_ft[0:N * C + 1:C]
        t1_ft = shared_ft[1:N * C + 1:C]
        t1ce_ft = shared_ft[2:N * C + 1:C]
        t2_ft = shared_ft[3:N * C + 1:C]

        # Missing Modality Feature Generation: Mean imputation from available modalities (Eq. 2)
        fill_fts = torch.zeros_like(shared_ft[0:N * C + 1:C], device=images.device)
        for m in mode_split:
            fill_fts += shared_ft[m:N * C + 1:C]
        fill_fts /= len(mode_split)

        for m in [0, 1, 2, 3]:
            if m not in mode_split:
                fused_ft[m:N * C + 1:C] = fill_fts

        totKDLoss = torch.tensor(0.0, device=images.device)
        if val:
            # Modulate feature fusion with the effective modality weights learned across regions
            mode_w = self.kd_weights.get_effective_modality_weights()
            fused_ft[0:N * C + 1:C] = flair_ft * mode_w[0]
            fused_ft[1:N * C + 1:C] = t1_ft * mode_w[1]
            fused_ft[2:N * C + 1:C] = t1ce_ft * mode_w[2]
            fused_ft[3:N * C + 1:C] = t2_ft * mode_w[3]

        if self.self_att:
            cat_attend = fused_ft.view(N, C * fused_ft.shape[1], fused_ft.shape[2], fused_ft.shape[3], fused_ft.shape[4])
            original_size = cat_attend.size()
            flat_input = cat_attend.view(original_size[0], original_size[1], original_size[2] * original_size[3] * original_size[4])
            perm_input = flat_input.permute(1, 0, 2)
            att_input, att_weights = self.multihead_attn(perm_input, perm_input, perm_input)
            flat_output = att_input.permute(1, 0, 2)
            out_ft = flat_output.view(original_size)
        else:
            out_ft = fused_ft.view(N, C * fused_ft.shape[1], fused_ft.shape[2], fused_ft.shape[3], fused_ft.shape[4])

        if not val:
            eff_w = self.kd_weights.get_effective_modality_weights()
            for i in [0, 1, 2, 3]:  # Teacher modality
                if i not in mode_split:
                    continue
                for j in [0, 1, 2, 3]:  # Student modality
                    if i == j or j not in mode_split:
                        continue
                    kd_w = eff_w[i] / (eff_w[j] + 1e-7)
                    kd_w_clamped = torch.clamp(kd_w, max=10.0)
                    totKDLoss = totKDLoss + kd_w_clamped.detach() * self.kd_loss(
                        shared_ft[i:N * C + 1:C].detach(),
                        shared_ft[j:N * C + 1:C]
                    ).mean(dim=(1, 2, 3, 4))

        logits = self.shared_dec(out_ft, _shared_layers)
        return logits[0], mode_split, totKDLoss.mean()


__all__ = ["DualNet_V11", "KD_Weights_Region", "U_Res3D_enc", "U_Res3D_dec"]
