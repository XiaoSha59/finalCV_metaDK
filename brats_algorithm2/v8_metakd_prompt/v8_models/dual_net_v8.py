import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BRATS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "brats"))
for p in [PARENT_DIR, BRATS_DIR, os.path.join(BRATS_DIR, "process")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from models.dual_net import DualNet, KD_Weights, U_Res3D_enc, U_Res3D_dec
except ImportError:
    from process.models.dual_net import DualNet, KD_Weights, U_Res3D_enc, U_Res3D_dec


class DualNet_V8(nn.Module):
    """
    MetaKD-Prompt Architecture: Modality-Aware Learnable Prompt Imputation
    for Missing Modality Medical Segmentation (PromptKD).
    
    Instead of passive mean feature imputation (Eq. 2 in MetaKD), DualNet_V8
    learns modality-specific prompt tokens P_m in R^(256 x 1 x 1 x 1) for each MRI modality
    (FLAIR, T1, T1ce, T2). When a modality is missing, the latent feature is actively
    restored via: F_m = F_mean + alpha * P_m.
    """
    def __init__(self, args=None, norm_cfg='IN', activation_cfg='LeakyReLU', num_classes=3,
                 weight_std=True, self_att=False, cross_att=False, prompt_dim=256):
        super().__init__()
        self.shared_enc = U_Res3D_enc(norm_cfg, activation_cfg, num_classes, weight_std)
        self.shared_dec = U_Res3D_dec(norm_cfg, activation_cfg, num_classes, weight_std)

        self.self_att = self_att
        self.cross_att = cross_att
        embed_dim = 125
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=5).cuda()
        self.kd_loss = nn.L1Loss(reduction='none')
        self.kd_weights = KD_Weights(shape=4)

        # Modality Prompt Dictionary: 4 modalities (0: FLAIR, 1: T1, 2: T1ce, 3: T2)
        self.modality_prompts = nn.Parameter(torch.randn(4, prompt_dim, 1, 1, 1) * 0.02)
        self.prompt_alpha = nn.Parameter(torch.tensor([0.2]))

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

        # 1. Base Mean Imputation from Available Modalities
        fill_fts = torch.zeros_like(shared_ft[0:N * C + 1:C], device=images.device)
        for m in mode_split:
            fill_fts += shared_ft[m:N * C + 1:C]
        fill_fts /= len(mode_split)

        # 2. Modality-Aware Prompt Residual Imputation (PromptKD)
        for m in [0, 1, 2, 3]:
            if m not in mode_split:
                prompt_m = self.modality_prompts[m]
                fused_ft[m:N * C + 1:C] = fill_fts + self.prompt_alpha * prompt_m

        totKDLoss = torch.tensor(0.0, device=images.device)
        if val:
            mode_w = self.kd_weights.get_kdWeights()
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
            for i in [0, 1, 2, 3]:
                if i not in mode_split:
                    continue
                for j in [0, 1, 2, 3]:
                    if i == j or j not in mode_split:
                        continue
                    kd_w = self.kd_weights(i, j)
                    totKDLoss = totKDLoss + kd_w.detach() * self.kd_loss(
                        shared_ft[i:N * C + 1:C].detach(),
                        shared_ft[j:N * C + 1:C]
                    ).mean(dim=(1, 2, 3, 4))

        logits = self.shared_dec(out_ft, _shared_layers)
        return logits[0], mode_split, totKDLoss.mean()


__all__ = ["DualNet_V8", "KD_Weights", "U_Res3D_enc", "U_Res3D_dec"]
