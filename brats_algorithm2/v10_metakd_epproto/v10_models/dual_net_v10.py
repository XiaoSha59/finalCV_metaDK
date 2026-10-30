import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
V10_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
ALG2_DIR = os.path.abspath(os.path.join(V10_DIR, ".."))
REPO_DIR = os.path.abspath(os.path.join(ALG2_DIR, ".."))
BRATS_DIR = os.path.abspath(os.path.join(REPO_DIR, "brats"))

for p in [CURRENT_DIR, V10_DIR, ALG2_DIR, REPO_DIR, BRATS_DIR, os.path.join(BRATS_DIR, "process")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from models.dual_net import DualNet, KD_Weights, U_Res3D_enc, U_Res3D_dec
except ImportError:
    from process.models.dual_net import DualNet, KD_Weights, U_Res3D_enc, U_Res3D_dec


class DualNet_V10(DualNet):
    """
    MetaKD-EPProto (V10) Unified Architecture:
    Combines Evidential Uncertainty Distillation (Job 7) with Class Semantic Prototype Alignment (Job 9).
    
    Supports:
    1. Standard segmentation prediction across all 15 incomplete modality combinations.
    2. Cross-modal meta-weighted feature knowledge distillation (Vanilla MetaKD).
    3. Returning bottleneck feature representations (out_ft) for class semantic prototype extraction.
    """
    def __init__(self, args=None, norm_cfg='IN', activation_cfg='LeakyReLU', num_classes=3,
                 weight_std=True, self_att=False, cross_att=False):
        super().__init__(args=args, norm_cfg=norm_cfg, activation_cfg=activation_cfg,
                         num_classes=num_classes, weight_std=weight_std,
                         self_att=self_att, cross_att=cross_att)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.multihead_attn = nn.MultiheadAttention(embed_dim=125, num_heads=5).to(device)

    def forward(self, images, val=False, mode='0,1,2,3', return_feat=False):
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

        # Missing Modality Mean Imputation
        fill_fts = torch.zeros_like(shared_ft[0:N * C + 1:C], device=images.device)
        for m in mode_split:
            fill_fts += shared_ft[m:N * C + 1:C]
        fill_fts /= len(mode_split)

        for m in [0, 1, 2, 3]:
            if m not in mode_split:
                fused_ft[m:N * C + 1:C] = fill_fts

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

        if not val and self.training:
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
        if return_feat:
            return logits[0], mode_split, totKDLoss.mean(), out_ft
        return logits[0], mode_split, totKDLoss.mean()


__all__ = ["DualNet_V10", "KD_Weights", "U_Res3D_enc", "U_Res3D_dec"]
