import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """
    SMIL Image Encoder Backbone for 28x28 MNIST Images:
    Conv layers with BatchNorm, ReLU, MaxPool, Dropout + FC layers.
    """
    def __init__(self, in_channels: int = 1, feat_dim: int = 128):
        super(ImageEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)


class AudioEncoder(nn.Module):
    """
    SMIL Sound Encoder Backbone for 20x20 Audio MFCCs:
    Conv layers with BatchNorm, ReLU, MaxPool, Dropout + FC layers.
    """
    def __init__(self, in_channels: int = 1, feat_dim: int = 128):
        super(AudioEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)


class KDWeights2D(nn.Module):
    """
    Learnable MetaKD Modality Importance Weight Vector (IWV) w = [w_v, w_a]^T.
    Normalized via Softmax as specified in Eq. (6) and Table 6.
    """
    def __init__(self):
        super(KDWeights2D, self).__init__()
        # Initialized equally for visual and audio
        self.weights = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return F.softmax(self.weights, dim=0)


class MetaKDAudioVision(nn.Module):
    """
    MetaKD Framework for AudioVision-MNIST Digit Classification (Paper Protocol).
    - ImageEncoder & AudioEncoder
    - Meta-Learned Importance Weight Vector (IWV) with Softmax
    - L1 Cross-modal Distillation Loss
    - Mean Feature Imputation (Eq. 2) for missing modalities
    """
    def __init__(self, num_classes: int = 10, feat_dim: int = 128):
        super(MetaKDAudioVision, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        # Modality Encoders
        self.img_enc = ImageEncoder(in_channels=1, feat_dim=feat_dim)
        self.aud_enc = AudioEncoder(in_channels=1, feat_dim=feat_dim)

        # Learnable MetaKD distillation IWV weights
        self.kd_weights = KDWeights2D()

        # Classification Heads (2 FC layers with dropout as specified in paper)
        self.img_classifier = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        self.aud_classifier = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        self.fusion_classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(
        self,
        images: torch.Tensor = None,
        audios: torch.Tensor = None,
        visual_mask: torch.Tensor = None,
        audio_mask: torch.Tensor = None,
        eval_mode: str = "full"
    ):
        img_feat = self.img_enc(images) if images is not None else None
        aud_feat = self.aud_enc(audios) if audios is not None else None

        # Eq. 2: Feature Imputation during Single-Modality Inference / Testing
        if eval_mode == "image_only":
            imputed_aud_feat = img_feat
            fused_feat = torch.cat([img_feat, imputed_aud_feat], dim=-1)
            fused_logits = self.fusion_classifier(fused_feat)
            img_logits = self.img_classifier(img_feat)
            logits = 0.5 * (fused_logits + img_logits)
            return logits, torch.tensor(0.0, device=images.device)

        if eval_mode == "audio_only":
            imputed_img_feat = aud_feat
            fused_feat = torch.cat([imputed_img_feat, aud_feat], dim=-1)
            fused_logits = self.fusion_classifier(fused_feat)
            aud_logits = self.aud_classifier(aud_feat)
            logits = 0.5 * (fused_logits + aud_logits)
            return logits, torch.tensor(0.0, device=audios.device)

        # Compute Cross-modal Knowledge Distillation Loss using IWV ratio & L1 loss (Eq. 5 & 6)
        iwv = self.kd_weights()
        w_v, w_a = iwv[0], iwv[1]
        ratio_v2a = (w_v / (w_a + 1e-6)).clamp(max=10.0)
        ratio_a2v = (w_a / (w_v + 1e-6)).clamp(max=10.0)

        device = images.device if images is not None else audios.device
        kd_loss = torch.tensor(0.0, device=device)

        if visual_mask is not None and audio_mask is not None:
            valid_pair = (visual_mask & audio_mask)
            if valid_pair.sum() > 0:
                diff_v2a = F.l1_loss(aud_feat[valid_pair], img_feat[valid_pair].detach())
                diff_a2v = F.l1_loss(img_feat[valid_pair], aud_feat[valid_pair].detach())
                kd_loss = ratio_v2a * diff_v2a + ratio_a2v * diff_a2v
        else:
            diff_v2a = F.l1_loss(aud_feat, img_feat.detach())
            diff_a2v = F.l1_loss(img_feat, aud_feat.detach())
            kd_loss = ratio_v2a * diff_v2a + ratio_a2v * diff_a2v

        # Feature imputation for partial samples in batch
        if visual_mask is not None and audio_mask is not None:
            eff_img_feat = img_feat.clone()
            eff_aud_feat = aud_feat.clone()
            eff_img_feat[~visual_mask] = aud_feat[~visual_mask].detach()
            eff_aud_feat[~audio_mask] = img_feat[~audio_mask].detach()
            fused_feat = torch.cat([eff_img_feat, eff_aud_feat], dim=-1)
        else:
            fused_feat = torch.cat([img_feat, aud_feat], dim=-1)

        logits = self.fusion_classifier(fused_feat)
        img_logits = self.img_classifier(img_feat)
        aud_logits = self.aud_classifier(aud_feat)

        return logits, img_logits, aud_logits, kd_loss
