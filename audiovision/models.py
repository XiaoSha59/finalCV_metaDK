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
    Learnable MetaKD Knowledge Distillation Weight Matrix (2x2: Visual <-> Audio).
    Optimized at the upper-level (Meta-Val) by Adam optimizer (lr=1e-2, weight_decay=1e-2).
    """
    def __init__(self):
        super(KDWeights2D, self).__init__()
        self.kd_weights = nn.Parameter(torch.ones(2, 2) * 0.5)

    def forward(self) -> torch.Tensor:
        return F.softmax(self.kd_weights, dim=-1)


class MetaKDAudioVision(nn.Module):
    """
    MetaKD Framework for AudioVision-MNIST Digit Classification (SMIL Protocol).
    - Conv + BN + Dropout + FC Encoders
    - 2 FC Layers with Dropout for Classification Head
    - Learnable MetaKD cross-modal distillation loss
    """
    def __init__(self, num_classes: int = 10, feat_dim: int = 128):
        super(MetaKDAudioVision, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        # Modality Encoders
        self.img_enc = ImageEncoder(in_channels=1, feat_dim=feat_dim)
        self.aud_enc = AudioEncoder(in_channels=1, feat_dim=feat_dim)

        # Learnable MetaKD distillation weights
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
        images: torch.Tensor,
        audios: torch.Tensor,
        visual_mask: torch.Tensor = None,
        audio_mask: torch.Tensor = None,
        eval_mode: str = "full"
    ):
        img_feat = self.img_enc(images) if images is not None else None
        aud_feat = self.aud_enc(audios) if audios is not None else None

        if eval_mode == "image_only":
            logits = self.img_classifier(img_feat)
            return logits, torch.tensor(0.0, device=images.device)

        if eval_mode == "audio_only":
            logits = self.aud_classifier(aud_feat)
            return logits, torch.tensor(0.0, device=audios.device)

        # Compute Cross-modal Knowledge Distillation Loss
        weights = self.kd_weights()
        kd_loss = torch.tensor(0.0, device=images.device)

        if visual_mask is not None and audio_mask is not None:
            valid_pair = (visual_mask & audio_mask)
            if valid_pair.sum() > 0:
                diff_v2a = F.mse_loss(img_feat[valid_pair], aud_feat[valid_pair].detach())
                diff_a2v = F.mse_loss(aud_feat[valid_pair], img_feat[valid_pair].detach())
                kd_loss = weights[0, 1] * diff_v2a + weights[1, 0] * diff_a2v
        else:
            diff_v2a = F.mse_loss(img_feat, aud_feat.detach())
            diff_a2v = F.mse_loss(aud_feat, img_feat.detach())
            kd_loss = weights[0, 1] * diff_v2a + weights[1, 0] * diff_a2v

        # Fused classification logits
        fused_feat = torch.cat([img_feat, aud_feat], dim=-1)
        logits = self.fusion_classifier(fused_feat)

        img_logits = self.img_classifier(img_feat)
        aud_logits = self.aud_classifier(aud_feat)

        return logits, img_logits, aud_logits, kd_loss
