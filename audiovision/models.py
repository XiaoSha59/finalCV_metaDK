import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """
    Standard LeNet-5 Backbone for 28x28 MNIST Images (SMIL setup).
    Conv1(1->6, k=5, pad=2) -> MaxPool -> Conv2(6->16, k=5) -> MaxPool -> FC(120) -> FC(128).
    """
    def __init__(self, in_channels: int = 1, feat_dim: int = 128):
        super(ImageEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)


class AudioEncoder(nn.Module):
    """
    3-Layer Multi-Layer Perceptron (MLP) for 20x20 Audio MFCCs (SMIL setup).
    Flattens 20x20=400 -> Linear(400, 256) -> ReLU -> Linear(256, 128) -> ReLU.
    """
    def __init__(self, in_channels: int = 1, feat_dim: int = 128):
        super(AudioEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(20 * 20, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.size(0), -1)
        return self.net(x_flat)


class KDWeights2D(nn.Module):
    """
    Learnable MetaKD Knowledge Distillation Weight Matrix (2x2: Visual <-> Audio).
    Optimized at the upper-level (Meta-Val) by Adam optimizer.
    """
    def __init__(self):
        super(KDWeights2D, self).__init__()
        # 2x2 matrix: (Visual->Audio, Audio->Visual)
        self.kd_weights = nn.Parameter(torch.ones(2, 2) * 0.5)

    def forward(self) -> torch.Tensor:
        # Normalize weights via softmax across source modalities
        return F.softmax(self.kd_weights, dim=-1)


class MetaKDAudioVision(nn.Module):
    """
    MetaKD Framework for AudioVision-MNIST Digit Classification.
    Features:
      - Multi-modal representation learning (Visual + Audio)
      - Shared and Specific feature projection
      - Cross-modal knowledge distillation with learnable MetaKD weights
      - Robust evaluation under missing modalities (Image-only or Audio-only)
    """
    def __init__(self, num_classes: int = 10, feat_dim: int = 128):
        super(MetaKDAudioVision, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        # Modality Encoders
        self.img_enc = ImageEncoder(in_channels=1, feat_dim=feat_dim)
        self.aud_enc = AudioEncoder(in_channels=1, feat_dim=feat_dim)

        # Shared and Specific feature projectors
        self.img_shared = nn.Linear(feat_dim, feat_dim)
        self.aud_shared = nn.Linear(feat_dim, feat_dim)

        # Learnable MetaKD distillation weights
        self.kd_weights = KDWeights2D()

        # Classification Heads
        self.img_classifier = nn.Linear(feat_dim, num_classes)
        self.aud_classifier = nn.Linear(feat_dim, num_classes)
        self.fusion_classifier = nn.Linear(feat_dim * 2, num_classes)

    def forward(
        self,
        images: torch.Tensor,
        audios: torch.Tensor,
        visual_mask: torch.Tensor = None,
        audio_mask: torch.Tensor = None,
        eval_mode: str = "full"  # "full", "image_only", "audio_only"
    ):
        """
        Forward pass with support for missing modality training and evaluation.
        """
        # Feature extraction
        img_feat = self.img_enc(images) if images is not None else None
        aud_feat = self.aud_enc(audios) if audios is not None else None

        # Evaluation modes
        if eval_mode == "image_only":
            logits = self.img_classifier(img_feat)
            return logits, torch.tensor(0.0, device=images.device)

        if eval_mode == "audio_only":
            logits = self.aud_classifier(aud_feat)
            return logits, torch.tensor(0.0, device=audios.device)

        # Full or training mode
        img_shared = self.img_shared(img_feat)
        aud_shared = self.aud_shared(aud_feat)

        # Compute Cross-modal Knowledge Distillation Loss (L2 distance on shared features)
        weights = self.kd_weights()
        kd_loss = torch.tensor(0.0, device=images.device)

        if visual_mask is not None and audio_mask is not None:
            # Only compute KD where at least one modality provides valid teacher representation
            valid_pair = (visual_mask & audio_mask)
            if valid_pair.sum() > 0:
                diff_v2a = F.mse_loss(img_shared[valid_pair], aud_shared[valid_pair].detach())
                diff_a2v = F.mse_loss(aud_shared[valid_pair], img_shared[valid_pair].detach())
                kd_loss = weights[0, 1] * diff_v2a + weights[1, 0] * diff_a2v
        else:
            diff_v2a = F.mse_loss(img_shared, aud_shared.detach())
            diff_a2v = F.mse_loss(aud_shared, img_shared.detach())
            kd_loss = weights[0, 1] * diff_v2a + weights[1, 0] * diff_a2v

        # Fused classification logits for multi-modal prediction
        fused_feat = torch.cat([img_feat, aud_feat], dim=-1)
        logits = self.fusion_classifier(fused_feat)

        # Individual modality logits for auxiliary supervision
        img_logits = self.img_classifier(img_feat)
        aud_logits = self.aud_classifier(aud_feat)

        return logits, img_logits, aud_logits, kd_loss
