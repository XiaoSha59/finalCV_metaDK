"""SMIL backbones (LeNet-5, SNetPluse) plus the MetaKD fusion head.

Image and sound conv stacks follow deep-real/SMIL:
https://github.com/deep-real/SMIL/blob/main/src/models/lenet5.py
https://github.com/deep-real/SMIL/blob/main/src/models/snet.py
https://github.com/deep-real/SMIL/blob/main/src/models/soundlenet5.py

SMIL concatenates a 160-d image feature with a 320-d sound feature.
MetaKD Eq. (2) and Eq. (5) need equal feature sizes, so the sound
encoder keeps SNetPluse.fc1 (320 -> 160). Both modality features are
then 160-d, which is the tensor immediately before the two fusion FC
layers. The fusion head keeps SMIL's width and dropout: Linear 64,
dropout 0.6, Linear 10.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """SMIL LeNet-5 conv stack. Output is 160-d, before any classifier FC."""

    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.pool1(self.conv1(x)))
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.relu(self.pool2(x))
        return x.view(x.size(0), -1)


class AudioEncoder(nn.Module):
    """SMIL SNetPluse conv stack plus fc1, mapping 20x20 MFCCs to 160-d."""

    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=2)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 20, kernel_size=2)
        self.bn3 = nn.BatchNorm2d(20)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=4)
        self.fc1 = nn.Linear(320, 160)
        self.dropout1 = nn.Dropout(p=0.25)
        self.relu = nn.ReLU(inplace=True)

    def conv_feature(self, x: torch.Tensor) -> torch.Tensor:
        """320-d map SMIL concatenates in SoundLenet5, taken before fc1."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout1(x)
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(self.conv_feature(x))
        x = self.dropout1(x)
        return self.relu(x)


class LeNet5(nn.Module):
    """SMIL image-only classifier. Table 3 LowerB for missing audio."""

    def __init__(self):
        super(LeNet5, self).__init__()
        self.encoder = ImageEncoder()
        self.fc1 = nn.Linear(160, 32)
        self.fc2 = nn.Linear(32, 10)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        hidden = self.relu(self.fc1(feat))
        return self.fc2(hidden)


class SNetPluse(nn.Module):
    """SMIL audio-only classifier. Table 3 LowerB for missing visual."""

    def __init__(self):
        super(SNetPluse, self).__init__()
        self.encoder = AudioEncoder()
        self.fc2 = nn.Linear(160, 32)
        self.fc3 = nn.Linear(32, 10)
        self.dropout2 = nn.Dropout(p=0.4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        hidden = self.relu(self.dropout2(self.fc2(feat)))
        return self.fc3(hidden)


class KDWeights2D(nn.Module):
    """Modality importance logits. Eq. (6) applies softmax at every use."""

    def __init__(self):
        super(KDWeights2D, self).__init__()
        self.weights = nn.Parameter(torch.randn(2) * 0.01)

    def forward(self) -> torch.Tensor:
        return F.softmax(self.weights, dim=0)


def _pairwise_l1(img_feat: torch.Tensor, aud_feat: torch.Tensor, w_v: torch.Tensor, w_a: torch.Tensor) -> torch.Tensor:
    """Eq. (5): L1 distance between paired features, weighted in both directions."""
    distance = F.l1_loss(img_feat, aud_feat)
    return (w_v / (w_a + 1e-6) + w_a / (w_v + 1e-6)) * distance


class MetaKDAudioVision(nn.Module):
    """MetaKD on SMIL Audiovision features.

    Task loss uses the concatenated modality features. The meta step uses
    the same features scaled by the importance weights, Eq. (4). Missing
    features are the mean of the available ones, Eq. (2). With one modality
    left, that mean is the remaining feature.
    """

    def __init__(self, num_classes: int = 10):
        super(MetaKDAudioVision, self).__init__()
        self.img_enc = ImageEncoder()
        self.aud_enc = AudioEncoder()
        self.kd_weights = KDWeights2D()
        self.fc1 = nn.Linear(320, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=0.6)
        self.relu = nn.ReLU(inplace=True)

    def encode(self, images: torch.Tensor, audios: torch.Tensor):
        return self.img_enc(images), self.aud_enc(audios)

    def fuse(self, img_feat: torch.Tensor, aud_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([img_feat, aud_feat], dim=-1)
        hidden = self.dropout(self.relu(self.fc1(fused)))
        return self.fc2(hidden)

    def forward(
        self,
        images: torch.Tensor = None,
        audios: torch.Tensor = None,
        visual_mask: torch.Tensor = None,
        audio_mask: torch.Tensor = None,
        eval_mode: str = "full",
    ):
        if eval_mode == "image_only":
            img_feat = self.img_enc(images)
            return self.fuse(img_feat, img_feat), img_feat.new_zeros(())

        if eval_mode == "audio_only":
            aud_feat = self.aud_enc(audios)
            return self.fuse(aud_feat, aud_feat), aud_feat.new_zeros(())

        img_feat, aud_feat = self.encode(images, audios)
        weights = self.kd_weights()

        if eval_mode == "meta_val":
            logits = self.fuse(weights[0] * img_feat, weights[1] * aud_feat)
            return logits, img_feat.new_zeros(())

        weights = weights.detach()
        w_v, w_a = weights[0], weights[1]
        kd_loss = img_feat.new_zeros(())

        if visual_mask is not None and audio_mask is not None:
            paired = visual_mask & audio_mask
            if paired.any():
                kd_loss = _pairwise_l1(img_feat[paired], aud_feat[paired], w_v, w_a)

            eff_img = img_feat.clone()
            eff_aud = aud_feat.clone()
            missing_visual = ~visual_mask
            missing_audio = ~audio_mask
            if missing_visual.any():
                eff_img[missing_visual] = aud_feat[missing_visual]
            if missing_audio.any():
                eff_aud[missing_audio] = img_feat[missing_audio]
        else:
            kd_loss = _pairwise_l1(img_feat, aud_feat, w_v, w_a)
            eff_img, eff_aud = img_feat, aud_feat

        return self.fuse(eff_img, eff_aud), kd_loss
