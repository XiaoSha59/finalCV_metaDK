"""
AudioVision-MNIST Missing Modality Classification Package.
Reproducing MetaKD (arXiv:2405.07155) on Multi-Modal Image-Audio benchmark.
"""

from .models import MetaKDAudioVision, ImageEncoder, AudioEncoder, KDWeights2D
from .dataset import AudioVisionMNIST, get_audiovision_loaders

__all__ = [
    "MetaKDAudioVision",
    "ImageEncoder",
    "AudioEncoder",
    "KDWeights2D",
    "AudioVisionMNIST",
    "get_audiovision_loaders",
]
