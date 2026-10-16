"""
Loss functions package for MetaKD.
"""

from .loss_dual import (
    BinaryDiceLoss,
    DiceLoss4BraTS,
    BCELoss4BraTS,
    DiceLoss4MOTS,
    CELoss4MOTS,
    KDLoss,
    DomainBCELoss,
    DomainClsLoss
)

__all__ = [
    "BinaryDiceLoss",
    "DiceLoss4BraTS",
    "BCELoss4BraTS",
    "DiceLoss4MOTS",
    "CELoss4MOTS",
    "KDLoss",
    "DomainBCELoss",
    "DomainClsLoss"
]
