import os
import sys
import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import label as LAB


def continues_region_extract(label, MR=None, name=''):
    """
    Port of the author's BraTS postprocess (billhhh/MetaKD postprocess.py).
    """
    regions = np.where(label >= 1, np.ones_like(label), np.zeros_like(label))
    L, n = LAB(regions, background=0, connectivity=2, return_num=True)

    sizes = np.bincount(L.ravel(), minlength=n + 1)
    boxes = find_objects(L)
    max_num = sizes[1:].max() if n > 0 else 0

    for i in range(1, n + 1):
        sl = boxes[i - 1]
        in_i = L[sl] == i
        z_max = sl[2].start + np.where(in_i.any(axis=(0, 1)))[0].max()

        if sizes[i] < min(2000, max_num * 0.1) or z_max <= 25:
            label[sl][in_i] = 0

    NUM_pixels = 500
    if np.sum(label == 4) < NUM_pixels:
        label = np.where(label == 4, 1, label)

    return label


def regions_to_label(et, wt, tc):
    seg = np.zeros_like(et, dtype=np.int16)
    seg = np.where(wt == 1, 2, seg)
    seg = np.where(tc == 1, 1, seg)
    seg = np.where(et == 1, 4, seg)
    return seg


def label_to_regions(seg):
    return (seg == 4).astype(np.uint8), (seg >= 1).astype(np.uint8), np.isin(seg, (1, 4)).astype(np.uint8)


def enforce_topological_hierarchy(et, wt, tc):
    tc_clean = np.where(wt == 1, tc, 0)
    et_clean = np.where(tc_clean == 1, et, 0)
    return et_clean, wt, tc_clean


def remove_small_speckles(binary_mask, min_size=30):
    if binary_mask.sum() == 0:
        return binary_mask
    labeled, num_features = LAB(binary_mask, background=0, connectivity=2, return_num=True)
    if num_features == 0:
        return binary_mask
    sizes = np.bincount(labeled.ravel(), minlength=num_features + 1)
    too_small = sizes < min_size
    too_small_mask = too_small[labeled]
    binary_mask[too_small_mask] = 0
    return binary_mask


def adaptive_postprocess_case(pred_3class, min_speckle_size=30):
    """
    Standard fair postprocessing (MICCAI/CVPR benchmark compliant):
    1. Enforce anatomical containment: ET subset of TC subset of WT
    2. Filter out tiny sliding window speckles (< min_speckle_size voxels)
    """
    if pred_3class.shape[-1] == 3:
        et = pred_3class[..., 0].copy().astype(np.uint8)
        wt = pred_3class[..., 1].copy().astype(np.uint8)
        tc = pred_3class[..., 2].copy().astype(np.uint8)
        channel_last = True
    elif pred_3class.shape[0] == 3:
        et = pred_3class[0].copy().astype(np.uint8)
        wt = pred_3class[1].copy().astype(np.uint8)
        tc = pred_3class[2].copy().astype(np.uint8)
        channel_last = False
    else:
        raise ValueError(f"Expected shape with 3 channels, got {pred_3class.shape}")

    et_clean, wt_clean, tc_clean = enforce_topological_hierarchy(et, wt, tc)

    if min_speckle_size > 0:
        et_clean = remove_small_speckles(et_clean, min_size=min_speckle_size)
        tc_clean = remove_small_speckles(tc_clean, min_size=min_speckle_size)
        wt_clean = remove_small_speckles(wt_clean, min_size=min_speckle_size)
        et_clean, wt_clean, tc_clean = enforce_topological_hierarchy(et_clean, wt_clean, tc_clean)

    if channel_last:
        return np.stack([et_clean, wt_clean, tc_clean], axis=-1)
    else:
        return np.stack([et_clean, wt_clean, tc_clean], axis=0)


__all__ = [
    "continues_region_extract",
    "regions_to_label",
    "label_to_regions",
    "enforce_topological_hierarchy",
    "remove_small_speckles",
    "adaptive_postprocess_case",
]
