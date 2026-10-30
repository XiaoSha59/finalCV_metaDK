"""
Medical Domain Post-Processing Module for BraTS 2018
====================================================
Contains the official medical post-processing pipeline adapted from the MetaKD
author's framework and MICCAI BraTS challenge standard practices:

1. 3D Connected Component Filtering (`continues_region_extract`):
   - Removes small disconnected false-positive artifacts (< 2,000 voxels or 10% of max).
   - Filters out extreme boundary slices (z_max <= 25) near cranial base.
2. K-means Feature Refinement:
   - When T1ce is available, recovers necrotic cores from edema clusters.
3. Enhancing Tumor (ET) Reclassification:
   - Disconnected ET clusters with total volume < 500 voxels are reclassified to
     necrotic / non-enhancing core (Label 1) to eliminate false-positive penalties.
4. Topological Consistency Enforcement:
   - Ensures anatomical nesting: ET subset of TC subset of WT.
"""

import numpy as np
import cv2
from scipy.ndimage import find_objects
from skimage.measure import label as LAB


def continues_region_extract(label, MR=None, name='', min_et_voxels=500):
    """
    Official BraTS medical post-processing algorithm.

    Args:
        label (np.ndarray): 3D volume with labels (0: Background, 1: Necrosis, 2: Edema, 4: Enhancing Tumor).
                            Orientation: (H, W, D).
        MR (np.ndarray or None): Raw T1ce volume in identical orientation.
                                 Must be None when T1ce is absent (missing modality).
        name (str): Patient case identifier (optional).
        min_et_voxels (int): Minimum volume threshold for ET. Default: 500.

    Returns:
        label (np.ndarray): Cleaned post-processed segmentation label map.
    """
    regions = np.where(label >= 1, np.ones_like(label), np.zeros_like(label))
    L, n = LAB(regions, background=0, connectivity=2, return_num=True)

    if n == 0:
        return label

    sizes = np.bincount(L.ravel(), minlength=n + 1)
    boxes = find_objects(L)
    max_num = sizes[1:].max() if n > 0 else 0

    vol_e_ratio = np.sum(label == 4) / max(np.sum(label >= 1), 1)

    for i in range(1, n + 1):
        sl = boxes[i - 1]
        in_i = L[sl] == i
        z_max = sl[2].start + np.where(in_i.any(axis=(0, 1)))[0].max()

        # Step 1: Delete tiny floating components or edge artifacts
        if sizes[i] < min(2000, max_num * 0.1) or z_max <= 25:
            label[sl][in_i] = 0
        if sizes[i] >= min(2000, max_num * 0.1) and z_max <= 25:
            continue

        # Step 2: K-means clustering on T1ce edema (active only when T1ce is present)
        if MR is None:
            continue

        lab_i = label[sl][in_i]
        vol_e_i = lab_i[lab_i == 4].sum()
        vol_t_i = lab_i[lab_i >= 1].sum()

        with np.errstate(divide='ignore', invalid='ignore'):
            cond = (vol_e_ratio < 0.12 and np.float64(vol_e_i) / max(vol_t_i, 1) < 0.05
                    and vol_e_i < 450 and np.sum(label == 1) < 10000)

        if cond:
            label_i = np.where(L == i, label, np.zeros_like(label))
            MR_edema = np.where(label_i == 2, MR, np.zeros_like(MR))
            MR_edema_ = np.float32(np.reshape(MR_edema, (-1)))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            num_clusters = 3
            ret, matrix, center = cv2.kmeans(
                MR_edema_, num_clusters, None, criteria, num_clusters, cv2.KMEANS_RANDOM_CENTERS
            )
            matrix = np.reshape(matrix, newshape=MR_edema.shape)
            center[np.argmax(center)] = np.min(center)
            index = np.argmax(center)
            matrix_convert = np.where(matrix == index, np.ones_like(matrix), np.zeros_like(matrix))
            label = np.where(matrix_convert == 1, 1 * np.ones_like(label), label)

    # Step 3: Reclassify sparse ET clusters (< 500 voxels) into necrotic core (Label 1)
    if np.sum(label == 4) < min_et_voxels:
        label = np.where(label == 4, 1, label)

    return label


def regions_to_label(et, wt, tc):
    """
    Converts 3 overlapping binary evaluation regions into standard BraTS multi-class labels.
    - ET (Enhancing Tumor) -> Label 4
    - TC (Tumor Core)      -> Label 1 (Necrosis / Non-enhancing) + Label 4 (ET)
    - WT (Whole Tumor)     -> Label 2 (Edema) + TC

    Args:
        et (np.ndarray): Binary mask for Enhancing Tumor.
        wt (np.ndarray): Binary mask for Whole Tumor.
        tc (np.ndarray): Binary mask for Tumor Core.
    Returns:
        seg (np.ndarray): BraTS label map (0, 1, 2, 4).
    """
    seg = np.zeros_like(et, dtype=np.int16)
    seg = np.where(wt == 1, 2, seg)
    seg = np.where(tc == 1, 1, seg)
    seg = np.where(et == 1, 4, seg)
    return seg


def label_to_regions(seg):
    """
    Converts a multi-class BraTS label map (0, 1, 2, 4) into the 3 evaluation target regions:
    - ET: Enhancing Tumor (Label 4)
    - WT: Whole Tumor (Labels 1, 2, 4)
    - TC: Tumor Core (Labels 1, 4)

    Args:
        seg (np.ndarray): BraTS label map.
    Returns:
        tuple: (et, wt, tc) binary masks.
    """
    et = np.where(seg == 4, 1, 0).astype(np.uint8)
    wt = np.where(seg > 0, 1, 0).astype(np.uint8)
    tc = np.where(np.logical_or(seg == 1, seg == 4), 1, 0).astype(np.uint8)
    return et, wt, tc


def enforce_topological_hierarchy(et, wt, tc):
    """
    Enforces biological containment hierarchy:
        ET subset of TC subset of WT

    Args:
        et, wt, tc: Binary masks (0 or 1).
    Returns:
        tuple: (et_clean, wt_clean, tc_clean) adhering to hierarchy.
    """
    wt_clean = (wt > 0).astype(np.uint8)
    tc_clean = np.where(wt_clean == 1, tc, 0).astype(np.uint8)
    et_clean = np.where(tc_clean == 1, et, 0).astype(np.uint8)
    return et_clean, wt_clean, tc_clean


def remove_small_speckles(binary_mask, min_size=30):
    """
    Removes isolated pixel clusters smaller than min_size voxels.
    """
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


__all__ = [
    "continues_region_extract",
    "regions_to_label",
    "label_to_regions",
    "enforce_topological_hierarchy",
    "remove_small_speckles"
]
