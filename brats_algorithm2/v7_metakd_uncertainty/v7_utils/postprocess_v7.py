import os
import sys
import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import label as LAB


def continues_region_extract(label, MR=None, name=''):
    """
    Port of the author's BraTS postprocess (billhhh/MetaKD postprocess.py).

    label: BraTS label map (0/1/2/4) in the original NIfTI orientation (H, W, D).
    MR:    raw T1ce volume, same orientation. Pass None when T1ce is a missing
           modality, so the k-means step cannot use unavailable information.
    """
    regions = np.where(label >= 1, np.ones_like(label), np.zeros_like(label))
    L, n = LAB(regions, background=0, connectivity=2, return_num=True)

    sizes = np.bincount(L.ravel(), minlength=n + 1)
    boxes = find_objects(L)
    max_num = sizes[1:].max() if n > 0 else 0

    vol_e_ratio = np.sum(label == 4) / max(np.sum(label >= 1), 1)
    for i in range(1, n + 1):
        sl = boxes[i - 1]
        in_i = L[sl] == i
        z_max = sl[2].start + np.where(in_i.any(axis=(0, 1)))[0].max()

        # STEP 1: delete components that are too small or close to the edge
        if sizes[i] < min(2000, max_num * 0.1) or z_max <= 25:
            label[sl][in_i] = 0
        if sizes[i] >= min(2000, max_num * 0.1) and z_max <= 25:
            continue

        # STEP 2: k-means on T1ce edema to recover tumor core
        if MR is None:
            continue
        lab_i = label[sl][in_i]
        vol_e_i = lab_i[lab_i == 4].sum()
        vol_t_i = lab_i[lab_i >= 1].sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            cond = (vol_e_ratio < 0.12 and np.float64(vol_e_i) / vol_t_i < 0.05
                    and vol_e_i < 450 and np.sum(label == 1) < 10000)
        if cond:
            label_i = np.where(L == i, label, np.zeros_like(label))
            MR_edema = np.where(label_i == 2, MR, np.zeros_like(MR))
            MR_edema_ = np.float32(np.reshape(MR_edema, (-1)))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            num_clusters = 3
            ret, matrix, center = cv2.kmeans(MR_edema_, num_clusters, None, criteria, num_clusters, cv2.KMEANS_RANDOM_CENTERS)
            matrix = np.reshape(matrix, newshape=MR_edema.shape)
            center[np.argmax(center)] = np.min(center)
            index = np.argmax(center)
            matrix_convert = np.where(matrix == index, np.ones_like(matrix), np.zeros_like(matrix))
            label = np.where(matrix_convert == 1, 1 * np.ones_like(label), label)

    # STEP 3: too few ET voxels -> treat them as necrosis / non-enhancing core
    NUM_pixels = 500
    if np.sum(label == 4) < NUM_pixels:
        label = np.where(label == 4, 1, label)

    return label


def regions_to_label(et, wt, tc):
    """Merge ET/WT/TC binary maps into a BraTS label map (0, 1, 2, 4)."""
    seg = np.zeros_like(et, dtype=np.int16)
    seg = np.where(wt == 1, 2, seg)
    seg = np.where(tc == 1, 1, seg)
    seg = np.where(et == 1, 4, seg)
    return seg


def label_to_regions(seg):
    """Convert BraTS label map (0, 1, 2, 4) back to ET, WT, TC binary maps."""
    return (seg == 4).astype(np.uint8), (seg >= 1).astype(np.uint8), np.isin(seg, (1, 4)).astype(np.uint8)


def enforce_topological_hierarchy(et, wt, tc):
    """Enforce biological containment: ET subset of TC subset of WT."""
    tc_clean = np.where(wt == 1, tc, 0)
    et_clean = np.where(tc_clean == 1, et, 0)
    return et_clean, wt, tc_clean


def remove_small_speckles(binary_mask, min_size=30):
    """Remove small isolated island artifacts < min_size voxels."""
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
    No K-Means, no intensity clustering, strictly fair and representation-based.
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


def postprocess_prediction(pred_binary, et_thresh=200):
    return adaptive_postprocess_case(pred_binary, min_speckle_size=30)


def postprocess_batch(pred_batch, et_thresh=200):
    return pred_batch


__all__ = [
    "continues_region_extract",
    "regions_to_label",
    "label_to_regions",
    "enforce_topological_hierarchy",
    "remove_small_speckles",
    "adaptive_postprocess_case",
    "postprocess_prediction",
    "postprocess_batch",
]
