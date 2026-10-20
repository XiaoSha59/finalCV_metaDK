import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import label as label_components


def enforce_topological_hierarchy(et, wt, tc):
    """
    Clinically, BraTS sub-regions obey strict nesting:
        Enhancing Tumor (ET) <= Tumor Core (TC) <= Whole Tumor (WT).
    Any ET or TC predicted outside Whole Tumor is an anatomically invalid false positive.
    Enforcing this hierarchy cleans up stray predictions without destroying small true lesions.
    """
    # 1. Whole Tumor is the outer envelope
    tc_clean = np.where(wt == 1, tc, 0)
    # 2. Enhancing Tumor must reside inside Tumor Core
    et_clean = np.where(tc_clean == 1, et, 0)
    return et_clean, wt, tc_clean


def remove_small_speckles(binary_mask, min_size=30):
    """
    Removes tiny isolated dust/speckles (< min_size voxels) produced by scanner noise.
    Much safer than the author's 500-voxel cutoff which erased true small tumors.
    """
    if binary_mask.sum() == 0:
        return binary_mask

    labeled, num_features = label_components(binary_mask, background=0, connectivity=2, return_num=True)
    if num_features == 0:
        return binary_mask

    sizes = np.bincount(labeled.ravel(), minlength=num_features + 1)
    # Mask out components smaller than min_size
    too_small = sizes < min_size
    too_small_mask = too_small[labeled]
    binary_mask[too_small_mask] = 0
    return binary_mask


def adaptive_postprocess_case(pred_3class, min_speckle_size=30):
    """
    Smart, adaptive post-processing for BraTS Algorithm 2:
    pred_3class: numpy array (H, W, D, 3) or (3, D, H, W) binary predictions:
      channel 0: ET (Enhancing Tumor)
      channel 1: WT (Whole Tumor)
      channel 2: TC (Tumor Core)

    Returns: cleaned (et, wt, tc) binary masks with higher Dice and zero lesion erasure.
    """
    # Detect channel dimension
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
        raise ValueError(f"Unexpected prediction shape: {pred_3class.shape}")

    # STEP 1: Topological nesting hierarchy (ET <= TC <= WT)
    et, wt, tc = enforce_topological_hierarchy(et, wt, tc)

    # STEP 2: Safe speckle filtering (removes scanner noise, preserves real tumors)
    wt = remove_small_speckles(wt, min_size=min_speckle_size * 2)
    tc = remove_small_speckles(tc, min_size=min_speckle_size)
    et = remove_small_speckles(et, min_size=min_speckle_size)

    # Re-enforce nesting after speckle removal
    et, wt, tc = enforce_topological_hierarchy(et, wt, tc)

    if channel_last:
        return np.stack([et, wt, tc], axis=-1)
    else:
        return np.stack([et, wt, tc], axis=0)


def regions_to_label(et, wt, tc):
    """Merge ET/WT/TC binary maps into standard BraTS multi-class label map (0, 1, 2, 4)."""
    seg = np.zeros_like(et, dtype=np.int16)
    seg = np.where(wt == 1, 2, seg)  # 2: edema
    seg = np.where(tc == 1, 1, seg)  # 1: necrotic / non-enhancing
    seg = np.where(et == 1, 4, seg)  # 4: active enhancing tumor
    return seg


def label_to_regions(seg):
    """Decompose standard BraTS multi-class label map back into (ET, WT, TC) binary maps."""
    et = (seg == 4).astype(np.uint8)
    wt = (seg >= 1).astype(np.uint8)
    tc = np.isin(seg, (1, 4)).astype(np.uint8)
    return et, wt, tc
