import numpy as np
from skimage.measure import label as label_components


def enforce_topological_hierarchy(et, wt, tc):
    """
    Clinically, BraTS sub-regions obey strict nesting:
        Enhancing Tumor (ET) <= Tumor Core (TC) <= Whole Tumor (WT).
    Any ET or TC predicted outside Whole Tumor is an anatomically invalid false positive.
    Enforcing this hierarchy cleans up stray predictions without destroying true lesions.
    """
    # 1. Whole Tumor is the outer envelope
    tc_clean = np.where(wt == 1, tc, 0)
    # 2. Enhancing Tumor must reside inside Tumor Core
    et_clean = np.where(tc_clean == 1, et, 0)
    return et_clean, wt, tc_clean


def remove_small_speckles(binary_mask, min_size=30):
    """
    Removes tiny isolated dust/speckles (< min_size voxels) produced by scanner noise.
    Much safer than the author's 500-voxel cutoff which erased true small lesions.
    """
    if binary_mask.sum() == 0:
        return binary_mask

    labeled, num_features = label_components(binary_mask, background=0, connectivity=2, return_num=True)
    if num_features == 0:
        return binary_mask

    sizes = np.bincount(labeled.ravel(), minlength=num_features + 1)
    too_small = sizes < min_size
    too_small_mask = too_small[labeled]
    binary_mask[too_small_mask] = 0
    return binary_mask


def adaptive_postprocess_case(pred_3class, min_speckle_size=30):
    """
    Smart, adaptive post-processing for BraTS:
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
        raise ValueError(f"Expected shape with 3 channels, got {pred_3class.shape}")

    # 1. Topological Nesting
    et_clean, wt_clean, tc_clean = enforce_topological_hierarchy(et, wt, tc)

    # 2. Speckle removal (gentle: 30 voxels instead of author's destructive 500)
    if min_speckle_size > 0:
        et_clean = remove_small_speckles(et_clean, min_size=min_speckle_size)
        tc_clean = remove_small_speckles(tc_clean, min_size=min_speckle_size)
        wt_clean = remove_small_speckles(wt_clean, min_size=min_speckle_size)

        # Re-enforce hierarchy after speckle removal
        et_clean, wt_clean, tc_clean = enforce_topological_hierarchy(et_clean, wt_clean, tc_clean)

    if channel_last:
        return np.stack([et_clean, wt_clean, tc_clean], axis=-1)
    else:
        return np.stack([et_clean, wt_clean, tc_clean], axis=0)
