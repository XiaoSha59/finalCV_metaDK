import os
import numpy as np
import nibabel as nib
import cv2
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

    # per-component work is restricted to its bounding box; noisy predictions can have thousands of components
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
    """Merge ET/WT/TC binary maps into a BraTS label map, same order as eval.py."""
    seg = np.zeros_like(et, dtype=np.int16)
    seg = np.where(wt == 1, 2, seg)
    seg = np.where(tc == 1, 1, seg)
    seg = np.where(et == 1, 4, seg)
    return seg


def label_to_regions(seg):
    return (seg == 4).astype(np.uint8), (seg >= 1).astype(np.uint8), np.isin(seg, (1, 4)).astype(np.uint8)


def postprocess_segmentations(pred_dir='outputs', output_dir='outputs_postprocessed', mr_dir=None):
    """
    Postprocess a folder of predicted label maps. mr_dir holds case folders with
    <case>/<case>_t1ce.nii(.gz); leave it None to skip the T1ce k-means step.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(pred_dir) if f.endswith('.nii.gz') or f.endswith('.nii'))
    print(f"Postprocessing {len(files)} predictions from {pred_dir}...")

    for f in files:
        img = nib.load(os.path.join(pred_dir, f))
        pred = np.asarray(img.dataobj)
        mr = None
        if mr_dir is not None:
            case = f[:-7] if f.endswith('.nii.gz') else f[:-4]
            for ext in ('.nii', '.nii.gz'):
                mr_file = os.path.join(mr_dir, case, case + '_t1ce' + ext)
                if os.path.exists(mr_file):
                    mr = np.asarray(nib.load(mr_file).dataobj)
                    break
        cleaned = continues_region_extract(pred.copy(), mr, f)
        nib.save(nib.Nifti1Image(cleaned.astype(np.int16), affine=img.affine), os.path.join(output_dir, f))

    print(f"Saved postprocessed predictions to {output_dir}")


if __name__ == '__main__':
    postprocess_segmentations()
