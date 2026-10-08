import os
import os.path as osp
import numpy as np
import nibabel as nib
from skimage.measure import label as LAB
import SimpleITK as sitk


def continues_region_extract(label, MR, name):
    regions = np.where(label >= 1, np.ones_like(label), np.zeros_like(label))
    L, n = LAB(regions, neighbors=4, background=0, connectivity=2, return_num=True)

    max_num = 0
    max_label = 0
    for i in range(0, n + 1):
        if np.sum(L == i) > max_num and i != 0:
            max_num = np.sum(L == i)
            max_label = i

    for i in range(1, n + 1):
        if np.sum(L == i) < min(2000, max_num * 0.1) or np.where(L == i)[2].max() <= 25:
            label = np.where(L == i, np.zeros_like(label), label)
            continue

    return label


def postprocess_segmentations(pred_dir='outputs', output_dir='outputs_postprocessed'):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(pred_dir) if f.endswith('.nii.gz') or f.endswith('.nii')]
    print(f"Postprocessing {len(files)} predictions from {pred_dir}...")

    for f in files:
        pred_path = os.path.join(pred_dir, f)
        img = nib.load(pred_path)
        data = img.get_data()
        cleaned_data = continues_region_extract(data, None, f)
        
        save_img = nib.Nifti1Image(cleaned_data.astype(np.int16), affine=img.affine)
        save_path = os.path.join(output_dir, f)
        nib.save(save_img, save_path)

    print(f"Saved postprocessed predictions to {output_dir}")


if __name__ == '__main__':
    postprocess_segmentations()
