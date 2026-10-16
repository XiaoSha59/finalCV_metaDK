import numpy as np
import torch
import nibabel as nib


def dice_score(preds, labels, eps=1e-5):
    """
    Computes Dice similarity coefficient for binary or multi-region segmentations.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    preds = preds.astype(np.bool_)
    labels = labels.astype(np.bool_)

    intersection = 2.0 * np.sum(preds & labels)
    union = np.sum(preds) + np.sum(labels)

    if union == 0:
        return 1.0 if np.sum(preds) == 0 and np.sum(labels) == 0 else 0.0

    return (intersection + eps) / (union + eps)


def compute_brats_subregion_dice(seg_pred_3class, seg_gt_3class):
    """
    Computes Dice score for the 3 BraTS sub-regions:
    - ET (Enhancing Tumor): Channel 0 / label 4
    - WT (Whole Tumor): Channel 1 / labels 1, 2, 4
    - TC (Tumor Core): Channel 2 / labels 1, 4
    """
    if seg_pred_3class.shape[-1] == 3:
        pred_et = seg_pred_3class[:, :, :, 0]
        pred_wt = seg_pred_3class[:, :, :, 1]
        pred_tc = seg_pred_3class[:, :, :, 2]
    else:
        pred_et = seg_pred_3class[0]
        pred_wt = seg_pred_3class[1]
        pred_tc = seg_pred_3class[2]

    if seg_gt_3class.shape[-1] == 3:
        gt_et = seg_gt_3class[:, :, :, 0]
        gt_wt = seg_gt_3class[:, :, :, 1]
        gt_tc = seg_gt_3class[:, :, :, 2]
    else:
        gt_et = seg_gt_3class[0]
        gt_wt = seg_gt_3class[1]
        gt_tc = seg_gt_3class[2]

    dice_et = dice_score(pred_et, gt_et)
    dice_wt = dice_score(pred_wt, gt_wt)
    dice_tc = dice_score(pred_tc, gt_tc)
    dice_avg = (dice_et + dice_wt + dice_tc) / 3.0

    return {
        'ET': dice_et,
        'WT': dice_wt,
        'TC': dice_tc,
        'Avg': dice_avg
    }


def convert_multiclass_to_brats_regions(label_volume):
    """
    Converts standard BraTS label volume (0: BG, 1: NCR/NET, 2: ED, 4: ET)
    into 3 binary channels (ET, WT, TC).
    """
    shape = label_volume.shape
    results_map = np.zeros((3, shape[0], shape[1], shape[2]), dtype=np.uint8)

    ncr_net = (label_volume == 1)
    ed = (label_volume == 2)
    et = (label_volume == 4)

    results_map[0] = np.where(et, 1, 0)
    results_map[1] = np.where(label_volume >= 1, 1, 0)
    results_map[2] = np.where(np.logical_or(ncr_net, et), 1, 0)

    return results_map


def print_brats_summary_table(results_per_modality):
    """
    Prints a clean summary table for all missing modality combinations.
    """
    modality_names = {
        '0,1,2,3': 'All (Fl+T1+T1c+T2)',
        '0,1,2': 'Fl+T1+T1c',
        '0,1,3': 'Fl+T1+T2',
        '0,2,3': 'Fl+T1c+T2',
        '1,2,3': 'T1+T1c+T2',
        '0,1': 'Fl+T1',
        '0,2': 'Fl+T1c',
        '0,3': 'Fl+T2',
        '1,2': 'T1+T1c',
        '1,3': 'T1+T2',
        '2,3': 'T1c+T2',
        '0': 'Flair only',
        '1': 'T1 only',
        '2': 'T1c only',
        '3': 'T2 only'
    }

    header = f"{'Modalities':<22} | {'Enhancing Tumor (ET)':<20} | {'Tumor Core (TC)':<16} | {'Whole Tumor (WT)':<16} | {'Average':<10}"
    divider = "-" * len(header)
    print("\n" + divider)
    print(header)
    print(divider)

    for mode, metrics in results_per_modality.items():
        name = modality_names.get(mode, mode)
        et = metrics['ET'] * 100.0
        tc = metrics['TC'] * 100.0
        wt = metrics['WT'] * 100.0
        avg = metrics['Avg'] * 100.0
        print(f"{name:<22} | {et:>18.2f}% | {tc:>14.2f}% | {wt:>14.2f}% | {avg:>8.2f}%")

    print(divider + "\n")
