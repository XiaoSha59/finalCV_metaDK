"""
Official 15-Modality Evaluation Script for Plan 1 Extended
==========================================================
Evaluates the trained Plan 1 Extended checkpoint across all 15 possible
combinations of missing MRI modalities on the BraTS 2018 test benchmark.

Features:
- Fast RAM-cached sliding window 3D inference.
- Dual metrics: Raw Predictions vs Post-Processed Predictions (Author Pipeline).
- Direct comparison with MetaKD Paper Table 1 and Reproduce Baseline.
- Automatic CSV summary export.
"""

import argparse
import os
import os.path as osp
import sys
import csv
import time
from math import ceil
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils import data

CURRENT_DIR = osp.dirname(osp.abspath(__file__))
REPO_DIR = osp.abspath(osp.join(CURRENT_DIR, ".."))
BRATS_DIR = osp.join(REPO_DIR, "brats")

for p in [CURRENT_DIR, REPO_DIR, BRATS_DIR, osp.join(BRATS_DIR, "process"), osp.join(BRATS_DIR, "data")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from models.dual_net import DualNet
except ImportError:
    from models import DualNet

try:
    from datasets.brats_dataset import BraTSEvalDataSet
except ImportError:
    from datasets import BraTSEvalDataSet

try:
    from utils.metrics import dice_score
except ImportError:
    def dice_score(pred, target):
        intersection = 2.0 * (pred * target).sum()
        union = pred.sum() + target.sum()
        if union == 0:
            return 1.0
        return float(intersection / union)

from improvements.postprocess import (
    continues_region_extract,
    regions_to_label,
    label_to_regions,
    enforce_topological_hierarchy
)

# 15 Combinations of BraTS Modalities
MODES_15 = [
    ('0',       'Flair only'),
    ('1',       'T1 only'),
    ('2',       'T1c only'),
    ('3',       'T2 only'),
    ('0,1',     'Fl+T1'),
    ('0,2',     'Fl+T1c'),
    ('0,3',     'Fl+T2'),
    ('1,2',     'T1+T1c'),
    ('1,3',     'T1+T2'),
    ('2,3',     'T1c+T2'),
    ('0,1,2',   'Fl+T1+T1c'),
    ('0,1,3',   'Fl+T1+T2'),
    ('0,2,3',   'Fl+T1c+T2'),
    ('1,2,3',   'T1+T1c+T2'),
    ('0,1,2,3', 'All (Fl+T1+T1c+T2)'),
]

PAPER_TABLE_1 = {
    '0':       {'ET': 47.37, 'TC': 73.01, 'WT': 89.11},
    '1':       {'ET': 44.94, 'TC': 67.16, 'WT': 77.47},
    '2':       {'ET': 76.22, 'TC': 81.98, 'WT': 77.72},
    '3':       {'ET': 46.93, 'TC': 69.72, 'WT': 84.44},
    '0,1':     {'ET': 49.64, 'TC': 75.66, 'WT': 90.23},
    '0,2':     {'ET': 77.49, 'TC': 84.82, 'WT': 90.30},
    '0,3':     {'ET': 50.27, 'TC': 75.88, 'WT': 90.39},
    '1,2':     {'ET': 75.41, 'TC': 81.85, 'WT': 80.59},
    '1,3':     {'ET': 49.77, 'TC': 74.40, 'WT': 87.05},
    '2,3':     {'ET': 78.24, 'TC': 84.84, 'WT': 86.18},
    '0,1,2':   {'ET': 77.66, 'TC': 85.36, 'WT': 90.70},
    '0,1,3':   {'ET': 51.05, 'TC': 76.73, 'WT': 90.59},
    '0,2,3':   {'ET': 75.58, 'TC': 85.08, 'WT': 90.87},
    '1,2,3':   {'ET': 75.87, 'TC': 84.57, 'WT': 86.94},
    '0,1,2,3': {'ET': 79.74, 'TC': 86.20, 'WT': 90.92},
}


def pad_image(img, target_size):
    deps_missing = target_size[0] - img.shape[2]
    rows_missing = target_size[1] - img.shape[3]
    cols_missing = target_size[2] - img.shape[4]
    return np.pad(img, ((0, 0), (0, 0), (0, deps_missing), (0, rows_missing), (0, cols_missing)), 'constant')


def predict_sliding(args, net, image, tile_size, classes):
    interp = nn.Upsample(size=tile_size, mode='trilinear', align_corners=True)
    image_size = image.shape
    overlap = 1 / 3

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    full_probs = torch.zeros((classes, image_size[2], image_size[3], image_size[4]))
    count_predictions = torch.zeros((classes, image_size[2], image_size[3], image_size[4]))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                y1 = int(row * strideHW)
                x1 = int(col * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                y2 = min(y1 + tile_size[1], image_size[3])
                x2 = min(x1 + tile_size[2], image_size[4])
                d1 = max(int(d2 - tile_size[0]), 0)
                y1 = max(int(y2 - tile_size[1]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                padded_img = pad_image(img, tile_size)
                input_tensor = torch.from_numpy(padded_img).to(device)
                padded_prediction, _, _ = net(input_tensor, mode=args.mode)
                padded_prediction = torch.sigmoid(padded_prediction)

                padded_prediction = interp(padded_prediction).cpu().data[0]
                prediction = padded_prediction[0:img.shape[2], 0:img.shape[3], 0:img.shape[4], :]
                count_predictions[:, d1:d2, y1:y2, x1:x2] += 1
                full_probs[:, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    return full_probs.numpy().transpose(1, 2, 3, 0)


def make_row(mode_str, mode_desc, et_list, tc_list, wt_list):
    our = {
        'et': float(np.mean(et_list)) * 100.0,
        'tc': float(np.mean(tc_list)) * 100.0,
        'wt': float(np.mean(wt_list)) * 100.0
    }
    pap = {
        'et': PAPER_TABLE_1[mode_str]['ET'],
        'tc': PAPER_TABLE_1[mode_str]['TC'],
        'wt': PAPER_TABLE_1[mode_str]['WT']
    }
    return {
        'mode': mode_str, 'desc': mode_desc,
        'our_et': our['et'], 'paper_et': pap['et'], 'diff_et': our['et'] - pap['et'],
        'our_tc': our['tc'], 'paper_tc': pap['tc'], 'diff_tc': our['tc'] - pap['tc'],
        'our_wt': our['wt'], 'paper_wt': pap['wt'], 'diff_wt': our['wt'] - pap['wt'],
        'our_avg': (our['et'] + our['tc'] + our['wt']) / 3.0,
        'paper_avg': (pap['et'] + pap['tc'] + pap['wt']) / 3.0,
    }


def export_report(results_table, title, csv_path):
    header = (f"{'Modalities':<22} | {'ET Our':<8} {'ET Pap':<8} {'Diff':<8} | "
              f"{'TC Our':<8} {'TC Pap':<8} {'Diff':<8} | "
              f"{'WT Our':<8} {'WT Pap':<8} {'Diff':<8} | "
              f"{'Avg Our':<8} {'Avg Pap':<8}")
    divider = "=" * len(header)
    print("\n" + divider)
    print(title)
    print(divider)
    print(header)
    print("-" * len(header))
    for r in results_table:
        print(f"{r['desc']:<22} | {r['our_et']:<8.2f} {r['paper_et']:<8.2f} {r['diff_et']:+6.2f}% | "
              f"{r['our_tc']:<8.2f} {r['paper_tc']:<8.2f} {r['diff_tc']:+6.2f}% | "
              f"{r['our_wt']:<8.2f} {r['paper_wt']:<8.2f} {r['diff_wt']:+6.2f}% | "
              f"{r['our_avg']:<8.2f} {r['paper_avg']:<8.2f}")

    mean = {
        k: np.mean([r[k] for r in results_table])
        for k in ('our_et', 'paper_et', 'our_tc', 'paper_tc', 'our_wt', 'paper_wt', 'our_avg', 'paper_avg')
    }
    print("-" * len(header))
    print(f"{'AVERAGE (15 combos)':<22} | {mean['our_et']:<8.2f} {mean['paper_et']:<8.2f} {mean['our_et']-mean['paper_et']:+6.2f}% | "
          f"{mean['our_tc']:<8.2f} {mean['paper_tc']:<8.2f} {mean['our_tc']-mean['paper_tc']:+6.2f}% | "
          f"{mean['our_wt']:<8.2f} {mean['paper_wt']:<8.2f} {mean['our_wt']-mean['paper_wt']:+6.2f}% | "
          f"{mean['our_avg']:<8.2f} {mean['paper_avg']:<8.2f}")
    print(divider + "\n")

    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Mode', 'Description', 'Our_ET', 'Paper_ET', 'Diff_ET',
                         'Our_TC', 'Paper_TC', 'Diff_TC', 'Our_WT', 'Paper_WT', 'Diff_WT',
                         'Our_Avg', 'Paper_Avg'])
        for r in results_table:
            writer.writerow([r['mode'], r['desc'], f"{r['our_et']:.2f}", f"{r['paper_et']:.2f}", f"{r['diff_et']:+.2f}",
                             f"{r['our_tc']:.2f}", f"{r['paper_tc']:.2f}", f"{r['diff_tc']:+.2f}",
                             f"{r['our_wt']:.2f}", f"{r['paper_wt']:.2f}", f"{r['diff_wt']:+.2f}",
                             f"{r['our_avg']:.2f}", f"{r['paper_avg']:.2f}"])
        writer.writerow(['Average', 'Overall 15 Modalities', f"{mean['our_et']:.2f}", f"{mean['paper_et']:.2f}", f"{mean['our_et']-mean['paper_et']:+.2f}",
                         f"{mean['our_tc']:.2f}", f"{mean['paper_tc']:.2f}", f"{mean['our_tc']-mean['paper_tc']:+.2f}",
                         f"{mean['our_wt']:.2f}", f"{mean['paper_wt']:.2f}", f"{mean['our_wt']-mean['paper_wt']:+.2f}",
                         f"{mean['our_avg']:.2f}", f"{mean['paper_avg']:.2f}"])
    print(f">>> Exported CSV: {csv_path}")


def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Plan 1 Extended across 15 missing modalities.")
    parser.add_argument("--data_dir", type=str, default='./data/')
    parser.add_argument("--data_list", type=str, default='datalist/BraTS18/BraTS18_test.csv')
    parser.add_argument("--input_size", type=str, default='80,160,160')
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--restore_from", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--output_csv", type=str, default='improvements/results/results_eval.csv')
    parser.add_argument("--postprocess", type=lambda v: v.lower() in ('true', '1', 'yes', 'y', 't'), default=True)
    return parser.parse_args()


def main():
    args = get_arguments()
    d, h, w = map(int, args.input_size.split(','))
    input_size = (d, h, w)

    print(f"Loading Plan 1 checkpoint: {args.restore_from}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.restore_from, map_location=device, weights_only=False)

    model = DualNet(args=args, norm_cfg='IN', activation_cfg='LeakyReLU',
                    num_classes=args.num_classes, weight_std=True, self_att=False, cross_att=False)

    if 'model' in checkpoint:
        m = checkpoint['model']
        model.load_state_dict(m.module.state_dict() if hasattr(m, 'module') else m)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)

    test_dataset = BraTSEvalDataSet(args.data_dir, args.data_list)
    testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    print(f"Loaded {len(testloader)} test volumes.")

    print(">>> Pre-loading test set into memory...")
    preloaded = []
    t0 = time.time()
    for index, batch in enumerate(testloader):
        image, image_res, label, size, name, affine = batch
        size_np = size[0].numpy()
        seg_gt = np.asarray(label[0].numpy()[:, :size_np[0], :size_np[1], :size_np[2]], dtype=np.int32)
        mr = np.asarray(nib.load(test_dataset.files[index]["t1ce"]).dataobj) if args.postprocess else None
        preloaded.append({
            'img': image.numpy(),
            'gt': seg_gt,
            'name': name[0],
            'mr': mr
        })
    print(f"Pre-loaded {len(preloaded)} volumes in {time.time()-t0:.1f}s.")

    results_table, results_table_pp = [], []

    for mode_str, mode_desc in MODES_15:
        m_t0 = time.time()
        args.mode = mode_str
        has_t1ce = '2' in mode_str.split(',')
        dice_ET_list, dice_WT_list, dice_TC_list = [], [], []
        pp_ET_list, pp_WT_list, pp_TC_list = [], [], []

        for item in preloaded:
            with torch.no_grad():
                output = predict_sliding(args, model, item['img'], input_size, args.num_classes)

            seg_pred_3class = np.asarray(np.around(output), dtype=np.uint8)
            seg_gt = item['gt']

            d_ET = dice_score(seg_pred_3class[:, :, :, 0], seg_gt[0])
            d_WT = dice_score(seg_pred_3class[:, :, :, 1], seg_gt[1])
            d_TC = dice_score(seg_pred_3class[:, :, :, 2], seg_gt[2])

            dice_ET_list.append(d_ET)
            dice_WT_list.append(d_WT)
            dice_TC_list.append(d_TC)

            if args.postprocess:
                seg = regions_to_label(seg_pred_3class[:, :, :, 0], seg_pred_3class[:, :, :, 1],
                                       seg_pred_3class[:, :, :, 2]).transpose((1, 2, 0))
                mr_case = item['mr'] if has_t1ce else None
                seg = continues_region_extract(seg, mr_case, item['name']).transpose((2, 0, 1))
                pp_et, pp_wt, pp_tc = label_to_regions(seg)
                pp_ET_list.append(dice_score(pp_et, seg_gt[0]))
                pp_WT_list.append(dice_score(pp_wt, seg_gt[1]))
                pp_TC_list.append(dice_score(pp_tc, seg_gt[2]))

        row = make_row(mode_str, mode_desc, dice_ET_list, dice_TC_list, dice_WT_list)
        results_table.append(row)
        msg = f"{mode_desc:<22} | raw ET {row['our_et']:.2f} TC {row['our_tc']:.2f} WT {row['our_wt']:.2f}"
        if args.postprocess:
            row_pp = make_row(mode_str, mode_desc, pp_ET_list, pp_TC_list, pp_WT_list)
            results_table_pp.append(row_pp)
            msg += f" | post ET {row_pp['our_et']:.2f} TC {row_pp['our_tc']:.2f} WT {row_pp['our_wt']:.2f}"
        msg += f" ({time.time()-m_t0:.1f}s)"
        print(msg, flush=True)

    export_report(results_table, "Plan 1 Extended: Raw Predictions (No Post-Processing)", args.output_csv)
    if args.postprocess:
        root, ext = os.path.splitext(args.output_csv)
        export_report(results_table_pp, "Plan 1 Extended: Official Author Post-Processed", root + '_postproc' + (ext or '.csv'))


if __name__ == '__main__':
    main()
