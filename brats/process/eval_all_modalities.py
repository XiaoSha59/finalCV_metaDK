import argparse
import csv
import os
import sys
from math import ceil

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

_PROCESS_DIR = os.path.dirname(os.path.abspath(__file__))
_BRATS_ROOT = os.path.dirname(_PROCESS_DIR)
_REPO_ROOT = os.path.dirname(_BRATS_ROOT)
sys.path.insert(0, _PROCESS_DIR)
sys.path.insert(0, _REPO_ROOT)

from models import DualNet
from brats.data import BraTSEvalDataSet
from utils.metrics import dice_score
from utils.postprocess import continues_region_extract, regions_to_label, label_to_regions

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

# Ground truth from MetaKD Paper Table 1 (BraTS 2018)
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
    """Pad an image up to the target size."""
    deps_missing = target_size[0] - img.shape[2]
    rows_missing = target_size[1] - img.shape[3]
    cols_missing = target_size[2] - img.shape[4]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, deps_missing), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img

def predict_sliding(args, net, img_list, tile_size, classes):
    image, image_res = img_list
    interp = nn.Upsample(size=tile_size, mode='trilinear', align_corners=True)
    image_size = image.shape
    overlap = 1/3

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    full_probs = torch.zeros((classes, image_size[2], image_size[3], image_size[4]))
    count_predictions = torch.zeros((classes, image_size[2], image_size[3], image_size[4]))

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
                padded_prediction, _, _ = net(torch.from_numpy(padded_img).cuda(), mode=args.mode)
                padded_prediction = torch.sigmoid(padded_prediction)

                padded_prediction = interp(padded_prediction).cpu().data
                padded_prediction = padded_prediction[0].cpu()
                prediction = padded_prediction[0:img.shape[2], 0:img.shape[3], 0:img.shape[4], :]
                count_predictions[:, d1:d2, y1:y2, x1:x2] += 1
                full_probs[:, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    full_probs = full_probs.numpy().transpose(1, 2, 3, 0)
    return full_probs

def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate MetaKD across all 15 missing modality combinations.")
    parser.add_argument("--data_dir", type=str, default=os.path.join(_BRATS_ROOT, "data"))
    parser.add_argument("--data_list", type=str, default="BraTS18_test.csv")
    parser.add_argument("--input_size", type=str, default='80,160,160')
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--restore_from", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--weight_std", type=bool, default=True)
    parser.add_argument("--norm_cfg", type=str, default='IN')
    parser.add_argument("--activation_cfg", type=str, default='LeakyReLU')
    parser.add_argument("--output_csv", type=str, default=os.path.join(_BRATS_ROOT, "logs", "brats_table1_results.csv"))
    parser.add_argument("--postprocess", type=lambda v: v.lower() in ('true', '1', 'yes', 'y', 't'), default=True,
                        help="Also report Dice after the author's postprocess (written to *_postproc.csv)")
    return parser.parse_args()


def report(results_table, title, csv_path):
    header = f"{'Modalities':<22} | {'ET Our':<8} {'ET Pap':<8} {'Diff':<8} | {'TC Our':<8} {'TC Pap':<8} {'Diff':<8} | {'WT Our':<8} {'WT Pap':<8} {'Diff':<8} | {'Avg Our':<8} {'Avg Pap':<8}"
    divider = "=" * len(header)
    print("\n" + divider)
    print(title)
    print(divider)
    print(header)
    print("-" * len(header))
    for r in results_table:
        print(f"{r['desc']:<22} | {r['our_et']:<8.2f} {r['paper_et']:<8.2f} {r['diff_et']:+6.2f}% | {r['our_tc']:<8.2f} {r['paper_tc']:<8.2f} {r['diff_tc']:+6.2f}% | {r['our_wt']:<8.2f} {r['paper_wt']:<8.2f} {r['diff_wt']:+6.2f}% | {r['our_avg']:<8.2f} {r['paper_avg']:<8.2f}")

    mean = {k: np.mean([r[k] for r in results_table]) for k in
            ('our_et', 'paper_et', 'our_tc', 'paper_tc', 'our_wt', 'paper_wt', 'our_avg', 'paper_avg')}
    print("-" * len(header))
    print(f"{'AVERAGE (15 combos)':<22} | {mean['our_et']:<8.2f} {mean['paper_et']:<8.2f} {mean['our_et']-mean['paper_et']:+6.2f}% | {mean['our_tc']:<8.2f} {mean['paper_tc']:<8.2f} {mean['our_tc']-mean['paper_tc']:+6.2f}% | {mean['our_wt']:<8.2f} {mean['paper_wt']:<8.2f} {mean['our_wt']-mean['paper_wt']:+6.2f}% | {mean['our_avg']:<8.2f} {mean['paper_avg']:<8.2f}")
    print(divider + "\n")

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Mode', 'Description', 'Our_ET', 'Paper_ET', 'Diff_ET', 'Our_TC', 'Paper_TC', 'Diff_TC', 'Our_WT', 'Paper_WT', 'Diff_WT', 'Our_Avg', 'Paper_Avg'])
        for r in results_table:
            writer.writerow([r['mode'], r['desc'], f"{r['our_et']:.2f}", f"{r['paper_et']:.2f}", f"{r['diff_et']:+.2f}",
                             f"{r['our_tc']:.2f}", f"{r['paper_tc']:.2f}", f"{r['diff_tc']:+.2f}",
                             f"{r['our_wt']:.2f}", f"{r['paper_wt']:.2f}", f"{r['diff_wt']:+.2f}",
                             f"{r['our_avg']:.2f}", f"{r['paper_avg']:.2f}"])
        writer.writerow(['Average', 'Overall 15 Modalities', f"{mean['our_et']:.2f}", f"{mean['paper_et']:.2f}", f"{mean['our_et']-mean['paper_et']:+.2f}",
                         f"{mean['our_tc']:.2f}", f"{mean['paper_tc']:.2f}", f"{mean['our_tc']-mean['paper_tc']:+.2f}",
                         f"{mean['our_wt']:.2f}", f"{mean['paper_wt']:.2f}", f"{mean['our_wt']-mean['paper_wt']:+.2f}",
                         f"{mean['our_avg']:.2f}", f"{mean['paper_avg']:.2f}"])
    print(f"Results exported to: {csv_path}")


def make_row(mode_str, mode_desc, et_list, tc_list, wt_list):
    our = {'et': float(np.mean(et_list)) * 100.0, 'tc': float(np.mean(tc_list)) * 100.0, 'wt': float(np.mean(wt_list)) * 100.0}
    pap = {'et': PAPER_TABLE_1[mode_str]['ET'], 'tc': PAPER_TABLE_1[mode_str]['TC'], 'wt': PAPER_TABLE_1[mode_str]['WT']}
    return {
        'mode': mode_str, 'desc': mode_desc,
        'our_et': our['et'], 'paper_et': pap['et'], 'diff_et': our['et'] - pap['et'],
        'our_tc': our['tc'], 'paper_tc': pap['tc'], 'diff_tc': our['tc'] - pap['tc'],
        'our_wt': our['wt'], 'paper_wt': pap['wt'], 'diff_wt': our['wt'] - pap['wt'],
        'our_avg': (our['et'] + our['tc'] + our['wt']) / 3.0,
        'paper_avg': (pap['et'] + pap['tc'] + pap['wt']) / 3.0,
    }

def main():
    args = get_arguments()
    d, h, w = map(int, args.input_size.split(','))
    input_size = (d, h, w)

    if not os.path.exists(args.restore_from):
        print(f"Checkpoint file not found: {args.restore_from}")
        sys.exit(1)

    print(f"Loading checkpoint from: {args.restore_from}")
    checkpoint = torch.load(args.restore_from, map_location='cuda:0' if torch.cuda.is_available() else 'cpu', weights_only=False)
    if 'model' in checkpoint:
        model = checkpoint['model']
        if isinstance(model, nn.DataParallel):
            model = model.module
    else:
        model = DualNet(args=args, norm_cfg=args.norm_cfg, activation_cfg=args.activation_cfg,
                        num_classes=args.num_classes, weight_std=args.weight_std, self_att=False, cross_att=False)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    test_dataset = BraTSEvalDataSet(args.data_dir, args.data_list)
    testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    print(f"Loaded test dataset: {len(testloader)} cases.")

    results_table, results_table_pp = [], []
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)

    for mode_str, mode_desc in MODES_15:
        args.mode = mode_str
        has_t1ce = '2' in mode_str.split(',')
        dice_ET_list, dice_WT_list, dice_TC_list = [], [], []
        pp_ET_list, pp_WT_list, pp_TC_list = [], [], []

        for index, batch in enumerate(testloader):
            if len(batch) == 6:
                image, image_res, label, size, name, affine = batch
            else:
                print("Error: Test data must have labels for offline Dice evaluation!")
                sys.exit(1)

            size = size[0].numpy()
            with torch.no_grad():
                output = predict_sliding(args, model, [image.numpy(), image_res.numpy()], input_size, args.num_classes)

            seg_pred_3class = np.asarray(np.around(output), dtype=np.uint8)
            seg_gt = np.asarray(label[0].numpy()[:, :size[0], :size[1], :size[2]], dtype=np.int32)

            d_ET = dice_score(seg_pred_3class[:, :, :, 0], seg_gt[0])
            d_WT = dice_score(seg_pred_3class[:, :, :, 1], seg_gt[1])
            d_TC = dice_score(seg_pred_3class[:, :, :, 2], seg_gt[2])

            dice_ET_list.append(d_ET)
            dice_WT_list.append(d_WT)
            dice_TC_list.append(d_TC)

            if args.postprocess:
                # postprocess runs in NIfTI orientation (H, W, D), like the author's saved predictions
                seg = regions_to_label(seg_pred_3class[:, :, :, 0], seg_pred_3class[:, :, :, 1],
                                       seg_pred_3class[:, :, :, 2]).transpose((1, 2, 0))
                mr = np.asarray(nib.load(test_dataset.files[index]["t1ce"]).dataobj) if has_t1ce else None
                seg = continues_region_extract(seg, mr, name[0]).transpose((2, 0, 1))
                pp_et, pp_wt, pp_tc = label_to_regions(seg)
                pp_ET_list.append(dice_score(pp_et, seg_gt[0]))
                pp_WT_list.append(dice_score(pp_wt, seg_gt[1]))
                pp_TC_list.append(dice_score(pp_tc, seg_gt[2]))
                del seg, mr

            del output, seg_pred_3class, seg_gt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        row = make_row(mode_str, mode_desc, dice_ET_list, dice_TC_list, dice_WT_list)
        results_table.append(row)
        msg = f"{mode_desc:<22} | raw ET {row['our_et']:.2f} TC {row['our_tc']:.2f} WT {row['our_wt']:.2f}"
        if args.postprocess:
            row_pp = make_row(mode_str, mode_desc, pp_ET_list, pp_TC_list, pp_WT_list)
            results_table_pp.append(row_pp)
            msg += f" | post ET {row_pp['our_et']:.2f} TC {row_pp['our_tc']:.2f} WT {row_pp['our_wt']:.2f}"
        print(msg, flush=True)

    report(results_table, "BRATS 2018 vs PAPER TABLE 1 (raw prediction, no postprocess)", args.output_csv)
    if args.postprocess:
        root, ext = os.path.splitext(args.output_csv)
        report(results_table_pp, "BRATS 2018 vs PAPER TABLE 1 (after author's postprocess)", root + '_postproc' + (ext or '.csv'))

if __name__ == '__main__':
    main()
