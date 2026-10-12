import argparse
import os
import sys
import csv
from math import ceil
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

from models import DualNet
from datasets import BraTSEvalDataSet
from utils.metrics import dice_score

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
    parser.add_argument("--data_dir", type=str, default='./datalist/')
    parser.add_argument("--data_list", type=str, default='BraTS18/BraTS18_test.csv')
    parser.add_argument("--input_size", type=str, default='80,160,160')
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--restore_from", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--weight_std", type=bool, default=True)
    parser.add_argument("--norm_cfg", type=str, default='IN')
    parser.add_argument("--activation_cfg", type=str, default='LeakyReLU')
    parser.add_argument("--output_csv", type=str, default='logs/brats_table1_results.csv')
    return parser.parse_args()

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

    testloader = data.DataLoader(
        BraTSEvalDataSet(args.data_dir, args.data_list),
        batch_size=1, shuffle=False, pin_memory=True
    )
    print(f"Loaded test dataset: {len(testloader)} cases.")

    results_table = []
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    header = f"{'Modalities':<22} | {'ET Our':<8} {'ET Pap':<8} {'Diff':<8} | {'TC Our':<8} {'TC Pap':<8} {'Diff':<8} | {'WT Our':<8} {'WT Pap':<8} {'Diff':<8} | {'Avg Our':<8} {'Avg Pap':<8}"
    divider = "=" * len(header)
    print("\n" + divider)
    print("🏆 BRATS 2018 EVALUATION VS PAPER TABLE 1")
    print(divider)
    print(header)
    print("-" * len(header))

    for mode_str, mode_desc in MODES_15:
        args.mode = mode_str
        dice_ET_list, dice_WT_list, dice_TC_list = [], [], []

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

            del output, seg_pred_3class, seg_gt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_et = float(np.mean(dice_ET_list)) * 100.0
        avg_tc = float(np.mean(dice_TC_list)) * 100.0
        avg_wt = float(np.mean(dice_WT_list)) * 100.0
        avg_our = (avg_et + avg_tc + avg_wt) / 3.0

        p_et = PAPER_TABLE_1[mode_str]['ET']
        p_tc = PAPER_TABLE_1[mode_str]['TC']
        p_wt = PAPER_TABLE_1[mode_str]['WT']
        p_avg = (p_et + p_tc + p_wt) / 3.0

        diff_et = avg_et - p_et
        diff_tc = avg_tc - p_tc
        diff_wt = avg_wt - p_wt

        results_table.append({
            'mode': mode_str,
            'desc': mode_desc,
            'our_et': avg_et, 'paper_et': p_et, 'diff_et': diff_et,
            'our_tc': avg_tc, 'paper_tc': p_tc, 'diff_tc': diff_tc,
            'our_wt': avg_wt, 'paper_wt': p_wt, 'diff_wt': diff_wt,
            'our_avg': avg_our, 'paper_avg': p_avg
        })

        print(f"{mode_desc:<22} | {avg_et:<8.2f} {p_et:<8.2f} {diff_et:+6.2f}% | {avg_tc:<8.2f} {p_tc:<8.2f} {diff_tc:+6.2f}% | {avg_wt:<8.2f} {p_wt:<8.2f} {diff_wt:+6.2f}% | {avg_our:<8.2f} {p_avg:<8.2f}")

    # Summary Average
    mean_our_et = np.mean([r['our_et'] for r in results_table])
    mean_pap_et = np.mean([r['paper_et'] for r in results_table])
    mean_our_tc = np.mean([r['our_tc'] for r in results_table])
    mean_pap_tc = np.mean([r['paper_tc'] for r in results_table])
    mean_our_wt = np.mean([r['our_wt'] for r in results_table])
    mean_pap_wt = np.mean([r['paper_wt'] for r in results_table])
    mean_our_avg = np.mean([r['our_avg'] for r in results_table])
    mean_pap_avg = np.mean([r['paper_avg'] for r in results_table])

    print("-" * len(header))
    print(f"{'AVERAGE (15 combos)':<22} | {mean_our_et:<8.2f} {mean_pap_et:<8.2f} {mean_our_et-mean_pap_et:+6.2f}% | {mean_our_tc:<8.2f} {mean_pap_tc:<8.2f} {mean_our_tc-mean_pap_tc:+6.2f}% | {mean_our_wt:<8.2f} {mean_pap_wt:<8.2f} {mean_our_wt-mean_pap_wt:+6.2f}% | {mean_our_avg:<8.2f} {mean_pap_avg:<8.2f}")
    print(divider + "\n")

    # Export to CSV
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Mode', 'Description', 'Our_ET', 'Paper_ET', 'Diff_ET', 'Our_TC', 'Paper_TC', 'Diff_TC', 'Our_WT', 'Paper_WT', 'Diff_WT', 'Our_Avg', 'Paper_Avg'])
        for r in results_table:
            writer.writerow([r['mode'], r['desc'], f"{r['our_et']:.2f}", f"{r['paper_et']:.2f}", f"{r['diff_et']:+.2f}",
                             f"{r['our_tc']:.2f}", f"{r['paper_tc']:.2f}", f"{r['diff_tc']:+.2f}",
                             f"{r['our_wt']:.2f}", f"{r['paper_wt']:.2f}", f"{r['diff_wt']:+.2f}",
                             f"{r['our_avg']:.2f}", f"{r['paper_avg']:.2f}"])
        writer.writerow(['Average', 'Overall 15 Modalities', f"{mean_our_et:.2f}", f"{mean_pap_et:.2f}", f"{mean_our_et-mean_pap_et:+.2f}",
                         f"{mean_our_tc:.2f}", f"{mean_pap_tc:.2f}", f"{mean_our_tc-mean_pap_tc:+.2f}",
                         f"{mean_our_wt:.2f}", f"{mean_pap_wt:.2f}", f"{mean_our_wt-mean_pap_wt:+.2f}",
                         f"{mean_our_avg:.2f}", f"{mean_pap_avg:.2f}"])
    print(f"Results exported to: {args.output_csv}")

if __name__ == '__main__':
    main()
