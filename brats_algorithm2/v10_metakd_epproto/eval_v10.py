import argparse
import csv
import os
import os.path as osp
import sys
from math import ceil

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

CURRENT_DIR = osp.dirname(osp.abspath(__file__))
ALG2_DIR = osp.abspath(osp.join(CURRENT_DIR, ".."))
REPO_DIR = osp.abspath(osp.join(CURRENT_DIR, "..", ".."))
BRATS_DIR = osp.abspath(osp.join(REPO_DIR, "brats"))

for p in [CURRENT_DIR, ALG2_DIR, REPO_DIR, BRATS_DIR, osp.join(BRATS_DIR, "process")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from v10_models.dual_net_v10 import DualNet_V10
from v10_utils.postprocess_v10 import adaptive_postprocess_case

try:
    from datasets.brats_dataset import BraTSEvalDataSet
except ImportError:
    try:
        from data.dataset import BraTSEvalDataSet
    except ImportError:
        from brats.data.dataset import BraTSEvalDataSet


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

METAKD_REPRO_RAW = {
    '0':       {'ET': 37.51, 'TC': 66.92, 'WT': 89.24, 'Mean': 64.56},
    '1':       {'ET': 35.55, 'TC': 66.21, 'WT': 74.42, 'Mean': 58.73},
    '2':       {'ET': 68.08, 'TC': 78.28, 'WT': 73.67, 'Mean': 73.34},
    '3':       {'ET': 38.90, 'TC': 62.88, 'WT': 82.67, 'Mean': 61.48},
    '0,1':     {'ET': 42.28, 'TC': 72.57, 'WT': 90.28, 'Mean': 68.38},
    '0,2':     {'ET': 68.89, 'TC': 83.16, 'WT': 90.18, 'Mean': 80.74},
    '0,3':     {'ET': 42.33, 'TC': 68.56, 'WT': 90.17, 'Mean': 67.02},
    '1,2':     {'ET': 67.62, 'TC': 81.16, 'WT': 77.59, 'Mean': 75.46},
    '1,3':     {'ET': 41.38, 'TC': 69.86, 'WT': 85.80, 'Mean': 65.68},
    '2,3':     {'ET': 71.96, 'TC': 81.24, 'WT': 85.79, 'Mean': 79.66},
    '0,1,2':   {'ET': 71.37, 'TC': 84.35, 'WT': 90.49, 'Mean': 82.07},
    '0,1,3':   {'ET': 43.41, 'TC': 71.23, 'WT': 90.39, 'Mean': 68.34},
    '0,2,3':   {'ET': 69.22, 'TC': 82.88, 'WT': 90.67, 'Mean': 80.93},
    '1,2,3':   {'ET': 70.97, 'TC': 82.52, 'WT': 86.31, 'Mean': 79.93},
    '0,1,2,3': {'ET': 71.32, 'TC': 83.89, 'WT': 90.76, 'Mean': 81.99},
}


def dice_eval(pred_bin, target_bin):
    num = 2.0 * np.sum(pred_bin * target_bin)
    den = np.sum(pred_bin) + np.sum(target_bin)
    if den == 0:
        return 100.0
    return (num / den) * 100.0


def predict_sliding_eval(model, image, tile_size, classes, mode='0,1,2,3'):
    image_size = image.shape
    overlap = 1.0 / 3.0

    stride_x = ceil(tile_size[0] * (1 - overlap))
    stride_y = ceil(tile_size[1] * (1 - overlap))
    stride_z = ceil(tile_size[2] * (1 - overlap))

    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride_x)) + 1
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride_y)) + 1
    num_slcs = int(ceil((image_size[4] - tile_size[2]) / stride_z)) + 1

    total_predictions = torch.zeros((classes, image_size[2], image_size[3], image_size[4]), device=image.device)
    count_predictions = torch.zeros((classes, image_size[2], image_size[3], image_size[4]), device=image.device)

    for row in range(num_rows):
        for col in range(num_cols):
            for slc in range(num_slcs):
                x1 = int(row * stride_x)
                y1 = int(col * stride_y)
                z1 = int(slc * stride_z)
                x2 = min(x1 + tile_size[0], image_size[2])
                y2 = min(y1 + tile_size[1], image_size[3])
                z2 = min(z1 + tile_size[2], image_size[4])
                x1 = max(int(x2 - tile_size[0]), 0)
                y1 = max(int(y2 - tile_size[1]), 0)
                z1 = max(int(z2 - tile_size[2]), 0)

                tile = image[:, :, x1:x2, y1:y2, z1:z2]
                out, _, _ = model(tile, val=False, mode=mode)
                count_predictions[:, x1:x2, y1:y2, z1:z2] += 1.0
                total_predictions[:, x1:x2, y1:y2, z1:z2] += torch.sigmoid(out).squeeze(0)

    total_predictions = total_predictions / count_predictions
    return (total_predictions > 0.5).byte().cpu().numpy()


def evaluate_model(model, test_loader, tile_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results_raw = {}
    results_std_pp = {}

    all_raw_et, all_raw_tc, all_raw_wt, all_raw_mean = [], [], [], []
    all_pp_et, all_pp_tc, all_pp_wt, all_pp_mean = [], [], [], []

    print("\n" + "=" * 130)
    print(f"| {'Modality Combination':<20} | {'Raw ET':>7} {'Raw TC':>7} {'Raw WT':>7} {'Raw Mean':>9} | {'PP ET':>7} {'PP TC':>7} {'PP WT':>7} {'PP Mean':>9} | {'MK Repro':>9} | {'Raw Diff':>8} |")
    print("=" * 130)

    for mode, desc in MODES_15:
        mode_raw_et, mode_raw_tc, mode_raw_wt = [], [], []
        mode_pp_et, mode_pp_tc, mode_pp_wt = [], [], []

        for index, batch in enumerate(test_loader):
            if isinstance(batch, (list, tuple)):
                image = batch[0].cuda().float()
                gt_bin = (batch[2][0].numpy() > 0.5).astype(np.uint8)
                size = batch[3][0].numpy() if len(batch) > 3 and hasattr(batch[3][0], 'numpy') else None
            else:
                image = batch['image'].cuda().float()
                gt_bin = (batch['label'][0].numpy() > 0.5).astype(np.uint8)
                size = None

            with torch.no_grad():
                pred_bin = predict_sliding_eval(model, image, tile_size, classes=3, mode=mode)

            if size is not None:
                pred_bin = pred_bin[:, :size[0], :size[1], :size[2]]

            # 1. Raw Metrics
            d_raw_et = dice_eval(pred_bin[0], gt_bin[0])
            d_raw_wt = dice_eval(pred_bin[1], gt_bin[1])
            d_raw_tc = dice_eval(pred_bin[2], gt_bin[2])

            mode_raw_et.append(d_raw_et)
            mode_raw_wt.append(d_raw_wt)
            mode_raw_tc.append(d_raw_tc)

            # 2. Standard Fair Post-Processing (Topological hierarchy + min speckle 30)
            std_pred = adaptive_postprocess_case(pred_bin, min_speckle_size=30)
            d_pp_et = dice_eval(std_pred[0], gt_bin[0])
            d_pp_wt = dice_eval(std_pred[1], gt_bin[1])
            d_pp_tc = dice_eval(std_pred[2], gt_bin[2])

            mode_pp_et.append(d_pp_et)
            mode_pp_wt.append(d_pp_wt)
            mode_pp_tc.append(d_pp_tc)

        m_raw_et = np.mean(mode_raw_et)
        m_raw_tc = np.mean(mode_raw_tc)
        m_raw_wt = np.mean(mode_raw_wt)
        m_raw_avg = (m_raw_et + m_raw_tc + m_raw_wt) / 3.0

        m_pp_et = np.mean(mode_pp_et)
        m_pp_tc = np.mean(mode_pp_tc)
        m_pp_wt = np.mean(mode_pp_wt)
        m_pp_avg = (m_pp_et + m_pp_tc + m_pp_wt) / 3.0

        mk_ref = METAKD_REPRO_RAW[mode]['Mean']
        raw_diff = m_raw_avg - mk_ref

        results_raw[mode] = {'desc': desc, 'ET': m_raw_et, 'TC': m_raw_tc, 'WT': m_raw_wt, 'Mean': m_raw_avg, 'MK_Ref': mk_ref, 'Diff': raw_diff}
        results_std_pp[mode] = {'desc': desc, 'ET': m_pp_et, 'TC': m_pp_tc, 'WT': m_pp_wt, 'Mean': m_pp_avg, 'MK_Ref': mk_ref, 'Diff': m_pp_avg - mk_ref}

        all_raw_et.append(m_raw_et)
        all_raw_tc.append(m_raw_tc)
        all_raw_wt.append(m_raw_wt)
        all_raw_mean.append(m_raw_avg)

        all_pp_et.append(m_pp_et)
        all_pp_tc.append(m_pp_tc)
        all_pp_wt.append(m_pp_wt)
        all_pp_mean.append(m_pp_avg)

        print(f"| {desc:<20} | {m_raw_et:>7.2f} {m_raw_tc:>7.2f} {m_raw_wt:>7.2f} {m_raw_avg:>9.2f} | {m_pp_et:>7.2f} {m_pp_tc:>7.2f} {m_pp_wt:>7.2f} {m_pp_avg:>9.2f} | {mk_ref:>9.2f} | {raw_diff:>+8.2f} |")

    overall_raw_et = np.mean(all_raw_et)
    overall_raw_tc = np.mean(all_raw_tc)
    overall_raw_wt = np.mean(all_raw_wt)
    overall_raw_avg = np.mean(all_raw_mean)

    overall_pp_et = np.mean(all_pp_et)
    overall_pp_tc = np.mean(all_pp_tc)
    overall_pp_wt = np.mean(all_pp_wt)
    overall_pp_avg = np.mean(all_pp_mean)

    overall_mk_ref = 72.55
    overall_raw_diff = overall_raw_avg - overall_mk_ref
    overall_pp_diff = overall_pp_avg - overall_mk_ref

    print("=" * 130)
    print(f"| {'AVERAGE (15 MODES)':<20} | {overall_raw_et:>7.2f} {overall_raw_tc:>7.2f} {overall_raw_wt:>7.2f} {overall_raw_avg:>9.2f} | {overall_pp_et:>7.2f} {overall_pp_tc:>7.2f} {overall_pp_wt:>7.2f} {overall_pp_avg:>9.2f} | {overall_mk_ref:>9.2f} | {overall_raw_diff:>+8.2f} |")
    print("=" * 130 + "\n")

    # 1. Save Raw CSV
    csv_raw = osp.join(output_dir, "results_v10_table1_raw.csv")
    with open(csv_raw, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Mode', 'Description', 'ET_Dice', 'TC_Dice', 'WT_Dice', 'Mean_Dice', 'MetaKD_Repro_Raw', 'Diff_vs_MetaKD'])
        for mode, d in results_raw.items():
            writer.writerow([mode, d['desc'], f"{d['ET']:.2f}", f"{d['TC']:.2f}", f"{d['WT']:.2f}", f"{d['Mean']:.2f}", f"{d['MK_Ref']:.2f}", f"{d['Diff']:+.2f}"])
        writer.writerow(['ALL', 'Average 15 Modes', f"{overall_raw_et:.2f}", f"{overall_raw_tc:.2f}", f"{overall_raw_wt:.2f}", f"{overall_raw_avg:.2f}", f"{overall_mk_ref:.2f}", f"{overall_raw_diff:+.2f}"])

    # 2. Save Standard Postprocessed CSV
    for fname in ["results_v10_table1_standard_pp.csv", "results_v10_table1.csv"]:
        csv_pp = osp.join(output_dir, fname)
        with open(csv_pp, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Mode', 'Description', 'ET_Dice', 'TC_Dice', 'WT_Dice', 'Mean_Dice', 'MetaKD_Repro_Raw', 'Diff_vs_MetaKD'])
            for mode, d in results_std_pp.items():
                writer.writerow([mode, d['desc'], f"{d['ET']:.2f}", f"{d['TC']:.2f}", f"{d['WT']:.2f}", f"{d['Mean']:.2f}", f"{d['MK_Ref']:.2f}", f"{d['Diff']:+.2f}"])
            writer.writerow(['ALL', 'Average 15 Modes', f"{overall_pp_et:.2f}", f"{overall_pp_tc:.2f}", f"{overall_pp_wt:.2f}", f"{overall_pp_avg:.2f}", f"{overall_mk_ref:.2f}", f"{overall_pp_diff:+.2f}"])

    print(f"Results saved to: {output_dir}")
    return results_raw, results_std_pp


def main():
    parser = argparse.ArgumentParser(description="Evaluate MetaKD-EPProto (V10) on 57 Test Cases Across 15 Modality Combinations")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to final.pth")
    parser.add_argument("--data_dir", type=str, default=osp.join(BRATS_DIR, "data"))
    parser.add_argument("--test_list", type=str, default="datalist/BraTS18/BraTS18_test.csv")
    parser.add_argument("--tile_size", type=str, default="64,64,64")
    parser.add_argument("--output_dir", type=str, default=osp.join(CURRENT_DIR, "results_57test"))
    args = parser.parse_args()

    tile_size = tuple(map(int, args.tile_size.split(',')))
    model = DualNet_V10(num_classes=3)
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt)
    if hasattr(state, 'state_dict'):
        state = state.state_dict()
    model.load_state_dict(state, strict=False)
    model.cuda()
    model.eval()

    test_dataset = BraTSEvalDataSet(args.data_dir, args.test_list)
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    evaluate_model(model, test_loader, tile_size, args.output_dir)


if __name__ == '__main__':
    main()
