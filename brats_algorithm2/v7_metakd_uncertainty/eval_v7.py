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

from v7_models.dual_net_v7 import DualNet_V7
from v7_utils.postprocess_v7 import postprocess_prediction

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

PAPER_TABLE_1 = {
    '0':       {'ET': 47.37, 'TC': 73.01, 'WT': 89.11},
    '1':       {'ET': 44.94, 'TC': 67.16, 'WT': 77.47},
    '2':       {'ET': 76.22, 'TC': 81.98, 'WT': 77.72},
    '3':       {'ET': 46.93, 'TC': 69.72, 'WT': 84.44},
    '0,1':     {'ET': 49.64, 'TC': 75.66, 'WT': 90.23},
    '0,2':     {'ET': 77.49, 'TC': 84.82, 'WT': 90.30},
    '0,3':     {'ET': 50.27, 'TC': 75.88, 'WT': 90.39},
    '1,2':     {'ET': 75.41, 'TC': 81.85, 'WT': 80.59},
    '1,3':     {'ET': 49.20, 'TC': 72.85, 'WT': 86.82},
    '2,3':     {'ET': 76.47, 'TC': 82.35, 'WT': 87.05},
    '0,1,2':   {'ET': 76.99, 'TC': 84.77, 'WT': 90.49},
    '0,1,3':   {'ET': 51.52, 'TC': 76.81, 'WT': 90.87},
    '0,2,3':   {'ET': 77.56, 'TC': 85.04, 'WT': 90.73},
    '1,2,3':   {'ET': 77.29, 'TC': 83.18, 'WT': 88.54},
    '0,1,2,3': {'ET': 77.72, 'TC': 85.18, 'WT': 90.99},
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

                img_tile = image[:, :, x1:x2, y1:y2, z1:z2]
                pred, _, _ = model(img_tile, val=False, mode=mode)
                total_predictions[:, x1:x2, y1:y2, z1:z2] += pred[0]
                count_predictions[:, x1:x2, y1:y2, z1:z2] += 1

    total_predictions /= count_predictions
    return total_predictions


def evaluate_model(model, test_loader, tile_size, output_dir, postprocess=True):
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print("\n" + "=" * 95)
    print(f"| {'Modality Combination':<22} | {'ET Dice':>8} | {'TC Dice':>8} | {'WT Dice':>8} | {'Mean Dice':>10} | {'MetaKD Ref':>10} | {'Delta':>8} |")
    print("=" * 95)

    all_et, all_tc, all_wt, all_mean = [], [], [], []

    for mode, desc in MODES_15:
        mode_et, mode_tc, mode_wt = [], [], []

        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                image, image_res, label, size, name, affine = batch
                size = size[0].numpy()
                image = image.cuda().float()
                gt_bin = (label[0].numpy()[:, :size[0], :size[1], :size[2]] > 0.5).astype(np.uint8)
            else:
                image = batch['image'].cuda().float()
                gt_bin = (batch['label'][0].numpy() > 0.5).astype(np.uint8)
                size = None

            with torch.no_grad():
                pred = predict_sliding_eval(model, image, tile_size, classes=3, mode=mode)
            pred = torch.sigmoid(pred).cpu().numpy()
            pred_bin = (pred > 0.5).astype(np.uint8)

            if size is not None:
                pred_bin = pred_bin[:, :size[0], :size[1], :size[2]]

            if postprocess:
                pred_bin = postprocess_prediction(pred_bin)

            d_et = dice_eval(pred_bin[0], gt_bin[0])
            d_wt = dice_eval(pred_bin[1], gt_bin[1])
            d_tc = dice_eval(pred_bin[2], gt_bin[2])

            mode_et.append(d_et)
            mode_wt.append(d_wt)
            mode_tc.append(d_tc)

        m_et = np.mean(mode_et)
        m_tc = np.mean(mode_tc)
        m_wt = np.mean(mode_wt)
        m_avg = (m_et + m_tc + m_wt) / 3.0

        ref_et = PAPER_TABLE_1[mode]['ET']
        ref_tc = PAPER_TABLE_1[mode]['TC']
        ref_wt = PAPER_TABLE_1[mode]['WT']
        ref_avg = (ref_et + ref_tc + ref_wt) / 3.0
        delta = m_avg - ref_avg

        results[mode] = {
            'desc': desc,
            'ET': m_et,
            'TC': m_tc,
            'WT': m_wt,
            'Mean': m_avg,
            'Ref': ref_avg,
            'Delta': delta
        }

        all_et.append(m_et)
        all_tc.append(m_tc)
        all_wt.append(m_wt)
        all_mean.append(m_avg)

        print(f"| {desc:<22} | {m_et:>8.2f} | {m_tc:>8.2f} | {m_wt:>8.2f} | {m_avg:>10.2f} | {ref_avg:>10.2f} | {delta:>+8.2f} |")

    overall_et = np.mean(all_et)
    overall_tc = np.mean(all_tc)
    overall_wt = np.mean(all_wt)
    overall_avg = np.mean(all_mean)
    overall_ref = 73.27
    overall_delta = overall_avg - overall_ref

    print("=" * 95)
    print(f"| {'AVERAGE (15 MODES)':<22} | {overall_et:>8.2f} | {overall_tc:>8.2f} | {overall_wt:>8.2f} | {overall_avg:>10.2f} | {overall_ref:>10.2f} | {overall_delta:>+8.2f} |")
    print("=" * 95 + "\n")

    # Save to CSV
    csv_path = osp.join(output_dir, "results_v7_table1.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Mode', 'Description', 'ET_Dice', 'TC_Dice', 'WT_Dice', 'Mean_Dice', 'Ref_MetaKD', 'Delta'])
        for mode, d in results.items():
            writer.writerow([mode, d['desc'], f"{d['ET']:.2f}", f"{d['TC']:.2f}", f"{d['WT']:.2f}", f"{d['Mean']:.2f}", f"{d['Ref']:.2f}", f"{d['Delta']:+.2f}"])
        writer.writerow(['ALL', 'Average 15 Modes', f"{overall_et:.2f}", f"{overall_tc:.2f}", f"{overall_wt:.2f}", f"{overall_avg:.2f}", f"{overall_ref:.2f}", f"{overall_delta:+.2f}"])

    print(f"Results saved to: {csv_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate MetaKD-Uncertainty (V7) on 15 Missing Modality Combinations")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to final.pth")
    parser.add_argument("--data_dir", type=str, default=osp.join(BRATS_DIR, "data"))
    parser.add_argument("--test_list", type=str, default="BraTS18_metaVal.csv")
    parser.add_argument("--tile_size", type=str, default="64,64,64")
    parser.add_argument("--output_dir", type=str, default=osp.join(CURRENT_DIR, "results"))
    parser.add_argument("--no_postprocess", action="store_true", default=False)
    args = parser.parse_args()

    tile_size = tuple(map(int, args.tile_size.split(',')))

    model = DualNet_V7(num_classes=3)
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

    evaluate_model(model, test_loader, tile_size, args.output_dir, postprocess=not args.no_postprocess)


if __name__ == '__main__':
    main()
