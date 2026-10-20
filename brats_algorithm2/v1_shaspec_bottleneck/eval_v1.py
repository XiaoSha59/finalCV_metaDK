import argparse
import csv
import os
import os.path as osp
import sys
from math import ceil

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

CURRENT_DIR = osp.dirname(osp.abspath(__file__))
REPO_DIR = osp.abspath(osp.join(CURRENT_DIR, ".."))
BRATS_DIR = osp.abspath(osp.join(REPO_DIR, "brats"))

for p in [CURRENT_DIR, REPO_DIR, BRATS_DIR, osp.join(BRATS_DIR, "process")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from v1_models.dual_net_v1 import DualNet_Algorithm2
from v1_utils.postprocess_v1 import adaptive_postprocess_case

try:
    from datasets.brats_dataset import BraTSEvalDataSet
except ImportError:
    try:
        from data.dataset import BraTSEvalDataSet
    except ImportError:
        from brats.data import BraTSEvalDataSet


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


def dice_metric(pred, gt):
    p = (pred > 0).astype(np.float32).ravel()
    g = (gt > 0).astype(np.float32).ravel()
    intersection = np.sum(p * g)
    total = np.sum(p) + np.sum(g)
    if total == 0:
        return 1.0
    return (2.0 * intersection) / (total + 1e-7)


def predict_sliding_eval(model, image, tile_size, num_classes, mode, impute_mode='mean'):
    image_size = image.shape
    overlap = 1.0 / 3.0

    strideHW = ceil(tile_size[1] * (1.0 - overlap))
    strideD = ceil(tile_size[0] * (1.0 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)

    full_probs = torch.zeros((image_size[0], num_classes, image_size[2], image_size[3], image_size[4]),
                             dtype=torch.float32, device=image.device)
    count_predictions = torch.zeros_like(full_probs)

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                x1 = int(col * strideHW)
                y1 = int(row * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                x2 = min(x1 + tile_size[2], image_size[4])
                y2 = min(y1 + tile_size[1], image_size[3])
                d1 = max(int(d2 - tile_size[0]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)
                y1 = max(int(y2 - tile_size[1]), 0)

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                pred, _, _, _, _ = model(img, val=False, mode=mode, impute_mode=impute_mode)
                count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1.0
                full_probs[:, :, d1:d2, y1:y2, x1:x2] += pred

    full_probs /= count_predictions
    return full_probs


def run_evaluation(checkpoint_path, data_dir, test_list, output_dir, use_shaspec=True, impute_mode='mean'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Imputation Mode: {impute_mode.upper()} (Shared-Mean Imputation for missing modalities)")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = DualNet_Algorithm2(num_classes=3, use_shaspec=use_shaspec, impute_mode=impute_mode).to(device)
    if 'model' in checkpoint:
        ckpt_model = checkpoint['model']
        if hasattr(ckpt_model, 'state_dict'):
            model.load_state_dict(ckpt_model.state_dict(), strict=False)
        elif isinstance(ckpt_model, dict):
            model.load_state_dict(ckpt_model, strict=False)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    raw_csv = osp.join(output_dir, f"alg2_table1_{impute_mode}_raw.csv")
    post_csv = osp.join(output_dir, f"alg2_table1_{impute_mode}_postproc.csv")

    test_dataset = BraTSEvalDataSet(data_dir, test_list)
    testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    rows_raw = []
    rows_post = []

    print("\n" + "=" * 80)
    print("EVALUATING ALGORITHM 2 ACROSS ALL 15 INCOMPLETE MODALITY COMBINATIONS")
    print("=" * 80)

    for mode_str, mode_desc in MODES_15:
        dice_ET_raw, dice_WT_raw, dice_TC_raw = [], [], []
        dice_ET_post, dice_WT_post, dice_TC_post = [], [], []

        for index, batch in enumerate(testloader):
            image, image_res, label, size, name, affine = batch
            size = size[0].numpy()
            image = image.to(device)

            with torch.no_grad():
                out = predict_sliding_eval(model, image, (80, 160, 160), 3, mode_str, impute_mode=impute_mode)

            # (1, 3, D, H, W) -> sigmoid and binary threshold
            probs = torch.sigmoid(out).cpu().numpy()[0]
            pred_binary = (probs > 0.5).astype(np.uint8)  # (3, D, H, W)
            gt = label[0].numpy()[:, :size[0], :size[1], :size[2]]  # (3, D, H, W)

            # Crop prediction to valid size
            pred_binary = pred_binary[:, :size[0], :size[1], :size[2]]

            # 1. Raw Dice
            d_et = dice_metric(pred_binary[0], gt[0]) * 100.0
            d_wt = dice_metric(pred_binary[1], gt[1]) * 100.0
            d_tc = dice_metric(pred_binary[2], gt[2]) * 100.0
            dice_ET_raw.append(d_et)
            dice_WT_raw.append(d_wt)
            dice_TC_raw.append(d_tc)

            # 2. Adaptive Post-processed Dice
            cleaned_pred = adaptive_postprocess_case(pred_binary, min_speckle_size=30)
            d_et_p = dice_metric(cleaned_pred[0], gt[0]) * 100.0
            d_wt_p = dice_metric(cleaned_pred[1], gt[1]) * 100.0
            d_tc_p = dice_metric(cleaned_pred[2], gt[2]) * 100.0
            dice_ET_post.append(d_et_p)
            dice_WT_post.append(d_wt_p)
            dice_TC_post.append(d_tc_p)

        paper = PAPER_TABLE_1[mode_str]
        m_et_raw = np.mean(dice_ET_raw)
        m_tc_raw = np.mean(dice_TC_raw)
        m_wt_raw = np.mean(dice_WT_raw)
        m_avg_raw = (m_et_raw + m_tc_raw + m_wt_raw) / 3.0

        m_et_post = np.mean(dice_ET_post)
        m_tc_post = np.mean(dice_TC_post)
        m_wt_post = np.mean(dice_WT_post)
        m_avg_post = (m_et_post + m_tc_post + m_wt_post) / 3.0

        rows_raw.append({
            'Mode': mode_str, 'Desc': mode_desc,
            'Our_ET': f"{m_et_raw:.2f}", 'Our_TC': f"{m_tc_raw:.2f}", 'Our_WT': f"{m_wt_raw:.2f}", 'Our_Avg': f"{m_avg_raw:.2f}",
            'Paper_ET': f"{paper['ET']:.2f}", 'Paper_TC': f"{paper['TC']:.2f}", 'Paper_WT': f"{paper['WT']:.2f}"
        })
        rows_post.append({
            'Mode': mode_str, 'Desc': mode_desc,
            'Our_ET': f"{m_et_post:.2f}", 'Our_TC': f"{m_tc_post:.2f}", 'Our_WT': f"{m_wt_post:.2f}", 'Our_Avg': f"{m_avg_post:.2f}",
            'Paper_ET': f"{paper['ET']:.2f}", 'Paper_TC': f"{paper['TC']:.2f}", 'Paper_WT': f"{paper['WT']:.2f}"
        })

        print(f"[{mode_desc:<20}] Raw: ET={m_et_raw:5.2f}% TC={m_tc_raw:5.2f}% WT={m_wt_raw:5.2f}% | Post: ET={m_et_post:5.2f}% TC={m_tc_post:5.2f}% WT={m_wt_post:5.2f}%", flush=True)

    def save_csv(rows, path):
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {path}", flush=True)

    save_csv(rows_raw, raw_csv)
    save_csv(rows_post, post_csv)
    print("\nBenchmark Complete!", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--test_list", type=str, default="datalist/BraTS18/BraTS18_test.csv")
    parser.add_argument("--output_dir", type=str, default="logs")
    parser.add_argument("--use_shaspec", type=str, default="True")
    parser.add_argument("--impute_mode", type=str, default="mean", choices=["mean", "zero"])
    args = parser.parse_args()

    run_evaluation(
        args.checkpoint,
        args.data_dir,
        args.test_list,
        args.output_dir,
        use_shaspec=(args.use_shaspec.lower() in ('true', '1')),
        impute_mode=args.impute_mode
    )
