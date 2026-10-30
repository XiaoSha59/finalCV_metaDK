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

if osp.exists("/home/XiaoSha/finalCV_metaDK"):
    BASE_DIR = "/home/XiaoSha/finalCV_metaDK"
else:
    BASE_DIR = REPO_DIR

for p in [CURRENT_DIR, ALG2_DIR, BASE_DIR, osp.join(BASE_DIR, "models"), osp.join(BASE_DIR, "datasets"), osp.join(BASE_DIR, "utils")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import models
from models.dual_net import DualNet
from brats_algorithm2.v7_metakd_uncertainty.v7_models.dual_net_v7 import DualNet_V7
from brats_algorithm2.v7_metakd_uncertainty.v7_utils.postprocess_v7 import adaptive_postprocess_case

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

METAKD_REPRO_PP = {
    '0':       {'ET': 37.44, 'TC': 66.94, 'WT': 89.35, 'Mean': 64.57},
    '1':       {'ET': 35.51, 'TC': 66.29, 'WT': 74.76, 'Mean': 58.85},
    '2':       {'ET': 71.13, 'TC': 78.68, 'WT': 73.80, 'Mean': 74.54},
    '3':       {'ET': 40.71, 'TC': 62.97, 'WT': 82.86, 'Mean': 62.18},
    '0,1':     {'ET': 42.28, 'TC': 72.55, 'WT': 90.30, 'Mean': 68.38},
    '0,2':     {'ET': 73.74, 'TC': 83.46, 'WT': 90.29, 'Mean': 82.50},
    '0,3':     {'ET': 42.33, 'TC': 68.57, 'WT': 90.27, 'Mean': 67.05},
    '1,2':     {'ET': 72.88, 'TC': 81.20, 'WT': 77.65, 'Mean': 77.24},
    '1,3':     {'ET': 41.30, 'TC': 69.87, 'WT': 85.97, 'Mean': 65.71},
    '2,3':     {'ET': 72.65, 'TC': 82.17, 'WT': 85.90, 'Mean': 80.24},
    '0,1,2':   {'ET': 74.43, 'TC': 84.38, 'WT': 90.60, 'Mean': 83.13},
    '0,1,3':   {'ET': 43.41, 'TC': 71.21, 'WT': 90.43, 'Mean': 68.35},
    '0,2,3':   {'ET': 74.05, 'TC': 83.39, 'WT': 90.79, 'Mean': 82.74},
    '1,2,3':   {'ET': 73.16, 'TC': 82.35, 'WT': 86.36, 'Mean': 80.63},
    '0,1,2,3': {'ET': 74.41, 'TC': 83.67, 'WT': 90.85, 'Mean': 82.98},
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


def evaluate_ensemble(model_a, model_b, test_loader, tile_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    records_blend = {}
    records_routing = {}
    records_weighted = {}

    print("\n" + "=" * 135)
    print(f"| {'Modality Combination':<20} | {'Repro MetaKD':>12} | {'Job 7 (V7)':>11} | {'Ensemble Blend':>15} | {'Ensemble Route':>15} | {'Weighted Blend':>15} |")
    print("=" * 135)

    all_res_blend = []
    all_res_routing = []
    all_res_weighted = []

    for mode, desc in MODES_15:
        mode_et_blend, mode_tc_blend, mode_wt_blend = [], [], []
        mode_et_route, mode_tc_route, mode_wt_route = [], [], []
        mode_et_wb, mode_tc_wb, mode_wt_wb = [], [], []

        for index, batch in enumerate(test_loader):
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
                pred_a = predict_sliding_eval(model_a, image, tile_size, classes=3, mode=mode)
                pred_b = predict_sliding_eval(model_b, image, tile_size, classes=3, mode=mode)

            prob_a = torch.sigmoid(pred_a).cpu().numpy()
            prob_b = torch.sigmoid(pred_b).cpu().numpy()

            if size is not None:
                prob_a = prob_a[:, :size[0], :size[1], :size[2]]
                prob_b = prob_b[:, :size[0], :size[1], :size[2]]

            # -----------------------------------------------------------------
            # Strategy 1: Soft Average (50-50 Blend)
            # -----------------------------------------------------------------
            prob_blend = 0.5 * prob_a + 0.5 * prob_b
            pred_blend_bin = (prob_blend > 0.5).astype(np.uint8)
            pp_blend = adaptive_postprocess_case(pred_blend_bin, min_speckle_size=30)

            mode_et_blend.append(dice_eval(pp_blend[0], gt_bin[0]))
            mode_wt_blend.append(dice_eval(pp_blend[1], gt_bin[1]))
            mode_tc_blend.append(dice_eval(pp_blend[2], gt_bin[2]))

            # -----------------------------------------------------------------
            # Strategy 2: Region-Routing (WT from MetaKD, TC and ET from Job 7)
            # -----------------------------------------------------------------
            pred_route_bin = np.zeros_like(pred_blend_bin)
            pred_route_bin[0] = (prob_b[0] > 0.5).astype(np.uint8)  # ET: Job 7
            pred_route_bin[1] = (prob_a[1] > 0.5).astype(np.uint8)  # WT: MetaKD
            pred_route_bin[2] = (prob_b[2] > 0.5).astype(np.uint8)  # TC: Job 7
            pp_route = adaptive_postprocess_case(pred_route_bin, min_speckle_size=30)

            mode_et_route.append(dice_eval(pp_route[0], gt_bin[0]))
            mode_wt_route.append(dice_eval(pp_route[1], gt_bin[1]))
            mode_tc_route.append(dice_eval(pp_route[2], gt_bin[2]))

            # -----------------------------------------------------------------
            # Strategy 3: Weighted Probability Blending
            # (ET: 0.40 A + 0.60 B, WT: 0.70 A + 0.30 B, TC: 0.30 A + 0.70 B)
            # -----------------------------------------------------------------
            prob_wb = np.zeros_like(prob_a)
            prob_wb[0] = 0.40 * prob_a[0] + 0.60 * prob_b[0]
            prob_wb[1] = 0.70 * prob_a[1] + 0.30 * prob_b[1]
            prob_wb[2] = 0.30 * prob_a[2] + 0.70 * prob_b[2]
            pred_wb_bin = (prob_wb > 0.5).astype(np.uint8)
            pp_wb = adaptive_postprocess_case(pred_wb_bin, min_speckle_size=30)

            mode_et_wb.append(dice_eval(pp_wb[0], gt_bin[0]))
            mode_wt_wb.append(dice_eval(pp_wb[1], gt_bin[1]))
            mode_tc_wb.append(dice_eval(pp_wb[2], gt_bin[2]))

        # Mean computations
        m_et_b, m_tc_b, m_wt_b = np.mean(mode_et_blend), np.mean(mode_tc_blend), np.mean(mode_wt_blend)
        mean_b = (m_et_b + m_tc_b + m_wt_b) / 3.0

        m_et_r, m_tc_r, m_wt_r = np.mean(mode_et_route), np.mean(mode_tc_route), np.mean(mode_wt_route)
        mean_r = (m_et_r + m_tc_r + m_wt_r) / 3.0

        m_et_wb, m_tc_wb, m_wt_wb = np.mean(mode_et_wb), np.mean(mode_tc_wb), np.mean(mode_wt_wb)
        mean_wb = (m_et_wb + m_tc_wb + m_wt_wb) / 3.0

        repro_m = METAKD_REPRO_PP[mode]['Mean']

        records_blend[mode] = {'desc': desc, 'ET': m_et_b, 'TC': m_tc_b, 'WT': m_wt_b, 'Mean': mean_b}
        records_routing[mode] = {'desc': desc, 'ET': m_et_r, 'TC': m_tc_r, 'WT': m_wt_r, 'Mean': mean_r}
        records_weighted[mode] = {'desc': desc, 'ET': m_et_wb, 'TC': m_tc_wb, 'WT': m_wt_wb, 'Mean': mean_wb}

        all_res_blend.append((m_et_b, m_tc_b, m_wt_b, mean_b))
        all_res_routing.append((m_et_r, m_tc_r, m_wt_r, mean_r))
        all_res_weighted.append((m_et_wb, m_tc_wb, m_wt_wb, mean_wb))

        print(f"| {desc:<20} | {repro_m:>11.2f}% | {'--':>11} | {mean_b:>14.2f}% | {mean_r:>14.2f}% | {mean_wb:>14.2f}% |")

    # Overall averages
    ov_blend = np.mean([x[3] for x in all_res_blend])
    ov_route = np.mean([x[3] for x in all_res_routing])
    ov_wb = np.mean([x[3] for x in all_res_weighted])
    ov_repro = 73.27

    print("=" * 135)
    print(f"| {'AVERAGE (15 MODES)':<20} | {ov_repro:>11.2f}% | {'71.29%':>11} | {ov_blend:>14.2f}% | {ov_route:>14.2f}% | {ov_wb:>14.2f}% |")
    print("=" * 135)

    def write_csv(records, filename, ov_mean):
        csv_path = osp.join(output_dir, filename)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Mode', 'Description', 'ET_Dice', 'TC_Dice', 'WT_Dice', 'Mean_Dice', 'MetaKD_Repro', 'Delta'])
            for m, d in records.items():
                r_m = METAKD_REPRO_PP[m]['Mean']
                diff = d['Mean'] - r_m
                writer.writerow([m, d['desc'], f"{d['ET']:.2f}", f"{d['TC']:.2f}", f"{d['WT']:.2f}", f"{d['Mean']:.2f}", f"{r_m:.2f}", f"{diff:+.2f}"])
            writer.writerow(['ALL', 'Average 15 Modes', '', '', '', f"{ov_mean:.2f}", f"{ov_repro:.2f}", f"{ov_mean - ov_repro:+.2f}"])
        print(f"Saved: {csv_path}")

    write_csv(records_blend, 'results_ensemble_blend5050.csv', ov_blend)
    write_csv(records_routing, 'results_ensemble_routing.csv', ov_route)
    write_csv(records_weighted, 'results_ensemble_weighted.csv', ov_wb)

    # Master comparison CSV
    master_csv = osp.join(output_dir, 'results_ensemble_comparison_master.csv')
    with open(master_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Mode', 'Description', 'MetaKD_Repro', 'Job7_Uncertainty', 'Ensemble_Blend5050', 'Ensemble_Routing', 'Ensemble_WeightedBlend'])
        for m, desc in MODES_15:
            writer.writerow([
                m, desc,
                f"{METAKD_REPRO_PP[m]['Mean']:.2f}",
                '--',
                f"{records_blend[m]['Mean']:.2f}",
                f"{records_routing[m]['Mean']:.2f}",
                f"{records_weighted[m]['Mean']:.2f}"
            ])
        writer.writerow(['ALL', 'Average 15 Modes', f"{ov_repro:.2f}", '71.29', f"{ov_blend:.2f}", f"{ov_route:.2f}", f"{ov_wb:.2f}"])
    print(f"Master comparison saved: {master_csv}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dual-Model Ensemble (MetaKD + Job 7)")
    parser.add_argument("--ckpt_metakd", type=str, default="/home/XiaoSha/finalCV_metaDK/snapshots/BraTS18_phase2_115k/final.pth")
    parser.add_argument("--ckpt_job7", type=str, default="/home/XiaoSha/finalCV_metaDK/snapshots/BraTS18_phase2_v7_metakd_uncertainty/final.pth")
    parser.add_argument("--data_dir", type=str, default=osp.join(BASE_DIR, "data"))
    parser.add_argument("--test_list", type=str, default="datalist/BraTS18/BraTS18_test.csv")
    parser.add_argument("--tile_size", type=str, default="64,64,64")
    parser.add_argument("--output_dir", type=str, default=osp.join(CURRENT_DIR, "results_57test"))
    args = parser.parse_args()

    tile_size = tuple(map(int, args.tile_size.split(',')))

    print("=================================================================")
    print("Dual-Model Complementary Ensemble Pipeline")
    print(f"Model A (MetaKD): {args.ckpt_metakd}")
    print(f"Model B (Job 7):  {args.ckpt_job7}")
    print(f"Output Directory: {args.output_dir}")
    print("=================================================================")

    # Load Model A (MetaKD)
    print("Loading Model A (MetaKD)...")
    model_a = DualNet(num_classes=3)
    ckpt_a = torch.load(args.ckpt_metakd, map_location='cpu', weights_only=False)
    state_a = ckpt_a.get('model', ckpt_a)
    if hasattr(state_a, 'state_dict'):
        state_a = state_a.state_dict()
    model_a.load_state_dict(state_a, strict=False)
    model_a.cuda().eval()

    # Load Model B (Job 7)
    print("Loading Model B (Job 7 Uncertainty)...")
    model_b = DualNet_V7(num_classes=3)
    ckpt_b = torch.load(args.ckpt_job7, map_location='cpu', weights_only=False)
    state_b = ckpt_b.get('model', ckpt_b)
    if hasattr(state_b, 'state_dict'):
        state_b = state_b.state_dict()
    model_b.load_state_dict(state_b, strict=False)
    model_b.cuda().eval()

    if not osp.isabs(args.test_list):
        test_list_path = osp.join(BASE_DIR, args.test_list)
    else:
        test_list_path = args.test_list

    test_dataset = BraTSEvalDataSet(args.data_dir, test_list_path)
    print(f"Loaded {len(test_dataset)} test images.")
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    evaluate_ensemble(model_a, model_b, test_loader, tile_size, args.output_dir)


if __name__ == '__main__':
    main()
