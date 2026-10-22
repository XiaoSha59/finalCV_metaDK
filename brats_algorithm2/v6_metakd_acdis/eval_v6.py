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

from v6_models.dual_net_v6 import DualNet_V6
from v6_utils.postprocess_v6 import adaptive_postprocess_case

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
    '1,3':     {'ET': 49.77, 'TC': 74.40, 'WT': 87.05},
    '2,3':     {'ET': 78.24, 'TC': 84.84, 'WT': 86.18},
    '0,1,2':   {'ET': 77.66, 'TC': 85.36, 'WT': 90.70},
    '0,1,3':   {'ET': 51.05, 'TC': 76.73, 'WT': 90.59},
    '0,2,3':   {'ET': 75.58, 'TC': 85.08, 'WT': 90.87},
    '1,2,3':   {'ET': 75.87, 'TC': 84.57, 'WT': 86.94},
    '0,1,2,3': {'ET': 79.74, 'TC': 86.20, 'WT': 90.92},
}


def dice_metric(pred, label):
    pred = (pred > 0).astype(np.float32)
    label = (label > 0).astype(np.float32)
    intersection = (pred * label).sum()
    total = pred.sum() + label.sum()
    if total == 0:
        return 1.0
    return float((2.0 * intersection) / (total + 1e-7))


def predict_sliding_eval(model, image, tile_size, num_classes, mode):
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
                pred, _, _ = model(img, val=False, mode=mode)
                count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1.0
                full_probs[:, :, d1:d2, y1:y2, x1:x2] += pred

    full_probs /= count_predictions
    return full_probs


def resolve_list_path(data_dir, list_name):
    if osp.isabs(list_name) and osp.isfile(list_name):
        return list_name
    candidates = [
        osp.abspath(list_name),
        osp.abspath(osp.join("datalist", "BraTS18", osp.basename(list_name))),
        osp.abspath(osp.join(REPO_DIR, "datalist", "BraTS18", osp.basename(list_name))),
        osp.abspath(osp.join(data_dir, list_name)),
        osp.abspath(osp.join(data_dir, osp.basename(list_name))),
        osp.abspath(osp.join(data_dir, "datalist", "BraTS18", osp.basename(list_name))),
        osp.abspath(osp.join(BRATS_DIR, "data", osp.basename(list_name))),
    ]
    for c in candidates:
        if osp.isfile(c):
            print(f">>> Resolved list '{list_name}' -> '{c}'")
            return c
    return osp.abspath(osp.join(data_dir, list_name))


def run_evaluation(checkpoint_path, data_dir, test_list, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = DualNet_V6(num_classes=3).to(device)
    if 'model' in checkpoint:
        ckpt_model = checkpoint['model']
        if hasattr(ckpt_model, 'state_dict'):
            model.load_state_dict(ckpt_model.state_dict(), strict=False)
        elif isinstance(ckpt_model, dict):
            model.load_state_dict(ckpt_model, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    raw_csv = osp.join(output_dir, "v6_table1_raw.csv")
    post_csv = osp.join(output_dir, "v6_table1_postproc.csv")
    summary_md = osp.join(output_dir, "v6_table1_summary.md")

    test_list_resolved = resolve_list_path(data_dir, test_list)
    test_dataset = BraTSEvalDataSet(data_dir, test_list_resolved)
    testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    rows_raw = []
    rows_post = []

    print("\n" + "=" * 80)
    print("EVALUATING METAKD-ACDIS (V6) ACROSS ALL 15 INCOMPLETE MODALITY COMBINATIONS")
    print("=" * 80)

    for mode_str, mode_desc in MODES_15:
        dice_ET_raw, dice_WT_raw, dice_TC_raw = [], [], []
        dice_ET_post, dice_WT_post, dice_TC_post = [], [], []

        for index, batch in enumerate(testloader):
            image, image_res, label, size, name, affine = batch
            size = size[0].numpy()
            image = image.to(device)

            with torch.no_grad():
                out = predict_sliding_eval(model, image, (80, 160, 160), 3, mode_str)

            probs = torch.sigmoid(out).cpu().numpy()[0]
            pred_binary = (probs > 0.5).astype(np.uint8)
            gt = label[0].numpy()[:, :size[0], :size[1], :size[2]]

            pred_binary = pred_binary[:, :size[0], :size[1], :size[2]]

            d_et = dice_metric(pred_binary[0], gt[0]) * 100.0
            d_wt = dice_metric(pred_binary[1], gt[1]) * 100.0
            d_tc = dice_metric(pred_binary[2], gt[2]) * 100.0
            dice_ET_raw.append(d_et)
            dice_WT_raw.append(d_wt)
            dice_TC_raw.append(d_tc)

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

        print(f"[{mode_desc:<20}] Raw: ET={m_et_raw:5.2f}% TC={m_tc_raw:5.2f}% WT={m_wt_raw:5.2f}% (Avg={m_avg_raw:5.2f}%) | Post: Avg={m_avg_post:5.2f}%", flush=True)

    raw_all_et = np.mean([float(r['Our_ET']) for r in rows_raw])
    raw_all_tc = np.mean([float(r['Our_TC']) for r in rows_raw])
    raw_all_wt = np.mean([float(r['Our_WT']) for r in rows_raw])
    raw_all_avg = np.mean([float(r['Our_Avg']) for r in rows_raw])

    post_all_et = np.mean([float(r['Our_ET']) for r in rows_post])
    post_all_tc = np.mean([float(r['Our_TC']) for r in rows_post])
    post_all_wt = np.mean([float(r['Our_WT']) for r in rows_post])
    post_all_avg = np.mean([float(r['Our_Avg']) for r in rows_post])

    rows_raw.append({
        'Mode': 'ALL_15_AVG', 'Desc': '15-Modality Overall Mean',
        'Our_ET': f"{raw_all_et:.2f}", 'Our_TC': f"{raw_all_tc:.2f}", 'Our_WT': f"{raw_all_wt:.2f}", 'Our_Avg': f"{raw_all_avg:.2f}",
        'Paper_ET': '63.09', 'Paper_TC': '79.79', 'Paper_WT': '87.51'
    })
    rows_post.append({
        'Mode': 'ALL_15_AVG', 'Desc': '15-Modality Overall Mean',
        'Our_ET': f"{post_all_et:.2f}", 'Our_TC': f"{post_all_tc:.2f}", 'Our_WT': f"{post_all_wt:.2f}", 'Our_Avg': f"{post_all_avg:.2f}",
        'Paper_ET': '63.09', 'Paper_TC': '79.79', 'Paper_WT': '87.51'
    })

    def save_csv(rows, path):
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    save_csv(rows_raw, raw_csv)
    save_csv(rows_post, post_csv)

    md_content = f"""# MetaKD-ACDIS (V6) Evaluation Results across 15 Modality Combinations

## Summary Overview
- **Raw 15-Modality Overall Average Dice:** **{raw_all_avg:.2f}%** (ET={raw_all_et:.2f}%, TC={raw_all_tc:.2f}%, WT={raw_all_wt:.2f}%)
- **Postprocessed Overall Average Dice:** **{post_all_avg:.2f}%** (ET={post_all_et:.2f}%, TC={post_all_tc:.2f}%, WT={post_all_wt:.2f}%)
- **Paper MetaKD Baseline:** **76.80%** (ET=63.09%, TC=79.79%, WT=87.51%)
- **Reproduced MetaKD Baseline:** **73.27%**

## Full 15-Modality Breakdown (Postprocessed)
| Modality Subset | Description | Our ET (%) | Our TC (%) | Our WT (%) | Our Avg (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for r in rows_post:
        md_content += f"| `{r['Mode']}` | {r['Desc']} | {r['Our_ET']} | {r['Our_TC']} | {r['Our_WT']} | **{r['Our_Avg']}** |\n"

    with open(summary_md, 'w') as f:
        f.write(md_content)

    print("\n" + "=" * 80)
    print(f"OVERALL 15-MODALITY AVERAGE: Raw={raw_all_avg:.2f}% | Postprocessed={post_all_avg:.2f}%")
    print(f"Saved CSVs: {raw_csv} and {post_csv}")
    print(f"Saved Markdown Summary: {summary_md}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MetaKD-ACDIS (V6) on 15 missing modality combinations")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--test_list", type=str, default="datalist/BraTS18/BraTS18_metaVal.csv")
    parser.add_argument("--output_dir", type=str, default="brats_algorithm2/v6_metakd_acdis/results")
    args = parser.parse_args()

    run_evaluation(args.checkpoint, args.data_dir, args.test_list, args.output_dir)


if __name__ == '__main__':
    main()
