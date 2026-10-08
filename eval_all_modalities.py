import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

from models import DualNet
from datasets import BraTSEvalDataSet
from eval import predict_sliding
from utils import dice_score, print_brats_summary_table

MODES_15 = [
    ('0,1,2,3', 'All (Fl+T1+T1c+T2)'),
    ('0,1,2',   'Fl+T1+T1c'),
    ('0,1,3',   'Fl+T1+T2'),
    ('0,2,3',   'Fl+T1c+T2'),
    ('1,2,3',   'T1+T1c+T2'),
    ('0,1',     'Fl+T1'),
    ('0,2',     'Fl+T1c'),
    ('0,3',     'Fl+T2'),
    ('1,2',     'T1+T1c'),
    ('1,3',     'T1+T2'),
    ('2,3',     'T1c+T2'),
    ('0',       'Flair only'),
    ('1',       'T1 only'),
    ('2',       'T1c only'),
    ('3',       'T2 only'),
]

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
    return parser.parse_args()

def main():
    args = get_arguments()
    d, h, w = map(int, args.input_size.split(','))
    input_size = (d, h, w)

    if not os.path.exists(args.restore_from):
        print(f"Checkpoint file not found: {args.restore_from}")
        sys.exit(1)

    print(f"Loading checkpoint from: {args.restore_from}")
    checkpoint = torch.load(args.restore_from)
    model = DualNet(args=args, norm_cfg=args.norm_cfg, activation_cfg=args.activation_cfg,
                    num_classes=args.num_classes, weight_std=args.weight_std, self_att=True, cross_att=False)
    model = nn.DataParallel(model)
    model = checkpoint['model'] if 'model' in checkpoint else model
    model.eval()
    model.cuda()

    testloader = data.DataLoader(
        BraTSEvalDataSet(args.data_dir, args.data_list),
        batch_size=1, shuffle=False, pin_memory=True
    )
    print(f"Loaded test dataset: {len(testloader)} cases.")

    results_table = {}

    for mode_str, mode_desc in MODES_15:
        args.mode = mode_str
        print(f"\nEvaluating combination: {mode_desc} (mode='{mode_str}') ...")
        
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

        avg_et = np.mean(dice_ET_list)
        avg_tc = np.mean(dice_TC_list)
        avg_wt = np.mean(dice_WT_list)
        avg_overall = (avg_et + avg_tc + avg_wt) / 3.0

        results_table[mode_str] = {
            'ET': avg_et,
            'TC': avg_tc,
            'WT': avg_wt,
            'Avg': avg_overall
        }
        print(f" -> Result for {mode_desc}: ET={avg_et*100:.2f}%, TC={avg_tc*100:.2f}%, WT={avg_wt*100:.2f}%")

    print_brats_summary_table(results_table)

if __name__ == '__main__':
    main()
