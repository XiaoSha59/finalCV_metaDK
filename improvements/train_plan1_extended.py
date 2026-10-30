"""
Plan 1 Extended: Class-Decoupled Uncertainty KD Training Script
==============================================================
Implements the full end-to-end training pipeline for Plan 1 Extended on BraTS 2018:
- Phase 1 (0 -> 25,000 steps): Supervised training on complete modalities.
- Phase 2 (25,000 -> 50,000 steps): Missing modality training with CDU-KD distillation.
- Fine-tune (50,000 -> 51,500 steps): ET gradient unblocking with tau_et = 0.25.

Author: Team finalCV_metaDK
"""

import argparse
import os
import os.path as osp
import sys
import timeit
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

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
    from brats.data import BraTSDataSet, my_collate
except ImportError:
    from data.dataset import BraTSDataSet, my_collate

from improvements.loss_cdukd import (
    ClassDecoupledUncertaintyKDLoss,
    DiceLoss4BraTS,
    BCELoss4BraTS
)


def get_arguments():
    parser = argparse.ArgumentParser(description="Train Plan 1 Extended (CDU-KD) on BraTS 2018.")
    parser.add_argument("--snapshot_dir", type=str, default="./snapshots/plan1_extended", help="Where to save checkpoints.")
    parser.add_argument("--reload_from_checkpoint", action="store_true", default=False, help="Resume training.")
    parser.add_argument("--reload_path", type=str, default="", help="Path to checkpoint to reload.")
    parser.add_argument("--reset_optimizer", action="store_true", default=False, help="Reset optimizer states when fine-tuning.")

    parser.add_argument("--data_dir", type=str, default="data", help="Path to BraTS 2018 dataset root.")
    parser.add_argument("--train_list", type=str, default="datalist/BraTS18/BraTS18_metaTrain.csv")
    parser.add_argument("--val_list", type=str, default="datalist/BraTS18/BraTS18_metaVal.csv")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--input_size", type=str, default="80,160,160")
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--num_steps", type=int, default=50000, help="Total training steps.")
    parser.add_argument("--start_iters", type=int, default=0, help="Iteration offset for fine-tuning.")
    parser.add_argument("--phase1_steps", type=int, default=25000, help="Cutoff step between Phase 1 and Phase 2.")
    parser.add_argument("--power", type=float, default=0.9, help="Polynomial decay exponent.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # CDU-KD Hyperparameters
    parser.add_argument("--lambda_kd", type=float, default=0.1, help="Feature-level KD weight.")
    parser.add_argument("--lambda_ukd", type=float, default=0.05, help="CDU-KD uncertainty distillation weight.")
    parser.add_argument("--tau_et", type=float, default=0.25, help="ET uncertainty confidence temperature.")
    parser.add_argument("--tau_tc", type=float, default=0.50, help="TC uncertainty confidence temperature.")
    parser.add_argument("--tau_wt", type=float, default=-1.0, help="WT temperature (< 0 denotes None = 100% preservation).")
    parser.add_argument("--mode", type=str, default="random", help="Modality sampling mode for Phase 2.")

    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


def lr_poly(base_lr, iter_num, max_iter, power):
    return base_lr * ((1.0 - float(iter_num) / max_iter) ** power)


def adjust_learning_rate(optimizer, i_iter, learning_rate, num_steps, power):
    lr = lr_poly(learning_rate, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def main():
    args = get_arguments()
    os.makedirs(args.snapshot_dir, exist_ok=True)
    writer = SummaryWriter(args.snapshot_dir)

    torch.manual_seed(args.random_seed)
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DualNet(args=args, norm_cfg='IN', activation_cfg='LeakyReLU',
                    num_classes=3, weight_std=True, self_att=False, cross_att=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=0.9, weight_decay=args.weight_decay)

    # Fallback secondary optimizer if required by architecture
    kd_optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                               momentum=0.9, weight_decay=args.weight_decay)

    # Resume from checkpoint if specified
    if args.reload_from_checkpoint and args.reload_path:
        print(f">>> Resuming from checkpoint: {args.reload_path}")
        checkpoint = torch.load(args.reload_path, map_location=device, weights_only=False)
        if 'model' in checkpoint:
            m = checkpoint['model']
            model.load_state_dict(m.module.state_dict() if hasattr(m, 'module') else m)
            if not args.reset_optimizer and 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            model.load_state_dict(checkpoint)
        print(">>> Checkpoint loaded successfully.")

    model.to(device)

    tau_wt_val = None if args.tau_wt < 0 else args.tau_wt
    loss_CDUKD = ClassDecoupledUncertaintyKDLoss(
        tau_et=args.tau_et,
        tau_tc=args.tau_tc,
        tau_wt=tau_wt_val,
        lambda_dice=1.0,
        lambda_bce=0.5
    ).to(device)

    loss_D = DiceLoss4BraTS().to(device)
    loss_BCE = BCELoss4BraTS().to(device)

    d, h, w = map(int, args.input_size.split(','))
    input_size = (d, h, w)

    train_dataset = BraTSDataSet(
        args.data_dir,
        args.train_list,
        max_iters=args.num_steps * args.batch_size,
        crop_size=input_size,
        scale=False,
        mirror=True
    )
    trainloader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=my_collate
    )

    print("=" * 74)
    print("Plan 1 Extended: Class-Decoupled Uncertainty KD Training Initialized")
    print(f"Steps: {args.start_iters} -> {args.num_steps} | Phase 1 cut: {args.phase1_steps}")
    print(f"Tau ET: {args.tau_et} | Tau TC: {args.tau_tc} | Tau WT: {tau_wt_val} (Full preservation: {tau_wt_val is None})")
    print(f"Lambda KD: {args.lambda_kd} | Lambda UKD: {args.lambda_ukd}")
    print("=" * 74)

    model.train()
    start_time = timeit.default_timer()

    for i_iter, batch in enumerate(trainloader):
        current_iter = args.start_iters + i_iter
        if current_iter >= args.num_steps:
            break

        images = torch.from_numpy(batch['image']).to(device)
        labels = torch.from_numpy(batch['label']).to(device)

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, current_iter, args.learning_rate, args.num_steps, args.power)

        is_phase1 = (current_iter < args.phase1_steps)
        current_mode = "0,1,2,3" if is_phase1 else args.mode

        student_preds, mode_split, totKDLoss = model(images, mode=current_mode)

        # 1. Supervised Task Segmentation Loss
        term_seg_Dice = loss_D(student_preds, labels)
        term_seg_BCE = loss_BCE(student_preds, labels)
        seg_loss = term_seg_Dice + term_seg_BCE

        # 2. Class-Decoupled Uncertainty KD (Active during Phase 2 when modalities are missing)
        if (not is_phase1) and len(mode_split) < 4 and args.lambda_ukd > 0:
            with torch.no_grad():
                teacher_preds, _, _ = model(images, mode="0,1,2,3")
            l_ukd, _, _, conf_triplet = loss_CDUKD(student_preds, teacher_preds)
            c_et, c_wt, c_tc = conf_triplet
        else:
            l_ukd = torch.tensor(0.0, device=device)
            c_et = c_wt = c_tc = torch.tensor(1.0, device=device)

        # 3. Total Combined Loss
        total_loss = seg_loss + args.lambda_kd * totKDLoss + args.lambda_ukd * l_ukd
        total_loss.backward()
        optimizer.step()

        if current_iter % 100 == 0 and args.local_rank == 0:
            writer.add_scalar('lr', lr, current_iter)
            writer.add_scalar('total_loss', total_loss.item(), current_iter)
            writer.add_scalar('seg_loss', seg_loss.item(), current_iter)
            writer.add_scalar('kd_loss', totKDLoss.item(), current_iter)
            writer.add_scalar('ukd_loss', l_ukd.item(), current_iter)
            writer.add_scalar('conf_ET', c_et.item(), current_iter)
            writer.add_scalar('conf_WT', c_wt.item(), current_iter)
            writer.add_scalar('conf_TC', c_tc.item(), current_iter)
            print(f"[Iter {current_iter:05d}/{args.num_steps:05d}] (Phase {'1' if is_phase1 else '2'}) "
                  f"lr={lr:.6f} | seg={seg_loss.item():.4f} | kd={totKDLoss.item():.4f} | ukd={l_ukd.item():.4f} "
                  f"(conf_WT={c_wt.item():.2f}, conf_TC={c_tc.item():.2f}, conf_ET={c_et.item():.2f})")

        # Periodic checkpointing
        if (current_iter % 1000 == 0 or current_iter == args.num_steps - 1) and args.local_rank == 0 and current_iter > 0:
            ckpt = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'kd_optim': kd_optim.state_dict(),
                'iter': current_iter
            }
            save_name = 'final.pth' if (current_iter == args.num_steps - 1) else 'last.pth'
            torch.save(ckpt, osp.join(args.snapshot_dir, save_name))

    # Guarantee final checkpoint saved
    if args.local_rank == 0:
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'kd_optim': kd_optim.state_dict(),
            'iter': args.num_steps
        }
        torch.save(ckpt, osp.join(args.snapshot_dir, 'final.pth'))

    elapsed_hours = (timeit.default_timer() - start_time) / 3600.0
    print("=" * 74)
    print(f"Plan 1 Extended Training completed in {elapsed_hours:.2f} hours!")
    print(f"Final model checkpoint saved: {osp.join(args.snapshot_dir, 'final.pth')}")
    print("=" * 74)


if __name__ == '__main__':
    main()
