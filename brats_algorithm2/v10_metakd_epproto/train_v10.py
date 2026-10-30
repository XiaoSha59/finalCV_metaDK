import argparse
import os
import os.path as osp
import random
import sys
import timeit
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils import data

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ALG2_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
REPO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
BRATS_DIR = os.path.abspath(os.path.join(REPO_DIR, "brats"))

for p in [CURRENT_DIR, ALG2_DIR, REPO_DIR, BRATS_DIR, os.path.join(BRATS_DIR, "process")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from v10_models.dual_net_v10 import DualNet_V10
from v10_losses.loss_v10 import (
    DecoupledUncertaintyWeightedKDLoss,
    SemanticPrototypeAlignmentLoss,
    TopologicalContainmentLoss,
    DiceLoss4BraTS,
    BCELoss4BraTS,
)

try:
    from datasets.brats_dataset import BraTSDataSet, my_collate
except ImportError:
    try:
        from data.dataset import BraTSDataSet, my_collate
    except ImportError:
        from brats.data.dataset import BraTSDataSet, my_collate


def get_arguments():
    parser = argparse.ArgumentParser(description="MetaKD-EPProto (V10) Unified Phase 2 Training")
    parser.add_argument("--data_dir", type=str, default=osp.join(BRATS_DIR, "data"))
    parser.add_argument("--train_list", type=str, default="datalist/BraTS18/BraTS18_metaTrain.csv")
    parser.add_argument("--val_list", type=str, default="datalist/BraTS18/BraTS18_metaVal.csv")
    parser.add_argument("--snapshot_dir", type=str, default=osp.join(REPO_DIR, "snapshots", "BraTS18_phase2_v10_metakd_epproto"))
    parser.add_argument("--reload_from_checkpoint", type=str, default="True")
    parser.add_argument("--reload_path", type=str, default=osp.join(BRATS_DIR, "checkpoints", "phase1_80k", "final.pth"))
    parser.add_argument("--reset_optimizer", action="store_true", default=False)
    parser.add_argument("--input_size", type=str, default="64,64,64")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--num_steps", type=int, default=115000)
    parser.add_argument("--start_iters", type=int, default=80000)
    parser.add_argument("--val_pred_every", type=int, default=500)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--random_mirror", action="store_true", default=True)
    parser.add_argument("--random_scale", action="store_true", default=False)
    parser.add_argument("--train_only", action="store_true", default=False)
    parser.add_argument("--mode", type=str, default="random")

    # Hyperparameters for Unified V10
    parser.add_argument("--lambda_kd", type=float, default=0.1, help="Vanilla MetaKD cross-modal feature KD weight")
    parser.add_argument("--lambda_proto", type=float, default=0.05, help="Semantic prototype alignment loss weight")
    parser.add_argument("--lambda_sep", type=float, default=0.1, help="Inter-class prototype separation weight")
    parser.add_argument("--proto_margin", type=float, default=0.2, help="Inter-class cosine separation margin")

    parser.add_argument("--lambda_ukd", type=float, default=0.05, help="Decoupled Uncertainty-Weighted KD loss weight")
    parser.add_argument("--tau_unc", type=float, default=0.5, help="Uncertainty temperature scaling factor")
    parser.add_argument("--min_wt_conf", type=float, default=0.70, help="Bounded minimum confidence floor for Whole Tumor")
    parser.add_argument("--ukd_lambda_dice", type=float, default=1.0)
    parser.add_argument("--ukd_lambda_bce", type=float, default=0.5)

    parser.add_argument("--lambda_topo", type=float, default=0.05, help="Differentiable topological containment loss weight")

    return parser.parse_args()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, learning_rate, num_steps, power):
    lr = lr_poly(learning_rate, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def dice_score(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)
    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1.0
    dice = 2.0 * num / den
    return dice.mean()


def validate_meta_update(model, valloader, input_size, num_classes, loss_D, loss_BCE, kd_optim):
    model.eval()
    val_loss = 0.0
    dice_ET, dice_WT, dice_TC = [], [], []

    for batch in valloader:
        val_img = batch['image']
        val_lbl = batch['label']
        images = torch.from_numpy(val_img).cuda() if isinstance(val_img, np.ndarray) else val_img.cuda()
        labels = torch.from_numpy(val_lbl).cuda() if isinstance(val_lbl, np.ndarray) else val_lbl.cuda()
        with torch.no_grad():
            preds, _, _ = model(images, val=True, mode='0,1,2,3')
            loss_dice = loss_D.forward(preds, labels)
            loss_bce = loss_BCE.forward(preds, labels)
            loss = loss_dice + loss_bce
            val_loss += loss.item()

            probs = torch.sigmoid(preds)
            pred_bin = (probs > 0.5).float()
            dice_ET.append(dice_score(pred_bin[:, 0], labels[:, 0]).item() * 100.0)
            dice_WT.append(dice_score(pred_bin[:, 1], labels[:, 1]).item() * 100.0)
            dice_TC.append(dice_score(pred_bin[:, 2], labels[:, 2]).item() * 100.0)

    # Meta-weight update step
    kd_optim.zero_grad()
    for batch in valloader:
        val_img = batch['image']
        val_lbl = batch['label']
        images = torch.from_numpy(val_img).cuda() if isinstance(val_img, np.ndarray) else val_img.cuda()
        labels = torch.from_numpy(val_lbl).cuda() if isinstance(val_lbl, np.ndarray) else val_lbl.cuda()
        preds, _, _ = model(images, val=True, mode='0,1,2,3')
        loss = loss_D.forward(preds, labels) + loss_BCE.forward(preds, labels)
        loss.backward()
        kd_optim.step()
        break

    model.train()
    mean_ET = np.mean(dice_ET)
    mean_WT = np.mean(dice_WT)
    mean_TC = np.mean(dice_TC)
    return mean_ET, mean_WT, mean_TC, val_loss / max(len(valloader), 1)


def main():
    args = get_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    if not osp.exists(args.snapshot_dir) and args.local_rank == 0:
        os.makedirs(args.snapshot_dir, exist_ok=True)

    writer = SummaryWriter(args.snapshot_dir)
    model = DualNet_V10(args=args, num_classes=args.num_classes)
    model.to(device)

    final_path = osp.join(args.snapshot_dir, "final.pth")
    if osp.exists(final_path) and osp.getsize(final_path) > 100000000:
        print(f">>> Final checkpoint already exists at {final_path}. Training complete, skipping!")
        return

    base_params = [p for n, p in model.named_parameters() if 'kd_weights' not in n]
    optimizer = torch.optim.SGD(
        base_params,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    kd_optim = torch.optim.Adam(model.kd_weights.parameters(), lr=1e-3)

    reload_flag = args.reload_from_checkpoint.lower() == 'true'
    reload_target = args.reload_path
    last_path = osp.join(args.snapshot_dir, "last.pth")
    resumed = False
    if osp.exists(last_path) and osp.getsize(last_path) > 100000000:
        reload_target = last_path
        resumed = True
        print(f">>> Found intermediate checkpoint. Resuming from: {reload_target}")
    elif reload_flag and osp.exists(args.reload_path):
        print(f">>> Loading base checkpoint from: {reload_target}")
    else:
        print(f">>> Starting from scratch (no reload from {args.reload_path})")

    if (resumed or reload_flag) and osp.exists(reload_target):
        saved_state = torch.load(reload_target, map_location=device, weights_only=False)
        state_dict = saved_state.get('model', saved_state)
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        model_dict = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(filtered)}/{len(model_dict)} tensors from {reload_target}")
        if resumed and isinstance(saved_state, dict):
            args.start_iters = saved_state.get('iter', args.start_iters)
            print(f">>> Resumed training at iter {args.start_iters}")

    loss_D = DiceLoss4BraTS()
    loss_BCE = BCELoss4BraTS()
    loss_UKD = DecoupledUncertaintyWeightedKDLoss(
        tau_unc=args.tau_unc,
        min_wt_conf=args.min_wt_conf,
        lambda_dice=args.ukd_lambda_dice,
        lambda_bce=args.ukd_lambda_bce
    )
    loss_Proto = SemanticPrototypeAlignmentLoss(
        lambda_sep=args.lambda_sep,
        margin=args.proto_margin
    )
    loss_Topo = TopologicalContainmentLoss()

    input_size = tuple(map(int, args.input_size.split(',')))
    train_dataset = BraTSDataSet(
        args.data_dir,
        args.train_list,
        max_iters=args.num_steps * args.batch_size,
        crop_size=input_size,
        scale=args.random_scale,
        mirror=args.random_mirror
    )
    trainloader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=my_collate
    )

    valloader = None
    if not args.train_only and args.val_list:
        val_dataset = BraTSDataSet(
            args.data_dir,
            args.val_list,
            crop_size=input_size,
            scale=False,
            mirror=False
        )
        valloader = data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=my_collate
        )

    print(f"============================================================")
    print(f"MetaKD-EPProto (V10) Unified Phase 2 Training Starting...")
    print(f"Steps: {args.start_iters} -> {args.num_steps} (Total {args.num_steps - args.start_iters} steps)")
    print(f"Mode: {args.mode} | Lambda KD: {args.lambda_kd} | Lambda Proto: {args.lambda_proto} | Lambda UKD: {args.lambda_ukd} | Lambda Topo: {args.lambda_topo}")
    print(f"Min WT Conf floor: {args.min_wt_conf} | Tau Unc: {args.tau_unc} | Proto Margin: {args.proto_margin}")
    print(f"============================================================")

    model.train()
    for i_iter, batch in enumerate(trainloader):
        current_iter = args.start_iters + i_iter
        if current_iter >= args.num_steps:
            break

        images = torch.from_numpy(batch['image']).to(device)
        labels = torch.from_numpy(batch['label']).to(device)

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, current_iter, args.learning_rate, args.num_steps, args.power)

        # 1. Student Forward Pass (Missing Modality Mode) with Feature output
        student_preds, mode_split, totKDLoss, stu_feats = model(images, mode=args.mode, return_feat=True)

        # 2. Supervised Task Segmentation Loss
        term_seg_Dice = loss_D.forward(student_preds, labels)
        term_seg_BCE = loss_BCE.forward(student_preds, labels)
        seg_loss = term_seg_Dice + term_seg_BCE

        # 3. Dual Evidential Prototype & Decoupled Uncertainty Distillation
        if len(mode_split) < 4:
            with torch.no_grad():
                teacher_preds, _, _, tea_feats = model(images, mode="0,1,2,3", return_feat=True)

            # A. Prototype Alignment (Job 9)
            if args.lambda_proto > 0:
                l_proto, l_align, l_sep = loss_Proto(stu_feats, tea_feats, labels)
            else:
                l_proto = torch.tensor(0.0, device=device)

            # B. Decoupled Uncertainty KD (Job 7 Upgraded with WT protection)
            if args.lambda_ukd > 0:
                l_ukd, l_u_dice, l_u_bce, mean_conf = loss_UKD(student_preds, teacher_preds)
            else:
                l_ukd = torch.tensor(0.0, device=device)
                mean_conf = torch.tensor(1.0, device=device)
        else:
            l_proto = torch.tensor(0.0, device=device)
            l_ukd = torch.tensor(0.0, device=device)
            mean_conf = torch.tensor(1.0, device=device)

        # 4. Topological Containment Loss (Differentiable ET subset of TC subset of WT)
        l_topo = loss_Topo(student_preds) if args.lambda_topo > 0 else torch.tensor(0.0, device=device)

        # 5. Total Loss & Optimization
        total_loss = (
            seg_loss
            + args.lambda_kd * totKDLoss
            + args.lambda_proto * l_proto
            + args.lambda_ukd * l_ukd
            + args.lambda_topo * l_topo
        )
        total_loss.backward()
        optimizer.step()

        if current_iter % 100 == 0 and args.local_rank == 0:
            writer.add_scalar('lr', lr, current_iter)
            writer.add_scalar('total_loss', total_loss.item(), current_iter)
            writer.add_scalar('seg_loss', seg_loss.item(), current_iter)
            writer.add_scalar('kd_loss', totKDLoss.item(), current_iter)
            writer.add_scalar('proto_loss', l_proto.item(), current_iter)
            writer.add_scalar('ukd_loss', l_ukd.item(), current_iter)
            writer.add_scalar('topo_loss', l_topo.item(), current_iter)
            writer.add_scalar('teacher_confidence', mean_conf.item(), current_iter)
            print(f"[Iter {current_iter}/{args.num_steps}] lr={lr:.5f}, seg={seg_loss.item():.4f}, kd={totKDLoss.item():.4f}, proto={l_proto.item():.4f}, ukd={l_ukd.item():.4f}, topo={l_topo.item():.4f} (conf={mean_conf.item():.3f})")

        if (current_iter % 500 == 0 or current_iter % args.val_pred_every == args.val_pred_every - 1) and args.local_rank == 0:
            checkpoint = {
                'model': model,
                'optimizer': optimizer,
                'kd_optim': kd_optim,
                'iter': current_iter
            }
            torch.save(checkpoint, osp.join(args.snapshot_dir, 'last.pth'))

        if not args.train_only and current_iter % args.val_pred_every == 0 and i_iter > 0:
            print(f"\n--- Validation & Meta-Update at iter {current_iter} ---")
            val_ET, val_WT, val_TC, val_loss = validate_meta_update(
                model, valloader, input_size, args.num_classes, loss_D, loss_BCE, kd_optim
            )
            val_avg = (val_ET + val_WT + val_TC) / 3.0
            print(f"Val Results: ET={val_ET:.2f}%, WT={val_WT:.2f}%, TC={val_TC:.2f}%, Mean={val_avg:.2f}% | Loss={val_loss:.4f}\n")

    torch.save({'model': model, 'iter': args.num_steps}, final_path)
    print(f">>> V10 Phase 2 Training Completed Successfully! Final model saved to: {final_path}")


if __name__ == '__main__':
    main()
