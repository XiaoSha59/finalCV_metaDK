import argparse
import os
import os.path as osp
import random
import sys
import timeit
from math import ceil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        class DummySummaryWriter:
            def __init__(self, *args, **kwargs): pass
            def add_scalar(self, *args, **kwargs): pass
            def close(self): pass
        SummaryWriter = DummySummaryWriter

CURRENT_DIR = osp.dirname(osp.abspath(__file__))
ALG2_DIR = osp.abspath(osp.join(CURRENT_DIR, ".."))
BRATS_DIR = osp.abspath(osp.join(CURRENT_DIR, "..", "..", "brats"))
REPO_DIR = osp.abspath(osp.join(CURRENT_DIR, "..", ".."))
for p in [CURRENT_DIR, ALG2_DIR, BRATS_DIR, osp.join(BRATS_DIR, "process"), REPO_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from datasets.brats_dataset import BraTSDataSet, my_collate
except ImportError:
    try:
        from data.dataset import BraTSDataSet, my_collate
    except ImportError:
        from brats.data.dataset import BraTSDataSet, my_collate

from v6_models.dual_net_v6 import DualNet_V6
from v6_losses.loss_v6 import AnatomicalConsistencyLoss, DiceLoss4BraTS, BCELoss4BraTS


def get_arguments():
    parser = argparse.ArgumentParser(description="MetaKD-ACDIS (V6): Anatomical Consistency Distillation from ACDIS")
    parser.add_argument("--data_dir", type=str, default=osp.join(BRATS_DIR, "data"))
    parser.add_argument("--train_list", type=str, default="datalist/BraTS18/BraTS18_metaTrain.csv")
    parser.add_argument("--val_list", type=str, default="datalist/BraTS18/BraTS18_metaVal.csv")
    parser.add_argument("--snapshot_dir", type=str, default=osp.join(BRATS_DIR, "checkpoints", "phase2_v6_metakd_acdis"))
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
    parser.add_argument("--mode", type=str, default="random")
    parser.add_argument("--random_mirror", type=str, default="True")
    parser.add_argument("--random_scale", type=str, default="True")
    parser.add_argument("--random_seed", type=int, default=999)
    parser.add_argument("--norm_cfg", type=str, default="IN")
    parser.add_argument("--activation_cfg", type=str, default="LeakyReLU")
    parser.add_argument("--weight_std", type=bool, default=True)
    parser.add_argument("--train_only", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=0)

    # ACDIS Anatomical Consistency Hyperparameters
    parser.add_argument("--lambda_kd", type=float, default=0.1, help="MetaKD feature KD loss weight")
    parser.add_argument("--lambda_acct", type=float, default=0.05, help="Weight of ACCT variance+covariance distillation")
    return parser


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "1")


def adjust_learning_rate(optimizer, i_iter, learning_rate, num_steps, power):
    lr = learning_rate * ((1 - float(i_iter) / num_steps) ** power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def dice_score(preds, labels):
    assert preds.shape[0] == labels.shape[0]
    preds = preds.contiguous().view(preds.shape[0], -1)
    labels = labels.contiguous().view(labels.shape[0], -1)
    num = torch.sum(torch.mul(preds, labels), dim=1)
    den = torch.sum(preds, dim=1) + torch.sum(labels, dim=1) + 1.0
    dice = 2.0 * num / den
    return dice.mean()


def compute_dice_score(preds, labels):
    preds = torch.sigmoid(preds)
    dice_ET = dice_score(preds[:, 0], labels[:, 0]).cpu().data.numpy()
    dice_WT = dice_score(preds[:, 1], labels[:, 1]).cpu().data.numpy()
    dice_TC = dice_score(preds[:, 2], labels[:, 2]).cpu().data.numpy()
    return dice_ET, dice_WT, dice_TC


def predict_sliding(model, image, tile_size, classes, mode='0,1,2,3'):
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
                pred, _, _ = model(img_tile, val=True, mode=mode)
                total_predictions[:, x1:x2, y1:y2, z1:z2] += pred[0]
                count_predictions[:, x1:x2, y1:y2, z1:z2] += 1

    total_predictions /= count_predictions
    return total_predictions


def validate_meta_update(model, valloader, input_size, num_classes, loss_D, loss_BCE, kd_optim):
    model.eval()
    val_ET = 0.0
    val_WT = 0.0
    val_TC = 0.0
    total_val_loss = 0.0

    for index, batch in enumerate(valloader):
        image = torch.from_numpy(batch['image']).cuda()
        label = torch.from_numpy(batch['label']).cuda()
        kd_optim.zero_grad()

        pred = predict_sliding(model, image, input_size, num_classes, mode='0,1,2,3')

        dice_ET, dice_WT, dice_TC = compute_dice_score(pred.unsqueeze(0), label)
        val_ET += dice_ET
        val_WT += dice_WT
        val_TC += dice_TC

        val_Dice = loss_D.forward(pred.unsqueeze(0), label)
        val_BCE = loss_BCE.forward(pred.unsqueeze(0), label)
        val_loss = val_Dice + val_BCE
        val_loss.backward()
        kd_optim.step()

        total_val_loss += val_loss.item()
        del pred, val_loss, val_Dice, val_BCE, image, label
        torch.cuda.empty_cache()

    num_samples = len(valloader)
    model.train()
    return val_ET / num_samples, val_WT / num_samples, val_TC / num_samples, total_val_loss / num_samples


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


def main():
    parser = get_arguments()
    args = parser.parse_args()

    os.makedirs(args.snapshot_dir, exist_ok=True)
    writer = SummaryWriter(args.snapshot_dir)

    d, h, w = map(int, args.input_size.split(','))
    input_size = (d, h, w)

    cudnn.benchmark = True
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = DualNet_V6(
        args=args,
        norm_cfg=args.norm_cfg,
        activation_cfg=args.activation_cfg,
        num_classes=args.num_classes,
        weight_std=args.weight_std,
        self_att=False,
        cross_att=False
    )
    model.to(device)

    optimizer = torch.optim.SGD(
        [p for n, p in model.named_parameters() if 'kd_weights' not in n],
        args.learning_rate,
        momentum=args.momentum,
        nesterov=True
    )
    kd_optim = torch.optim.Adam(
        model.kd_weights.parameters(),
        args.learning_rate,
        weight_decay=args.weight_decay
    )

    if str2bool(args.reload_from_checkpoint):
        reload_target = args.reload_path
        last_ckpt = osp.join(args.snapshot_dir, "last.pth")
        if osp.exists(last_ckpt):
            reload_target = last_ckpt
            print(f">>> Resuming training from intermediate checkpoint: {reload_target}")
        elif osp.exists(args.reload_path):
            print(f">>> Initializing from specified checkpoint: {reload_target}")
        else:
            print(f"ERROR: Checkpoint not found at {args.reload_path}!")
            sys.exit(1)

        checkpoint = torch.load(reload_target, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        model.load_state_dict(state_dict, strict=False)
        args.start_iters = checkpoint.get('iter', args.start_iters) if isinstance(checkpoint, dict) else args.start_iters
        if isinstance(checkpoint, dict) and not args.reset_optimizer and 'optimizer' in checkpoint:
            try:
                opt_sd = checkpoint['optimizer'].state_dict() if hasattr(checkpoint['optimizer'], 'state_dict') else checkpoint['optimizer']
                optimizer.load_state_dict(opt_sd)
                if 'kd_optim' in checkpoint:
                    kd_sd = checkpoint['kd_optim'].state_dict() if hasattr(checkpoint['kd_optim'], 'state_dict') else checkpoint['kd_optim']
                    kd_optim.load_state_dict(kd_sd)
                print(f">>> Loaded model weights and optimizer at iter {args.start_iters}")
            except Exception as e:
                print(f">>> Fresh optimizer initialized (optimizer reload skipped: {e})")
        else:
            print(f">>> Loaded model weights at iter {args.start_iters} with fresh optimizer")

    model.train()

    loss_D = DiceLoss4BraTS().to(device)
    loss_BCE = BCELoss4BraTS().to(device)
    loss_ACCT = AnatomicalConsistencyLoss().to(device)

    train_list_resolved = resolve_list_path(args.data_dir, args.train_list)
    val_list_resolved = resolve_list_path(args.data_dir, args.val_list)

    train_dataset = BraTSDataSet(
        args.data_dir,
        train_list_resolved,
        max_iters=args.num_steps * args.batch_size,
        crop_size=input_size,
        scale=str2bool(args.random_scale),
        mirror=str2bool(args.random_mirror)
    )
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=my_collate
    )

    val_dataset = BraTSDataSet(
        args.data_dir,
        val_list_resolved,
        max_iters=None,
        crop_size=input_size,
        scale=False,
        mirror=False
    )
    valloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=my_collate
    )

    print(f"============================================================")
    print(f"MetaKD-ACDIS (V6) Phase 2 Training Starting...")
    print(f"Steps: {args.start_iters} -> {args.num_steps} (Total {args.num_steps - args.start_iters} steps)")
    print(f"Mode: {args.mode} | Batch: {args.batch_size} | Workers: {args.num_workers}")
    print(f"Lambda KD: {args.lambda_kd} | Lambda ACCT: {args.lambda_acct} (Variance + Covariance)")
    print(f"============================================================")

    for i_iter, batch in enumerate(trainloader):
        current_iter = args.start_iters + i_iter
        if current_iter >= args.num_steps:
            break

        images = torch.from_numpy(batch['image']).to(device)
        labels = torch.from_numpy(batch['label']).to(device)

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, current_iter, args.learning_rate, args.num_steps, args.power)

        # 1. Student Forward with intermediate feature extraction
        student_preds, mode_split, totKDLoss, student_out_ft, _ = model(
            images, mode=args.mode, return_features=True
        )

        # 2. Supervised Segmentation Loss
        seg_loss = loss_D(student_preds, labels) + loss_BCE(student_preds, labels)

        # 3. ACDIS Anatomical Consistency Distillation Loss (ACCT)
        if len(mode_split) < 4 and args.lambda_acct > 0:
            with torch.no_grad():
                _, _, _, teacher_out_ft, _ = model(images, mode="0,1,2,3", return_features=True)
            l_acct, l_var, l_covar = loss_ACCT(student_out_ft, teacher_out_ft)
        else:
            l_acct = torch.tensor(0.0, device=device)
            l_var = torch.tensor(0.0, device=device)
            l_covar = torch.tensor(0.0, device=device)

        # 4. Total Objective
        total_loss = seg_loss + args.lambda_kd * totKDLoss + args.lambda_acct * l_acct
        total_loss.backward()
        optimizer.step()

        if current_iter % 100 == 0 and args.local_rank == 0:
            writer.add_scalar('lr', lr, current_iter)
            writer.add_scalar('total_loss', total_loss.item(), current_iter)
            writer.add_scalar('seg_loss', seg_loss.item(), current_iter)
            writer.add_scalar('kd_loss', totKDLoss.item(), current_iter)
            writer.add_scalar('acct_loss', l_acct.item(), current_iter)
            print(f"[Iter {current_iter}/{args.num_steps}] lr={lr:.5f}, seg={seg_loss.item():.4f}, kd={totKDLoss.item():.4f}, acct={l_acct.item():.4f} (var={l_var.item():.4f}, cov={l_covar.item():.4f})")

        # Save checkpoint periodically
        if (current_iter % 500 == 0 or current_iter % args.val_pred_every == args.val_pred_every - 1) and args.local_rank == 0:
            checkpoint = {
                'model': model,
                'optimizer': optimizer,
                'kd_optim': kd_optim,
                'iter': current_iter
            }
            torch.save(checkpoint, osp.join(args.snapshot_dir, 'last.pth'))

        # Validation & Meta-Update
        if not args.train_only and current_iter % args.val_pred_every == 0 and i_iter > 0:
            print(f"\n--- Validation & Meta-Update at iter {current_iter} ---")
            val_start = timeit.default_timer()
            val_ET, val_WT, val_TC, val_loss = validate_meta_update(
                model, valloader, input_size, args.num_classes, loss_D, loss_BCE, kd_optim
            )
            val_avg = (val_ET + val_WT + val_TC) / 3.0

            if args.local_rank == 0:
                writer.add_scalar('Val_ET_Dice', val_ET, current_iter)
                writer.add_scalar('Val_WT_Dice', val_WT, current_iter)
                writer.add_scalar('Val_TC_Dice', val_TC, current_iter)
                writer.add_scalar('Val_Avg_Dice', val_avg, current_iter)
                print(f"Validation: ET={val_ET:.4f}, WT={val_WT:.4f}, TC={val_TC:.4f}, Avg={val_avg:.4f}")
                iwv = model.kd_weights.get_kdWeights().detach().cpu().numpy()
                print(f"IWV Softmax Weights: FLAIR={iwv[0]:.4f}, T1={iwv[1]:.4f}, T1ce={iwv[2]:.4f}, T2={iwv[3]:.4f}")
            print(f"Validation time: {timeit.default_timer() - val_start:.2f}s\n")

    # Save final model
    if args.local_rank == 0:
        final_ckpt = {
            'model': model,
            'optimizer': optimizer,
            'kd_optim': kd_optim,
            'iter': args.num_steps
        }
        torch.save(final_ckpt, osp.join(args.snapshot_dir, 'final.pth'))
        print(f"Training completed successfully! Saved final checkpoint to {osp.join(args.snapshot_dir, 'final.pth')}")


if __name__ == '__main__':
    main()
