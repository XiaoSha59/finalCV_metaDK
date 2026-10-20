import argparse
import os
import os.path as osp
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


# Cleanly add paths for existing dataset and model utilities without modifying them
CURRENT_DIR = osp.dirname(osp.abspath(__file__))
BRATS_DIR = osp.abspath(osp.join(CURRENT_DIR, "..", "brats"))
REPO_DIR = osp.abspath(osp.join(CURRENT_DIR, ".."))
for p in [CURRENT_DIR, BRATS_DIR, osp.join(BRATS_DIR, "process"), REPO_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from datasets.brats_dataset import BraTSDataSet, my_collate
except ImportError:
    try:
        from data.dataset import BraTSDataSet, my_collate
    except ImportError:
        from brats.data.dataset import BraTSDataSet, my_collate

from v1_models.dual_net_v1 import DualNet_Algorithm2
from v1_losses.loss_v1 import BinaryDiceLoss, RegionDecomposedLoss


def get_arguments():
    parser = argparse.ArgumentParser(description="Algorithm 2: Region-Aware Shared-Specific MetaKD for BraTS")
    parser.add_argument("--data_dir", type=str, default=osp.join(BRATS_DIR, "data"))
    parser.add_argument("--train_list", type=str, default="BraTS18_metaTrain.csv")
    parser.add_argument("--val_list", type=str, default="BraTS18_metaVal.csv")
    parser.add_argument("--snapshot_dir", type=str, default=osp.join(BRATS_DIR, "checkpoints", "phase2_alg2"))
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
    # Algorithm 2 Hyperparameters
    parser.add_argument("--use_shaspec", type=str, default="True", help="Toggle ShaSpec Shared/Specific decomposition")
    parser.add_argument("--lambda_kd", type=float, default=1.0, help="Weight for Region-Masked KD loss")
    parser.add_argument("--lambda_ortho", type=float, default=0.1, help="Weight for ShaSpec Orthogonal loss")
    parser.add_argument("--lambda_shared", type=float, default=0.1, help="Weight for ShaSpec Shared consistency loss")
    return parser


def str2bool(v):
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
    # Channel 0: ET, Channel 1: WT, Channel 2: TC
    dice_ET = dice_score(preds[:, 0], labels[:, 0]).cpu().data.numpy()
    dice_WT = dice_score(preds[:, 1], labels[:, 1]).cpu().data.numpy()
    dice_TC = dice_score(preds[:, 2], labels[:, 2]).cpu().data.numpy()
    return dice_ET, dice_WT, dice_TC


def predict_sliding(model, image, tile_size, classes, mode='0,1,2,3'):
    """
    Standard sliding-window inference evaluating full 3D volume piece by piece.
    """
    image_size = image.shape
    overlap = 1.0 / 3.0

    strideHW = ceil(tile_size[1] * (1.0 - overlap))
    strideD = ceil(tile_size[0] * (1.0 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)

    full_probs = torch.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4]),
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
                prediction, _, _, _, _ = model(img, val=True, mode=mode)

                count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1.0
                full_probs[:, :, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    return full_probs


def validate_meta_update(model, valloader, input_size, num_classes, loss_fn, kd_optim):
    """
    Validation with 1-pass sliding window & Outer Loop meta-update for w_ET, w_TC, w_WT.
    """
    val_ET = 0.0
    val_WT = 0.0
    val_TC = 0.0
    val_loss_total = 0.0

    num_samples = len(valloader)
    for index, batch in enumerate(valloader):
        image = torch.from_numpy(batch['image']).cuda()
        label = torch.from_numpy(batch['label']).cuda()
        kd_optim.zero_grad()

        # 1-pass sliding window over full brain volume
        pred = predict_sliding(model, image, input_size, num_classes, mode='0,1,2,3')

        # Logging metrics
        d_ET, d_WT, d_TC = compute_dice_score(pred, label)
        val_ET += d_ET
        val_WT += d_WT
        val_TC += d_TC

        # Isolated sub-region loss computation
        total_loss, (l_ET, l_WT, l_TC) = loss_fn(pred, label)

        # Meta-gradient backpropagation into region weight matrix W
        total_loss.backward()
        kd_optim.step()

        val_loss_total += total_loss.item()

        del pred, total_loss, l_ET, l_WT, l_TC, image, label
        torch.cuda.empty_cache()

    avg_ET = val_ET / num_samples
    avg_WT = val_WT / num_samples
    avg_TC = val_TC / num_samples
    avg_loss = val_loss_total / num_samples
    return avg_ET, avg_WT, avg_TC, avg_loss


def main():
    parser = get_arguments()
    args = parser.parse_args()

    os.makedirs(args.snapshot_dir, exist_ok=True)
    if osp.isdir(osp.join(args.data_dir, "BraTS2018")):
        os.chdir(osp.dirname(osp.abspath(args.data_dir)))
    writer = SummaryWriter(args.snapshot_dir)

    d, h, w = map(int, args.input_size.split(','))
    input_size = (d, h, w)

    cudnn.benchmark = True
    seed = args.random_seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    use_shaspec = str2bool(args.use_shaspec)
    print(f"=== Initializing DualNet_Algorithm2 (use_shaspec={use_shaspec}) ===")
    model = DualNet_Algorithm2(
        norm_cfg=args.norm_cfg,
        activation_cfg=args.activation_cfg,
        num_classes=args.num_classes,
        weight_std=args.weight_std,
        use_shaspec=use_shaspec
    )
    model.train()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Separate optimizers: SGD for backbone, Adam for region weights
    optimizer = torch.optim.SGD(
        [p for n, p in model.named_parameters() if 'region_weights' not in n],
        args.learning_rate, momentum=args.momentum, nesterov=True
    )
    kd_optim = torch.optim.Adam(
        model.region_weights.parameters(),
        args.learning_rate, weight_decay=args.weight_decay
    )

    # Safe Checkpoint Loading
    if str2bool(args.reload_from_checkpoint):
        print(f"Loading checkpoint from: {args.reload_path}")
        if os.path.exists(args.reload_path):
            checkpoint = torch.load(args.reload_path, weights_only=False)
            old_model = checkpoint['model']

            # Robust loading of pretrained backbone layers (handles state_dict or model object)
            if hasattr(old_model, 'shared_enc') and hasattr(old_model, 'shared_dec'):
                model.shared_enc.load_state_dict(old_model.shared_enc.state_dict())
                model.shared_dec.load_state_dict(old_model.shared_dec.state_dict())
                print(">>> Successfully transferred shared_enc and shared_dec from Phase 1 checkpoint!")
            elif isinstance(old_model, dict):
                # State dict loading with prefix matching
                enc_dict = {k.replace('shared_enc.', ''): v for k, v in old_model.items() if k.startswith('shared_enc.')}
                dec_dict = {k.replace('shared_dec.', ''): v for k, v in old_model.items() if k.startswith('shared_dec.')}
                if enc_dict and dec_dict:
                    model.shared_enc.load_state_dict(enc_dict)
                    model.shared_dec.load_state_dict(dec_dict)
                    print(">>> Successfully loaded shared encoder and decoder from state_dict!")

            args.start_iters = checkpoint.get('iter', args.start_iters)
            if not args.reset_optimizer and 'optimizer' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
                    print("Optimizer state restored.")
                except Exception:
                    print("Using fresh optimizer state.")
        else:
            print(f"Warning: Checkpoint path does not exist: {args.reload_path}. Initializing from scratch.")

    loss_fn = RegionDecomposedLoss().to(device)

    def resolve_list_path(data_dir, list_name):
        if osp.isabs(list_name) and osp.exists(list_name):
            return list_name
        candidates = [
            list_name,
            osp.join(data_dir, list_name),
            osp.join(data_dir, "datalist", "BraTS18", list_name),
            osp.join(osp.dirname(data_dir), "datalist", "BraTS18", list_name),
            osp.join(CURRENT_DIR, "..", "datalist", "BraTS18", list_name),
            osp.join(BRATS_DIR, "data", list_name),
        ]
        for c in candidates:
            if osp.exists(c):
                return c
        return osp.join(data_dir, list_name)

    train_list_path = resolve_list_path(args.data_dir, args.train_list)
    val_list_path = resolve_list_path(args.data_dir, args.val_list)
    print(f"Using train list: {train_list_path}")
    print(f"Using val list:   {val_list_path}")

    train_dataset = BraTSDataSet(
        args.data_dir,
        train_list_path,
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
        val_list_path,
        max_iters=20,
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

    print(f"Starting Algorithm 2 Training from iter {args.start_iters} to {args.num_steps}...")

    for i_iter, batch in enumerate(trainloader):
        start = timeit.default_timer()
        current_iter = args.start_iters + i_iter
        if current_iter >= args.num_steps:
            break

        images = torch.from_numpy(batch['image']).cuda()
        labels = torch.from_numpy(batch['label']).cuda()

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, current_iter, args.learning_rate, args.num_steps, args.power)

        # Forward pass: Student + Masked KD + ShaSpec losses
        preds, mode_split, totKDLoss, loss_ortho, loss_shared = model(
            images, labels=labels, val=False, mode=args.mode
        )

        # Segmentation loss
        seg_loss, (l_et, l_wt, l_tc) = loss_fn(preds, labels)

        # Total Inner Loop Loss
        total_loss = (
            seg_loss
            + args.lambda_kd * totKDLoss
            + (args.lambda_ortho * loss_ortho if use_shaspec else 0.0)
            + (args.lambda_shared * loss_shared if use_shaspec else 0.0)
        )

        total_loss.backward()
        optimizer.step()

        if current_iter % 100 == 0 and args.local_rank == 0:
            writer.add_scalar('lr', lr, current_iter)
            writer.add_scalar('seg_loss', seg_loss.item(), current_iter)
            writer.add_scalar('kd_loss', totKDLoss.item(), current_iter)
            if use_shaspec:
                writer.add_scalar('loss_ortho', loss_ortho.item(), current_iter)
                writer.add_scalar('loss_shared', loss_shared.item(), current_iter)
            print(f"[Iter {current_iter}/{args.num_steps}] lr={lr:.5f}, seg_loss={seg_loss.item():.4f}, kd_loss={totKDLoss.item():.4f}")

        # Regular Checkpoint Saving
        if (current_iter % 500 == 0 or current_iter % args.val_pred_every == args.val_pred_every - 1) and args.local_rank == 0:
            checkpoint = {
                'model': model,
                'optimizer': optimizer,
                'kd_optim': kd_optim,
                'iter': current_iter
            }
            torch.save(checkpoint, osp.join(args.snapshot_dir, 'last.pth'))

        # Validation & Outer Loop Meta-Update
        if not args.train_only and current_iter % args.val_pred_every == 0 and current_iter != args.start_iters:
            print(f"\n--- Running Validation & Meta-Update at iter {current_iter} ---")
            val_start = timeit.default_timer()
            val_ET, val_WT, val_TC, val_loss = validate_meta_update(
                model, valloader, input_size, args.num_classes, loss_fn, kd_optim
            )
            val_avg = (val_ET + val_WT + val_TC) / 3.0

            if args.local_rank == 0:
                writer.add_scalar('Val_ET_Dice', val_ET, current_iter)
                writer.add_scalar('Val_WT_Dice', val_WT, current_iter)
                writer.add_scalar('Val_TC_Dice', val_TC, current_iter)
                writer.add_scalar('Val_Avg_Dice', val_avg, current_iter)
                print(f"Validation: ET={val_ET:.4f}, WT={val_WT:.4f}, TC={val_TC:.4f}, Avg={val_avg:.4f}")
                norm_w = model.region_weights.get_normalized_weights().detach().cpu().numpy()
                print(f"Region Weights W (softmax):")
                print(f"  ET (Enhancing) : FLAIR={norm_w[0,0]:.3f}, T1={norm_w[0,1]:.3f}, T1ce={norm_w[0,2]:.3f}, T2={norm_w[0,3]:.3f}")
                print(f"  WT (Whole)     : FLAIR={norm_w[1,0]:.3f}, T1={norm_w[1,1]:.3f}, T1ce={norm_w[1,2]:.3f}, T2={norm_w[1,3]:.3f}")
                print(f"  TC (Core)      : FLAIR={norm_w[2,0]:.3f}, T1={norm_w[2,1]:.3f}, T1ce={norm_w[2,2]:.3f}, T2={norm_w[2,3]:.3f}")
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
