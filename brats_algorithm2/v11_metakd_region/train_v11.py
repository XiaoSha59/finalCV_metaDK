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

from v11_models.dual_net_v11 import DualNet_V11
from v11_losses.loss_v11 import RegionDecoupledUncertaintyKDLoss, DiceLoss4BraTS, BCELoss4BraTS


def get_arguments():
    parser = argparse.ArgumentParser(description="MetaKD-V11: Region-Disentangled Meta-Distillation with Region-Decoupled Uncertainty")
    parser.add_argument("--data_dir", type=str, default=osp.join(BRATS_DIR, "data"))
    parser.add_argument("--train_list", type=str, default="BraTS18_metaTrain.csv")
    parser.add_argument("--val_list", type=str, default="BraTS18_metaVal.csv")
    parser.add_argument("--snapshot_dir", type=str, default=osp.join(BRATS_DIR, "checkpoints", "phase2_v11_metakd_region"))
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

    # MetaKD Hyperparameters
    parser.add_argument("--lambda_kd", type=float, default=0.1, help="MetaKD cross-modal feature KD weight")
    parser.add_argument("--lambda_ukd", type=float, default=0.05, help="Region-Decoupled Uncertainty KD weight")
    
    # Region-Decoupled Uncertainty Temperatures
    parser.add_argument("--tau_ET", type=float, default=0.5, help="Sharp confidence threshold for Enhancing Tumor")
    parser.add_argument("--tau_WT", type=float, default=1.0, help="Relaxed confidence threshold for Whole Tumor (edema protection)")
    parser.add_argument("--tau_TC", type=float, default=0.5, help="Sharp confidence threshold for Tumor Core")
    parser.add_argument("--ukd_lambda_dice", type=float, default=1.0)
    parser.add_argument("--ukd_lambda_bce", type=float, default=0.5)

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
    val_loss = 0.0
    val_Dice = 0.0
    val_BCE = 0.0
    dice_ET = []
    dice_WT = []
    dice_TC = []

    for batch in valloader:
        val_image = torch.from_numpy(batch['image']).cuda()
        val_label = torch.from_numpy(batch['label']).cuda()

        kd_optim.zero_grad()
        pred = predict_sliding(model, val_image, input_size, num_classes, mode='0,1,2,3')
        pred = pred.unsqueeze(0)

        d_ET, d_WT, d_TC = compute_dice_score(pred, val_label)
        dice_ET.append(d_ET)
        dice_WT.append(d_WT)
        dice_TC.append(d_TC)

        loss_dice = loss_D.forward(pred, val_label)
        loss_bce = loss_BCE.forward(pred, val_label)
        total_loss = loss_dice + loss_bce

        total_loss.backward()
        kd_optim.step()

        val_loss += total_loss.item()
        val_Dice += loss_dice.item()
        val_BCE += loss_bce.item()
        del pred, total_loss, loss_dice, loss_bce, val_image, val_label
        torch.cuda.empty_cache()

    model.train()
    mean_ET = np.mean(dice_ET)
    mean_WT = np.mean(dice_WT)
    mean_TC = np.mean(dice_TC)
    return mean_ET, mean_WT, mean_TC, val_loss / len(valloader)


def main():
    args = get_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    if not osp.exists(args.snapshot_dir) and args.local_rank == 0:
        os.makedirs(args.snapshot_dir, exist_ok=True)

    writer = SummaryWriter(args.snapshot_dir)

    model = DualNet_V11(args=args, num_classes=args.num_classes)
    model.to(device)

    # Check if final.pth already exists
    final_path = osp.join(args.snapshot_dir, "final.pth")
    if osp.exists(final_path) and osp.getsize(final_path) > 100000000:
        print(f">>> Final checkpoint already exists at {final_path}. Training complete, skipping!")
        return

    # Optimizers
    base_params = [p for n, p in model.named_parameters() if 'kd_weights' not in n]
    optimizer = torch.optim.SGD(
        base_params,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    # Meta optimizer for Region Weights W in R^(3 x 4)
    kd_optim = torch.optim.Adam(model.kd_weights.parameters(), lr=1e-3)

    # Reload from checkpoint (resume from last.pth if exists, else reload_path)
    reload_flag = args.reload_from_checkpoint.lower() == 'true'
    reload_target = args.reload_path
    last_path = osp.join(args.snapshot_dir, "last.pth")
    if osp.exists(last_path) and osp.getsize(last_path) > 100000000:
        print(f">>> Found active checkpoint at {last_path}. Auto-resuming...")
        reload_target = last_path
        reload_flag = True

    if reload_flag and osp.exists(reload_target):
        print(f">>> Loading checkpoint from {reload_target}")
        ckpt = torch.load(reload_target, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
            # Load matching keys into model (filtering out shape mismatches for kd_weights if upgrading from 1x4 to 3x4)
            model_dict = model.state_dict()
            filtered_dict = {}
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                else:
                    print(f"    [Notice] Re-initializing {k} due to shape/key change: ckpt {getattr(v, 'shape', None)} vs model {getattr(model_dict.get(k), 'shape', None)}")
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)

            if not args.reset_optimizer:
                if 'optimizer' in ckpt and hasattr(ckpt['optimizer'], 'state_dict'):
                    try:
                        optimizer.load_state_dict(ckpt['optimizer'].state_dict())
                    except Exception as e:
                        print(f"    [Notice] Optimizer reload failed, using fresh optimizer: {e}")
                if 'iter' in ckpt:
                    args.start_iters = ckpt['iter'] + 1
                    print(f">>> Resuming iteration counter from step {args.start_iters}")
        else:
            model.load_state_dict(ckpt, strict=False)
        print(">>> Model parameters successfully loaded.")
    else:
        print(f">>> WARNING: No checkpoint loaded! Training from random initialization!")

    # Loss Functions
    loss_D = DiceLoss4BraTS()
    loss_BCE = BCELoss4BraTS()
    loss_UKD = RegionDecoupledUncertaintyKDLoss(
        tau_ET=args.tau_ET,
        tau_WT=args.tau_WT,
        tau_TC=args.tau_TC,
        lambda_dice=args.ukd_lambda_dice,
        lambda_bce=args.ukd_lambda_bce
    )

    # DataLoaders
    w, h, d = map(int, args.input_size.split(','))
    input_size = (w, h, d)

    train_dataset = BraTSDataSet(
        args.data_dir,
        args.train_list,
        max_iters=args.num_steps * args.batch_size,
        crop_size=input_size,
        scale=args.random_scale,
        mirror=args.random_mirror
    )
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=my_collate
    )

    val_dataset = BraTSDataSet(
        args.data_dir,
        args.val_list,
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
    print(f"MetaKD-Region (V11) Phase 2 Training Starting...")
    print(f"Steps: {args.start_iters} -> {args.num_steps} (Total {args.num_steps - args.start_iters} steps)")
    print(f"Mode: {args.mode} | Lambda KD: {args.lambda_kd} | Lambda UKD: {args.lambda_ukd}")
    print(f"Region Tau: ET={args.tau_ET} | WT={args.tau_WT} (Edema Protected) | TC={args.tau_TC}")
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

        # 1. Student Forward Pass (Missing Modality Mode)
        student_preds, mode_split, totKDLoss = model(images, mode=args.mode)

        # 2. Supervised Task Segmentation Loss
        term_seg_Dice = loss_D.forward(student_preds, labels)
        term_seg_BCE = loss_BCE.forward(student_preds, labels)
        seg_loss = term_seg_Dice + term_seg_BCE

        # 3. Region-Decoupled Uncertainty-Weighted KD
        if len(mode_split) < 4 and args.lambda_ukd > 0:
            with torch.no_grad():
                teacher_preds, _, _ = model(images, mode="0,1,2,3")
            l_ukd, l_et, l_wt, l_tc, c_et, c_wt, c_tc = loss_UKD(
                student_preds,
                teacher_preds,
                kd_weights_module=model.kd_weights,
                mode_split=mode_split
            )
        else:
            l_ukd = torch.tensor(0.0, device=device)
            l_et = torch.tensor(0.0, device=device)
            l_wt = torch.tensor(0.0, device=device)
            l_tc = torch.tensor(0.0, device=device)
            c_et = torch.tensor(1.0, device=device)
            c_wt = torch.tensor(1.0, device=device)
            c_tc = torch.tensor(1.0, device=device)

        # 4. Total Loss & Optimization
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
            print(f"[Iter {current_iter}/{args.num_steps}] lr={lr:.5f}, seg={seg_loss.item():.4f}, kd={totKDLoss.item():.4f}, ukd={l_ukd.item():.4f} (conf: ET={c_et.item():.2f}, WT={c_wt.item():.2f}, TC={c_tc.item():.2f})")

        # Save checkpoint periodically
        if (current_iter % 500 == 0 or current_iter % args.val_pred_every == args.val_pred_every - 1) and args.local_rank == 0:
            checkpoint = {
                'model': model,
                'optimizer': optimizer,
                'kd_optim': kd_optim,
                'iter': current_iter
            }
            torch.save(checkpoint, osp.join(args.snapshot_dir, 'last.pth'))

        # Validation & Bi-level Meta-Update of Region Weights W
        if current_iter % args.val_pred_every == 0 and current_iter > args.start_iters and not args.train_only:
            print(f">>> Step {current_iter}: Executing Meta-Validation & Region Weights Update...")
            m_ET, m_WT, m_TC, v_loss = validate_meta_update(
                model, valloader, input_size, args.num_classes, loss_D, loss_BCE, kd_optim
            )
            writer.add_scalar('val_Dice_ET', m_ET, current_iter)
            writer.add_scalar('val_Dice_WT', m_WT, current_iter)
            writer.add_scalar('val_Dice_TC', m_TC, current_iter)
            writer.add_scalar('val_mean_Dice', (m_ET + m_WT + m_TC) / 3.0, current_iter)
            writer.add_scalar('val_loss', v_loss, current_iter)

            # Log learned region weights matrix
            norm_W = model.kd_weights.get_normalized_weights().cpu().detach().numpy()
            print(f"    [Val Step {current_iter}] Mean Dice: {(m_ET + m_WT + m_TC) / 3.0:.4f} | ET: {m_ET:.4f}, WT: {m_WT:.4f}, TC: {m_TC:.4f} | Loss: {v_loss:.4f}")
            print(f"    Learned Region Weights (Softmax W):")
            print(f"      ET (Row 0): FLAIR={norm_W[0,0]:.3f}, T1={norm_W[0,1]:.3f}, T1ce={norm_W[0,2]:.3f}, T2={norm_W[0,3]:.3f}")
            print(f"      WT (Row 1): FLAIR={norm_W[1,0]:.3f}, T1={norm_W[1,1]:.3f}, T1ce={norm_W[1,2]:.3f}, T2={norm_W[1,3]:.3f}")
            print(f"      TC (Row 2): FLAIR={norm_W[2,0]:.3f}, T1={norm_W[2,1]:.3f}, T1ce={norm_W[2,2]:.3f}, T2={norm_W[2,3]:.3f}")

    # Final Save
    if args.local_rank == 0:
        final_ckpt = osp.join(args.snapshot_dir, 'final.pth')
        checkpoint = {
            'model': model,
            'optimizer': optimizer,
            'kd_optim': kd_optim,
            'iter': args.num_steps
        }
        torch.save(checkpoint, final_ckpt)
        print(f"============================================================")
        print(f"Training Complete! Saved final checkpoint: {final_ckpt}")
        print(f"============================================================")


if __name__ == '__main__':
    main()
