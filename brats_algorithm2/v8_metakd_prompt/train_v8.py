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

from v8_models.dual_net_v8 import DualNet_V8
from v8_losses.loss_v8 import PromptDiversityRegularizer, DiceLoss4BraTS, BCELoss4BraTS


def get_arguments():
    parser = argparse.ArgumentParser(description="MetaKD-Prompt: Modality-Aware Prompt Learning for Incomplete Multi-Modal Segmentation")
    parser.add_argument("--data_dir", type=str, default=osp.join(BRATS_DIR, "data"))
    parser.add_argument("--train_list", type=str, default="BraTS18_metaTrain.csv")
    parser.add_argument("--val_list", type=str, default="BraTS18_metaVal.csv")
    parser.add_argument("--snapshot_dir", type=str, default=osp.join(BRATS_DIR, "checkpoints", "phase2_v8_metakd_prompt"))
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

    # MetaKD-Prompt Hyperparameters
    parser.add_argument("--lambda_kd", type=float, default=0.1, help="MetaKD cross-modal feature KD weight")
    parser.add_argument("--lambda_ortho", type=float, default=0.01, help="Prompt diversity regularizer weight")
    parser.add_argument("--prompt_lr", type=float, default=1e-3, help="Learning rate for prompt parameters")

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

    model = DualNet_V8(args=args, num_classes=args.num_classes)
    model.to(device)

    # Check if final.pth already exists
    final_path = osp.join(args.snapshot_dir, "final.pth")
    if osp.exists(final_path) and osp.getsize(final_path) > 100000000:
        print(f">>> Final checkpoint already exists at {final_path}. Training complete, skipping!")
        return

    # Parameter groups: Base Backbone vs Modality Prompts
    prompt_params = [model.modality_prompts, model.prompt_alpha]
    base_params = [p for n, p in model.named_parameters() if 'kd_weights' not in n and 'modality_prompts' not in n and 'prompt_alpha' not in n]

    optimizer = torch.optim.SGD([
        {'params': base_params, 'lr': args.learning_rate},
        {'params': prompt_params, 'lr': args.prompt_lr}
    ], momentum=args.momentum, weight_decay=args.weight_decay)

    kd_optim = torch.optim.Adam(model.kd_weights.parameters(), lr=1e-3)

    # Reload from checkpoint (resume from last.pth if exists, else reload_path)
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
            if not args.reset_optimizer and 'optimizer' in saved_state:
                try:
                    opt_sd = saved_state['optimizer'].state_dict() if hasattr(saved_state['optimizer'], 'state_dict') else saved_state['optimizer']
                    optimizer.load_state_dict(opt_sd)
                    if 'kd_optim' in saved_state:
                        kd_sd = saved_state['kd_optim'].state_dict() if hasattr(saved_state['kd_optim'], 'state_dict') else saved_state['kd_optim']
                        kd_optim.load_state_dict(kd_sd)
                    print(">>> Restored optimizer state successfully")
                except Exception as e:
                    print(f">>> Fresh optimizer used (skip restore: {e})")

    # Losses
    loss_D = DiceLoss4BraTS()
    loss_BCE = BCELoss4BraTS()
    reg_diversity = PromptDiversityRegularizer()

    # Dataloaders
    input_size = tuple(map(int, args.input_size.split(',')))
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
    print(f"MetaKD-Prompt (V8) Phase 2 Training Starting...")
    print(f"Steps: {args.start_iters} -> {args.num_steps} (Total {args.num_steps - args.start_iters} steps)")
    print(f"Mode: {args.mode} | Lambda KD: {args.lambda_kd} | Lambda Ortho: {args.lambda_ortho}")
    print(f"Prompt Alpha (Initial): {model.prompt_alpha.item():.3f}")
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

        # 1. Forward Pass with Modality-Aware Prompting
        preds, mode_split, totKDLoss = model(images, mode=args.mode)

        # 2. Supervised Segmentation Task Loss
        term_seg_Dice = loss_D.forward(preds, labels)
        term_seg_BCE = loss_BCE.forward(preds, labels)
        seg_loss = term_seg_Dice + term_seg_BCE

        # 3. Prompt Orthogonality / Diversity Loss
        if args.lambda_ortho > 0:
            loss_ortho = reg_diversity(model.modality_prompts)
        else:
            loss_ortho = torch.tensor(0.0, device=device)

        # 4. Total Loss & Optimization
        total_loss = seg_loss + args.lambda_kd * totKDLoss + args.lambda_ortho * loss_ortho
        total_loss.backward()
        optimizer.step()

        if current_iter % 100 == 0 and args.local_rank == 0:
            writer.add_scalar('lr', lr, current_iter)
            writer.add_scalar('total_loss', total_loss.item(), current_iter)
            writer.add_scalar('seg_loss', seg_loss.item(), current_iter)
            writer.add_scalar('kd_loss', totKDLoss.item(), current_iter)
            writer.add_scalar('ortho_loss', loss_ortho.item(), current_iter)
            writer.add_scalar('prompt_alpha', model.prompt_alpha.item(), current_iter)
            print(f"[Iter {current_iter}/{args.num_steps}] lr={lr:.5f}, seg={seg_loss.item():.4f}, kd={totKDLoss.item():.4f}, ortho={loss_ortho.item():.4f}, alpha={model.prompt_alpha.item():.3f}")

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
