#!/bin/bash
# =============================================================
# MetaKD BraTS 2018 — 2-Phase Training Pipeline
# Follows the EXACT methodology from billhhh/MetaKD run.sh:
#   Phase 1 (0 ? 80k):   Full 4 modalities  --mode=0,1,2,3
#   Phase 2 (80k ? 115k): Missing modality  --mode=random
# =============================================================
set -e
set -o pipefail

PYTHON="/home/XiaoSha/venv/bin/python"
WORKDIR="/home/XiaoSha/finalCV_metaDK"
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd "$WORKDIR"

mkdir -p logs snapshots/BraTS18_phase1_80k snapshots/BraTS18_phase2_115k

echo "=========================================================="
echo "MetaKD BraTS 2018 — 2-Phase Training"
echo "Date: $(date)"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "=========================================================="

PHASE1_SNAP="snapshots/BraTS18_phase1_80k"
PHASE1_CKPT="$PHASE1_SNAP/final.pth"

if [ -f "$PHASE1_CKPT" ]; then
    echo ""
    echo ">>> Phase 1 checkpoint found ($PHASE1_CKPT), skipping Phase 1."
else
    echo ""
    echo "=========================================================="
    echo ">>> [PHASE 1/2] Full-Modality Warmup: 0 to 80,000 steps"
    echo "    mode=0,1,2,3 | lr=1e-2 | momentum=0.99 | batch_size=2"
    echo "=========================================================="
    $PYTHON train.py \
        --snapshot_dir "$PHASE1_SNAP/" \
        --reload_from_checkpoint False \
        --num_steps 80000 \
        --val_pred_every 2000 \
        --batch_size 2 \
        --learning_rate 1e-2 \
        --mode "0,1,2,3" \
        --train_only \
        2>&1 | tee logs/brats_phase1.log
    echo ">>> Phase 1 completed at $(date)"
fi

PHASE2_SNAP="snapshots/BraTS18_phase2_115k"

echo ""
echo "=========================================================="
echo ">>> [PHASE 2/2] Missing-Modality MetaKD: 80k to 115,000 steps"
echo "    mode=random | lr=1e-2 | momentum=0.99 | batch_size=2"
echo "    Reload from: $PHASE1_CKPT"
echo "=========================================================="

RELOAD_PATH="$PHASE1_CKPT"
if [ -f "$PHASE2_SNAP/last.pth" ]; then
    RELOAD_PATH="$PHASE2_SNAP/last.pth"
    echo "    (Resuming interrupted Phase 2 from: $RELOAD_PATH)"
fi

$PYTHON train.py \
    --snapshot_dir "$PHASE2_SNAP/" \
    --reload_from_checkpoint True \
    --reload_path "$RELOAD_PATH" \
    --num_steps 115000 \
    --val_pred_every 2000 \
    --batch_size 2 \
    --learning_rate 1e-2 \
    --mode "random" \
    2>&1 | tee logs/brats_phase2.log

echo ""
echo "=========================================================="
echo "BraTS 2-Phase Training COMPLETED at $(date)"
echo "  Best model : $PHASE2_SNAP/best_model.pth"
echo "  Final model: $PHASE2_SNAP/final.pth"
echo "=========================================================="
