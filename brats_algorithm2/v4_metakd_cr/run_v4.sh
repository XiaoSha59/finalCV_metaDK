#!/bin/bash
# ==============================================================================
# Runner for MetaKD-CR (V4): Consistency Regularization Self-Distillation
# Phase 2 Finetuning (80k -> 115k steps) from Phase 1 Teacher
# Fully self-contained inside brats_algorithm2/v4_metakd_cr/
# ==============================================================================
set -e
set -o pipefail

V4_DIR="$(cd "$(dirname "$0")" && pwd)"
ALG2_DIR="$(cd "$V4_DIR/.." && pwd)"
REPO_DIR="$(cd "$ALG2_DIR/.." && pwd)"

if [ -d "/home/XiaoSha/finalCV_metaDK" ]; then
    WORKDIR="/home/XiaoSha/finalCV_metaDK"
    PYTHON="/home/XiaoSha/venv/bin/python"
    DATA_DIR="$WORKDIR/data"
    PHASE1_CKPT="$WORKDIR/snapshots/BraTS18_phase1_80k/final.pth"
    SNAPSHOT_BASE="$WORKDIR/snapshots"
    export PYTHONPATH="$V4_DIR:$ALG2_DIR:$WORKDIR:$PYTHONPATH"
else
    WORKDIR="$REPO_DIR"
    PYTHON="${PYTHON:-python}"
    DATA_DIR="$REPO_DIR/brats/data"
    PHASE1_CKPT="$REPO_DIR/brats/checkpoints/phase1_80k/final.pth"
    SNAPSHOT_BASE="$REPO_DIR/brats/checkpoints"
    export PYTHONPATH="$V4_DIR:$ALG2_DIR:$REPO_DIR/brats:$REPO_DIR/brats/process:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

cd "$WORKDIR"

# Fast RAM caching for cloud instances
if [ -d "$DATA_DIR/cache_brats" ] && [ ! -d "/dev/shm/cache_brats" ]; then
    echo ">>> Populating RAM cache /dev/shm/cache_brats..."
    cp -r "$DATA_DIR/cache_brats" /dev/shm/cache_brats
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
NUM_WORKERS="${NUM_WORKERS:-6}"
ENABLE_SHUTDOWN="${ENABLE_SHUTDOWN:-true}"

cleanup_and_poweroff() {
    EXIT_CODE=$?
    echo ""
    echo "=========================================================="
    echo "[$(date)] MetaKD-CR Runner exited with status: $EXIT_CODE"
    echo "=========================================================="
    sync
    if [ "$ENABLE_SHUTDOWN" = "true" ]; then
        if [ $EXIT_CODE -ne 0 ]; then
            echo ">>> WARNING: Pipeline exited with error ($EXIT_CODE)."
            echo ">>> Powering off VM in 60s to protect user budget..."
            sleep 60
        else
            echo ">>> SUCCESS: Pipeline completed cleanly! Shutting down in 30s to protect user budget..."
            sleep 30
        fi
        sudo /usr/sbin/poweroff -f || sudo shutdown -h now
    fi
}
trap cleanup_and_poweroff EXIT

SNAP_NAME="BraTS18_phase2_v4_metakd_cr"
SNAPSHOT_DIR="$SNAPSHOT_BASE/$SNAP_NAME"
LOGS_DIR="$V4_DIR/logs"
RESULTS_DIR="$V4_DIR/results"
mkdir -p "$SNAPSHOT_DIR" "$LOGS_DIR" "$RESULTS_DIR"

RELOAD_PATH="$PHASE1_CKPT"
RELOAD_FLAG="True"

if [ -f "$SNAPSHOT_DIR/last.pth" ]; then
    RELOAD_PATH="$SNAPSHOT_DIR/last.pth"
    echo ">>> Resuming from existing intermediate checkpoint: $RELOAD_PATH"
elif [ -f "$PHASE1_CKPT" ]; then
    echo ">>> Initializing Phase 2 from Phase 1 Teacher: $PHASE1_CKPT"
else
    echo ">>> WARNING: Phase 1 checkpoint not found at $PHASE1_CKPT."
    RELOAD_FLAG="False"
fi

echo "=========================================================="
echo "Host: $(hostname) | Date: $(date)"
echo "Starting MetaKD-CR (V4) Phase 2 Training"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true
echo "=========================================================="

# Run Phase 2 Training (80k -> 115k)
$PYTHON "$V4_DIR/train_v4.py" \
    --snapshot_dir "$SNAPSHOT_DIR" \
    --reload_from_checkpoint "$RELOAD_FLAG" \
    --reload_path "$RELOAD_PATH" \
    --data_dir "$DATA_DIR" \
    --train_list "datalist/BraTS18/BraTS18_metaTrain.csv" \
    --val_list "datalist/BraTS18/BraTS18_metaVal.csv" \
    --num_steps 115000 \
    --start_iters 80000 \
    --val_pred_every 500 \
    --batch_size 2 \
    --num_workers "$NUM_WORKERS" \
    --learning_rate 1e-2 \
    --mode "random" \
    --lambda_kd 0.1 \
    --lambda_cr 0.05 \
    --cr_temperature 4.0 \
    --cr_lambda_dice 1.0 \
    --cr_lambda_bce 0.5 \
    2>&1 | tee "$LOGS_DIR/train_v4.log"

echo ""
echo "=========================================================="
echo "Training Finished. Starting Full 15-Modality Evaluation..."
echo "=========================================================="

EVAL_CKPT="$SNAPSHOT_DIR/final.pth"
if [ ! -f "$EVAL_CKPT" ]; then
    EVAL_CKPT="$SNAPSHOT_DIR/last.pth"
fi

$PYTHON "$V4_DIR/eval_v4.py" \
    --checkpoint "$EVAL_CKPT" \
    --data_dir "$DATA_DIR" \
    --test_list "datalist/BraTS18/BraTS18_metaVal.csv" \
    --output_dir "$RESULTS_DIR" \
    2>&1 | tee "$LOGS_DIR/eval_v4.log"

echo ""
echo "=========================================================="
echo "Evaluation Finished! Summary saved to $RESULTS_DIR"
echo "=========================================================="
