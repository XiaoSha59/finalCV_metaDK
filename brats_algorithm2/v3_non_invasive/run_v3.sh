#!/bin/bash
# ==============================================================================
# Runner for DualNet-V3: Non-Invasive ShaSpec + Soft Region-Aware KD
# Phase 2 Finetuning (80k -> 115k steps) from Phase 1 Teacher
# Fully self-contained inside brats_algorithm2/v3_non_invasive/
# ==============================================================================
set -e
set -o pipefail

V3_DIR="$(cd "$(dirname "$0")" && pwd)"
ALG2_DIR="$(cd "$V3_DIR/.." && pwd)"
REPO_DIR="$(cd "$ALG2_DIR/.." && pwd)"

if [ -d "/home/XiaoSha/finalCV_metaDK" ]; then
    WORKDIR="/home/XiaoSha/finalCV_metaDK"
    PYTHON="/home/XiaoSha/venv/bin/python"
    DATA_DIR="$WORKDIR/data"
    PHASE1_CKPT="$WORKDIR/snapshots/BraTS18_phase1_80k/final.pth"
    SNAPSHOT_BASE="$WORKDIR/snapshots"
    export PYTHONPATH="$V3_DIR:$ALG2_DIR:$WORKDIR:$PYTHONPATH"
else
    WORKDIR="$REPO_DIR"
    PYTHON="${PYTHON:-python}"
    DATA_DIR="$REPO_DIR/brats/data"
    PHASE1_CKPT="$REPO_DIR/brats/checkpoints/phase1_80k/final.pth"
    SNAPSHOT_BASE="$REPO_DIR/brats/checkpoints"
    export PYTHONPATH="$V3_DIR:$ALG2_DIR:$REPO_DIR/brats:$REPO_DIR/brats/process:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

cd "$WORKDIR"

# Fast RAM caching for cloud instances
if [ -d "$DATA_DIR/cache_brats" ] && [ ! -d "/dev/shm/cache_brats" ]; then
    echo ">>> Populating RAM cache /dev/shm/cache_brats..."
    cp -r "$DATA_DIR/cache_brats" /dev/shm/cache_brats
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
NUM_WORKERS="${NUM_WORKERS:-6}"
ENABLE_SHUTDOWN="${ENABLE_SHUTDOWN:-false}"

cleanup_and_poweroff() {
    EXIT_CODE=$?
    echo ""
    echo "=========================================================="
    echo "[$(date)] DualNet-V3 Runner exited with status: $EXIT_CODE"
    echo "=========================================================="
    sync
    if [ "$ENABLE_SHUTDOWN" = "true" ]; then
        if [ $EXIT_CODE -ne 0 ]; then
            echo ">>> WARNING: Pipeline exited with error ($EXIT_CODE). Keeping VM alive for 5 minutes for inspection..."
            sleep 300
        else
            echo ">>> SUCCESS: Pipeline completed cleanly! Shutting down in 30s to protect budget..."
            sleep 30
        fi
        sudo /usr/sbin/poweroff -f || sudo shutdown -h now
    fi
}
trap cleanup_and_poweroff EXIT

SNAP_NAME="BraTS18_phase2_v3_non_invasive"
SNAPSHOT_DIR="$SNAPSHOT_BASE/$SNAP_NAME"
LOGS_DIR="$V3_DIR/logs"
mkdir -p "$SNAPSHOT_DIR" "$LOGS_DIR"

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
echo "Starting DualNet-V3: Non-Invasive ShaSpec + Soft Region KD"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true
echo "=========================================================="

$PYTHON "$V3_DIR/train_v3.py" \
    --snapshot_dir "$SNAPSHOT_DIR" \
    --reload_from_checkpoint "$RELOAD_FLAG" \
    --reload_path "$RELOAD_PATH" \
    --data_dir "$DATA_DIR" \
    --train_list BraTS18_metaTrain.csv \
    --val_list BraTS18_metaVal.csv \
    --num_steps 115000 \
    --start_iters 80000 \
    --val_pred_every 500 \
    --batch_size 2 \
    --num_workers "$NUM_WORKERS" \
    --learning_rate 1e-2 \
    --mode "random" \
    --use_shaspec "True" \
    --lambda_kd 0.1 \
    --lambda_ortho 0.05 \
    --lambda_shared 0.05 \
    --tumor_boost 2.0 \
    2>&1 | tee -a "$LOGS_DIR/${SNAP_NAME}.log"

echo ""
echo "=========================================================="
echo "DualNet-V3 Training completed successfully at $(date)!"
echo "Checkpoints saved in: $SNAPSHOT_DIR"
echo "=========================================================="

echo ""
echo "=========================================================="
echo "Starting Automated 15-Modality Benchmark Evaluation..."
echo "=========================================================="
bash "$V3_DIR/run_eval_v3.sh"

