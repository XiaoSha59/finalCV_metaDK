#!/bin/bash
# ==============================================================================
# Runner for Algorithm 2: Region-Aware Shared-Specific MetaKD (MetaKD-v2)
# Fully self-contained inside brats_algorithm2/ — zero modifications to brats/
# ==============================================================================
set -e
set -o pipefail

ALG2_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$ALG2_DIR/.." && pwd)"

# Detect if running on GCP cloud instance vs local machine
if [ -d "/home/XiaoSha/finalCV_metaDK" ]; then
    WORKDIR="/home/XiaoSha/finalCV_metaDK"
    PYTHON="/home/XiaoSha/venv/bin/python"
    DATA_DIR="$WORKDIR/data"
    PHASE1_CKPT="$WORKDIR/snapshots/BraTS18_phase1_80k/final.pth"
    SNAPSHOT_BASE="$WORKDIR/snapshots"
    export PYTHONPATH="$ALG2_DIR:$WORKDIR:$PYTHONPATH"
else
    WORKDIR="$REPO_DIR"
    PYTHON="${PYTHON:-python}"
    DATA_DIR="$REPO_DIR/brats/data"
    PHASE1_CKPT="$REPO_DIR/brats/checkpoints/phase1_80k/final.pth"
    SNAPSHOT_BASE="$REPO_DIR/brats/checkpoints"
    export PYTHONPATH="$ALG2_DIR:$REPO_DIR/brats:$REPO_DIR/brats/process:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

cd "$WORKDIR"

if [ -d "$DATA_DIR/cache_brats" ] && [ ! -d "/dev/shm/cache_brats" ]; then
    echo ">>> Populating RAM cache /dev/shm/cache_brats..."
    cp -r "$DATA_DIR/cache_brats" /dev/shm/cache_brats
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
NUM_WORKERS="${NUM_WORKERS:-6}"
ENABLE_SHUTDOWN="${ENABLE_SHUTDOWN:-false}"

# ------------------------------------------------------------------------------
# Auto-Shutdown & Resource Protection Trap
# ------------------------------------------------------------------------------
cleanup_and_poweroff() {
    EXIT_CODE=$?
    echo ""
    echo "=========================================================="
    echo "[$(date)] Algorithm 2 Runner exited with status: $EXIT_CODE"
    echo "=========================================================="
    sync
    sleep 10
    if [ "$ENABLE_SHUTDOWN" = "true" ]; then
        echo ">>> AUTO-SHUTDOWN ACTIVATED: Powering off VM to protect budget..."
        sudo /usr/sbin/poweroff -f || sudo shutdown -h now
    fi
}
trap cleanup_and_poweroff EXIT

# ------------------------------------------------------------------------------
# Experiment Configuration
# TRACK 1 (Default): use_shaspec=True  (Experiment 1: With ShaSpec)
# TRACK 2:           use_shaspec=False (Experiment 2: Without ShaSpec)
# ------------------------------------------------------------------------------
USE_SHASPEC="${1:-True}"
if [ "$USE_SHASPEC" = "True" ]; then
    SNAP_NAME="BraTS18_phase2_alg2_shaspec"
    echo ">>> Running Track 1: Algorithm 2 WITH ShaSpec"
else
    SNAP_NAME="BraTS18_phase2_alg2_no_shaspec"
    echo ">>> Running Track 2: Algorithm 2 WITHOUT ShaSpec (Ablation)"
fi

SNAPSHOT_DIR="$SNAPSHOT_BASE/$SNAP_NAME"
LOGS_DIR="$ALG2_DIR/logs"
mkdir -p "$SNAPSHOT_DIR" "$LOGS_DIR"

RELOAD_PATH="$PHASE1_CKPT"
RELOAD_FLAG="True"

# Auto-resume from latest checkpoint if interrupted
if [ -f "$SNAPSHOT_DIR/last.pth" ]; then
    RELOAD_PATH="$SNAPSHOT_DIR/last.pth"
    echo ">>> Resuming from existing intermediate checkpoint: $RELOAD_PATH"
elif [ -f "$PHASE1_CKPT" ]; then
    echo ">>> Initializing Phase 2 from Phase 1 Teacher: $PHASE1_CKPT"
else
    echo ">>> WARNING: Phase 1 checkpoint not found at $PHASE1_CKPT. Training from scratch."
    RELOAD_FLAG="False"
fi

echo "=========================================================="
echo "Host: $(hostname) | Date: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true
echo "=========================================================="

$PYTHON "$ALG2_DIR/train_v1.py" \
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
    --use_shaspec "$USE_SHASPEC" \
    --lambda_kd 1.0 \
    --lambda_ortho 0.1 \
    --lambda_shared 0.1 \
    2>&1 | tee -a "$LOGS_DIR/${SNAP_NAME}.log"


echo ""
echo "=========================================================="
echo "Algorithm 2 Training completed successfully at $(date)!"
echo "Checkpoints saved in: $SNAPSHOT_DIR"
echo "=========================================================="
