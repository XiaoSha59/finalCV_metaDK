#!/bin/bash
set -u

WORKDIR="/home/XiaoSha/finalCV_metaDK"
cd "$WORKDIR"

export PYTHONPATH="$WORKDIR/brats_algorithm2/v11_metakd_region:$WORKDIR/brats_algorithm2:$WORKDIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

V11_DIR="$WORKDIR/brats_algorithm2/v11_metakd_region"
LOG_DIR="$V11_DIR/logs"
mkdir -p "$LOG_DIR"
mkdir -p "$V11_DIR/results_57test"

LOG="$LOG_DIR/run_v11_master.log"
exec >> "$LOG" 2>&1

echo "=========================================================="
echo "STARTING METAKD-REGION (JOB 11: V11) MASTER PIPELINE"
echo "Time: $(date)"
echo "Host: $(hostname) | User: $(whoami)"
echo "=========================================================="

# 1. Check RAM Cache
if [ -d "$WORKDIR/data/cache_brats" ] && [ ! -d "/dev/shm/cache_brats" ]; then
    echo ">>> Populating RAM cache /dev/shm/cache_brats..."
    cp -r "$WORKDIR/data/cache_brats" /dev/shm/cache_brats || true
fi

PYTHON="/home/XiaoSha/venv/bin/python"
SNAPSHOT_DIR="$WORKDIR/snapshots/BraTS18_phase2_v11_metakd_region"
mkdir -p "$SNAPSHOT_DIR"

# 2. Phase 2 Training (80,000 -> 115,000 steps)
FINAL_CKPT="$SNAPSHOT_DIR/final.pth"
if [ ! -f "$FINAL_CKPT" ] || [ $(stat -c%s "$FINAL_CKPT" 2>/dev/null || echo 0) -lt 100000000 ]; then
    echo ">>> [STAGE 1/2] Starting V11 Training (80k -> 115k steps)..."
    $PYTHON "$V11_DIR/train_v11.py" \
        --data_dir "$WORKDIR/data" \
        --train_list "datalist/BraTS18/BraTS18_metaTrain.csv" \
        --val_list "datalist/BraTS18/BraTS18_metaVal.csv" \
        --reload_path "$WORKDIR/snapshots/BraTS18_phase1_80k/final.pth" \
        --snapshot_dir "$SNAPSHOT_DIR" \
        --batch_size 2 \
        --num_workers 6 \
        --learning_rate 1e-2 \
        --num_steps 115000 \
        --start_iters 80000 \
        --lambda_kd 0.1 \
        --lambda_ukd 0.05 \
        --tau_ET 0.5 \
        --tau_WT 1.0 \
        --tau_TC 0.5 \
        --ukd_lambda_dice 1.0 \
        --ukd_lambda_bce 0.5
    TRAIN_EXIT=$?
    if [ $TRAIN_EXIT -ne 0 ] || [ ! -f "$FINAL_CKPT" ]; then
        echo ">>> ERROR: V11 Training failed (exit code $TRAIN_EXIT). Aborting pipeline!"
        exit 1
    fi
    echo ">>> V11 Training finished at $(date)!"
else
    echo ">>> V11 Final checkpoint already exists at $FINAL_CKPT. Skipping training!"
fi

# 3. 57-Case Full 15-Modality Evaluation
echo ">>> [STAGE 2/2] Running V11 Evaluation across all 57 test cases..."
$PYTHON "$V11_DIR/eval_v11.py" \
    --checkpoint "$FINAL_CKPT" \
    --data_dir "$WORKDIR/data" \
    --test_list "datalist/BraTS18/BraTS18_test.csv" \
    --output_dir "$V11_DIR/results_57test"

echo "=========================================================="
echo "JOB 11 (V11) COMPLETED SUCCESSFULLY AT $(date)!"
echo "Verifying result files:"
ls -lh "$V11_DIR/results_57test/results_v11_table1_raw.csv" || true
ls -lh "$V11_DIR/results_57test/results_v11_table1_standard_pp.csv" || true
ls -lh "$V11_DIR/results_57test/results_v11_table1.csv" || true
echo "=========================================================="

# 4. Safe Auto-Poweroff (protect GPU budget)
echo "Cleaning all cron jobs and shutting down VM in 30 seconds to protect budget..."
crontab -r 2>/dev/null || true
sync
sleep 30
echo "POWERING OFF VM NOW: sudo /usr/sbin/poweroff -f"
sudo /usr/sbin/poweroff -f || sudo shutdown -h now
