#!/bin/bash
set -u

WORKDIR="/home/XiaoSha/finalCV_metaDK"
cd "$WORKDIR"

export PYTHONPATH="$WORKDIR/brats_algorithm2/v10_metakd_epproto:$WORKDIR/brats_algorithm2:$WORKDIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

V10_DIR="$WORKDIR/brats_algorithm2/v10_metakd_epproto"
LOG_DIR="$V10_DIR/logs"
mkdir -p "$LOG_DIR"
mkdir -p "$V10_DIR/results_57test"

LOG="$LOG_DIR/run_v10_master.log"
exec >> "$LOG" 2>&1

echo "=========================================================="
echo "STARTING METAKD-EPPROTO (JOB 10: V10) MASTER PIPELINE"
echo "Time: $(date)"
echo "Host: $(hostname) | User: $(whoami)"
echo "=========================================================="

# 1. Check RAM Cache
if [ -d "$WORKDIR/data/cache_brats" ] && [ ! -d "/dev/shm/cache_brats" ]; then
    echo ">>> Populating RAM cache /dev/shm/cache_brats..."
    cp -r "$WORKDIR/data/cache_brats" /dev/shm/cache_brats || true
fi

PYTHON="/home/XiaoSha/venv/bin/python"
SNAPSHOT_DIR="$WORKDIR/snapshots/BraTS18_phase2_v10_metakd_epproto"
mkdir -p "$SNAPSHOT_DIR"

# 2. Phase 2 Training (80,000 -> 115,000 steps)
FINAL_CKPT="$SNAPSHOT_DIR/final.pth"
if [ ! -f "$FINAL_CKPT" ] || [ $(stat -c%s "$FINAL_CKPT" 2>/dev/null || echo 0) -lt 100000000 ]; then
    echo ">>> [STAGE 1/2] Starting V10 Training..."
    $PYTHON "$V10_DIR/train_v10.py" \
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
        --lambda_proto 0.05 \
        --lambda_sep 0.05 \
        --proto_margin 0.5 \
        --lambda_ukd 0.05 \
        --tau_unc 0.5 \
        --min_wt_conf 0.70 \
        --lambda_topo 0.0
    TRAIN_EXIT=$?
    if [ $TRAIN_EXIT -ne 0 ] || [ ! -f "$FINAL_CKPT" ]; then
        echo ">>> ERROR: V10 Training failed (exit code $TRAIN_EXIT). Aborting pipeline!"
        exit 1
    fi
    echo ">>> V10 Training finished at $(date)!"
else
    echo ">>> V10 Final checkpoint already exists at $FINAL_CKPT. Skipping training!"
fi

# 3. 57-Case Full 15-Modality Evaluation
echo ">>> [STAGE 2/2] Running V10 Evaluation across 57 test cases..."
$PYTHON "$V10_DIR/eval_v10.py" \
    --checkpoint "$FINAL_CKPT" \
    --data_dir "$WORKDIR/data" \
    --test_list "datalist/BraTS18/BraTS18_test.csv" \
    --output_dir "$V10_DIR/results_57test"

echo "=========================================================="
echo "JOB 10 (V10) COMPLETED SUCCESSFULLY AT $(date)!"
echo "Verifying result files:"
ls -lh "$V10_DIR/results_57test/results_v10_table1_raw.csv" || true
ls -lh "$V10_DIR/results_57test/results_v10_table1_standard_pp.csv" || true
ls -lh "$V10_DIR/results_57test/results_v10_table1.csv" || true
echo "=========================================================="

# 4. Safe Auto-Poweroff
echo "Cleaning all cron jobs and shutting down VM in 30 seconds..."
crontab -r 2>/dev/null || true
sync
sleep 30
echo "POWERING OFF VM NOW: sudo /usr/sbin/poweroff -f"
sudo /usr/sbin/poweroff -f || sudo shutdown -h now
