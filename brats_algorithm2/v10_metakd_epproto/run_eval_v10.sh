#!/bin/bash
set -e
WORKDIR="/home/XiaoSha/finalCV_metaDK"
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR/brats_algorithm2/v10_metakd_epproto:$WORKDIR/brats_algorithm2:$WORKDIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

V10_DIR="$WORKDIR/brats_algorithm2/v10_metakd_epproto"
FINAL_CKPT="$WORKDIR/snapshots/BraTS18_phase2_v10_metakd_epproto/final.pth"
PYTHON="/home/XiaoSha/venv/bin/python"

mkdir -p "$V10_DIR/logs"
mkdir -p "$V10_DIR/results_57test"

echo ">>> Starting V10 Evaluation at $(date)..."
$PYTHON "$V10_DIR/eval_v10.py" \
    --checkpoint "$FINAL_CKPT" \
    --data_dir "$WORKDIR/data" \
    --test_list "datalist/BraTS18/BraTS18_test.csv" \
    --output_dir "$V10_DIR/results_57test" \
    2>&1 | tee "$V10_DIR/logs/eval_v10_fixed.log"

echo ">>> Evaluation completed successfully at $(date)!"
