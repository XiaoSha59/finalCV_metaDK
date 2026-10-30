#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="/home/XiaoSha/finalCV_metaDK"
PYTHON="/home/XiaoSha/venv/bin/python"

mkdir -p "$DIR/logs" "$DIR/results_57test"

cd "$BASE_DIR"

echo "=== STARTING ENSEMBLE EVALUATION: $(date) ===" > "$DIR/logs/eval_ensemble.log"

$PYTHON "$DIR/eval_ensemble.py" \
    --ckpt_metakd "$BASE_DIR/snapshots/BraTS18_phase2_115k/final.pth" \
    --ckpt_job7 "$BASE_DIR/snapshots/BraTS18_phase2_v7_metakd_uncertainty/final.pth" \
    --data_dir "$BASE_DIR/data" \
    --test_list "datalist/BraTS18/BraTS18_test.csv" \
    --output_dir "$DIR/results_57test" >> "$DIR/logs/eval_ensemble.log" 2>&1

echo "=== ENSEMBLE EVALUATION FINISHED: $(date) ===" >> "$DIR/logs/eval_ensemble.log"

# Auto-poweroff VM after 30 seconds
sleep 30
sudo /usr/sbin/poweroff -f
