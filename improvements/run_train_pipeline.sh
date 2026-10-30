#!/usr/bin/env bash
# ==============================================================================
# Pipeline Runner for Plan 1 Extended (CDU-KD on BraTS 2018)
# Executes:
#   1. Phase 1 & 2 End-to-End Training (50,000 steps)
#   2. Fast Fine-Tuning with Unblocked ET Gradient (50,000 -> 51,500 steps, tau_et=0.25)
#   3. Full 15-Modality Benchmark Evaluation with Official Medical Post-Processing
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SNAPSHOT_DIR="$REPO_DIR/snapshots/plan1_extended_50k"
FINETUNE_DIR="$REPO_DIR/snapshots/plan1_extended_finetune"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$SNAPSHOT_DIR" "$FINETUNE_DIR" "$RESULTS_DIR"

echo "========================================================================"
echo "Step 1: Training Plan 1 Extended (50,000 steps)"
echo "========================================================================"
python -u "$SCRIPT_DIR/train_plan1_extended.py" \
    --snapshot_dir "$SNAPSHOT_DIR" \
    --data_dir "$REPO_DIR/data" \
    --train_list "$REPO_DIR/datalist/BraTS18/BraTS18_metaTrain.csv" \
    --val_list "$REPO_DIR/datalist/BraTS18/BraTS18_metaVal.csv" \
    --start_iters 0 \
    --num_steps 50000 \
    --phase1_steps 25000 \
    --batch_size 2 \
    --learning_rate 1e-2 \
    --lambda_kd 0.1 \
    --lambda_ukd 0.05 \
    --tau_et 0.50 \
    --tau_tc 0.50 \
    --tau_wt -1.0 \
    --mode random

echo "========================================================================"
echo "Step 2: Fast Fine-Tuning ET Gradient Unblocking (50,000 -> 51,500 steps)"
echo "========================================================================"
python -u "$SCRIPT_DIR/train_plan1_extended.py" \
    --snapshot_dir "$FINETUNE_DIR" \
    --reload_from_checkpoint \
    --reload_path "$SNAPSHOT_DIR/final.pth" \
    --reset_optimizer \
    --data_dir "$REPO_DIR/data" \
    --train_list "$REPO_DIR/datalist/BraTS18/BraTS18_metaTrain.csv" \
    --val_list "$REPO_DIR/datalist/BraTS18/BraTS18_metaVal.csv" \
    --start_iters 50000 \
    --num_steps 51500 \
    --phase1_steps 25000 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --lambda_kd 0.1 \
    --lambda_ukd 0.05 \
    --tau_et 0.25 \
    --tau_tc 0.50 \
    --tau_wt -1.0 \
    --mode random

echo "========================================================================"
echo "Step 3: 15-Modality Evaluation with Official Medical Post-Processing"
echo "========================================================================"
python -u "$SCRIPT_DIR/eval_plan1_extended.py" \
    --data_dir "$REPO_DIR/data" \
    --data_list "$REPO_DIR/datalist/BraTS18/BraTS18_test.csv" \
    --restore_from "$FINETUNE_DIR/final.pth" \
    --output_csv "$RESULTS_DIR/results_plan1_extended_eval.csv" \
    --postprocess true

echo "========================================================================"
echo "Pipeline finished successfully! Results saved to $RESULTS_DIR"
echo "========================================================================"
