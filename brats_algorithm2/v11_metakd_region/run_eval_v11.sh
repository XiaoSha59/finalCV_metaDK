#!/bin/bash
# ==============================================================================
# Standalone Evaluator for MetaKD-V11 on All 15 Missing Modality Combinations
# Evaluates on the exact 57 held-out test cases (datalist/BraTS18/BraTS18_test.csv)
# ==============================================================================
set -e
set -o pipefail

V11_DIR="$(cd "$(dirname "$0")" && pwd)"
ALG2_DIR="$(cd "$V11_DIR/.." && pwd)"
REPO_DIR="$(cd "$ALG2_DIR/.." && pwd)"

if [ -d "/home/XiaoSha/finalCV_metaDK" ]; then
    WORKDIR="/home/XiaoSha/finalCV_metaDK"
    PYTHON="/home/XiaoSha/venv/bin/python"
    DATA_DIR="$WORKDIR/data"
    SNAPSHOT_DIR="$WORKDIR/snapshots/BraTS18_phase2_v11_metakd_region"
    export PYTHONPATH="$V11_DIR:$ALG2_DIR:$WORKDIR:$PYTHONPATH"
else
    WORKDIR="$REPO_DIR"
    PYTHON="${PYTHON:-python}"
    DATA_DIR="$REPO_DIR/brats/data"
    SNAPSHOT_DIR="$REPO_DIR/brats/checkpoints/phase2_v11_metakd_region"
    export PYTHONPATH="$V11_DIR:$ALG2_DIR:$REPO_DIR/brats:$REPO_DIR/brats/process:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

cd "$WORKDIR"

CKPT="${1:-$SNAPSHOT_DIR/final.pth}"
if [ ! -f "$CKPT" ]; then
    CKPT="$SNAPSHOT_DIR/last.pth"
fi

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found at $CKPT"
    exit 1
fi

RESULTS_DIR="$V11_DIR/results_57test"
mkdir -p "$RESULTS_DIR"

echo "=========================================================="
echo "Evaluating MetaKD-V11 Checkpoint: $CKPT"
echo "Test set: datalist/BraTS18/BraTS18_test.csv (57 cases)"
echo "Output: $RESULTS_DIR"
echo "=========================================================="

$PYTHON "$V11_DIR/eval_v11.py" \
    --checkpoint "$CKPT" \
    --data_dir "$DATA_DIR" \
    --test_list "datalist/BraTS18/BraTS18_test.csv" \
    --output_dir "$RESULTS_DIR"

echo "=========================================================="
echo "Evaluation Finished! Results exported to:"
echo "  - $RESULTS_DIR/results_v11_table1_raw.csv"
echo "  - $RESULTS_DIR/results_v11_table1_standard_pp.csv"
echo "=========================================================="
