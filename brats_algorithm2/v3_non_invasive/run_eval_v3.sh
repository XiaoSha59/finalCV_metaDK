#!/bin/bash
# ==============================================================================
# Runner for DualNet-V3 15-Modality Benchmark Evaluation
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
    TEST_LIST="$WORKDIR/datalist/BraTS18/BraTS18_test.csv"
    CKPT="$WORKDIR/snapshots/BraTS18_phase2_v3_non_invasive/final.pth"
    OUTPUT_DIR="$V3_DIR/results"
    export PYTHONPATH="$V3_DIR:$ALG2_DIR:$WORKDIR:$PYTHONPATH"
else
    WORKDIR="$REPO_DIR"
    PYTHON="${PYTHON:-python}"
    DATA_DIR="$REPO_DIR/brats/data"
    TEST_LIST="$REPO_DIR/brats/data/BraTS18_test.csv"
    CKPT="$ALG2_DIR/snapshots/BraTS18_phase2_v3_non_invasive/final.pth"
    OUTPUT_DIR="$V3_DIR/results"
    export PYTHONPATH="$V3_DIR:$ALG2_DIR:$REPO_DIR/brats:$REPO_DIR/brats/process:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

cd "$WORKDIR"
mkdir -p "$OUTPUT_DIR"

echo "=========================================================="
echo "Starting Evaluation: DualNet-V3 (Non-Invasive ShaSpec + Soft Region KD)..."
echo "Checkpoint: $CKPT"
echo "=========================================================="

$PYTHON "$V3_DIR/eval_v3.py" \
    --checkpoint "$CKPT" \
    --data_dir "$DATA_DIR" \
    --test_list "$TEST_LIST" \
    --output_dir "$OUTPUT_DIR" \
    --use_shaspec "True" \
    2>&1 | tee "$OUTPUT_DIR/eval_v3.log"

echo "=========================================================="
echo "Evaluation Finished! Output saved in: $OUTPUT_DIR"
echo "=========================================================="
