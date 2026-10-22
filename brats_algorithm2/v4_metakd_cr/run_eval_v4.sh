#!/bin/bash
set -e
set -o pipefail

V4_DIR="$(cd "$(dirname "$0")" && pwd)"
ALG2_DIR="$(cd "$V4_DIR/.." && pwd)"
REPO_DIR="$(cd "$ALG2_DIR/.." && pwd)"

if [ -d "/home/XiaoSha/finalCV_metaDK" ]; then
    WORKDIR="/home/XiaoSha/finalCV_metaDK"
    PYTHON="/home/XiaoSha/venv/bin/python"
    DATA_DIR="$WORKDIR/data"
    SNAPSHOT_BASE="$WORKDIR/snapshots"
    export PYTHONPATH="$V4_DIR:$ALG2_DIR:$WORKDIR:$PYTHONPATH"
else
    WORKDIR="$REPO_DIR"
    PYTHON="${PYTHON:-python}"
    DATA_DIR="$REPO_DIR/brats/data"
    SNAPSHOT_BASE="$REPO_DIR/brats/checkpoints"
    export PYTHONPATH="$V4_DIR:$ALG2_DIR:$REPO_DIR/brats:$REPO_DIR/brats/process:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

cd "$WORKDIR"

CKPT="${1:-$SNAPSHOT_BASE/BraTS18_phase2_v4_metakd_cr/final.pth}"
if [ ! -f "$CKPT" ]; then
    CKPT="$SNAPSHOT_BASE/BraTS18_phase2_v4_metakd_cr/last.pth"
fi

RESULTS_DIR="$V4_DIR/results"
mkdir -p "$RESULTS_DIR"

echo "Evaluating checkpoint: $CKPT"
$PYTHON "$V4_DIR/eval_v4.py" \
    --checkpoint "$CKPT" \
    --data_dir "$DATA_DIR" \
    --test_list "BraTS18_metaVal.csv" \
    --output_dir "$RESULTS_DIR"
