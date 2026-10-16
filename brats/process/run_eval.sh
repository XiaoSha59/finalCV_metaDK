#!/bin/bash
PROCESS="$(cd "$(dirname "$0")" && pwd)"
BRATS="$(cd "$PROCESS/.." && pwd)"
REPO="$(cd "$BRATS/.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$PROCESS:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd "$REPO"

echo "=========================================================="
echo "RUNNING BRATS 2018 15-MODALITY BENCHMARK (TABLE 1)"
echo "Date: $(date)"
echo "=========================================================="

CKPT="$BRATS/checkpoints/phase2_115k/final.pth"
if [ ! -f "$CKPT" ]; then
    CKPT="$BRATS/checkpoints/phase2_115k/last.pth"
    if [ ! -f "$CKPT" ]; then
        echo "ERROR: no final.pth / last.pth found under $BRATS/checkpoints/phase2_115k/"
        exit 1
    fi
    echo "WARNING: final.pth not found, evaluating UNFINISHED checkpoint $CKPT (not comparable to the paper)."
fi
echo "Checkpoint: $CKPT"
mkdir -p "$BRATS/logs"

$PYTHON -u "$PROCESS/eval_all_modalities.py" \
    --restore_from "$CKPT" \
    --data_dir "$BRATS/data" \
    --data_list BraTS18_test.csv \
    --output_csv "$BRATS/logs/brats_table1_results.csv" \
    2>&1 | tee "$BRATS/logs/brats_eval.log"

echo "=========================================================="
echo "Finished evaluation at $(date)"
echo "=========================================================="
