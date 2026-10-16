#!/bin/bash
PYTHON="/home/XiaoSha/venv/bin/python"
WORKDIR="/home/XiaoSha/finalCV_metaDK"
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=========================================================="
echo "🚀 RUNNING BRATS 2018 15-MODALITY BENCHMARK (TABLE 1)"
echo "Date: $(date)"
echo "=========================================================="

# The author evaluates the final 115k-step model; no checkpoint selection on the test set.
CKPT="snapshots/BraTS18_phase2_115k/final.pth"
if [ ! -f "$CKPT" ]; then
    CKPT="snapshots/BraTS18_MetaKD_115k/final.pth"
fi
if [ ! -f "$CKPT" ]; then
    CKPT="snapshots/BraTS18_phase2_115k/last.pth"
    if [ ! -f "$CKPT" ]; then
        echo "ERROR: no final.pth / last.pth found, nothing to evaluate."
        exit 1
    fi
    echo "WARNING: final.pth not found, evaluating UNFINISHED checkpoint $CKPT (not comparable to the paper)."
fi
echo "Checkpoint: $CKPT"
mkdir -p logs

$PYTHON -u eval_all_modalities.py \
    --restore_from "$CKPT" \
    --data_dir ./datalist/ \
    --data_list BraTS18/BraTS18_test.csv \
    --output_csv logs/brats_table1_results.csv \
    2>&1 | tee logs/brats_eval.log

echo "=========================================================="
echo "Finished evaluation at $(date)"
echo "=========================================================="
