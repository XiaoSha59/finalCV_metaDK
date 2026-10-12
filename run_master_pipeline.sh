#!/bin/bash
# =============================================================
# MetaKD Master Pipeline: BraTS 2-Phase + ADNI + Auto Shutdown
# =============================================================
set -e
set -o pipefail

PYTHON="/home/XiaoSha/venv/bin/python"
WORKDIR="/home/XiaoSha/finalCV_metaDK"
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd "$WORKDIR"

mkdir -p logs checkpoints data/BraTS2018 adni_raw_flexmoe

echo "=========================================================="
echo "MetaKD MASTER PIPELINE: BraTS 2-Phase → ADNI → POWEROFF"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
echo "=========================================================="

# ----------------------------------------------------------
# STEP 1: BraTS 2018 — 2-Phase MetaKD Training
#   Phase 1: 80k steps, mode=0,1,2,3 (Full Modality Warmup)
#   Phase 2: 80k→115k steps, mode=random (Missing Modality KD)
# ----------------------------------------------------------
echo ""
echo "=========================================================="
echo ">>> [1/2] BraTS 2018 — 2-Phase MetaKD Training"
echo "=========================================================="
bash run_brats_2phase.sh

echo ">>> [1/2] BraTS 2018 COMPLETED at $(date)"

# ----------------------------------------------------------
# STEP 2: ADNI Multimodal Benchmark (Table 2 Reproduction)
# ----------------------------------------------------------
echo ""
echo "=========================================================="
echo ">>> [2/2] ADNI Multimodal Training & Benchmark"
echo "=========================================================="
if [ ! -f "logs/adni_table2_results.csv" ]; then
    $PYTHON -u run_adni_pipeline.py 2>&1 | tee logs/adni_train.log
else
    echo "ADNI Benchmark already done. (Found logs/adni_table2_results.csv)"
fi

echo ">>> [2/2] ADNI COMPLETED at $(date)"

# ----------------------------------------------------------
# STEP 3: Automatic VM Shutdown
# ----------------------------------------------------------
echo ""
echo "=========================================================="
echo "ALL TASKS FINISHED at $(date)!"
echo "Shutting down VM in 30 seconds..."
echo "=========================================================="
sleep 30
sudo /usr/sbin/poweroff -f
