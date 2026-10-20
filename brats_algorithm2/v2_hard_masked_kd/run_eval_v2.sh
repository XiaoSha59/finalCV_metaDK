#!/bin/bash
set -e
cd /home/XiaoSha/finalCV_metaDK
mkdir -p brats_algorithm2/results
rm -f brats_algorithm2/results_eval_mean.log
export PYTHONPATH="/home/XiaoSha/finalCV_metaDK/brats_algorithm2:/home/XiaoSha/finalCV_metaDK"
echo "Starting Algorithm 2 (Shared-Mean Imputation) Evaluation with Tile Size 80x160x160..."
/home/XiaoSha/venv/bin/python brats_algorithm2/eval_v2.py \
    --checkpoint snapshots/BraTS18_phase2_alg2_no_shaspec/final.pth \
    --data_dir data \
    --test_list datalist/BraTS18/BraTS18_test.csv \
    --output_dir brats_algorithm2/results \
    --impute_mode mean \
    2>&1 | tee brats_algorithm2/results_eval_mean.log
echo "Evaluation Finished!"
