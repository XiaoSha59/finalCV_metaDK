#!/bin/bash
set -e
export PYTHONPATH="/home/XiaoSha/finalCV_metaDK:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd /home/XiaoSha/finalCV_metaDK

echo "=========================================================="
echo "🚀 STARTING BRATS 2018 METAKD PRODUCTION RUN (84k -> 115k)"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU Details:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "=========================================================="

/home/XiaoSha/venv/bin/python train.py \
    --reload_from_checkpoint True \
    --reload_path checkpoints/last_84k.pth \
    --num_steps 115000 \
    --val_pred_every 500 \
    --batch_size 2 \
    --mode random \
    2>&1 | tee logs/brats_train.log

echo ""
echo "=========================================================="
echo "🎉 BRATS 2018 METAKD TRAINING COMPLETED AT $(date)"
echo "Checkpoints saved in snapshots/BraTS18_MetaKD_115k/"
echo "Shutting down VM in 30 seconds..."
echo "=========================================================="
sleep 30
sudo poweroff
