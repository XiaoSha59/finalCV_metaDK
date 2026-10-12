#!/bin/bash
TARGET_PID=$(pgrep -f "train.py" | head -n 1)
if [ -z "$TARGET_PID" ]; then
    echo "[$(date)] No train.py found" >> /home/XiaoSha/finalCV_metaDK/logs/train.log
    exit 1
fi
echo "[$(date)] Auto-shutdown watcher active for PID $TARGET_PID" >> /home/XiaoSha/finalCV_metaDK/logs/train.log
while kill -0 "$TARGET_PID" 2>/dev/null; do
    sleep 30
done
echo "[$(date)] Training finished. Starting auto-eval..." >> /home/XiaoSha/finalCV_metaDK/logs/train.log
cd /home/XiaoSha/finalCV_metaDK
source venv/bin/activate
python eval_all_modalities.py --restore_from checkpoints/final.pth >> logs/eval_final.log 2>&1 || true
echo "[$(date)] Eval finished. Shutting down VM in 60s to save cost..." >> /home/XiaoSha/finalCV_metaDK/logs/train.log
sleep 60
sudo poweroff