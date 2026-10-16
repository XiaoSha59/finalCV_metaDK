#!/bin/bash
# Resumable BraTS pipeline: 2-phase training -> Table 1 evaluation -> poweroff.
# Started by metakd.service on every boot, so a GCP host-maintenance restart resumes from last.pth.
WORKDIR="/home/XiaoSha/finalCV_metaDK"
cd "$WORKDIR"
mkdir -p logs
LOG=logs/autopilot.log
DONE=logs/PIPELINE_DONE
log() { echo "[$(date -u '+%F %T') UTC] $*" | tee -a "$LOG"; }

if [ -f "$DONE" ]; then
    log "Pipeline already finished ($DONE exists); nothing to do, VM stays on."
    exit 0
fi

for _ in $(seq 1 60); do
    nvidia-smi >/dev/null 2>&1 && break
    sleep 5
done
log "Boot/start: GPU $(nvidia-smi --query-gpu=name --format=csv,noheader)"

for attempt in 1 2 3; do
    log "Training attempt $attempt"
    if bash run_brats_2phase.sh >> logs/brats_2phase_driver.log 2>&1; then
        break
    fi
    log "Training attempt $attempt failed (see logs/brats_phase*.log)"
    if [ "$attempt" = 3 ]; then
        log "3 consecutive failures; powering off to stop billing."
        sudo -n /usr/sbin/poweroff
        exit 1
    fi
    sleep 120
done

if [ ! -f logs/brats_table1_results_postproc.csv ]; then
    log "Running 15-modality evaluation"
    bash run_eval_pipeline.sh >> logs/brats_eval_driver.log 2>&1
fi
if [ ! -f logs/brats_table1_results.csv ]; then
    log "Evaluation did not produce results; powering off anyway (checkpoints are on disk)."
    sudo -n /usr/sbin/poweroff
    exit 1
fi

touch "$DONE"
log "Pipeline finished. Powering off in 60 s."
sleep 60
sudo -n /usr/sbin/poweroff
