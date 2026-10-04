#!/bin/bash
# The lab training job: 2 nodes, all 4 GPUs of each, requeued (not failed) when a node breaks.
#SBATCH --job-name=lab-train
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/shared/runs/%x-%j.out

echo "$(date -u +%FT%TZ) run ${SLURM_RESTART_COUNT:-0} on ${SLURM_JOB_NODELIST}"
exec python3 -m gpuops train --steps "${STEPS:-150}" --checkpoint-every "${CHECKPOINT_EVERY:-25}" "$@"
