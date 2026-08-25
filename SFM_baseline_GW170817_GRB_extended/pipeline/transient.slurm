#!/bin/bash
#SBATCH --job-name=subthreshold
#SBATCH --account=dp429
#SBATCH --partition=gpu-a100-80
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=07:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "host=$(hostname) date=$(date --iso-8601=seconds) job=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if (( $# == 0 )); then
    echo "usage: sbatch tursa.sbatch SFM_WT_sims.py [ARG ...]" >&2
    exit 2
fi

exec "$HOME/spackming311/bin/python" -u "$@"

