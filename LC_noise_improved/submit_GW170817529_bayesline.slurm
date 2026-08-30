#!/bin/bash
#SBATCH --job-name=lc-gw170817-bayesline
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

project_dir=/home/dp429/dp429/dc-craw2/Subthreshold-Bright-Sirens
work_dir="$project_dir/LC_noise_improved"

cd "$work_dir"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export GRB_DATA_DIR="$project_dir/SFM_baseline_GW170817_GRB_extended/data"
export GRB_CATALOGUE="$GRB_DATA_DIR/grb_catalog.csv"

echo "host=$(hostname) date=$(date --iso-8601=seconds) job=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

exec "$project_dir/.venv/bin/python" -u LC_SFM_sampling_GRB.py \
    --grb-names GRB170817529 \
    --psd-source bayesline \
    --bayesline-psd-dir /home/dp429/dp429/dc-craw2/assist_bayeswave \
    --bayesline-segment-duration 128 \
    --output-dir "$work_dir/results/GRB170817529"
