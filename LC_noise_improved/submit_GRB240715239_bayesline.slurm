#!/bin/bash
#SBATCH --job-name=lc-grb240715239-bayesline
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
grb_name=GRB240715239
bayesline_psd_dir=/home/dp429/dp429/dc-craw2/assist_bayeswave/GRB24

cd "$work_dir"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export GRB_DATA_DIR="$project_dir/SFM_baseline_GW170817_GRB_extended/data"
export GRB_CATALOGUE="$GRB_DATA_DIR/grb_catalog.csv"

echo "host=$(hostname) date=$(date --iso-8601=seconds) job=$SLURM_JOB_ID"
echo "grb=$grb_name bayesline_psd_dir=$bayesline_psd_dir"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

exec "$project_dir/.venv/bin/python" -u LC_SFM_sampling_GRB.py \
    --grb-names "$grb_name" \
    --psd-source bayesline \
    --bayesline-psd-dir "$bayesline_psd_dir" \
    --bayesline-segment-duration 128 \
    --output-dir "$work_dir/results/$grb_name"