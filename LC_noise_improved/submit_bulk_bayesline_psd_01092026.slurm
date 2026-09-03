#!/bin/bash
#SBATCH --job-name=lc-grb-bayesline
#SBATCH --account=dp429
#SBATCH --partition=gpu-a100-80
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=07:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

project_dir=/home/dp429/dp429/dc-craw2/Subthreshold-Bright-Sirens
work_dir="$project_dir/LC_noise_improved"
psd_root=/home/dp429/dp429/dc-craw2/GRB_bayesline_psd_runs_01092026

mapfile -t psd_dir_names < <(
    find "$psd_root" -mindepth 1 -maxdepth 1 -type d -name 'GRB*_' -printf '%f\n' \
        | sort \
        | grep -Ev '^(GRB150922234_|GRB190905985_|GRB191227723_|GRB240715239_)$'
)

array_index=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set}
if (( array_index < 0 || array_index >= ${#psd_dir_names[@]} )); then
    echo "Invalid array index: $array_index (available: ${#psd_dir_names[@]})" >&2
    exit 2
fi

psd_dir_name=${psd_dir_names[$array_index]}
grb_name=${psd_dir_name%_}
shopt -s nullglob
clean_dirs=("$psd_root/$psd_dir_name"/run/trigtime_*/post/clean)
if (( ${#clean_dirs[@]} != 1 )); then
    echo "Expected one BayesWave clean PSD directory for $psd_dir_name; found ${#clean_dirs[@]}" >&2
    exit 1
fi
bayesline_psd_dir=${clean_dirs[0]}
if ! compgen -G "$bayesline_psd_dir/glitch_median_PSD_*.dat" > /dev/null; then
    echo "No raw BayesWave glitch median PSD files found for $psd_dir_name" >&2
    exit 1
fi

cd "$work_dir"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export GRB_DATA_DIR="$project_dir/SFM_baseline_GW170817_GRB_extended/data"
export GRB_CATALOGUE="$GRB_DATA_DIR/grb_catalog.csv"

echo "host=$(hostname) date=$(date --iso-8601=seconds) job=${SLURM_JOB_ID:-dry-run}"
echo "grb=$grb_name bayesline_psd_dir=$bayesline_psd_dir"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [[ ${DRY_RUN:-0} == 1 ]]; then
    exit 0
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

exec "$project_dir/.venv/bin/python" -u LC_SFM_sampling_GRB.py \
    --grb-names "$grb_name" \
    --psd-source bayesline \
    --bayesline-psd-dir "$bayesline_psd_dir" \
    --bayesline-segment-duration 128 \
    --output-dir "$work_dir/results/$grb_name"