#!/bin/bash
#SBATCH --job-name=ada-fine-contacts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores python3 demo.py highmem --map "${1:?supply the extracted contact map}" --repeats 1
