#!/bin/bash
#SBATCH --job-name=ada-contacts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=4G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores python3 demo.py scale
