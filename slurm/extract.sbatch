#!/bin/bash
#SBATCH --job-name=ada-contact-map
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export OPENBLAS_NUM_THREADS=1
srun .venv/bin/python scripts/extract.py "${1:?supply trajectory directory}" --stride "${2:?choose 2, 4 or 10}"
