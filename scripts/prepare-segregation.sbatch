#!/bin/sh
#SBATCH --job-name=segregation-index
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:00:00
set -eu
# Submit from the shared clone. Reuse the packed files from earlier versions.
if [ ! -f results/segregation-2024/manifest.json ]; then
    .venv/bin/python -m segregation.prepare results/Simulation_data.zip
fi
exec .venv/bin/python -m segregation.index
