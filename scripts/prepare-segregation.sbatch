#!/bin/sh
#SBATCH --job-name=segregation-index
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:00:00
set -eu
# Submit from the shared clone. Reuse the packed files from earlier versions.
mkdir -p inputs
exec 8>inputs/.ensemble-index.lock
flock 8
if [ ! -f inputs/segregation-2024/manifest.json ]; then
    .venv/bin/python -m segregation.prepare inputs/Simulation_data.zip
fi
exec .venv/bin/python -m segregation.index inputs/segregation-2024
