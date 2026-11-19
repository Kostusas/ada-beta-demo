#!/usr/bin/env bash
# Called on the submission host with the shared clone as the working directory.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p inputs results
exec 9>inputs/.ensemble-prep.lock
flock 9
folder=inputs/segregation-2024
archive=inputs/Simulation_data.zip
if [[ ! -f "$folder/stage-index-v1/index.json" ]] && ! command -v sbatch >/dev/null; then
    echo 'Preparation needs Slurm. Set submit_host = "inter01.labs.vu.nl" in slurm.toml and rerun prepare-ada.' >&2
    exit 1
fi
.venv/bin/python -c 'import h5py, numpy, psutil'
if [[ ! -f "$folder/manifest.json" && ! -f "$archive" ]]; then
    echo "Downloading the 10.47 GB coordinate archive to shared storage."
    curl --fail --location --retry 3 --continue-at - --output "$archive.part" \
        https://zenodo.org/records/10877077/files/Simulation_data.zip
    mv "$archive.part" "$archive"
fi
if [[ ! -f "$folder/stage-index-v1/index.json" || ! -f "$folder/manifest.json" ]]; then
    echo "Preparing the full ensemble in Slurm. Batch output is in results/ensemble-prep-JOBID.log."
    sbatch --wait --chdir="$PWD" --output="results/ensemble-prep-%j.log" "$@" scripts/prepare-segregation.sbatch
fi
.venv/bin/python -m segregation.index --check "$folder"
