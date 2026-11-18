#!/usr/bin/env bash
# Run on the Slurm submission host, from the same shared clone as the notebook.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
exec 9>results/bait-map-prep.lock
flock 9

if .venv/bin/python -c 'from notebooks.experiment import inputs; raise SystemExit(2020 not in inputs())'; then
    echo "2,020-locus map is ready."
    exit 0
fi
.venv/bin/python -c 'import h5py, numpy, scipy.spatial'
revision=3fb7dea343f8426140f38442c47eba000e80b9f2
folder=Extract_triplets/Example_Brandao
source_dir="trajectories/$folder"
if [[ ! -d "$source_dir" ]]; then
    checkout=results/bait-map-source
    if [[ ! -d "$checkout/.git" ]]; then
        git clone --depth 1 --filter=blob:none --no-checkout \
            https://github.com/PLSysGitHub/chromosomal_multi_contact_data.git "$checkout"
    fi
    git -C "$checkout" fetch --depth 1 origin "$revision"
    git -C "$checkout" restore --source="$revision" --worktree -- "$folder"
    source_dir="$checkout/$folder"
fi

echo "Preparing the 2 kb input. The extraction job will print its map path to results/bait-map-JOBID.log."
sbatch --wait --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=01:00:00 \
    --job-name=bait-map --chdir="$PWD" --output="results/bait-map-%j.log" "$@" \
    --wrap=".venv/bin/python scripts/extract.py $source_dir --stride 2"
echo "2,020-locus map is ready. Rerun the dashboard cell."
