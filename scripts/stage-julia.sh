#!/usr/bin/env bash
# Stage the tested official binary once on the shared ADA filesystem.
set -euo pipefail

if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    echo "Run this command on an x86-64 Linux ADA host from the shared clone." >&2
    exit 1
fi
cd "$(dirname "$0")/.."
version=1.12.6
runtime_dir=results/runtime
dest="$runtime_dir/julia-$version"
if [[ -x "$dest/bin/julia" ]]; then
    exec "$dest/bin/julia" --version
fi

mkdir -p "$runtime_dir"
staging=$(mktemp -d "$runtime_dir/.julia-XXXXXXXX")
trap 'rm -rf -- "$staging"' EXIT
archive="julia-$version-linux-x86_64.tar.gz"
curl --fail --location --retry 3 --output "$staging/$archive" \
    "https://julialang-s3.julialang.org/bin/linux/x64/1.12/$archive"
(cd "$staging" && echo "bbabf3bef19421a9dbd24a767d807606ab85e444323b5a1c73ffe293fa3d079a  $archive" | sha256sum -c -)
tar -xzf "$staging/$archive" -C "$staging"
"$staging/julia-$version/bin/julia" --version
# Parallel setup commands may finish together. Publish only a complete installation.
mv -T "$staging/julia-$version" "$dest" || [[ -x "$dest/bin/julia" ]]
"$dest/bin/julia" --version
