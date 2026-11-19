"""Reconstruct radius-5 pair contacts from the original 4,040-bead trajectories."""
import argparse
import hashlib
import json
from math import lcm
from pathlib import Path
import tempfile

import h5py
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
RADIUS = 5
REVISION = "3fb7dea343f8426140f38442c47eba000e80b9f2"


def add_frame(counts, xyz):
    # h5py sees (beads, xyz). Use the strict Euclidean cutoff of the original extractor.
    pairs = cKDTree(xyz).query_pairs(RADIUS, output_type="ndarray")
    a, b = pairs.T
    delta = xyz[a] - xyz[b]
    inside = np.einsum("ij,ij->i", delta, delta) < RADIUS**2
    a, b = a[inside], b[inside]
    counts[a, b] += 1
    counts[b, a] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--stride", type=int, choices=(2, 4, 10), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "inputs")
    args = parser.parse_args()
    n = 4040 // args.stride
    counts = np.zeros((n, n), dtype=np.uint32)
    frames = 0
    for entry in (ROOT / "data/trajectories.sha256").read_text().splitlines():
        expected, name = entry.split()
        path = args.folder / name
        with path.open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != expected:
                raise ValueError(f"Wrong trajectory contents: {name}")
        with h5py.File(path) as h5:
            for group in h5.values():
                pos = group["pos"]
                if pos.shape != (4040, 3):
                    raise ValueError(f"Unexpected position layout in {name}: {pos.shape}")
                add_frame(counts, pos[::args.stride].astype(np.float64))
                frames += 1
    if frames != 3060:
        raise ValueError(f"Expected 3060 snapshots, read {frames}")
    np.fill_diagonal(counts, frames)
    # A shared physical grid gives an independent indexing and normalization check.
    common = lcm(args.stride, 10)
    reference = np.loadtxt(ROOT / "data/condensin_hic_contact_r_5.txt")
    published_counts = np.rint(frames * reference[::common//10, ::common//10]).astype(int)
    difference = counts[::common//args.stride, ::common//args.stride].astype(int) - published_counts
    max_error = int(np.max(np.abs(difference)))
    # Upstream uses single-precision distances. Allow one borderline observation.
    if max_error > 1:
        raise ValueError(f"Published map differs by up to {max_error} frames on shared beads")
    args.output.mkdir(parents=True, exist_ok=True)
    dest = Path(tempfile.mkdtemp(prefix=f"map-{args.stride}kb-", dir=args.output))
    path = dest / "contacts.txt"
    pending = dest / "contacts.tmp"
    np.savetxt(pending, counts / frames, fmt="%.17g", delimiter="\t")
    with pending.open("rb") as file:
        sha = hashlib.file_digest(file, "sha256").hexdigest()
    metadata = dict(revision=REVISION, stride=args.stride, radius=RADIUS, frames=frames,
                    first_bead=1, common_grid_max_count_difference=max_error, sha256=sha)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    pending.replace(path)
    print(path.resolve())


if __name__ == "__main__":
    main()
