"""Pack the two matched dynamic conditions from Zenodo into trajectory files."""

import argparse
import hashlib
import io
import json
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from session_resources import cpu_capacity

ROOT = Path(__file__).resolve().parents[1]
DOI = "10.5281/zenodo.10877077"
MD5 = "aae8b0f6363764b838452707a1ae049f"
TAIL = "rateFork_0.07440000000000001_stoch_0.05_sps_2500_trunc_1.5_pullF_2.0"
PREFIX = "N_404_colRate_0.03_"
DIRS = {
    "No SMC": "Dynamic/No_smcs/" + PREFIX + TAIL,
    "SMC": "Dynamic/Nontopological_smcs/stallFork_0.0_"
    + PREFIX
    + "sep_101_parSstrength_4040.0_lifetime_40400_"
    + TAIL,
}
ARCHIVE = None


def open_archive(path):
    global ARCHIVE
    ARCHIVE = zipfile.ZipFile(path)


def pack(source, output, preview=False):
    with h5py.File(source) as h5:
        keys = list(h5)
        forks = np.array([h5[k]["Fork"][:] for k in keys])
        blocks = np.array([h5[k].attrs["block"] for k in keys], dtype=np.int32)
        if forks.shape != (len(keys), 2) or np.any(forks != forks.astype(int)):
            raise ValueError("Expected two integer fork positions per frame")
        if (
            np.any(forks < 0)
            or np.any(forks > 403)
            or np.any(forks[:, 0] > forks[:, 1])
        ):
            raise ValueError("Unexpected fork positions")
        order = np.argsort(blocks)
        if len(np.unique(blocks)) != len(blocks):
            raise ValueError("Duplicate block in trajectory")
        if preview:
            fraction = 1 - (forks[:, 1] - forks[:, 0]) / 404
            selected = set()
            for stage in (0.25, 0.50, 0.75):
                selected.update(np.argsort(abs(fraction - stage), kind="stable")[:3])
            order = [i for i in order if i in selected]
        positions = np.stack([h5[keys[i]]["pos"][:808] for i in order]).astype(
            np.float32
        )
        if positions.shape != (len(order), 808, 3) or not np.isfinite(positions).all():
            raise ValueError("Unexpected coordinate array")
        np.savez_compressed(
            output,
            pos=positions,
            forks=forks[order].astype(np.int16),
            block=blocks[order],
            group=np.asarray([keys[i] for i in order]),
        )
    return len(order)


def from_zip(job):
    member, output = job
    return pack(io.BytesIO(ARCHIVE.read(member)), output)


def prepare(source, dest, workers, preview=False):
    if dest.exists():
        raise FileExistsError(
            f"{dest} already exists. Choose a fresh output directory."
        )
    if source.is_file():
        with source.open("rb") as file:
            if hashlib.file_digest(file, "md5").hexdigest() != MD5:
                raise ValueError(f"Zenodo archive checksum mismatch. Move {source} aside and rerun prepare-ada to download it again.")
        with zipfile.ZipFile(source) as archive:
            members = archive.namelist()
    else:
        members = [
            p.relative_to(source).as_posix()
            for d in DIRS.values()
            for p in (source / d).glob("*.h5")
        ]
    tracks = []
    for condition, folder in DIRS.items():
        group = sorted(
            m for m in members if m.endswith(".h5") and str(Path(m).parent) == folder
        )
        if not group:
            raise ValueError(f"No trajectories in {folder}")
        for m in group:
            tracks.append(
                dict(member=m, condition=condition, file=f"track-{len(tracks):04d}.npz")
            )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as temp:
        staging = Path(temp) / "packed"
        staging.mkdir()
        jobs = [(t["member"], staging / t["file"]) for t in tracks]
        if source.is_file():
            with ProcessPoolExecutor(
                min(workers, len(jobs)), initializer=open_archive, initargs=(source,)
            ) as pool:
                counts = list(pool.map(from_zip, jobs))
        else:
            counts = [pack(source / member, out, preview) for member, out in jobs]
        for track, count in zip(tracks, counts):
            track["frames"] = count
        manifest = dict(version=1, doi=DOI, preview=preview, tracks=tracks)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(dest)
    print(f"{len(tracks)} trajectories, {sum(counts):,} frames → {dest}")


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("archive", type=Path)
    arg.add_argument("--output", type=Path, default=ROOT / "inputs/segregation-2024")
    arg.add_argument("--workers", type=int, default=min(60, cpu_capacity()))
    args = arg.parse_args()
    if not 1 <= args.workers <= cpu_capacity():
        arg.error("Choose a worker count inside the session CPU capacity")
    prepare(args.archive, args.output, args.workers)
