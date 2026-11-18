"""Stage-sorted, read-only coordinates shared by the batch jobs and the viewer."""

import argparse
import errno
import hashlib
import json
import tempfile
import threading
from pathlib import Path

import numpy as np

from session_resources import cpu_capacity
from .engine import ARRAY_NAMES, FULL, PREVIEW, ROOT, Coordinates, read_index_id


def index_path(folder):
    if folder == PREVIEW:
        return ROOT / "results/segregation-preview-index-v1"
    return folder / "stage-index-v1"


def build_index(folder, destination, workers):
    if destination.exists():
        return read_index_id(folder, destination)
    coords = Coordinates(folder, workers, threading.Event(), lambda *_: None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        staging = Path(tmp) / "index"
        staging.mkdir()
        manifest_hash = hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest()
        digest = hashlib.sha256(manifest_hash.encode())
        for key in ARRAY_NAMES:
            data = getattr(coords, key)
            digest.update(memoryview(data).cast("B"))
            np.save(staging / f"{key}.npy", data, allow_pickle=False)
        (staging / "index.json").write_text(json.dumps({
            "manifest_sha256": manifest_hash, "id": digest.hexdigest()
        }))
        try:
            staging.rename(destination)
        except OSError as exc:
            if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                raise
    return read_index_id(folder, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, default=FULL, nargs="?")
    args = parser.parse_args()
    folder = args.folder.resolve()
    build_index(folder, index_path(folder), min(60, cpu_capacity()))
    print(f"Ready: {index_path(folder)}")
