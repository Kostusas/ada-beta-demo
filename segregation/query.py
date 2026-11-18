"""Compute one stage and atomically save its map, scatter points and measurements."""

import argparse
import json
import os
from pathlib import Path
import resource
import socket
import sys
import tempfile
import threading
import time

import numpy as np

from session_resources import cpu_capacity
from .engine import Ensemble

# Bump when the persisted result format or the scientific calculation changes.
QUERY_VERSION = 1


def write_text(path, text):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        file.write(text)
    os.replace(file.name, path)


def compute(folder, index, stage, radius, backend, output, stop, progress):
    workers = min(2, cpu_capacity()) if backend == "local" else cpu_capacity()
    engine = Ensemble(folder, backend, workers, 0, stop, progress, index=index)
    result = engine.compare(stage / 100, radius, stop, progress)
    result.update(
        device=engine.device, node=socket.gethostname(), cpus=engine.workers,
        job=os.environ.get("SLURM_JOB_ID", "") if backend != "local" else "",
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        cuda_pool=engine.cp.get_default_memory_pool().total_bytes() if engine.cp else 0,
        index_id=engine.index_id, version=QUERY_VERSION,
    )
    for point in result["points"]:
        point["frames"] = point["frames"].tolist()
    difference = result.pop("difference")
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".result-", delete=False) as file:
            tmp = Path(file.name)
            np.savez_compressed(file, metadata=json.dumps(result), difference=difference)
        tmp.replace(output)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def load_result(path):
    with np.load(path, allow_pickle=False) as saved:
        result = json.loads(str(saved["metadata"]))
        result["difference"] = saved["difference"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("index", type=Path)
    parser.add_argument("stage", type=int)
    parser.add_argument("radius", type=float)
    parser.add_argument("backend", choices=("CUDA", "CPU"))
    parser.add_argument("output", type=Path)
    parser.add_argument("attempt", type=Path)
    args = parser.parse_args()
    last = 0.0

    def progress(label, done, total):
        global last
        now = time.monotonic()
        if now - last >= 0.5 or done == total:
            write_text(args.attempt / "progress", f"{label} · {done:,}/{total:,}")
            last = now

    try:
        write_text(args.attempt / "progress", f"Starting {args.backend} on {socket.gethostname()}")
        compute(args.folder, args.index, args.stage, args.radius, args.backend,
                args.output, threading.Event(), progress)
    except Exception as exc:
        write_text(args.attempt / "error", f"{type(exc).__name__}: {exc}")
        raise
