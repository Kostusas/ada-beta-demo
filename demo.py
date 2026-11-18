#!/usr/bin/env python3
"""Run one contact prediction experiment. Requires Python 3.11+ and Julia."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parent
MAP = ROOT / "data/condensin_hic_contact_r_5.txt"
MAP_SHA = "81f921d346eef394910f168ce3e551fc017275044d2755dccf13feb7450729bd"


def file_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=["preview", "smoke", "scale", "highmem"])
    parser.add_argument("--max-threads", type=int, help="ceiling for the thread sweep")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--map", type=Path, help="contact map from scripts/extract.py (highmem)")
    parser.add_argument("--output", type=Path, default=ROOT / "results", help="parent for unique run directories")
    args = parser.parse_args()
    profile = args.profile
    if args.map and profile != "highmem":
        parser.error("--map applies to highmem only")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    allocation = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    limit = args.max_threads if args.max_threads is not None else min(allocation, 2 if profile == "smoke" else 32)
    if limit < 1 or limit > allocation:
        parser.error(f"choose 1 through {allocation} threads")
    julia = shutil.which(os.environ.get("JULIA", "julia"))
    if not julia:
        parser.exit(2, "Julia was not found. Install Julia 1.12.6 or set JULIA to its executable.\n")
    path = MAP
    n, stride, spacing = (101, 4, 40) if profile == "smoke" else (404, 1, 10)
    if profile == "highmem":
        if not (args.map and os.environ.get("SLURM_JOB_ID")):
            parser.error("highmem requires a Slurm allocation and --map from scripts/extract.py")
        path = args.map.resolve()
        provenance = json.loads(path.with_suffix(".json").read_text())
        spacing = provenance["stride"]
        if spacing not in (2, 4) or provenance["frames"] != 3060 or provenance["radius"] != 5:
            parser.error("expected a complete 2 or 4 kb condensin map at radius 5")
        n = 4040 // spacing
        expected_hash = provenance["sha256"]
        mem_mb = int(os.environ.get("SLURM_MEM_PER_NODE", 0))
        if mem_mb < (96 if n == 2020 else 16) * 1024:
            parser.error("request at least 96G (2 kb) or 16G (4 kb) with --mem")
    else:
        expected_hash = MAP_SHA
    if file_hash(path) != expected_hash:
        parser.error("input map checksum mismatch")
    bait = 1 + round(1990 / spacing)
    args.output.mkdir(parents=True, exist_ok=True)
    dest = Path(tempfile.mkdtemp(prefix=profile + "-", dir=args.output)).resolve()
    print(f"Run directory: {dest}", flush=True)
    metadata = dict(profile=profile, n=n, spacing_kb=spacing, bait=bait,
                    tensor_bytes=8*n**3, map_sha256=expected_hash, byteorder=sys.byteorder,
                    started_utc=datetime.now(timezone.utc).isoformat(), platform=platform.system(),
                    architecture=platform.machine(), node=os.getenv("SLURMD_NODENAME", ""),
                    job=os.getenv("SLURM_JOB_ID", ""), partition=os.getenv("SLURM_JOB_PARTITION", ""),
                    allocated_cpus=os.getenv("SLURM_CPUS_PER_TASK", ""),
                    requested_mem_mb=os.getenv("SLURM_MEM_PER_NODE", ""))
    if profile == "highmem":
        metadata["map_path"] = str(path)
        metadata["extraction"] = provenance
    if profile == "preview":
        configs = [("preview", 1)]
    elif profile == "highmem":
        configs = [("threaded", limit)]
    else:
        configs = [("serial", 1)] + [("threaded", t) for t in (1, 2, 4, 8, 16, 32) if t <= limit]
    baseline_digest = None
    baseline_time = None
    fields = ["mode", "threads", "trial", "seconds", "cpu_seconds", "peak_rss_bytes", "digest"]
    summary = []
    for mode, threads in configs:
        command = [julia, "--startup-file=no", f"--threads={threads},0", str(ROOT / "src/worker.jl"),
                   mode, str(path), str(n), str(stride), str(bait), str(args.repeats), str(dest)]
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
        if mode == "preview":
            break
        metrics = tomllib.loads(result.stdout)
        metadata.update(cpu=metrics["cpu"], julia=metrics["julia"])
        if metrics["threads"] != threads:
            raise ValueError("Julia started with the wrong thread count")
        trials = metrics["trials"]
        if mode == "serial":
            baseline_digest = trials[0]["digest"]
        if baseline_digest and any(t["digest"] != baseline_digest for t in trials):
            raise ValueError(f"{mode}, {threads} threads: tensor differs from original serial result")
        csv_path = dest / "trials.csv"
        new = not csv_path.exists()
        with csv_path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
            if new:
                writer.writeheader()
            for index, trial in enumerate(trials, 1):
                writer.writerow(dict(mode=mode, threads=threads, trial=index, **trial))
        median = statistics.median(t["seconds"] for t in trials)
        if mode == "threaded" and threads == 1:
            baseline_time = median
        speedup = baseline_time / median if baseline_time and mode == "threaded" else None
        row = dict(mode=mode, threads=threads, median_s=median, speedup=speedup,
                   efficiency=speedup/threads if speedup else None,
                   cpu_cores=statistics.median(t["cpu_seconds"]/t["seconds"] for t in trials),
                   peak_rss_mib=max(t["peak_rss_bytes"] for t in trials)/2**20)
        summary.append(row)
        print(f"{mode:8} {threads:2d} thread(s)   {median:.6f} s   {row['peak_rss_mib']:.1f} MiB peak RSS", flush=True)
    metadata["summary"] = summary
    if profile in ("scale", "preview") and sys.byteorder == "little":
        expected = ROOT / "examples/published-slice/bait.f64"
        if file_hash(dest / "bait.f64") != file_hash(expected):
            raise ValueError("Bait 200 differs from the published-resolution reference plane")
    metadata["validation"] = ("full tensor SHA-256 vs original serial" if baseline_digest else
                              "sparse grid vs original serial" if profile == "highmem" else "slice only")
    # A manifest marks a completed run, so interrupted jobs never appear as finished results.
    (dest / "run.json.tmp").write_text(json.dumps(metadata, indent=2) + "\n")
    (dest / "run.json.tmp").replace(dest / "run.json")


if __name__ == "__main__":
    main()
