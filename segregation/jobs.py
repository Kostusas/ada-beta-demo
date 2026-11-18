"""One batch submission per new query, with a shared, versioned result cache."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import tomllib

from .engine import PREVIEW, ROOT, check_stop
from .query import QUERY_VERSION, compute, load_result, write_text

RESULTS = ROOT / "results"
CACHE = RESULTS / "segregation-queries"


def scheduler(command, *, check=True, env=None):
    return subprocess.run(command, capture_output=True, text=True, timeout=30, check=check, env=env)


def submit(folder, index, stage, radius, backend, output, attempt):
    config = RESULTS / "slurm.toml"
    options = tomllib.loads(config.read_text()) if config.is_file() else {}
    args = ["sbatch", "--parsable", "--job-name=segregation", "--nodes=1", "--ntasks=1",
            f"--chdir={ROOT}", f"--output={attempt / 'slurm.log'}",
            f"--cpus-per-task={options.get('cpus', 4)}", f"--mem={options.get('memory', '8G')}",
            f"--time={options.get('time', '00:10:00')}"]
    for key in ("partition", "account", "qos"):
        if options.get(key):
            args.append(f"--{key}={options[key]}")
    if backend == "CUDA":
        args.append(f"--gres={options.get('gres', 'gpu:1')}")
    shared_python = ROOT / ".venv/bin/python"
    python = str(shared_python) if shared_python.exists() else sys.executable
    args.extend([str(ROOT / "scripts/segregation-query.sbatch"), python, str(folder),
                 str(index), str(stage), str(radius), backend, str(output), str(attempt)])
    # Use the batch allocation's resources and GPU assignment. Retain site config and PATH.
    env = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_") and k not in
           {"CUDA_VISIBLE_DEVICES", "SLURM_EXPORT_ENV", "SLURM_MEM_PER_CPU", "SLURM_MEM_PER_NODE",
            "SLURM_CPUS_PER_TASK", "SLURM_NTASKS", "SLURM_HINT"}}
    call = scheduler(args, check=False, env=env)
    if call.returncode:
        raise RuntimeError(f"Slurm rejected the request: {call.stderr.strip()}")
    job = call.stdout.strip().split(";")[0]
    if not job.isdecimal():
        raise RuntimeError(f"Unexpected sbatch response: {call.stdout.strip()}")
    return job


def request(folder, index, index_id, stage, radius, backend, stop, status, *, cache=CACHE):
    if backend not in ("local", "CPU", "CUDA") or (backend == "local" and folder != PREVIEW):
        raise ValueError("In-session calculation is available for the tiny preview only.")
    if stage not in range(5, 100, 5) or not (1 <= radius <= 4 and radius * 4 == int(radius * 4)):
        raise ValueError("Choose a stage and contact radius supported by the dashboard.")
    directory = cache / f"v{QUERY_VERSION}" / index_id / f"{backend}-{stage}-{int(radius * 4)}"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "result.npz"
    active = directory / "active.json"
    with (directory / "lock").open("a") as lock:
        while True:
            check_stop(stop)
            if output.is_file():
                return load_result(output), "Saved result"
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                status("Waiting for the same query in another session")
                stop.wait(0.25)
        if output.is_file():
            return load_result(output), "Saved result"
        if backend == "local":
            compute(folder, index, stage, radius, backend, output, stop,
                    lambda label, done, total: status(f"{label} · {done:,}/{total:,}"))
            return load_result(output), "Computed in the preview session"
        if active.exists():
            attempt_name, job = json.loads(active.read_text())
            attempt = directory / attempt_name
        else:
            attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=directory))
            status("Submitting to Slurm")
            job = submit(folder, index, stage, radius, backend, output, attempt)
            write_text(active, json.dumps([attempt.name, job]))
        scheduler_state = "Submitted"
        next_check = time.monotonic() + 5
        missing = False
        while True:
            if output.is_file():
                active.unlink(missing_ok=True)
                return load_result(output), f"Slurm job {job}"
            if stop.is_set():
                cancel = scheduler(["scancel", job], check=False)
                if cancel.returncode and not output.exists():
                    raise RuntimeError(f"Could not stop job {job}: {cancel.stderr.strip()}")
                active.unlink(missing_ok=True)
                raise InterruptedError(f"Stopped job {job}")
            if (attempt / "error").exists():
                active.unlink(missing_ok=True)
                raise RuntimeError(f"Job {job}: {(attempt / 'error').read_text()} · {attempt / 'slurm.log'}")
            now = time.monotonic()
            if now >= next_check:
                state = scheduler(["squeue", "-h", "-j", job, "-o", "%T"], check=False)
                gone = (not state.returncode and not state.stdout.strip()) or "invalid job id" in state.stderr.lower()
                if state.returncode and not gone:
                    raise RuntimeError(f"Could not check job {job}: {state.stderr.strip()}. Compare again to reconnect.")
                if gone and missing:
                    active.unlink(missing_ok=True)
                    raise RuntimeError(f"Job {job} ended without a result. See {attempt / 'slurm.log'}")
                missing = gone
                scheduler_state = state.stdout.strip() or "Checking for output"
                next_check = now + 8
            progress_file = attempt / "progress"
            detail = progress_file.read_text() if progress_file.exists() else scheduler_state
            status(f"Job {job} · {detail}")
            stop.wait(0.25)
