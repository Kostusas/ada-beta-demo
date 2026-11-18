"""One batch submission per new query, with a shared, versioned result cache."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
import tomllib

from .engine import PREVIEW, ROOT, check_stop
from .query import QUERY_VERSION, compute, load_result, write_text

RESULTS = ROOT / "results"
CACHE = RESULTS / "segregation-queries"


def scheduler(command, *, submit_host="", env=None):
    executable = command[0]
    if submit_host:
        command = ["ssh", "-nT", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                   "-o", "ConnectTimeout=8", "--", submit_host, shlex.join(command)]
    try:
        call = subprocess.run(command, capture_output=True, text=True, timeout=30, env=env)
    except FileNotFoundError as exc:
        if submit_host:
            raise RuntimeError("The notebook needs an SSH client to use submit_host.") from exc
        raise RuntimeError(
            f"{executable} is unavailable in this environment. Set submit_host in results/slurm.toml "
            "to a Slurm-enabled ADA host, or run the check from an ADA host shell."
        ) from exc
    if submit_host and call.returncode == 255:
        raise RuntimeError(f"SSH to {submit_host} failed: {call.stderr.strip()}")
    return call


def submit(folder, index, stage, radius, backend, output, attempt, options):
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
    call = scheduler(args, submit_host=options.get("submit_host", ""), env=env)
    if call.returncode:
        raise RuntimeError(f"Slurm rejected the request: {call.stderr.strip()}")
    # Some SSH login environments print a greeting before sbatch's parsable line.
    job_ids = re.findall(r"^([0-9]+)(?:;[^\s;]+)?$", call.stdout, re.MULTILINE)
    if len(job_ids) != 1:
        raise RuntimeError(f"Unexpected sbatch response: {call.stdout.strip()}")
    return job_ids[0]


def file_stamp(path):
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    return info.st_ino, info.st_mtime_ns


def request(folder, index, index_id, stage, radius, backend, stop, status, *, cache=CACHE, fresh=False):
    if backend not in ("local", "CPU", "CUDA") or (backend == "local" and folder != PREVIEW):
        raise ValueError("In-session calculation is available for the tiny preview only.")
    if stage not in range(5, 100, 5) or not (1 <= radius <= 4 and radius * 4 == int(radius * 4)):
        raise ValueError("Choose a stage and contact radius supported by the dashboard.")
    directory = cache / f"v{QUERY_VERSION}" / index_id / f"{backend}-{stage}-{int(radius * 4)}"
    config = RESULTS / "slurm.toml"
    options = tomllib.loads(config.read_text()) if backend != "local" and config.is_file() else {}
    if backend == "CUDA":
        # A performance measurement belongs to the GPU allocation that produced it.
        allocation = json.dumps([options.get("partition", ""), options.get("gres", "gpu:1")])
        directory /= "gpu-" + hashlib.sha256(allocation.encode()).hexdigest()[:16]
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "result.npz"
    active = directory / "active.json"
    at_request = file_stamp(output) if fresh else None
    with (directory / "lock").open("a") as lock:
        while True:
            check_stop(stop)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                status("Waiting for the same query in another session")
                stop.wait(0.25)
        previous = file_stamp(output)
        running = json.loads(active.read_text()) if active.is_file() else None
        if previous is not None:
            saved = load_result(output)
            if running and saved["job"] == running[1]:
                active.unlink()
                running = None
            if not running and (not fresh or previous != at_request):
                return saved, "Saved result"
        if backend == "local":
            compute(folder, index, stage, radius, backend, output, stop,
                    lambda label, done, total: status(f"{label} · {done:,}/{total:,}"))
            return load_result(output), "Computed in the preview session"
        submit_host = options.get("submit_host", "")
        if running:
            attempt_name, job, *route = running
            # A job started directly on ADA can be resumed from its Jupyter container.
            if route and route[0]:
                submit_host = route[0]
            attempt = directory / attempt_name
        else:
            attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=directory))
            status(f"Submitting to Slurm via {submit_host}" if submit_host else "Submitting to Slurm")
            job = submit(folder, index, stage, radius, backend, output, attempt, options)
            write_text(active, json.dumps([attempt.name, job, submit_host]))
        scheduler_state = "Submitted"
        next_check = time.monotonic() + 5
        missing_since = None
        while True:
            completed = file_stamp(output)
            if completed is not None and completed != previous:
                active.unlink(missing_ok=True)
                return load_result(output), f"Slurm job {job}"
            if stop.is_set():
                cancel = scheduler(["scancel", job], submit_host=submit_host)
                if cancel.returncode and file_stamp(output) == previous:
                    raise RuntimeError(f"Could not stop job {job}: {cancel.stderr.strip()}")
                active.unlink(missing_ok=True)
                raise InterruptedError(f"Stopped job {job}")
            if (attempt / "error").exists():
                active.unlink(missing_ok=True)
                raise RuntimeError(f"Job {job}: {(attempt / 'error').read_text()} · {attempt / 'slurm.log'}")
            now = time.monotonic()
            if now >= next_check:
                state = scheduler(["squeue", "-h", "-j", job, "-o", "%i %T"], submit_host=submit_host)
                row = re.search(rf"^{re.escape(job)}\s+(\S+)$", state.stdout, re.MULTILINE)
                gone = (not state.returncode and row is None) or "invalid job id" in state.stderr.lower()
                if state.returncode and not gone:
                    raise RuntimeError(f"Could not check job {job}: {state.stderr.strip()}. Compare again to reconnect.")
                if gone:
                    if missing_since is None:
                        missing_since = now
                    elif now - missing_since >= 60:
                        active.unlink(missing_ok=True)
                        raise RuntimeError(f"Job {job} left the queue without a visible result. See {attempt / 'slurm.log'}")
                else:
                    missing_since = None
                scheduler_state = row[1] if row else "Checking for output"
                next_check = now + 8
            progress_file = attempt / "progress"
            detail = (progress_file.read_text()
                      if missing_since is None and progress_file.exists() else scheduler_state)
            status(f"Job {job} · {detail}")
            stop.wait(0.25)
