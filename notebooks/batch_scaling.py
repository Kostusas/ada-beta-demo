"""Run and observe a Julia thread sweep in its own Slurm CPU allocation."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import tempfile
import time

from notebooks.experiment import experiment
from segregation import jobs
from segregation.query import write_text


async def run_worker(args):
    if args.julia:
        os.environ["JULIA"] = args.julia
    task = asyncio.current_task()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        await experiment(args.levels, args.cpus, args.seconds, lambda _: None, lambda _: None,
                         directory=args.directory, thread_counts=args.threads)
    except Exception as exc:
        write_text(args.directory / "error.txt", str(exc))
        raise


async def batch_experiment(levels, ceiling, seconds, update, completed, status, *, thread_counts=None):
    options = jobs.settings()
    host = options.get("submit_host", "")
    jobs.RESULTS.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(dir=jobs.RESULTS, prefix="session-"))
    arguments = [directory, "--cpus", ceiling, "--seconds", seconds, "--levels", *levels]
    if thread_counts is not None:
        arguments.extend(["--threads", *thread_counts])
    if options.get("julia"):
        arguments.extend(["--julia", options["julia"]])
    # Large cubes need enough room for Julia, the cube, and the verification arrays.
    memory_gib = max(8, (max(levels)**3 * 8 + 2**30 - 1) // 2**30 + 3)
    allocation = dict(options, memory=options.get("benchmark_memory", f"{memory_gib}G"),
                      time=options.get("benchmark_time", "00:30:00"))
    launch = asyncio.create_task(asyncio.to_thread(jobs.submit_python, "bait-scaling", allocation,
        directory, "bait-scaling.sbatch", arguments, cpus=ceiling))
    job = None
    try:
        job = await asyncio.shield(launch)
        status(f"Job {job} · requested {ceiling} CPUs · waiting for worker")
        write_text(directory / "batch.json", json.dumps({"job": job, "submit_host": host, "cpus": ceiling}))
        offset = row_count = 0
        next_check = time.monotonic() + 5
        missing_since = None
        while True:
            error = directory / "error.txt"
            if error.is_file():
                raise RuntimeError(f"Job {job}: {error.read_text()} · {directory / 'slurm.log'}")
            meta = directory / "session.json"
            if meta.is_file():
                run = json.loads(meta.read_text())
                if run["status"] == "failed":
                    raise RuntimeError(run["error"])
                if run["status"] == "stopped":
                    raise RuntimeError(f"Job {job} stopped.")
                if len(run["rows"]) > row_count:
                    completed(run)
                    row_count = len(run["rows"])
                telemetry = directory / "telemetry.jsonl"
                if telemetry.is_file():
                    with telemetry.open("rb") as stream:
                        stream.seek(offset)
                        for line in stream:
                            if not line.endswith(b"\n"):
                                break
                            offset += len(line)
                            event = json.loads(line)
                            update(dict(event, node=run["node"], capacity=run["cpus"], job=job))
                if run["status"] == "complete":
                    return directory
            now = time.monotonic()
            if now >= next_check:
                state = await asyncio.to_thread(jobs.scheduler, ["squeue", "-h", "-j", job, "-o", "%i %T"], submit_host=host)
                if state.returncode and "invalid job id" not in state.stderr.lower():
                    raise RuntimeError(f"Could not monitor job {job}: {state.stderr.strip()} · {directory}")
                rows = [line.split() for line in state.stdout.splitlines()]
                current = next((r[1] for r in rows if len(r) == 2 and r[0] == job), None)
                if current:
                    missing_since = None
                    if not meta.is_file():
                        status(f"Job {job} · {current} · requested {ceiling} CPUs")
                elif missing_since is None:
                    missing_since = now
                elif now - missing_since >= 60:
                    raise RuntimeError(f"Job {job} ended without a completed run. See {directory / 'slurm.log'}")
                next_check = now + 8
            await asyncio.sleep(0.45)
    except asyncio.CancelledError:
        # Stop also works if the user clicks it before sbatch has returned the ID.
        job = job or await launch
        call = await asyncio.to_thread(jobs.scheduler, ["scancel", job], submit_host=host)
        if call.returncode and "invalid job id" not in call.stderr.lower():
            raise RuntimeError(f"Could not stop job {job}: {call.stderr.strip()}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--levels", type=int, nargs="+", required=True)
    parser.add_argument("--cpus", type=int, required=True)
    parser.add_argument("--threads", type=int, nargs="+")
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--julia")
    asyncio.run(run_worker(parser.parse_args()))
