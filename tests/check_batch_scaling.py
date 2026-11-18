"""Tiny end-to-end CPU scaling check with a stand-in for the Slurm command line."""
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from notebooks.batch_scaling import batch_experiment
from notebooks.experiment import configurations
from segregation import jobs


async def check():
    assert configurations(60) == [1, 2, 4, 8, 16, 32, 60]
    events, snapshots, statuses = [], [], []
    processes = []
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        def scheduler(command, *, submit_host, env=None):
            assert submit_host == "test-host"
            if command[0] == "squeue":
                return subprocess.CompletedProcess(command, 0, "501 RUNNING\n", "")
            assert command[0] == "sbatch" and "--cpus-per-task=2" in command
            assert "--partition=test" in command and not any(x.startswith("--gres") for x in command)
            args = command[command.index(str(ROOT / "scripts/bait-scaling.sbatch")):]
            logfile = next(x[9:] for x in command if x.startswith("--output="))
            with open(logfile, "wb") as log:
                processes.append(subprocess.Popen(["/bin/sh", *args], cwd=ROOT, stdout=log, stderr=log,
                    env={**env, "SLURM_CPUS_PER_TASK": "2", "SLURM_JOB_ID": "501"}))
            return subprocess.CompletedProcess(command, 0, "501\n", "")
        (parent / "slurm.toml").write_text('partition="test"\nsubmit_host="test-host"\n')
        with patch.object(jobs, "RESULTS", parent), patch.object(jobs, "scheduler", scheduler):
            run = await asyncio.wait_for(batch_experiment([101], 2, 1, events.append, snapshots.append,
                                                          statuses.append, thread_counts=[2, 1]), 90)
        assert processes[0].wait(timeout=5) == 0
        assert run.parent == parent and snapshots[-1]["cpus"] == 2
        fast, baseline = snapshots[-1]["rows"]
        assert [fast["threads"], baseline["threads"]] == [2, 1]
        assert fast["reference_threads"] == 1 and fast["speedup"] == baseline["median_s"] / fast["median_s"]
        assert events and all(e["capacity"] == 2 and e["job"] == "501" for e in events)
        assert any(e["sample"] and e["sample"]["cores"] > 0 for e in events)
        assert json.loads((run / "session.json").read_text())["status"] == "complete"
        assert (run / "bait-101.f64").stat().st_size == 8 * 101**2
        submitting = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        cancelled = []
        def delayed_submit(command, *, submit_host, env=None):
            assert submit_host == "test-host"
            if command[0] == "sbatch":
                loop.call_soon_threadsafe(submitting.set)
                assert release.wait(5)
                return subprocess.CompletedProcess(command, 0, "502\n", "")
            cancelled.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")
        with patch.object(jobs, "RESULTS", parent), patch.object(jobs, "scheduler", delayed_submit):
            task = asyncio.create_task(batch_experiment([101], 2, 1, events.append, snapshots.append, statuses.append))
            await asyncio.wait_for(submitting.wait(), 5)
            task.cancel()
            release.set()
            try:
                await task
            except asyncio.CancelledError:
                pass
            assert cancelled == [["scancel", "502"]]
    print("CPU batch submitted once; two actual Julia thread configurations, worker telemetry and saved results verified.")


if __name__ == "__main__":
    asyncio.run(check())
