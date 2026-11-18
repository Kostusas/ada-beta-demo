"""Tiny integration check for the real batch entry point with a simulated Slurm CLI."""

import json
import os
import shlex
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segregation import jobs
from segregation.engine import Coordinates, Ensemble, PREVIEW
from segregation.index import build_index, index_path
from segregation.query import load_result


def quiet(*_):
    pass


class Scheduler:
    def __init__(self):
        self.jobs = []
        self.reject = self.hold = False

    def __call__(self, cmd, *, submit_host="", env=None):
        assert submit_host == "test-host"
        command = cmd[0]
        if command == "sbatch":
            if self.reject:
                return subprocess.CompletedProcess(cmd, 1, "", "unknown partition")
            assert "--partition=test" in cmd and "--cpus-per-task=2" in cmd
            assert "SBATCH_QOS" not in env and "CUDA_VISIBLE_DEVICES" not in env
            job = str(100 + len(self.jobs))
            script_index = cmd.index(str(ROOT / "scripts/segregation-query.sbatch"))
            proc = None
            if not self.hold:
                proc = subprocess.Popen(["/bin/sh", *cmd[script_index:]], cwd=ROOT,
                                        env={**env, "SLURM_JOB_ID": job, "SLURM_CPUS_PER_TASK": "2"})
            self.jobs.append((job, proc))
            return subprocess.CompletedProcess(cmd, 0, "Welcome to the cluster\n" + job + ";ada\n", "")
        if command == "scancel":
            assert cmd[1] == self.jobs[-1][0]
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if command == "squeue" and self.hold:
            job = self.jobs[-1][0]
            assert cmd == ["squeue", "-h", "-j", job, "-o", "%i %T"]
            return subprocess.CompletedProcess(cmd, 0, f"Welcome to the cluster\n{job} PENDING\n", "")
        raise AssertionError(f"Unexpected command: {command}")


def run():
    stop = threading.Event()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        # OpenSSH sends a single shell command to the host. Preserve every argument
        # exactly, including shared paths with spaces and apostrophes.
        args = ["sbatch", "--chdir=/shared/a user's project", "worker.sbatch"]
        with patch.object(subprocess, "run") as invoke:
            invoke.return_value.returncode = 0
            jobs.scheduler(args, submit_host="test-host")
            ssh = invoke.call_args.args[0]
            assert ssh[0] == "ssh" and "BatchMode=yes" in ssh and "StrictHostKeyChecking=yes" in ssh
            assert ssh[-2] == "test-host" and shlex.split(ssh[-1]) == args
            jobs.scheduler(args)
            assert invoke.call_args.args[0] == args
            invoke.side_effect = FileNotFoundError()
            try:
                jobs.scheduler(args)
            except RuntimeError as exc:
                assert "submit_host" in str(exc)
            else:
                raise AssertionError("Missing local Slurm client must give setup instructions")
        index = root / "index"
        digest = build_index(PREVIEW, index, 2)
        assert digest == build_index(PREVIEW, index, 2)
        coords = Coordinates(PREVIEW, 1, stop, None, index=index)
        assert isinstance(coords.pos, np.memmap) and not coords.pos.flags.writeable
        (root / "slurm.toml").write_text('partition="test"\ncpus=2\nsubmit_host="test-host"\n')
        sim = Scheduler()

        def request(radius, **kwargs):
            return jobs.request(PREVIEW, index, digest, 50, radius, "CPU",
                                kwargs.get("stop", stop), kwargs.get("status", quiet), cache=root / "cache")

        with patch.object(jobs, "scheduler", sim), patch.object(jobs, "RESULTS", root), \
             patch.dict(os.environ, {"SBATCH_QOS": "ood", "CUDA_VISIBLE_DEVICES": ""}):
            with ThreadPoolExecutor(max_workers=2) as pool:
                replies = list(pool.map(request, [2, 2]))
            assert len(sim.jobs) == 1 and {r[1] for r in replies} == {"Saved result", "Slurm job 100"}
            assert sim.jobs[-1][1].wait(timeout=10) == 0
            result = replies[0][0]
            reference = Ensemble(PREVIEW, "CPU", 2, 0, stop, quiet).compare(0.5, 2, stop, quiet)
            np.testing.assert_allclose(reference["difference"], result["difference"], equal_nan=True)
            for a, b in zip(reference["points"], result["points"], strict=True):
                assert a["score"] == b["score"] and a["contacts"] == b["contacts"]
                np.testing.assert_array_equal(a["frames"], b["frames"])
            assert result["index_id"] == digest and result["job"] == "100"
            assert result["frames"] == 24 and result["pairs"] == 965373
            assert request(2)[1] == "Saved result" and len(sim.jobs) == 1

            sim.reject = True
            try:
                request(2.25)
            except RuntimeError as exc:
                assert "unknown partition" in str(exc)
            else:
                raise AssertionError("A rejected submission must surface an error")
            sim.reject = False
            sim.hold = True
            cancel = threading.Event()

            def cancel_submitted(message):
                if message.endswith("PENDING"):
                    cancel.set()

            try:
                with patch.object(jobs, "time") as clock:
                    clock.monotonic.side_effect = [0, 10]
                    request(2.25, stop=cancel, status=cancel_submitted)
            except InterruptedError:
                pass
            else:
                raise AssertionError("Pending job did not cancel")
            assert not list((root / "cache").rglob("active.json"))
            sim.hold = False
            result, source = request(2.25)
            assert source == "Slurm job 102" and result["radius"] == 2.25
            assert sim.jobs[-1][1].wait(timeout=10) == 0

            # Reconnect to a submitted job after a frontend exits, with no resubmission.
            folder = root / "cache" / "v1" / digest / "CPU-50-10"
            folder.mkdir()
            attempt = folder / "attempt-reconnect"
            attempt.mkdir()
            (folder / "active.json").write_text(json.dumps([attempt.name, "103"]))
            with subprocess.Popen([sys.executable, "-m", "segregation.query", str(PREVIEW), str(index),
                                   "50", "2.5", "CPU", str(folder / "result.npz"), str(attempt)],
                                  cwd=ROOT, env={**os.environ, "SLURM_JOB_ID": "103", "SLURM_CPUS_PER_TASK": "2"}) as proc:
                reconnected, _ = request(2.5)
                assert proc.wait(timeout=10) == 0
            assert len(sim.jobs) == 3 and reconnected["job"] == "103"
            assert load_result(folder / "result.npz")["stage"] == 0.5
    print("Verified stage-only batch execution, reference parity, deduplication, replay, rejection, cancellation, retry and reconnect.")


def check_ada_gpu():
    """Exercise the notebook's real submit/cache path on the eight archived tracks."""
    stop = threading.Event()
    index = index_path(PREVIEW)
    digest = build_index(PREVIEW, index, 2)
    previous = ""

    def status(message):
        nonlocal previous
        if message != previous:
            print(message, flush=True)
            previous = message

    result, source = jobs.request(PREVIEW, index, digest, 50, 2.0, "CUDA", stop, status)
    reference = Ensemble(PREVIEW, "CPU", 2, 0, stop, quiet).compare(0.5, 2, stop, quiet)
    np.testing.assert_allclose(result["difference"], reference["difference"], equal_nan=True)
    for gpu, cpu in zip(result["points"], reference["points"], strict=True):
        np.testing.assert_allclose([gpu["contacts"], gpu["score"]], [cpu["contacts"], cpu["score"]])
        np.testing.assert_array_equal(gpu["frames"], cpu["frames"])
    assert result["pairs"] == reference["pairs"] and result["job"]
    _, replay = jobs.request(PREVIEW, index, digest, 50, 2.0, "CUDA", stop, status)
    assert replay == "Saved result"
    print(f"{source} · {result['device']} on {result['node']} · CPU agreement and saved replay passed.")


if __name__ == "__main__":
    if sys.argv[1:] == ["--slurm-gpu"]:
        check_ada_gpu()
    elif not sys.argv[1:]:
        run()
    else:
        raise SystemExit("Usage: python tests/check_batch_queries.py [--slurm-gpu]")
