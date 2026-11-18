"""Launch and observe Julia workers inside the current Jupyter allocation."""
import asyncio
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import time

import psutil

from demo import MAP, MAP_SHA, ROOT, file_hash
from session_resources import cpu_capacity


def configurations(ceiling):
    return sorted({1, ceiling, *(n for n in (2, 4, 8, 16, 32, 64, 128) if n < ceiling)})


def inputs():
    maps = {n: dict(path=MAP, stride=404//n, spacing=4040//n, sha=MAP_SHA) for n in (101, 202, 404)}
    for path in sorted((ROOT / "results").glob("map-*kb-*/contacts.txt")):
        manifest = path.with_suffix(".json")
        if not manifest.is_file():
            continue
        meta = json.loads(manifest.read_text())
        if meta["stride"] in (2, 4) and meta["frames"] == 3060 and meta["radius"] == 5:
            maps[4040//meta["stride"]] = dict(path=path, stride=1, spacing=meta["stride"], sha=meta["sha256"])
    return maps


class Sampler:
    def __init__(self, pid):
        self.process = psutil.Process(pid)
        self.previous = None
        self.tasks = Path(f"/proc/{pid}/task")

    def sample(self):
        try:
            with self.process.oneshot():
                cpu = self.process.cpu_times()
                rss = self.process.memory_info().rss
                try:
                    tids = {t.id: t.user_time+t.system_time for t in self.process.threads()}
                    os_threads = len(tids)
                except psutil.AccessDenied:
                    tids = {}
                    os_threads = None
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
        now = time.monotonic()
        total = cpu.user + cpu.system
        previous, self.previous = self.previous, (now, total, tids)
        if previous is None:
            return None
        before, old_total, old_tids = previous
        dt = now - before
        activity = [(tid, max(0, seconds-old_tids[tid])/dt) for tid, seconds in tids.items() if tid in old_tids]
        activity.sort(key=lambda item: item[1], reverse=True)
        last_cpu = {}
        if self.tasks.is_dir():
            for tid, _ in activity[:8]:
                try:
                    # Fields after the final ')' begin with field 3 (thread state).
                    fields = (self.tasks / str(tid) / "stat").read_text().rsplit(")", 1)[1].split()
                    last_cpu[str(tid)] = int(fields[39 - 3])
                except (OSError, IndexError, ValueError):
                    continue  # thread exited between the two reads
        return dict(last_cpu=last_cpu, rss=rss, cores=max(0, total-old_total)/dt, os_threads=os_threads, busiest=activity[:12])


async def worker(command, n, threads, update, stream, origin, index, count):
    proc = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.STDOUT,
                                                env={**os.environ, "OPENBLAS_NUM_THREADS": "1"})
    info = dict(index=index,count=count,n=n, threads=threads, pid=proc.pid, phase="Starting Julia", passes=0, compute_s=0)
    errors = deque(maxlen=12)
    row = None

    async def read_events():
        nonlocal row
        async for line in proc.stdout:
            fields = line.decode(errors="replace").rstrip().split("\t")
            if fields[0] == "phase":
                info["phase"] = fields[1]
            elif fields[0] == "progress":
                info.update(passes=int(fields[1]), compute_s=float(fields[2]))
            elif fields[0] == "done":
                passes, total, median, busy, rss, digest, actual, julia, cpu = fields[1:]
                if int(actual) != threads:
                    raise RuntimeError("Worker thread count differs from requested configuration")
                row = dict(n=n, threads=threads, passes=int(passes), seconds=float(total),
                           median_s=float(median), busy_cores=float(busy), peak_rss=int(rss), digest=digest, julia=julia, cpu=cpu)
            else:
                errors.append(" ".join(fields))

    reader = asyncio.create_task(read_events())
    try:
        sampler = Sampler(proc.pid)
        while True:
            sample = sampler.sample()
            event = dict(**info, elapsed=time.monotonic()-origin, sample=sample)
            stream.write(json.dumps(event) + "\n")
            stream.flush()
            update(event)
            finished, _ = await asyncio.wait({reader}, timeout=0.45)
            if finished:
                break
        await reader
        code = await proc.wait()
        if code or row is None:
            raise RuntimeError(f"Julia exited with status {code}. " + "\n".join(errors))
        return row
    finally:
        if proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), 3)
            except TimeoutError:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                await proc.wait()
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)


async def experiment(levels, ceiling, seconds, update, completed, output=None, *, directory=None, thread_counts=None):
    configured = os.getenv("JULIA")
    if configured:
        julia = shutil.which(configured)
        if not julia:
            raise RuntimeError(f"Configured Julia executable {configured!r} is unavailable on {socket.gethostname()}. Check JULIA or the julia setting in results/slurm.toml.")
    else:
        julia = shutil.which(str(ROOT / "results/runtime/julia-1.12.6/bin/julia")) or shutil.which("julia")
    if not julia:
        raise RuntimeError(f"Julia is missing on {socket.gethostname()}. Run 'bash scripts/stage-julia.sh' once from the shared ADA clone, then retry. A site installation can also be selected with julia in results/slurm.toml.")
    if not 1 <= ceiling <= cpu_capacity() or seconds <= 0:
        raise ValueError("Invalid CPU ceiling or measurement duration")
    if thread_counts is not None and (not thread_counts or len(set(thread_counts)) != len(thread_counts)
                                      or any(t not in range(1, ceiling + 1) for t in thread_counts)):
        raise ValueError("Choose distinct thread counts within the allocation")
    maps = inputs()
    parent = output or ROOT / "results"
    parent.mkdir(parents=True, exist_ok=True)
    if directory is None:
        directory = Path(tempfile.mkdtemp(dir=parent, prefix="session-"))
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run = dict(started=started, node=socket.gethostname(), cpus=cpu_capacity(),
               kernel_pid=os.getpid(), window_s=seconds, rows=[], status="running", maps={})
    def save():
        temp = directory / "session.tmp"
        temp.write_text(json.dumps(run, indent=2) + "\n")
        temp.replace(directory / "session.json")
    save()
    try:
        counts = {n: thread_counts or (configurations(ceiling) if n <= 404 else [ceiling]) for n in levels}
        plan = [(n,t) for n in levels for t in counts[n]]
        hashes = set()
        baseline = {}
        origin = time.monotonic()
        with (directory / "telemetry.jsonl").open("w") as stream:
            for index, (n, threads) in enumerate(plan, 1):
                source = maps[n]
                path = source["path"]
                if path not in hashes:
                    if file_hash(path) != source["sha"]:
                        raise ValueError(f"Input checksum mismatch: {path.name}")
                    hashes.add(path)
                bait = 1 + round(1990/source["spacing"])
                run["maps"][n] = dict(sha256=source["sha"], spacing_kb=source["spacing"], bait=bait)
                args = [julia, "--startup-file=no", f"--threads={threads}", str(ROOT/"src/live.jl"),
                        str(path), str(n), str(source["stride"]), str(seconds), str(directory),
                        str(ROOT/"examples/published-slice/bait.f64"), str(bait)]
                row = await worker(args, n, threads, update, stream, origin,index,len(plan))
                if n <= 404:
                    first = baseline.setdefault(n, row.copy())
                    if first["digest"] != row["digest"]:
                        raise ValueError(f"Tensor mismatch at {n} loci, {threads} threads")
                run["rows"].append(row)
                reference = min(counts[n])
                group = [r for r in run["rows"] if r["n"] == n]
                base = next((r for r in group if r["threads"] == reference), None)
                if base and (len(counts[n]) > 1 or reference == 1):
                    for r in group:
                        r["reference_threads"] = reference
                        r["speedup"] = base["median_s"] / r["median_s"]
                        r["efficiency"] = r["speedup"] * reference / r["threads"]
                save()
                completed(run)
        run["status"] = "complete"
    except asyncio.CancelledError:
        run["status"] = "stopped"
        raise
    except Exception as exc:
        run.update(status="failed", error=str(exc))
        raise
    finally:
        save()
    return directory
