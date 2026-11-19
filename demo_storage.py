"""Prepare the shared ADA inputs and manage saved stage comparisons."""

import argparse
import errno
import fcntl
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parent


def slurm_config(root=ROOT):
    config = root / "slurm.toml"
    legacy = root / "results/slurm.toml"
    return legacy if legacy.is_file() and not config.exists() else config


def full_ensemble(root=ROOT):
    durable = root / "inputs/segregation-2024"
    legacy = root / "results/segregation-2024"
    if not (durable / "manifest.json").is_file() and (legacy / "manifest.json").is_file():
        return legacy
    return durable


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError as exc:
        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK, errno.ENOTSUP):
            raise
        shutil.copy2(src, dst)


def adopt_inputs(root=ROOT):
    """Publish completed old inputs atomically while retaining their original paths."""
    old = root / "results"
    inputs = root / "inputs"
    inputs.mkdir(exist_ok=True)
    entries = [(old / "Simulation_data.zip", inputs, None),
               (old / "segregation-2024", inputs, "manifest.json"),
               (old / "runtime/julia-1.12.6", root / "runtime", "bin/julia")]
    entries.extend((path, inputs, "contacts.txt") for path in old.glob("map-*kb-*"))
    with (inputs / ".setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for src, parent, marker in entries:
            dst = parent / src.name
            if dst.exists() or not (src / marker if marker else src).is_file():
                continue
            parent.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(dir=parent) as tmp:
                pending = Path(tmp) / src.name
                if marker:
                    shutil.copytree(src, pending, symlinks=True, copy_function=link_or_copy)
                else:
                    link_or_copy(src, pending)
                pending.rename(dst)
            print(f"Recovered {dst.relative_to(root)}", flush=True)
        config = root / "slurm.toml"
        if not config.exists():
            source = slurm_config(root)
            if not source.is_file():
                source = root / "config/ada-h200.toml"
            with tempfile.TemporaryDirectory(dir=root) as tmp:
                pending = Path(tmp) / config.name
                shutil.copy2(source, pending)
                pending.rename(config)
            print(f"Configured slurm.toml from {source.relative_to(root)}", flush=True)


def prepare_ada(with_bait):
    adopt_inputs()
    config = tomllib.loads(slurm_config().read_text())
    options = [f"--{key}={config[key]}" for key in ("partition", "account", "qos") if config.get(key)]
    host = config.get("submit_host")
    scripts = ["prepare-ensemble.sh"]
    if with_bait:
        scripts += ["stage-julia.sh", "prepare-bait-map.sh"]
    for script in scripts:
        args = ["bash", f"scripts/{script}"]
        if script != "stage-julia.sh":
            args += options
        if host:
            remote = f"cd {shlex.quote(str(ROOT))} && {shlex.join(args)}"
            args = ["ssh", "-nT", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                    "-o", "ConnectTimeout=8", "--", host, remote]
        completed = subprocess.run(args, cwd=ROOT)
        if completed.returncode:
            raise SystemExit(f"{script} failed (exit {completed.returncode}). Correct the reported error and rerun prepare-ada.")


def clear_comparisons(cache):
    """Use the request lock so a clear cannot remove an in-flight job's output."""
    removed = busy = 0
    for output in cache.rglob("result.npz"):
        with (output.parent / "lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                busy += 1
                continue
            if (output.parent / "active.json").exists():
                busy += 1
            elif output.is_file():
                output.unlink()
                removed += 1
    return removed, busy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare-ada", help="restore the full ensemble using the shared ADA clone")
    prep.add_argument("--with-bait", action="store_true", help="also prepare Julia and the 2,020-locus CPU map")
    commands.add_parser("clear-cache", help="remove completed stage answers, keeping inputs and run logs")
    args = parser.parse_args()
    if args.command == "clear-cache":
        removed, busy = clear_comparisons(ROOT / "results/segregation-queries")
        print(f"Cleared {removed} saved stage comparisons. Skipped {busy} with an active request.")
    else:
        prepare_ada(args.with_bait)
