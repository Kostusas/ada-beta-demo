"""Exercise recovery and cache clearing in an isolated clone with eight real tracks."""

import fcntl
from pathlib import Path
import shutil
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo_storage import adopt_inputs, clear_comparisons, full_ensemble, slurm_config
from segregation.engine import Coordinates, PREVIEW
from segregation.index import build_index, index_path


def check(root):
    old = root / "results"
    folder = old / "segregation-2024"
    shutil.copytree(PREVIEW, folder)
    index_id = build_index(folder, index_path(folder), 2)
    config = 'partition = "existing-partition"\n'
    (old / "slurm.toml").write_text(config)
    for path in ("Simulation_data.zip", "map-2kb-sample/contacts.txt", "runtime/julia-1.12.6/bin/julia"):
        target = old / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("sample")
    assert full_ensemble(root) == folder and slurm_config(root) == old / "slurm.toml"
    adopt_inputs(root)
    adopt_inputs(root)
    assert slurm_config(root) == root / "slurm.toml"
    assert slurm_config(root).read_text() == config
    assert full_ensemble(root) == root / "inputs/segregation-2024"

    cache = old / "segregation-queries"
    saved, locked, disconnected = [cache / "v1/digest" / key for key in ("CPU-90-8", "CPU-75-8", "CPU-50-8")]
    for path in (saved, locked, disconnected):
        path.mkdir(parents=True)
        (path / "result.npz").touch()
    (disconnected / "active.json").touch()
    (saved / "slurm.log").touch()
    with (locked / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert clear_comparisons(cache) == (1, 2)
        assert (locked / "result.npz").is_file() and (disconnected / "result.npz").is_file()
        assert (saved / "slurm.log").is_file() and not (saved / "result.npz").exists()

    # Repeat the original incident in this temporary clone.
    shutil.rmtree(old)
    assert clear_comparisons(cache) == (0, 0)
    folder = full_ensemble(root)
    coords = Coordinates(folder, 1, threading.Event(), None, index=index_path(folder))
    assert coords.index_id == index_id and len(coords.tracks) == 8
    for path in ("inputs/Simulation_data.zip", "inputs/map-2kb-sample/contacts.txt",
                 "runtime/julia-1.12.6/bin/julia", "slurm.toml"):
        assert (root / path).is_file()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        check(Path(temp))
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        shutil.copytree(ROOT / "config", root / "config")
        adopt_inputs(root)
        assert (root / "slurm.toml").read_bytes() == (root / "config/ada-h200.toml").read_bytes()
    print("Verified old and deleted results layouts, preserved inputs and settings, and lock-aware cache clearing.")
