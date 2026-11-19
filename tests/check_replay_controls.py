"""Check that replay restores the recorded batch route and workload without allocating a node."""

from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "notebooks"), str(ROOT)]
import dashboard


def check():
    maps = dashboard.inputs()
    maps[2020] = dict(spacing=2, sha="prepared-map-checksum")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with (patch.object(dashboard, "ROOT", root),
              patch.object(dashboard, "inputs", return_value=maps),
              patch.object(dashboard, "slurm_config", return_value=root / "slurm.toml"),
              patch.object(dashboard, "settings", return_value={"benchmark_cpus": 60})):
            ui = dashboard.Dashboard()
            old = dict(status="complete", rows=[dict(n=2020, threads=n) for n in (60, 16)],
                       window_s=10, maps={"2020": {"sha256": maps[2020]["sha"]}})
            current = dict(old, plan=dict(levels=[2020], ceiling=60, threads=[60, 16]))
            for manifest in (old, current):
                ui.recorded_config, reason = ui.replay_config(manifest, True, 60)
                assert not reason
                with patch.object(ui, "launch") as start:
                    ui.rerun_session()
                    start.assert_called_once()
                assert ui.tabs.selected_index == 1 and ui.location.value == "slurm"
                assert ui.level.value == (2020,) and ui.run_mode.value == (60, 16)
                assert ui.ceiling.value == 60 and ui.duration.value == 10
            old["maps"]["2020"]["sha256"] = "different-input"
            assert ui.replay_config(old, True)[0] is None
            ui.close()
    print("Verified 60-then-16 replay controls for current and older batch sessions and detection of a changed input.")


if __name__ == "__main__":
    check()
