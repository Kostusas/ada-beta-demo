"""Check that every preview stage displays its own limits and preserves unclipped values."""
import asyncio
from functools import partial
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segregation import view


async def check():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with (patch.object(view, "index_path", return_value=root / "index"),
              patch.object(view, "request", partial(view.request, cache=root / "cache")),
              patch.object(view, "slurm_config", return_value=root / "slurm.toml"),
              patch.object(view, "full_ensemble", return_value=root / "full")):
            widget = view.SegregationView()
            for stage in (50, 75):
                widget.stage.value = stage
                await widget.task
                assert widget.points, widget.progress.value
                trace = widget.map.data[0]
                data = np.array(trace.z)
                assert np.isnan(data).any()
                absolute = np.abs(data[np.isfinite(data)])
                assert 0 < trace.zmax == np.percentile(absolute, 95) < absolute.max()
                assert trace.zmin == -trace.zmax
                task = widget.task
                widget.map_scale.value = "full"
                assert trace.zmax == absolute.max() and trace.zmin == -trace.zmax
                np.testing.assert_equal(trace.z, data)
                widget.map_scale.value = "p95"
                assert widget.task is task
            for data in ([[np.nan]], [[0., np.nan]]):
                trace.z = data
                widget.scale_map()
                assert not trace.showscale
        # A prepared installation selects the late-stage GPU run without submitting on open.
        full = root / "staged"
        full.mkdir()
        (full / "manifest.json").write_bytes((view.PREVIEW / "manifest.json").read_bytes())
        with (patch.object(view, "full_ensemble", return_value=full),
              patch.object(view, "index_path", return_value=root / "index"),
              patch.object(view, "request", side_effect=ValueError("test submission")) as submit):
            widget = view.SegregationView()
            assert widget.dataset.value == str(full) and widget.backend.value == "CUDA"
            assert widget.stage.value == 90 and widget.task is None
            assert [value for label, value in widget.stage.options] == list(range(5, 100, 5))
            for reuse in (False, True):
                widget.reuse.value = reuse
                widget.compare_button.click()
                await widget.task
                assert submit.call_args.args[3] == 90
                assert submit.call_args.kwargs["fresh"] == (not reuse)


if __name__ == "__main__":
    asyncio.run(check())
