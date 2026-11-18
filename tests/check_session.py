"""Short integration check for the process sampler, validated sweep and cancellation."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "notebooks"), str(ROOT)]
import psutil
from experiment import cpu_capacity, experiment


async def check():
    events = []
    with tempfile.TemporaryDirectory() as temp:
        cap = min(2, cpu_capacity())
        path = await experiment([101], cap, 1.0, events.append, lambda _: None, Path(temp))
        saved = json.loads((path / "session.json").read_text())
        samples = [e["sample"] for e in events if e["phase"] == "Predicting triplets" and e["sample"]]
        assert saved["status"] == "complete" and len(saved["rows"]) == cap
        assert len({r["digest"] for r in saved["rows"]}) == 1
        assert all(r["passes"] > 0 and r["seconds"] >= 1 for r in saved["rows"])
        assert samples and any(s["cores"] > 0 and s["rss"] > 0 for s in samples)
        if sys.platform == "linux":
            assert any(s["last_cpu"] for s in samples)
            assert any(any(rate > 0 for _,rate in s["busiest"]) for s in samples)
        assert (path / "bait-101.f64").stat().st_size == 101 * 101 * 8
        started = asyncio.Event()
        def on_sample(event):
            events.append(event)
            started.set()
        task = asyncio.create_task(experiment([101], 1, 5, on_sample, lambda _: None, Path(temp)))
        await asyncio.wait_for(started.wait(), 30)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert not psutil.pid_exists(events[-1]["pid"])
        assert "stopped" in [json.loads(p.read_text())["status"] for p in Path(temp).glob("*/session.json")]


if __name__ == "__main__":
    asyncio.run(check())
