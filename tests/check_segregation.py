"""Tiny real-data test. Pass --cuda on the allocated H200 to check both CUDA paths."""

import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segregation.engine import PREVIEW, Ensemble, N, pair_support


def quiet(*args):
    pass


def run():
    stop = threading.Event()
    cpu = Ensemble(PREVIEW, "CPU", 2, 0, stop, quiet)
    reference = cpu.compare(0.5, 2.0, stop, quiet)
    assert len(reference["points"]) == 8 and reference["frames"] == 24
    # Check the fast support calculation at boundary forks and on archived frames.
    forks = np.vstack(([[0, N - 1], [0, 0], [N - 1, N - 1]], cpu.forks[::7]))
    locus = np.arange(N)
    masks = ((locus < forks[:, :1]) | (locus > forks[:, 1:])).astype(np.int64)
    np.testing.assert_array_equal(pair_support(forks), masks.T @ masks)
    # Independent direct Euclidean calculation for all eligible pairs of one trajectory.
    point = reference["points"][0]
    scores, fractions = [], []
    for idx in point["frames"]:
        a, b = cpu.forks[idx]
        eligible = np.r_[0:a, b + 1 : N]
        xyz = cpu.pos[idx]
        old, new = xyz[eligible], xyz[eligible + N]
        score = abs(np.sum(old[:, 2] < 0) + np.sum(new[:, 2] > 0) - len(old)) / len(old)
        scores.append(score)
        fractions.append(
            (np.linalg.norm(old[:, None] - new[None, :], axis=2) < 2.0).mean()
        )
    np.testing.assert_allclose(
        [point["score"], point["contacts"]], [np.mean(scores), np.mean(fractions)]
    )
    configs = [("CPU", 1)]
    if "--cuda" in sys.argv:
        configs += [("CUDA", 0), ("CUDA", 1)]
    for backend, budget in configs:
        e = Ensemble(PREVIEW, backend, 2, budget, stop, quiet)
        assert e.cached == bool(budget)
        result = e.compare(0.5, 2.0, stop, quiet)
        np.testing.assert_allclose(
            result["difference"], reference["difference"], equal_nan=True
        )
        np.testing.assert_allclose(
            [p["contacts"] for p in result["points"]],
            [p["contacts"] for p in reference["points"]],
        )
        assert e.compare(0.75, 3.0, stop, quiet)["frames"] == 24
        print(
            f"{backend}, cache {e.cached}: {result['frames']} archived frames, {result['pairs']:,} eligible pairs"
        )
    stop.set()
    try:
        cpu.compare(0.25, 2.0, stop, quiet)
    except InterruptedError:
        return
    raise AssertionError("Cancellation was ignored")


if __name__ == "__main__":
    run()
