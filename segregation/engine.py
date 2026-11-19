"""Resident trajectory coordinates and optional CPU/CUDA inter-sister distance cache."""

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil

from demo_storage import ROOT, full_ensemble
from session_resources import cpu_capacity

N = 404
WINDOW = 0.04
PREVIEW = ROOT / "data/segregation-preview"
FULL = full_ensemble()


def ranges(start, stop, chunk):
    return [slice(i, min(i + chunk, stop)) for i in range(start, stop, chunk)]


def check_stop(stop):
    if stop.is_set():
        raise InterruptedError("Comparison stopped")


def replicated(forks):
    locus = np.arange(N)
    return (locus < forks[:, 0, None]) | (locus > forks[:, 1, None])


def pair_support(forks):
    """Count frames in which both loci have replicated, using the two fork endpoints."""
    lo = forks[:, 0]
    end = forks[:, 1] + 1
    # Each unreplicated square [lo:end, lo:end] contributes four signed corners.
    corners = np.zeros((N + 1, N + 1), dtype=np.int64)
    for a, b, sign in ((lo, lo, 1), (lo, end, -1), (end, lo, -1), (end, end, 1)):
        np.add.at(corners, (a, b), sign)
    both_unreplicated = corners[:-1, :-1].cumsum(axis=0).cumsum(axis=1)
    unreplicated = both_unreplicated.diagonal()
    return len(forks) - unreplicated[:, None] - unreplicated[None, :] + both_unreplicated


def cpu_distances(pos, forks, out):
    old, new = pos[:, :N], pos[:, N:]
    scratch = np.empty_like(out)
    for axis in range(3):
        part = scratch if axis else out
        np.subtract(old[:, :, None, axis], new[:, None, :, axis], out=part)
        np.square(part, out=part)
        if axis:
            np.add(out, part, out=out)
    r = replicated(forks)
    np.copyto(out, np.inf, where=~(r[:, :, None] & r[:, None, :]))


CUDA_CODE = """
int a = (i/404)%404, b = i%404, f = i/(404*404);
int lo = forks[2*f], hi = forks[2*f+1];
if ((a >= lo && a <= hi) || (b >= lo && b <= hi)) {
    distance = __int_as_float(0x7f800000);
} else {
    int j=(f*808+a)*3, k=(f*808+404+b)*3;
    float x=pos[j]-pos[k], y=pos[j+1]-pos[k+1], z=pos[j+2]-pos[k+2];
    distance = (x*x+y*y)+z*z;
}
"""


ARRAY_NAMES = ("pos", "forks", "block", "group", "track", "score", "fraction")


def read_index_id(folder, index):
    meta = json.loads((index / "index.json").read_text())
    digest = hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest()
    if meta["manifest_sha256"] != digest:
        raise ValueError("The stage index belongs to a different trajectory manifest.")
    return meta["id"]


class Coordinates:
    def __init__(self, folder, workers, stop, progress, *, index=None):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "manifest.json").read_text())
        self.tracks = self.meta["tracks"]
        if self.meta["version"] != 1 or not self.tracks:
            raise ValueError("Unsupported or empty trajectory manifest")
        self.workers = min(max(1, workers), cpu_capacity(), 60)
        if index is not None:
            self.index_id = read_index_id(self.folder, index)
            for key in ARRAY_NAMES:
                setattr(self, key, np.load(index / f"{key}.npy", mmap_mode="r", allow_pickle=False))
            return
        offsets = np.cumsum([0] + [t["frames"] for t in self.tracks])
        size = int(offsets[-1])
        self.pos = np.empty((size, 2 * N, 3), np.float32)
        self.forks = np.empty((size, 2), np.int16)
        self.block = np.empty(size, np.int32)
        self.group = np.empty(size, np.int64)
        self.track = np.empty(size, np.int32)
        self.score = np.empty(size, np.float64)

        def load(i):
            check_stop(stop)
            span = slice(offsets[i], offsets[i + 1])
            with np.load(
                self.folder / self.tracks[i]["file"], allow_pickle=False
            ) as data:
                p, f = data["pos"], data["forks"]
                self.pos[span], self.forks[span] = p, f
                self.block[span], self.group[span] = (
                    data["block"],
                    data["group"].astype(np.int64),
                )
            self.track[span] = i
            mask = replicated(f)
            count = mask.sum(axis=1)
            below = ((p[:, :N, 2] < 0) & mask).sum(axis=1)
            above = ((p[:, N:, 2] > 0) & mask).sum(axis=1)
            self.score[span] = np.abs(below + above - count) / np.maximum(1, count)

        with ThreadPoolExecutor(self.workers) as pool:
            for i, _ in enumerate(pool.map(load, range(len(self.tracks))), 1):
                progress("Loading trajectories", i, len(self.tracks))
        self.fraction = 1 - (self.forks[:, 1] - self.forks[:, 0]).astype(float) / N
        # Adjacent stages become contiguous ranges in RAM and VRAM.
        order = np.argsort(self.fraction, kind="stable")
        for key in ARRAY_NAMES:
            setattr(self, key, getattr(self, key)[order])


class Ensemble(Coordinates):
    def __init__(self, folder, backend, workers, budget_gib, stop, progress, *, index=None):
        super().__init__(folder, workers, stop, progress, index=index)
        size = len(self.fraction)
        self.distance = None
        self.cp = None
        self.device = "CPU"
        if backend == "CUDA":
            import cupy as cp

            cp.cuda.Device().use()
            self.cp = cp
            self.device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            self.kernel = cp.ElementwiseKernel(
                "raw float32 pos, raw int16 forks",
                "float32 distance",
                CUDA_CODE,
                "sister_squared_distance",
                options=("--fmad=false",),
            )
        self.labels = np.array(
            [0 if t["condition"] == "No SMC" else 1 for t in self.tracks]
        )
        need = size * N * N * 4
        available = (
            self.cp.cuda.runtime.memGetInfo()[0]
            if self.cp
            else psutil.virtual_memory().available
        )
        self.cached = need <= min(budget_gib * 2**30, available * 0.75)
        self.pair_bytes = need if self.cached else 0
        if self.cached:
            if self.cp:
                # Own the large slab directly so changing datasets releases it.
                ptr = self.cp.cuda.MemoryPointer(self.cp.cuda.Memory(need), 0)
                self.distance = self.cp.ndarray(
                    (size, N, N), dtype=self.cp.float32, memptr=ptr
                )
            else:
                self.distance = np.empty((size, N, N), dtype=np.float32)
            chunks = ranges(0, size, 64 if self.cp else 8)

            def build(span):
                check_stop(stop)
                self.fill(span, self.distance[span])

            if self.cp:
                for i, span in enumerate(chunks, 1):
                    build(span)
                    self.cp.cuda.get_current_stream().synchronize()
                    progress("Caching squared distances on GPU", i, len(chunks))
            else:
                with ThreadPoolExecutor(self.workers) as pool:
                    for i, _ in enumerate(pool.map(build, chunks), 1):
                        progress("Caching squared distances in RAM", i, len(chunks))

    def fill(self, span, out):
        if self.cp:
            self.kernel(
                self.cp.asarray(self.pos[span]), self.cp.asarray(self.forks[span]), out
            )
        else:
            cpu_distances(self.pos[span], self.forks[span], out)

    def compare(self, stage, radius, stop, progress):
        start = time.perf_counter()
        cpu_start = time.process_time()
        lo, hi = (
            int(np.searchsorted(self.fraction, stage - WINDOW)),
            int(np.searchsorted(self.fraction, stage + WINDOW, side="right")),
        )
        if hi == lo:
            raise ValueError(
                "No frames within four percentage points of this replication stage"
            )
        ids = np.arange(lo, hi)
        xp = self.cp or np
        frame_hits = xp.empty(len(ids), dtype=xp.int32)
        hits = xp.zeros((2, N, N), dtype=xp.int64)
        valid_loci = N - (self.forks[ids, 1] - self.forks[ids, 0] + 1)
        label = self.labels[self.track]
        chunks = ranges(lo, hi, 64 if self.cp else 8)
        # Fold maps in the worker, keeping at most one accumulator per worker.
        lock = threading.Lock()
        processed = 0

        def reduce(span):
            nonlocal processed
            check_stop(stop)
            if self.cached:
                d2 = self.distance[span]
            else:
                d2 = xp.empty((span.stop - span.start, N, N), dtype=xp.float32)
                self.fill(span, d2)
            contact = d2 < np.float32(radius**2)
            idx = slice(span.start - lo, span.stop - lo)
            frame_hits[idx] = contact.sum(axis=(1, 2), dtype=xp.int32)
            for c in (0, 1):
                rows = np.flatnonzero(label[span] == c)
                if rows.size:
                    sel = xp.asarray(rows)
                    h = contact[sel].sum(axis=0, dtype=xp.int32)
                    with lock:
                        hits[c] += h
            with lock:
                processed += span.stop - span.start
                progress("Comparing archived frames", processed, len(ids))

        if self.cp:
            for index, chunk in enumerate(chunks):
                reduce(chunk)
                if index % 8 == 7:
                    self.cp.cuda.get_current_stream().synchronize()
            # Only the two aggregate maps and one count per frame cross to the host.
            hits = self.cp.asnumpy(hits)
            frame_hits = self.cp.asnumpy(frame_hits)
        else:
            with ThreadPoolExecutor(self.workers) as pool:
                for _ in pool.map(reduce, chunks):
                    pass
        scores = frame_hits / np.maximum(1, valid_loci.astype(np.int64) ** 2)
        support = np.stack([pair_support(self.forks[ids[label[ids] == c]]) for c in (0, 1)])
        probability = np.full(hits.shape, np.nan)
        np.divide(hits, support, out=probability, where=support > 0)
        points = []
        for track in np.unique(self.track[ids]):
            local = np.flatnonzero(self.track[ids] == track)
            frames = ids[local]
            frames = frames[np.argsort(self.block[frames])]
            points.append(
                dict(
                    track=int(track),
                    condition=self.tracks[track]["condition"],
                    score=float(self.score[frames].mean()),
                    contacts=float(scores[local].mean()),
                    frames=frames,
                    nearest=int(np.argmin(abs(self.fraction[frames] - stage))),
                )
            )
        elapsed = time.perf_counter() - start
        return dict(
            points=points,
            difference=probability[1] - probability[0],
            frames=len(ids),
            pairs=int(np.sum(valid_loci.astype(np.int64) ** 2)),
            seconds=elapsed,
            busy=(time.process_time() - cpu_start) / elapsed,
            stage=stage,
            radius=radius,
        )

    @property
    def coordinate_bytes(self):
        return self.pos.nbytes
