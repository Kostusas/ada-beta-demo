"""Stage-resolved segregation and linked 3-D trajectory inspection."""

import asyncio
import json
import threading
import time
from html import escape
from pathlib import Path

import ipywidgets as w
import numpy as np
import plotly.graph_objects as go
import psutil

from session_resources import cpu_capacity

from .engine import FULL, PREVIEW, Ensemble, N, WINDOW

COLORS = ("#63d9cb", "#ffab75")
THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#101f2d",
    plot_bgcolor="#152939",
    font=dict(color="#e2eff2"),
    margin=dict(l=55, r=18, t=44, b=48),
)
MICRONS = 0.129


class SegregationView:
    def __init__(self):
        options = [("Eight archived trajectories", str(PREVIEW))]
        if (FULL / "manifest.json").is_file():
            options.append(("Full matched ensemble", str(FULL)))
        self.dataset = w.Dropdown(
            options=options, value=options[-1][1], description="Dataset"
        )
        self.backend = w.Dropdown(options=["CPU", "CUDA"], description="Compute")
        if len(options) > 1:
            try:
                import cupy

                if cupy.cuda.runtime.getDeviceCount():
                    self.backend.value = "CUDA"
            except (ImportError, OSError, RuntimeError):
                pass
        cap = min(60, cpu_capacity())
        self.workers = w.BoundedIntText(
            min=1,
            max=cap,
            value=cap if len(options) > 1 else min(2, cap),
            description="CPU workers",
        )
        self.limit = w.BoundedIntText(min=0, max=512, value=64, description="Cache GiB")
        self.warm = w.Button(
            description="Warm ensemble", icon="bolt", button_style="primary"
        )
        self.stop = w.Button(description="Stop", icon="stop", disabled=True)
        self.stage = w.SelectionSlider(
            options=[("25%", 25), ("50%", 50), ("75%", 75)],
            value=50,
            continuous_update=False,
            description="Replicated",
        )
        self.radius = w.FloatSlider(
            min=1,
            max=4,
            value=2,
            step=0.25,
            continuous_update=False,
            description="Radius",
        )
        self.plan = w.HTML()
        self.progress = w.HTML()
        self.metrics = w.HTML()
        self.source = w.HTML()
        self.pick = w.Dropdown(
            options=[],
            description="Trajectory",
            layout=w.Layout(width="min(520px, 95%)"),
        )
        self.frame = w.IntSlider(
            min=0, max=0, description="Frame", continuous_update=False
        )
        self.pin = w.Button(description="Pin trajectory", icon="thumb-tack", disabled=True)
        self.pinned_track = None
        self.points = []
        self.task = None
        self.stop_event = threading.Event()
        self.engine = None
        self.scatter = go.FigureWidget()
        self.trace_points = [[], []]
        for index, label in enumerate(("No SMC", "SMC")):
            self.scatter.add_scatter(
                name=label,
                mode="markers",
                marker=dict(
                    color=COLORS[index],
                    size=10,
                    opacity=0.8,
                    line=dict(width=0.7, color="#101f2d"),
                ),
                hovertemplate="%{text}<br>Segregation %{x:.1%}<br>Contacts %{y:.2%}<extra></extra>",
            )
            self.scatter.data[-1].on_click(self.click)
        self.scatter.update_layout(
            **THEME,
            height=395,
            title="Trajectory comparison",
            legend=dict(orientation="h", y=-0.24, yanchor="top", x=0.0),
        )
        self.scatter.update_xaxes(
            title="Mean segregated fraction", range=[-0.025, 1.025], tickformat=".0%"
        )
        self.scatter.update_yaxes(
            title="Inter-sister contact fraction", tickformat=".1%", rangemode="tozero"
        )
        self.scene = go.FigureWidget()
        for label, color in [
            ("Sister A", COLORS[0]),
            ("Sister B", COLORS[1]),
            ("Unreplicated", "#b5b9ca"),
        ]:
            self.scene.add_scatter3d(
                name=label,
                mode="lines",
                line=dict(color=color, width=5),
                hovertemplate=label + "<extra></extra>",
            )
        axis = dict(showbackground=False, zeroline=False, gridcolor="#334755")
        self.scene.update_layout(
            **THEME,
            height=395,
            title="Drag to rotate the selected frame",
            legend=dict(orientation="h", y=0.05),
            scene=dict(
                aspectmode="data",
                xaxis=dict(title="Long axis (µm)", **axis),
                yaxis=dict(title="x (µm)", **axis),
                zaxis=dict(title="y (µm)", **axis),
            ),
            uirevision="same-camera",
        )
        self.history = go.FigureWidget()
        for name, color in (("Selected", "#63d9cb"), ("Pinned", "#ffab75")):
            self.history.add_scatter(
                name=name, mode="lines+markers", line=dict(color=color),
                marker=dict(size=5),
                hovertemplate="%{x:.1%} replicated · %{y:.1%} segregated<extra>%{fullData.name}</extra>",
            )
        self.history.add_scatter(
            name="3-D frame", mode="markers",
            marker=dict(symbol="diamond-open", color="white", size=13, line=dict(width=2)),
        )
        self.history.update_layout(
            **THEME, height=315, title="Segregation through replication",
            legend=dict(orientation="h", y=-0.23),
        )
        self.history.update_xaxes(title="Replicated fraction", range=[0, 1], tickformat=".0%")
        self.history.update_yaxes(title="Segregated fraction", range=[-0.03, 1.03], tickformat=".0%")
        self.map = go.FigureWidget(
            data=[
                go.Heatmap(
                    colorscale="RdBu_r",
                    zmid=0,
                    colorbar=dict(title="Δ frequency"),
                    hoverongaps=False,
                )
            ]
        )
        self.map.update_layout(
            **THEME,
            height=345,
            title="Inter-sister contact frequency · SMC minus no SMC",
        )
        self.map.update_xaxes(title="Locus on sister A (10 kb bins)")
        self.map.update_yaxes(title="Locus on sister B")
        boxes = [
            w.VBox([fig], layout=w.Layout(flex="1 1 390px", min_width="0"))
            for fig in (self.scatter, self.scene)
        ]
        self.pair = w.Box(
            boxes, layout=w.Layout(display="flex", flex_flow="row wrap", width="100%")
        )
        for chart in (self.scatter, self.scene):
            chart.layout.autosize = True

        def row(children):
            return w.Box(
                children,
                layout=w.Layout(display="flex", flex_flow="row wrap", gap="6px"),
            )

        settings = w.Accordion(
            children=[
                w.VBox(
                    [
                        row([self.dataset, self.backend, self.workers, self.limit]),
                        self.plan,
                    ]
                )
            ],
            titles=["Ensemble and compute settings"],
            selected_index=None if len(options) == 1 else 0,
        )
        self.view = w.VBox(
            [
                w.HTML(
                    "<h2>Sister chromosome segregation</h2>Choose a replication stage to compare matched tethered simulations with and without non-topological SMC. Click any point to inspect its sister chromosomes."
                ),
                settings,
                row([self.warm, self.stop]),
                self.progress,
                row([self.stage, self.radius]),
                self.metrics,
                self.pair,
                row([self.pick, self.frame, self.pin]),
                self.source,
                self.history,
                self.map,
            ]
        )
        self.warm.on_click(lambda _: self.launch(True))
        self.stop.on_click(lambda _: self.cancel())
        self.pick.observe(self.select, names="value")
        self.frame.observe(self.draw, names="value")
        self.pin.on_click(self.toggle_pin)
        for control in (self.stage, self.radius):
            control.observe(self.query_change, names="value")
        for control in (self.dataset, self.backend, self.limit, self.workers):
            control.observe(self.invalidate, names="value")
        self.describe()
        self.progress.value = "Warm once to enable live stage and radius changes."
        if self.dataset.value == str(PREVIEW):
            self.launch(True)

    def describe(self):
        meta = json.loads((Path(self.dataset.value) / "manifest.json").read_text())
        stages = (25, 50, 75) if meta["preview"] else range(5, 100, 5)
        selected = self.stage.value
        self.stage.options = [(f"{s}%", s) for s in stages]
        self.stage.value = selected if selected in stages else 50
        frames = sum(t["frames"] for t in meta["tracks"])
        size = frames * N * N * 4 / 2**30
        store = "VRAM" if self.backend.value == "CUDA" else "RAM"
        self.plan.value = (
            f"<b>{len(meta['tracks']):,} trajectories · {frames:,} frames</b> · "
            f"{size:,.2f} GiB for reusable squared distances in {store}. "
            "A smaller cache budget selects on-demand calculation."
        )

    def invalidate(self, *_):
        self.engine = None
        self.clear()
        self.describe()
        self.progress.value = "Warm the selected configuration to start."

    def clear(self):
        self.points = []
        self.pinned_track = None
        self.pin.disabled = True
        self.pin.description = "Pin trajectory"
        self.pick.options = []
        self.trace_points = [[], []]
        for trace in self.scatter.data:
            trace.x = []
            trace.y = []
        for trace in self.scene.data:
            trace.x = []
            trace.y = []
            trace.z = []
        self.map.data[0].z = []
        with self.history.batch_update():
            for trace in self.history.data:
                trace.x = trace.y = []
            self.history.layout.shapes = []
        self.source.value = self.metrics.value = ""

    def query_change(self, *_):
        if self.engine is not None:
            self.launch(False)

    def launch(self, warm):
        if self.task is None or self.task.done():
            self.stop_event = threading.Event()
            self.task = asyncio.create_task(self.run(warm))

    def cancel(self):
        self.stop_event.set()

    async def run(self, warm):
        controls = (
            self.dataset,
            self.backend,
            self.limit,
            self.workers,
            self.warm,
            self.stage,
            self.radius,
        )
        for c in controls:
            c.disabled = True
        self.stop.disabled = False
        latest = ("Starting", 0, 1)
        started = time.perf_counter()

        def progress(label, done, total):
            nonlocal latest
            latest = (label, done, total)

        async def work(function, *args):
            proc = psutil.Process()
            previous_cpu, previous_wall = time.process_time(), time.perf_counter()
            task = asyncio.create_task(asyncio.to_thread(function, *args))
            while not task.done():
                label, done, total = latest
                now_cpu, now_wall = time.process_time(), time.perf_counter()
                busy = (now_cpu - previous_cpu) / max(1e-6, now_wall - previous_wall)
                cpu_text = f" · {busy:.1f} busy CPU equivalents" if now_wall - previous_wall > 0.05 else ""
                previous_cpu, previous_wall = now_cpu, now_wall
                self.progress.value = (
                    f"<b>{label}</b> · {done:,} / {total:,} · {now_wall - started:.1f} s "
                    f"· kernel RSS {proc.memory_info().rss / 2**30:.2f} GiB{cpu_text}"
                )
                await asyncio.wait({task}, timeout=0.18)
            return await task

        try:
            if warm:
                self.engine = None
                self.clear()
                warm_cpu = time.process_time()
                warm_start = time.perf_counter()
                self.engine = await work(
                    Ensemble,
                    self.dataset.value,
                    self.backend.value,
                    self.workers.value,
                    self.limit.value,
                    self.stop_event,
                    progress,
                )
                self.warm_seconds = time.perf_counter() - warm_start
                self.warm_busy = (time.process_time() - warm_cpu) / self.warm_seconds
            result = await work(
                self.engine.compare,
                self.stage.value / 100,
                self.radius.value,
                self.stop_event,
                progress,
            )
            self.present(result)
            self.progress.value = (
                f"<b>{escape(self.engine.device)}</b> · warm-up {self.warm_seconds:.2f} s · {self.warm_busy:.1f} average busy CPUs · {self.engine.workers} configured workers · "
                + (
                    "resident distance cache"
                    if self.engine.cached
                    else "computing distances in bounded batches"
                )
            )
        except InterruptedError:
            self.clear()
            self.progress.value = "Stopped. Warm or change the stage to run again."
        except Exception as exc:
            self.clear()
            self.progress.value = "<b>Experiment error</b> · " + escape(str(exc))
        finally:
            for c in controls:
                c.disabled = False
            self.stop.disabled = True

    def present(self, result):
        previous = (
            self.points[self.pick.value]["track"]
            if self.points and self.pick.value is not None
            else None
        )
        self.points = result["points"]
        self.trace_points = [[], []]
        for idx, p in enumerate(self.points):
            self.trace_points[int(p["condition"] == "SMC")].append(idx)
        with self.scatter.batch_update():
            for trace, ids in zip(self.scatter.data, self.trace_points):
                trace.x = [self.points[i]["score"] for i in ids]
                trace.y = [self.points[i]["contacts"] for i in ids]
                trace.text = [self.name(i) for i in ids]
        # Send a small raster to the frontend. Average supported locus-pair differences per 4×4 tile.
        tiled = result["difference"].reshape(101, 4, 101, 4)
        support = np.isfinite(tiled).sum(axis=(1, 3))
        coarse = np.full(support.shape, np.nan)
        np.divide(np.nansum(tiled, axis=(1, 3)), support, out=coarse, where=support > 0)
        with self.map.batch_update():
            self.map.data[0].z = coarse.T
            self.map.data[0].x = self.map.data[0].y = np.arange(101) * 4 + 1.5
        e = self.engine
        store = "VRAM" if e.cp else "RAM"
        self.metrics.value = (
            f"<b>{len(self.points):,} trajectories</b> · {result['frames']:,} frames · "
            f"{result['pairs']:,} eligible locus pairs · <b>{result['seconds'] * 1000:,.0f} ms</b> analysis · "
            f"{result['pairs'] / result['seconds'] / 1e6:,.1f} M eligible pairs/s · "
            f"{result['busy']:.1f} busy CPU equivalents<br>"
            f"{e.coordinate_bytes / 2**20:,.1f} MiB coordinates in RAM · {e.pair_bytes / 2**30:,.3f} GiB distance cache in {store} · "
            f"stage {result['stage']:.0%} ± 4 percentage points · radius {result['radius']:.2f} monomers"
        )
        if e.cp:
            free, total = e.cp.cuda.runtime.memGetInfo()
            self.metrics.value += f" · device memory in use {(total - free) / 2**30:.1f} / {total / 2**30:.1f} GiB"
        self.pick.options = [(self.name(i), i) for i in range(len(self.points))]
        self.pick.value = next(
            (i for i, p in enumerate(self.points) if p["track"] == previous), 0
        )
        self.select()

    def name(self, i):
        point = self.points[i]
        return f"{point['condition']} · {Path(self.engine.tracks[point['track']]['member']).stem}"

    def click(self, trace, points, selector):
        if points.point_inds:
            ids = self.trace_points[int(trace.name == "SMC")]
            if points.point_inds[0] < len(ids):
                self.pick.value = ids[points.point_inds[0]]

    def select(self, *_):
        if self.pick.value is None or not self.points:
            return
        point = self.points[self.pick.value]
        self.pin.disabled = False
        with self.scatter.batch_update():
            for trace, ids in zip(self.scatter.data, self.trace_points):
                trace.marker.size = [15 if i == self.pick.value else 10 for i in ids]
                trace.marker.line.color = ["white" if i == self.pick.value else "#101f2d" for i in ids]
        self.frame.max = len(point["frames"]) - 1
        self.frame.value = point["nearest"]
        self.draw_histories(point["track"])
        self.draw()

    def toggle_pin(self, _):
        if not self.points or self.pick.value is None:
            return
        track = self.points[self.pick.value]["track"]
        self.pinned_track = track if self.pinned_track is None else None
        self.pin.description = "Unpin trajectory" if self.pinned_track is not None else "Pin trajectory"
        self.draw_histories(track)

    def draw_histories(self, selected):
        e = self.engine
        stage = self.stage.value / 100
        with self.history.batch_update():
            self.history.layout.shapes = [dict(
                type="rect", x0=stage - WINDOW, x1=stage + WINDOW,
                y0=0, y1=1, yref="paper", layer="below", line_width=0,
                fillcolor="rgba(190,210,230,0.13)",
            )]
            for trace, track, label in zip(
                self.history.data[:2], (selected, self.pinned_track), ("Selected", "Pinned")
            ):
                if track is None or (label == "Pinned" and track == selected):
                    trace.x = trace.y = []
                    trace.showlegend = False
                    continue
                ids = np.flatnonzero(e.track == track)
                ids = ids[np.argsort(e.block[ids])]
                entry = e.tracks[track]
                trace.name = f"{label} · {entry['condition']} · {Path(entry['member']).stem}"
                trace.mode = "markers" if e.meta["preview"] else "lines+markers"
                trace.showlegend = True
                trace.x, trace.y = e.fraction[ids], e.score[ids]

    def draw(self, *_):
        if self.pick.value is None or not self.points:
            return
        p = self.points[self.pick.value]
        i = p["frames"][min(self.frame.value, len(p["frames"]) - 1)]
        e = self.engine
        lo, hi = e.forks[i]
        with self.history.batch_update():
            self.history.data[2].x = [e.fraction[i]]
            self.history.data[2].y = [e.score[i]]
        replicated = np.r_[hi:N, 0 : lo + 1]
        xyz = e.pos[i] * MICRONS
        with self.scene.batch_update():
            for t, idx in zip(
                self.scene.data, (replicated, replicated + N, np.arange(lo, hi + 1))
            ):
                t.x, t.y, t.z = xyz[idx, 2], xyz[idx, 0], xyz[idx, 1]
        self.source.value = (
            f"<b>{escape(self.name(self.pick.value))}</b> · archive group {e.group[i]} · block {e.block[i]} · "
            f"{e.fraction[i]:.1%} replicated · frame segregation {e.score[i]:.1%}"
        )
