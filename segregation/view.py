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

from .engine import FULL, PREVIEW, Coordinates, N, WINDOW
from .index import build_index, index_path
from .jobs import request, settings as slurm_settings
from notebooks.panels import cards, panel, ram
from .timing import PlotTiming

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
        self.backend = w.Dropdown(description="Run on")
        self.compare_button = w.Button(
            description="Compare stage", icon="play", button_style="primary"
        )
        self.configuring = False
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
        for control in (self.stage, self.radius, self.compare_button):
            control.add_class("ada-query-trigger")
        self.timing = PlotTiming()
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
                    zauto=False,
                    colorbar=dict(title="Δ (pp)"),
                    hovertemplate="Sister A bin %{x} · Sister B bin %{y}<br>SMC − no SMC: %{z:+.4g} pp<extra></extra>",
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
        self.map_scale = w.Dropdown(description="Color range", options=[
            ("95th percentile of |Δ|", "p95"), ("Full range", "full")])
        self.map_scale.observe(self.scale_map, names="value")
        self.map_key = w.HTML()
        boxes = [
            w.VBox([fig], layout=w.Layout(flex="1 1 390px", min_width="0"))
            for fig in (self.scatter, self.scene)
        ]
        self.pair = w.Box(
            boxes, layout=w.Layout(display="flex", flex_flow="row wrap", width="100%")
        )
        for chart in (self.scatter, self.scene, self.history, self.map):
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
                        row([self.dataset, self.backend]),
                        self.plan,
                    ]
                )
            ],
            titles=["Ensemble and compute settings"],
            selected_index=None if len(options) == 1 else 0,
        )
        detail_plots = w.Box([
            w.VBox([self.history], layout=w.Layout(flex="1 1 390px", min_width="0")),
            w.VBox([self.map, self.map_scale, self.map_key], layout=w.Layout(flex="1 1 390px", min_width="0")),
        ], layout=w.Layout(display="flex", flex_flow="row wrap", width="100%"))
        self.results_view = w.VBox([
            self.metrics, self.pair, row([self.pick, self.frame, self.pin]),
            self.source, detail_plots,
        ], layout=w.Layout(display="none"))
        self.view = w.VBox(
            [
                w.HTML(panel("<h2>Sister chromosome segregation</h2>"
                             '<div class="adaMuted">Compare cells with and without non-topological SMC at the same replication stage. '
                             "Select a trajectory to inspect its chromosomes in 3-D.</div>")),
                settings,
                row([self.stage, self.radius, self.compare_button, self.stop]),
                self.progress,
                self.timing,
                self.results_view,
            ]
        )
        self.view.add_class("ada-segregation-view")
        self.compare_button.on_click(self.launch)
        self.stop.on_click(lambda _: self.cancel())
        self.pick.observe(self.select, names="value")
        self.frame.observe(self.draw, names="value")
        self.pin.on_click(self.toggle_pin)
        for control in (self.stage, self.radius):
            control.observe(self.query_change, names="value")
        self.dataset.observe(self.configure_dataset, names="value")
        self.backend.observe(self.change_backend, names="value")
        self.configure_dataset()
        if self.backend.value == "local":
            self.launch()

    def configure_dataset(self, *_):
        self.configuring = True
        try:
            folder = Path(self.dataset.value)
            preview = folder == PREVIEW
            meta = json.loads((folder / "manifest.json").read_text())
            frames = sum(t["frames"] for t in meta["tracks"])
            self.dataset_summary = f"{len(meta['tracks']):,} trajectories · {frames:,} archived frames"
            self.clear()
            self.engine = None
            choices = [("Slurm GPU", "CUDA"), ("Slurm CPU", "CPU")]
            if preview:
                choices.insert(0, ("Session preview", "local"))
            self.backend.options = choices
            self.backend.value = choices[0][1]
            stages = (25, 50, 75) if preview else range(5, 100, 5)
            self.stage.options = [(f"{s}%", s) for s in stages]
            self.stage.value = 50
            self.change_backend()
        finally:
            self.configuring = False

    def change_backend(self, *_):
        self.clear()
        if self.backend.value == "local":
            self.plan.value = f"<b>{self.dataset_summary}</b> · in-session preview"
            self.progress.value = "Select a stage or press Compare stage to run the tiny preview."
        else:
            opt = slurm_settings()
            device = f" · {escape(opt.get('gres', 'gpu:1'))}" if self.backend.value == "CUDA" else ""
            route = escape(opt.get("submit_host") or "local Slurm client")
            self.plan.value = (f"<b>{self.dataset_summary}</b> · batch request {opt.get('cpus', 4)} CPUs{device} · "
                               f"{escape(opt.get('partition') or 'default partition')} via {route}")
            self.progress.value = "Select a stage or press Compare stage. Each new query submits one batch job."

    def clear(self):
        self.results_view.layout.display = "none"
        self.timing.revision = ""
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
        self.map_key.value = ""
        with self.history.batch_update():
            for trace in self.history.data:
                trace.x = trace.y = []
            self.history.layout.shapes = []
        self.source.value = self.metrics.value = ""

    def query_change(self, *_):
        if not self.configuring:
            self.launch()

    def launch(self, *_):
        if self.task is not None and not self.task.done():
            return
        self.stop_event = threading.Event()
        for control in self.query_controls:
            control.disabled = True
        self.stop.disabled = False
        self.task = asyncio.create_task(self.run())

    @property
    def query_controls(self):
        return (self.dataset, self.backend, self.compare_button, self.stage, self.radius)

    def cancel(self):
        self.stop_event.set()

    async def run(self):
        started = time.perf_counter()
        latest = "Opening stage index"
        folder = Path(self.dataset.value)
        index = index_path(folder)
        backend, stage, radius = self.backend.value, self.stage.value, self.radius.value

        def status(message):
            nonlocal latest
            latest = message

        def compare():
            if folder == PREVIEW:
                build_index(folder, index, 2)
            elif not (index / "index.json").is_file():
                raise ValueError("Prepare the shared stage index with scripts/prepare-segregation.sbatch. See docs/segregation.md.")
            coords = Coordinates(folder, 1, self.stop_event, None, index=index)
            result, source = request(folder, index, coords.index_id, stage, radius, backend,
                                     self.stop_event, status)
            return coords, result, source

        try:
            work = asyncio.create_task(asyncio.to_thread(compare))
            while not work.done():
                self.progress.value = f"{escape(latest)} · {time.perf_counter() - started:.1f} s"
                await asyncio.wait({work}, timeout=0.2)
            self.engine, result, source = await work
            self.results_view.layout.display = ""
            revision = str(time.monotonic_ns())
            self.present(result, revision)
            self.progress.value = (
                f"<b>{escape(source)}</b> · {time.perf_counter() - started:.2f} s from request "
                "to plot update in kernel"
            )
            self.timing.revision = revision
        except InterruptedError:
            self.clear()
            self.progress.value = "Stopped. Choose a stage or press Compare stage to retry."
        except Exception as exc:
            self.clear()
            self.progress.value = "<b>Comparison failed</b> · " + escape(str(exc))
        finally:
            for control in self.query_controls:
                control.disabled = False
            self.stop.disabled = True

    def present(self, result, revision):
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
            self.scatter.layout.meta = revision
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
            self.map.layout.meta = revision
            self.map.data[0].z = 100 * coarse.T  # probability difference in percentage points
            self.map.data[0].x = self.map.data[0].y = np.arange(101) * 4 + 1.5
        self.scale_map()
        device = f"{escape(result['device'])} on {escape(result['node'])}"
        gpu = f" · {ram(result['cuda_pool'])} CuPy pool reserved at finish" if result["cuda_pool"] else ""
        self.metrics.value = panel(
            f"<h3>{device}</h3>" + cards([
                ("Trajectories", f"{len(self.points):,}"), ("Eligible pairs", f"{result['pairs']:,}"),
                ("Analysis time", f"{result['seconds'] * 1000:,.0f} ms"),
                ("Throughput", f"{result['pairs'] / result['seconds'] / 1e6:,.1f} M pairs/s"),
            ]) + f'<div class="adaMuted">{result["frames"]:,} frames · stage {result["stage"]:.0%} ± 4 percentage points · '
            f'radius {result["radius"]:.2f} monomers · {result["busy"]:.1f} busy CPU equivalents '
            f'of {result["cpus"]} configured · {ram(result["peak_rss"])} peak worker RSS{gpu}</div>')
        self.pick.options = [(self.name(i), i) for i in range(len(self.points))]
        self.pick.value = next(
            (i for i, p in enumerate(self.points) if p["track"] == previous), 0
        )
        self.select()

    def scale_map(self, *_):
        trace = self.map.data[0]
        data = np.asarray(trace.z, dtype=float)
        values = data[np.isfinite(data)]
        magnitude = np.abs(values)
        peak = float(magnitude.max(initial=0))
        limit = peak
        if peak and self.map_scale.value == "p95":
            limit = float(np.percentile(magnitude, 95)) or peak
        with self.map.batch_update():
            trace.zmin, trace.zmax = -(limit or 1), limit or 1
            trace.showscale = bool(peak)
        if not values.size:
            detail = "No tiles have observations in both conditions."
        elif not peak:
            detail = "All supported tiles have a mean difference of zero."
        else:
            saturated = np.count_nonzero(magnitude > limit)
            detail = f"Color limits ±{limit:.3g} pp. Tile means span {values.min():+.3g} to {values.max():+.3g} pp."
            if saturated:
                detail += f" {saturated:,} tiles exceed the color limits. Hover for their values."
        self.map_key.value = (detail + " Differences are in percentage points (pp). White marks zero. Dark gaps have no comparable locus pairs. "
                              "Red indicates higher contact frequency with SMC, blue indicates lower.")

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
