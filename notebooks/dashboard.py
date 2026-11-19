"""Interactive Open OnDemand presentation inside an existing Python kernel."""
import asyncio
from collections import deque
from html import escape
from io import BytesIO
import json
import os
import socket

import ipywidgets as w
import matplotlib.pyplot as plt
import numpy as np

from demo import ROOT
from demo_storage import slurm_config
from notebooks.batch_scaling import batch_experiment
from segregation.jobs import settings
from experiment import configurations, cpu_capacity, experiment, inputs
from figures import heatmaps, predict_slice, scaling_figure
from panels import STYLE, cards, comparison, panel, ram, receipt, study, telemetry


class Dashboard:
    def __init__(self):
        self.capacity = cpu_capacity()
        self.maps = inputs()
        self.task = None
        self.history = deque(maxlen=75)
        self.last_pid = None
        self.saved_plane = None
        self.header = w.HTML(
            f'<style>{STYLE}</style><header class="adaHero"><h2>Chromosome contact lab</h2>'
            f'<div class="adaHost"><b>Jupyter kernel</b>{escape(socket.gethostname())} · '
            f'{self.capacity} session CPUs · PID {os.getpid()}</div></header>').add_class("adaHeader")
        self.resolution = w.Dropdown(options=[(f"{n:,} loci · {m['spacing']} kb",n) for n,m in self.maps.items()],
                                     value=404, description="Resolution")
        self.bait = w.IntSlider(description="Bait bin", min=1, max=404, value=200,
                                continuous_update=False, layout=w.Layout(width="min(560px, 95%)"))
        self.map_image = w.Image(format="png", layout=w.Layout(width="100%", max_width="1050px"))
        self.science = w.HTML()
        self.plane_source = w.HTML()
        self.resolution.observe(self.change_map, names="value")
        self.bait.observe(self.draw_map, names="value")
        self.change_map()
        advance = w.Button(description="Measure CPU scaling", icon="tachometer", button_style="primary",
                           layout=w.Layout(width="180px"))
        advance.on_click(lambda _: setattr(self.tabs, "selected_index", 1))
        self.recalculate = w.Button(description="Recalculate plane", icon="refresh")
        self.recalculate.on_click(self.recalculate_plane)
        map_page = w.VBox([w.HTML(study("contacts", "Where do three chromosome regions meet?",
            "The authors’ condensin example predicts the bait-200 plane from pair contacts at 10 kb. "
            "Change the bait to explore another chromosome position.")),
            w.HBox([self.resolution, self.bait], layout=w.Layout(flex_flow="row wrap")).add_class("adaControls"),
            self.plane_source, self.map_image, self.science,
            w.HBox([self.recalculate, advance], layout=w.Layout(flex_flow="row wrap")).add_class("adaControls")])
        self.level = w.Dropdown(description="Workload", options=[("Resolution ladder · 101 → 202 → 404", (101,202,404))]
                                + [(f"{n:,} loci · {m['spacing']} kb", (n,)) for n,m in self.maps.items()],
                                layout=w.Layout(width="340px"))
        self.ceiling = w.Dropdown(description="CPU ceiling", options=configurations(self.capacity),
                                  value=max(c for c in configurations(self.capacity) if c <= 8))
        self.run_mode = w.Dropdown(description="Benchmark", options=[("Selected threads only", (self.ceiling.value,))],
                                   layout=w.Layout(width="320px"))
        self.location = w.Dropdown(description="Run on", options=[("Notebook session", "local"), ("Slurm CPU allocation", "slurm")],
                                   value="slurm" if slurm_config().is_file() else "local")
        self.duration = w.Dropdown(description="Time / config", options=[("4 seconds",4),("10 seconds",10),("20 seconds",20)],
                                   value=10)
        self.start = w.Button(description="Run experiment", icon="play", button_style="success",
                              layout=w.Layout(width="165px"))
        self.stop = w.Button(description="Stop", icon="stop", disabled=True)
        self.start.on_click(self.launch)
        self.stop.on_click(self.cancel)
        self.plan = w.HTML()
        for widget in (self.level, self.run_mode, self.duration):
            widget.observe(self.describe_plan, names="value")
        self.ceiling.observe(self.choose_threads, names="value")
        self.location.observe(self.choose_location, names="value")
        self.choose_location()
        self.live = w.HTML(panel('<h3>Worker monitor</h3><p>Start the experiment to sample the Julia process and its threads.</p>'))
        self.live_measurements = w.HTML()
        self.status = w.Label()
        controls = w.Box([self.location, self.level, self.ceiling, self.run_mode, self.duration], layout=w.Layout(display="flex",flex_flow="row wrap"))
        compute_page = w.VBox([w.HTML(study("contacts", "Measure triplet prediction",
            "Julia fills the complete probability tensor from the selected pair map. Thread scaling and fine-resolution "
            "workloads are ADA benchmarks of the published predictor.")),
            controls.add_class("adaControls"), self.plan, w.HBox([self.start,self.stop]).add_class("adaControls"),
            self.status,self.live,self.live_measurements])
        self.saved = w.Dropdown(description="Session", options=[], layout=w.Layout(width="min(600px, 95%)"))
        self.saved.observe(self.load_session, names="value")
        self.table = w.HTML(comparison({}))
        self.scaling = w.Image(format="png",layout=w.Layout(width="100%",max_width="1050px",display="none"))
        self.replay = w.IntSlider(description="Sample", min=0, max=0, continuous_update=False,disabled=True,
                                  layout=w.Layout(width="min(620px, 95%)"))
        self.replay.observe(self.draw_replay, names="value")
        self.replay_view = w.HTML()
        self.samples = []
        self.inspect = w.Button(description="Explore saved bait plane", icon="search", disabled=True,
                                layout=w.Layout(width="215px"))
        self.inspect.on_click(self.inspect_result)
        self.record_source = w.HTML()
        self.rerun = w.Button(description="Rerun this benchmark", icon="refresh", disabled=True,
                              layout=w.Layout(width="195px"))
        self.rerun.on_click(self.rerun_session)
        self.recorded_config = None
        results_page = w.VBox([w.HTML(study("contacts", "Recorded CPU measurements",
            "Select a Julia session to inspect the timings, worker samples and saved bait plane.")),
            self.saved,self.record_source,self.rerun,self.scaling,self.table,self.inspect,self.replay,self.replay_view])
        self.segregation = None
        self.segregation_page = w.VBox()
        self.tabs = w.Tab(children=[map_page,compute_page,results_page,self.segregation_page],
                          titles=["Bait-200 map","Live experiment","Results & replay","Sister segregation"],
                          selected_index=1 if self.location.value == "slurm" else 0)
        self.tabs.observe(self.open_segregation, names="selected_index")
        self.view = w.VBox([self.header,self.tabs],layout=w.Layout(width="100%")).add_class("adaLab")
        self.refresh_sessions()

    def open_segregation(self, *_):
        if self.tabs.selected_index != 3 or self.segregation is not None:
            return
        try:
            from segregation.view import SegregationView
            self.segregation = SegregationView()
            self.segregation_page.children = [self.segregation.view]
        except ImportError as exc:
            self.segregation_page.children = [w.HTML(panel(
                f'<h3>Segregation viewer dependencies</h3><p>{escape(str(exc))}</p>'
                '<p>Install the segregation extra in this kernel environment with '
                '<code>uv sync --locked --extra lab --extra segregation</code> from the clone, '
                'select that kernel and rerun the notebook.</p>'))]

    def close(self):
        self.cancel()
        if self.segregation is not None:
            self.segregation.cancel()

    def change_map(self, *_):
        n = self.resolution.value
        source = self.maps[n]
        self.contacts = np.loadtxt(source["path"])[::source["stride"], ::source["stride"]]
        self.saved_plane = None
        self.bait.max = n
        self.bait.value = 1 + round(1990/source["spacing"])
        self.draw_map()

    def draw_map(self, *_):
        n, bait = self.resolution.value, self.bait.value
        spacing = self.maps[n]["spacing"]
        if self.saved_plane is not None and self.saved_plane[0] == bait:
            plane = self.saved_plane[1]
            origin = receipt("Saved Julia plane", self.saved_plane[2], saved=True)
        else:
            plane = predict_slice(self.contacts, bait)
            origin = receipt("Calculated in Jupyter", f"NumPy · {socket.gethostname()} · one {n:,} × {n:,} plane from the loaded pair map")
        self.plane_source.value = origin
        if n > 404:
            input_label = "Pair map reconstructed from archived snapshots"
        elif n == 404:
            input_label = "Published pair-contact map"
        else:
            input_label = "Subsample of the published pair-contact map"
        self.science.value = panel(f'<div class="adaMuted">Input · {input_label} · {spacing} kb · contact radius 5</div>'
            +cards([("Bait position", f"{(bait-1)*spacing/1000:.2f} Mb"),("Predictions displayed",f"{n*n:,}")]))
        fig = heatmaps(self.contacts, plane, bait, spacing)
        self.show_figure(self.map_image, fig)

    def recalculate_plane(self, *_):
        self.saved_plane = None
        self.draw_map()

    @staticmethod
    def show_figure(widget, fig):
        buffer = BytesIO()
        try:
            fig.savefig(buffer, format="png", dpi=105)
            widget.value = buffer.getvalue()
        finally:
            plt.close(fig)

    def show_results(self, run):
        self.table.value = comparison(run)
        fig = scaling_figure(run["rows"])
        self.scaling.layout.display = "none" if fig is None else ""
        if fig is not None:
            self.show_figure(self.scaling, fig)

    def show_live_measurements(self, run):
        self.live_measurements.value = comparison(run)

    def choose_location(self, *_):
        remote = self.location.value == "slurm"
        limit = max(1, int(settings().get("benchmark_cpus", 60))) if remote else self.capacity
        steps = configurations(limit)
        self.ceiling.options = steps
        self.ceiling.value = limit if remote else min(2, limit)
        self.level.value = (max(self.maps) if remote else min(self.maps),)
        self.choose_threads()
        self.run_mode.index = 0

    def choose_threads(self, *_):
        steps = configurations(self.ceiling.value)
        options = [("Selected threads only", (self.ceiling.value,))]
        if len(steps) > 1:
            reference = max(n for n in steps[:-1] if n <= 16)
            options.extend([(f"Compare with {reference} thread{'s' if reference != 1 else ''}", (self.ceiling.value, reference)),
                            ("Thread sweep", tuple(steps))])
        previous = self.run_mode.index or 0
        self.run_mode.options = options
        self.run_mode.index = min(previous, len(options) - 1)
        self.describe_plan()

    def describe_plan(self, *_):
        levels = self.level.value
        threads = self.run_mode.value
        count = len(levels)*len(threads)
        stages = " → ".join(f"{n:,}" for n in levels)
        target = "Slurm CPU job" if self.location.value == "slurm" else "Notebook session"
        opt = settings() if self.location.value == "slurm" else {}
        route = (f'{escape(opt.get("partition") or "default partition")} via {escape(opt.get("submit_host") or "local Slurm client")}'
                 if self.location.value == "slurm" else escape(socket.gethostname()))
        self.plan.value = panel(f'<h3>{target} · {stages} loci</h3><div class="adaMuted">Measure the complete triplet cube at '
            + " → ".join(map(str,threads))+' thread(s). Each configuration repeats the prediction for the selected measurement window.</div>'
            +cards([("Largest output",ram(8*max(levels)**3)),("Configurations",str(count)),
                    ("Minimum timed compute",f"{count*self.duration.value} s")])
            +f'<div class="adaLabel">{route} · fresh Julia workers each run · prepared pair map reused. '
             'Pass times exclude startup, warm-up and numerical checks.</div>')

    def launch(self, *_):
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self.run())

    def cancel(self, *_):
        if self.task is not None and not self.task.done():
            self.task.cancel()

    def update(self, event):
        if self.last_pid != event["pid"]:
            self.history.clear()
            self.last_pid = event["pid"]
        if event["sample"]:
            self.history.append((event["elapsed"],event["sample"]))
        target = f'Job {event["job"]} · {event["node"]} · {event["capacity"]} allocated CPUs · ' if "job" in event else ""
        self.status.value = f'{target}{event["elapsed"]:.1f} s elapsed · {event["compute_s"]:.1f} s measured in this configuration'
        self.live.value = telemetry(event,self.history,event.get("capacity", self.capacity))

    async def run(self):
        controls = (self.location,self.level,self.ceiling,self.run_mode,self.duration,self.start)
        for widget in controls:
            widget.disabled = True
        self.stop.disabled = False
        self.rerun.disabled = True
        self.live.value = panel('<h3>Starting experiment</h3>')
        try:
            args = (self.level.value,self.ceiling.value,self.duration.value,self.update,self.show_live_measurements)
            if self.location.value == "slurm":
                self.status.value = "Submitting CPU scaling job"
                dest = await batch_experiment(*args, status=lambda s: setattr(self.status, "value", s),
                                              thread_counts=self.run_mode.value)
            else:
                dest = await experiment(*args, thread_counts=self.run_mode.value)
            self.status.value = f"Saved to results/{dest.name}"
            self.live.value = panel('<h3>Experiment complete</h3><p>Open Results &amp; replay to compare throughput and inspect recorded worker activity.</p>')
            self.refresh_sessions(dest)
            self.tabs.selected_index = 2
        except asyncio.CancelledError:
            self.status.value = "Stopped. Finished configurations and telemetry are saved."
            self.live.value = panel('<h3>Worker stopped</h3>')
            self.refresh_sessions()
        except Exception as exc:
            self.status.value = "Experiment failed"
            self.live.value = panel(f'<h3>Worker error</h3><pre>{escape(str(exc))}</pre>')
            self.refresh_sessions()
        finally:
            self.live_measurements.value = ""
            self.stop.disabled = True
            self.rerun.disabled = self.recorded_config is None
            for widget in controls:
                widget.disabled = False

    def refresh_sessions(self, selected=None):
        options = []
        for p in (ROOT/"results").glob("session-*/session.json"):
            meta = json.loads(p.read_text())
            options.append((f'{meta["started"]} · {meta["node"]} · {meta["status"]} · {p.parent.name}',p.parent))
        self.saved.options = sorted(options, reverse=True)
        if options:
            self.saved.value = selected or self.saved.options[0][1]
            self.load_session()

    def replay_config(self, run, remote, allocated=None):
        plan = run.get("plan")
        if plan and plan["threads"]:
            levels, ceiling, threads = tuple(plan["levels"]), plan["ceiling"], tuple(plan["threads"])
        elif run["status"] == "complete" and run["rows"]:
            levels = tuple(dict.fromkeys(row["n"] for row in run["rows"]))
            groups = {tuple(r["threads"] for r in run["rows"] if r["n"] == n) for n in levels}
            if len(groups) != 1:
                return None, "Select the per-resolution thread schedules separately in Live experiment."
            threads, = groups
            ceiling = allocated or max(threads)
        else:
            return None, "This session has no complete run plan."
        for n in levels:
            if n not in self.maps or run["maps"].get(str(n), {}).get("sha256") != self.maps[n]["sha"]:
                return None, f"Restore the matching {n:,}-locus input to rerun this measurement."
        limit = int(settings().get("benchmark_cpus", 60)) if remote else self.capacity
        if ceiling > limit:
            return None, f"The recorded run needs {ceiling} CPUs on its original execution route."
        return (remote, levels, ceiling, threads, run["window_s"]), ""

    def rerun_session(self, *_):
        if not self.recorded_config or (self.task and not self.task.done()):
            return
        remote, levels, ceiling, threads, seconds = self.recorded_config
        self.location.value = "slurm" if remote else "local"
        # Old sessions may have a valid ceiling or timing window outside today's presets.
        for control, value, label in ((self.level, levels, "Recorded loci · " + ", ".join(map(str, levels))),
                                      (self.ceiling, ceiling, str(ceiling)),
                                      (self.run_mode, threads, "Recorded threads · " + " → ".join(map(str, threads))),
                                      (self.duration, seconds, f"{seconds:g} seconds")):
            options = [(str(v), v) if not isinstance(v, tuple) else v for v in control.options]
            if value not in [v for _, v in options]:
                control.options = [*options, (label, value)]
            control.value = value
        self.tabs.selected_index = 1
        self.launch()

    def load_session(self, *_):
        path = self.saved.value
        if not path:
            return
        run = json.loads((path/"session.json").read_text())
        batch = path / "batch.json"
        meta = json.loads(batch.read_text()) if batch.is_file() else {}
        job = run.get("job") or meta.get("job")
        remote = run.get("execution") == "slurm" or bool(meta)
        where = ("Slurm CPU" if remote else "Jupyter session") + (f" · job {job}" if job else "")
        self.recorded_config, reason = self.replay_config(run, remote, meta.get("cpus"))
        self.recorded_worker = f'{where} · {run["node"]} · {path.name}'
        self.rerun.disabled = self.recorded_config is None or bool(self.task and not self.task.done())
        self.record_source.value = receipt("Recorded session", f'{where} · {run["node"]} · {run["started"]}. '
            + (reason or "Rerun repeats these settings using the current session or Slurm configuration."), saved=True)
        self.show_results(run)
        data = path/"telemetry.jsonl"
        self.samples = []
        if data.exists():
            for line in data.read_text().splitlines():
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue  # concurrent writer's last line
                if sample["sample"]:
                    self.samples.append(sample)
        self.replay_capacity = run["cpus"]
        self.replay.max = max(0,len(self.samples)-1)
        self.replay.value = max(
            (i for i, event in enumerate(self.samples) if event["phase"] == "Predicting triplets"),
            key=lambda i: (self.samples[i]["n"], self.samples[i]["threads"], i),
            default=self.replay.max)
        self.replay.disabled = not self.samples
        self.inspect.disabled = not run["rows"] or run["rows"][-1]["n"] not in self.maps
        self.draw_replay()

    def draw_replay(self, *_):
        if not self.samples:
            self.replay_view.value = ""
            return
        index = self.replay.value
        event = self.samples[index]
        history = [(s["elapsed"],s["sample"]) for s in self.samples[max(0,index-74):index+1] if s["pid"]==event["pid"]]
        self.replay_view.value = panel(f'<b>Recorded activity · {event["elapsed"]:.1f} s into session</b>') + telemetry(
            event,history,self.replay_capacity)

    def inspect_result(self, *_):
        path = self.saved.value
        run = json.loads((path/"session.json").read_text())
        n = run["rows"][-1]["n"]
        self.resolution.value = n
        bait = run["maps"][str(n)]["bait"]
        saved = np.fromfile(path/f"bait-{n}.f64", dtype=np.float64).reshape((n,n),order="F")
        np.testing.assert_array_equal(saved, predict_slice(self.contacts,bait))
        self.saved_plane = (bait, saved, f'{self.recorded_worker} · loaded from bait-{n}.f64 and checked in Jupyter')
        self.bait.value = bait
        self.draw_map()
        self.tabs.selected_index = 0
