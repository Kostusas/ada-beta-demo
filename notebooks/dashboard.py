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
from panels import cards, comparison, panel, ram, telemetry


class Dashboard:
    def __init__(self):
        self.capacity = cpu_capacity()
        self.maps = inputs()
        self.task = None
        self.history = deque(maxlen=75)
        self.last_pid = None
        self.saved_plane = None
        self.header = w.HTML(panel(
            '<h2>Chromosome contact lab</h2><div class="adaMuted">Jupyter interface · '
            f'{escape(socket.gethostname())} · {self.capacity} session CPUs · Python kernel PID {os.getpid()}</div>'))
        self.resolution = w.Dropdown(options=[(f"{n:,} loci · {m['spacing']} kb",n) for n,m in self.maps.items()],
                                     value=404, description="Resolution")
        self.bait = w.IntSlider(description="Bait bin", min=1, max=404, value=200,
                                continuous_update=False, layout=w.Layout(width="min(560px, 95%)"))
        self.map_image = w.Image(format="png", layout=w.Layout(width="100%", max_width="1050px"))
        self.science = w.HTML()
        self.resolution.observe(self.change_map, names="value")
        self.bait.observe(self.draw_map, names="value")
        self.change_map()
        advance = w.Button(description="Measure CPU scaling", icon="tachometer", button_style="primary",
                           layout=w.Layout(width="180px"))
        advance.on_click(lambda _: setattr(self.tabs, "selected_index", 1))
        map_page = w.VBox([self.science, w.HBox([self.resolution, self.bait], layout=w.Layout(flex_flow="row wrap")),
                           self.map_image, advance])
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
        self.status = w.Label()
        controls = w.Box([self.location, self.level, self.ceiling, self.run_mode, self.duration], layout=w.Layout(display="flex",flex_flow="row wrap"))
        compute_page = w.VBox([controls, self.plan, w.HBox([self.start,self.stop]), self.status,self.live])
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
        results_page = w.VBox([self.saved,self.scaling,self.table,self.inspect,self.replay,self.replay_view])
        self.segregation = None
        self.segregation_page = w.VBox()
        self.tabs = w.Tab(children=[map_page,compute_page,results_page,self.segregation_page],
                          titles=["Bait-200 map","Live experiment","Results & replay","Sister segregation"],
                          selected_index=1 if self.location.value == "slurm" else 0)
        self.tabs.observe(self.open_segregation, names="selected_index")
        self.view = w.VBox([self.header,self.tabs],layout=w.Layout(width="100%"))
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
            origin = "Saved Julia prediction"
        else:
            plane = predict_slice(self.contacts, bait)
            origin = "Loop-extruder prediction"
        self.science.value = panel('<h3>Where do three chromosome regions meet?</h3>'
            '<div class="adaMuted">Move the bait along the chromosome. Each pixel on the right predicts a contact between the bait and two other loci.</div>'
            +cards([("Bait position", f"{(bait-1)*spacing/1000:.2f} Mb"),("Entries in this plane",f"{n*n:,}"),
                    ("Entries in full cube",f"{n**3:,}")])+f'<span class="adaGlow">{origin}</span> · radius 5')
        fig = heatmaps(self.contacts, plane, bait, spacing)
        self.show_figure(self.map_image, fig)

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
        self.plan.value = panel(f'<h3>{target} · {stages} loci</h3><div class="adaMuted">Measure the complete triplet cube at '
            + " → ".join(map(str,threads))+' thread(s). Each configuration repeats the prediction for the selected measurement window.</div>'
            +cards([("Largest output",ram(8*max(levels)**3)),("Configurations",str(count)),
                    ("Minimum timed compute",f"{count*self.duration.value} s")])
            +'<div class="adaLabel">Startup, warm-up and numerical checks add time.</div>')

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
        self.live.value = panel('<h3>Starting experiment</h3>')
        try:
            args = (self.level.value,self.ceiling.value,self.duration.value,self.update,self.show_results)
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
            self.stop.disabled = True
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

    def load_session(self, *_):
        path = self.saved.value
        if not path:
            return
        run = json.loads((path/"session.json").read_text())
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
        self.saved_plane = (bait,saved)
        self.bait.value = bait
        self.draw_map()
        self.tabs.selected_index = 0
