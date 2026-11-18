"""Browser timing for stage comparisons, from the user's gesture to plot display."""

from pathlib import Path

import anywidget
import traitlets


class PlotTiming(anywidget.AnyWidget):
    _esm = Path(__file__).with_name("timing.js")
    revision = traitlets.Unicode().tag(sync=True)
