"""Display a 2-D section of the periodic predictor at the requested bait."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_run(folder):
    run = json.loads((folder / "run.json").read_text())
    if run["profile"] == "highmem":
        source = Path(run["map_path"])
    else:
        source = ROOT / "data/condensin_hic_contact_r_5.txt"
    contacts = np.loadtxt(source)
    if run["profile"] == "smoke":
        contacts = contacts[::4, ::4]
    n = run["n"]
    dtype = "<f8" if run["byteorder"] == "little" else ">f8"
    saved = np.fromfile(folder / "bait.f64", dtype=dtype).reshape((n, n), order="F")
    np.testing.assert_array_equal(saved, predict_slice(contacts, run["bait"]))
    return run, contacts, saved


def predict_slice(p, bait):
    i = bait - 1
    row, col = p[i], p[:, i]
    # Output axes are k (rows), j (columns), matching Julia T[:, :, i].
    plane = (row[None, :] * p.T + row[:, None] * p + col[None, :] * row[:, None]
             - 2 * row[None, :] * p.T * col[:, None])
    np.minimum(plane, 1, out=plane)
    bins = np.arange(len(p))
    for delta in (-1, 0, 1):
        other = (bins + delta) % len(p)
        plane[bins, other] = 0
        neighbour = (i + delta) % len(p)
        plane[neighbour, :] = 0
        plane[:, neighbour] = 0
    return plane


def heatmaps(p, plane, bait, spacing):
    step = spacing / 1000
    bounds = (-step/2, (len(p)-0.5)*step) * 2
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1), layout="constrained")
    fig.set_facecolor("#101f2d")
    for ax, data, title, cap, cmap in zip(
        axes, (p, plane), ("Pairwise contacts", f"Three-point contacts · bait {bait}"),
        (0.25, 0.005), ("viridis", "magma"),
    ):
        im = ax.imshow(data, origin="lower", extent=bounds, vmin=0, vmax=cap,
                       cmap=cmap, interpolation="nearest")
        ax.set(title=title, xlabel="Chromosome position (Mb)", ylabel="Chromosome position (Mb)")
        ax.title.set_color("#eff7fa")
        ax.xaxis.label.set_color("#d4e5e9")
        ax.yaxis.label.set_color("#d4e5e9")
        ax.tick_params(colors="#d4e5e9")
        bar = fig.colorbar(im, ax=ax, shrink=0.82, label="Contact probability", extend="max")
        bar.ax.tick_params(colors="#d4e5e9")
        bar.ax.yaxis.label.set_color("#d4e5e9")
    locus = (bait-1)*step
    axes[0].axvline(locus, color="white", alpha=.65, linestyle=":", linewidth=.8)
    axes[0].axhline(locus, color="white", alpha=.65, linestyle=":", linewidth=.8)
    return fig


def scaling_figure(rows):
    """Compare only configurations measured on the same input in this session."""
    series = sorted({r["n"] for r in rows if "speedup" in r})
    if not series or max(r["threads"] for r in rows) < 2:
        return None
    with plt.style.context("dark_background"):
        fig, (speed, use) = plt.subplots(1, 2, figsize=(10, 3.2), layout="constrained")
        cap = max(r["threads"] for r in rows)
        for ax in (speed, use):
            ax.set_facecolor("#101f2d")
            ax.grid(alpha=.13)
            ax.set_xlabel("Julia compute threads")
            ax.set_xticks(sorted({r["threads"] for r in rows}))
            ax.plot([1,cap], [1,cap], "--", color="#78949c", linewidth=1)
        for n, color in zip(series, ("#5ce1c0","#ffa873","#95b9ff")):
            group = [r for r in rows if r["n"] == n]
            threads = [r["threads"] for r in group]
            speed.plot(threads, [r["speedup"] for r in group], "o-", color=color, label=f"{n} loci")
            use.plot(threads, [r["busy_cores"] for r in group], "o-", color=color)
        speed.set(ylabel="Speedup over one thread",title="Prediction speedup")
        use.set(ylabel="Average busy CPU equivalents",title="CPU use during prediction")
        speed.legend(frameon=False)
        fig.set_facecolor("#101f2d")
        return fig
