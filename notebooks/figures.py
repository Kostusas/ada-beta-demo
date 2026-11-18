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
    for ax, data, title, cap, cmap in zip(
        axes, (p, plane), ("Pairwise contacts", f"Three-point contacts · bait {bait}"),
        (0.25, 0.005), ("viridis", "magma"),
    ):
        im = ax.imshow(data, origin="lower", extent=bounds, vmin=0, vmax=cap,
                       cmap=cmap, interpolation="nearest")
        ax.set(title=title, xlabel="Chromosome position (Mb)", ylabel="Chromosome position (Mb)")
        fig.colorbar(im, ax=ax, shrink=0.82, label="Contact probability", extend="max")
    return fig
