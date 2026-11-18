# Compare sister chromosomes in Open OnDemand

Open **Sister segregation** beside the bait-200 map. The included 2024 preview loads eight real trajectories on CPU. Pick 25%, 50% or 75% replication. Click a point in the scatter plot, rotate the two sisters in 3-D, and move **Frame** to inspect the observations behind that point. Cyan and orange lines in the 3-D scene are the two replicated copies. Gray traces the shared unreplicated segment. The label below the scene includes the original HDF5 group and simulation block.

## Stage the full ensemble once on ADA

Run these commands from the clone in the allocated OnDemand session. The published archive is **10.47 GB**. Allow space for both the ZIP and the packed coordinate files on shared project storage. The tool verifies the archive MD5 before conversion and selects the two parameter-matched directories. The ZIP contains 303 no-SMC trajectories and 300 non-topological-SMC trajectories in this comparison.

```sh
uv sync --locked --extra lab --extra segregation --extra cuda12
.venv/bin/python -m ipykernel install --user --name ada-contacts --display-name "ADA chromosome contacts"
mkdir -p results
curl --fail --location --retry 3 --continue-at - --output results/Simulation_data.zip https://zenodo.org/records/10877077/files/Simulation_data.zip
.venv/bin/python -m segregation.prepare results/Simulation_data.zip
.venv/bin/python tests/check_segregation.py --cuda
```

The optional `cuda12` extra supplies CuPy and CUDA 12 userspace packages for Linux x86-64. It needs a compatible NVIDIA driver. A kernel already providing compatible CuPy can use it directly. The CPU environment uses the `lab` and `segregation` extras alone. The CUDA check compares cached and streaming device results with the CPU on a few real frames before you warm the full ensemble. [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)

Packing uses up to 60 independent CPU processes, capped by the visible session capacity and trajectory count. Each process reads its own ZIP member and prepares a compact trajectory file. Pass `--workers 8` for a smaller allocation. A successful conversion publishes a complete `results/segregation-2024/` directory. Keep it for subsequent sessions.

Select **ADA chromosome contacts** in JupyterLab, restart the kernel after updating the clone, and run the notebook. The existing JupyterLab frontend needs ipywidgets and the anywidget frontend extension to render Plotly FigureWidgets. If the OnDemand server uses a separate environment, expose the locked environment’s `.venv/share/jupyter` through the app’s `JUPYTER_PATH` at launch, or install anywidget in the server environment. [Plotly widget documentation](https://plotly.com/python/figurewidget/)

## Warm before the meeting

Select **Full matched ensemble**, **CUDA**, up to **60 CPU workers**, and a **64 GiB cache budget**. The tab reports the exact frame count and cache size from the prepared manifest. Click **Warm ensemble**. CPU workers decompress tracks, compute per-frame segregation and put coordinates in host RAM. CUDA builds a contiguous distance array ordered by replication stage. It contains one Float32 squared distance for each old–new locus pair per frame, with out-of-scope pairs masked. Each of the eight inspected source trajectories had 149 frames. If all 603 have that length, the reusable distances occupy **54.6 GiB** of VRAM.

Warming happens once per dataset/backend configuration in the live kernel. Selecting another stage or dragging the radius changes the query against those distances. **Cache GiB** limits the distance array. If it exceeds that limit or 75% of available backend memory, the engine computes the requested stage in batches. Set the budget to zero to compare this streaming path with a warm cache. The CPU backend can keep the same distance array in host RAM and reduce it using multiple threads. The displayed RSS and cache size describe different allocations. CUDA device occupancy includes memory in use by other processes on that device.

For a presentation, start halfway through replication. Click an unusually low-segregation trajectory and press **Pin trajectory**. Click a point from the other condition with a similar contact fraction. The history plot now follows both trajectories across replication. Its shaded band is the chosen stage window. The diamond marks the archived frame in the 3-D view. Scrub the frames, then change the radius and watch where the contact difference map changes. The metrics give the number of eligible pairs, pairs per second, comparison wall time and average process CPU equivalents. The latency excludes widget rendering and the initial warm-up.

## What is measured

- **Stage.** With 404 genomic loci and zero-based fork endpoints `lo, hi`, the archive’s replication-stage coordinate is `1 − (hi − lo)/404`. Queries include frames within four percentage points of the chosen stage. The full ensemble supports 5% through 95% in five-point increments.
- **Horizontal axis.** Use the authors’ absolute segregated-fraction formula on raw coordinates. For replicated loci strictly outside the fork interval, count sister A loci below the cell midpoint and sister B loci above it. Subtract the number of replicated loci, take the absolute value, and divide by that number. The plotted point is the mean over that trajectory’s included frames.
- **Vertical axis.** Our additional observable is the fraction of eligible inter-sister locus pairs closer than the selected 3-D Euclidean radius (simulation monomer lengths). Both loci must have replicated. Each eligible locus on A is compared with every eligible locus on B, including its homolog. Each trajectory contributes one point. Its initial 3-D frame is the one nearest the requested stage.
- **Difference map.** At each locus pair, divide contact counts by the number of eligible frames separately for each condition. Subtract the no-SMC frequency from the SMC frequency. Empty support stays blank. Display pixels average supported differences over 4 × 4 locus tiles. The map pools frames and can weight a trajectory with a longer dwell time more heavily. The scatter averages within each trajectory.
- **Geometry.** The viewer uses the archive’s original locus ordering and long axis, with the authors’ conversion of 0.129 µm per simulation length. Distances and stage comparisons run on the compute node. The browser receives aggregate plots and the selected conformation. These are archived polymer trajectories analyzed live.

[Harju, van Teeseling & Broedersz, Nature Communications (2024)](https://doi.org/10.1038/s41467-024-49039-w) · [Coordinate archive](https://doi.org/10.5281/zenodo.10877077) · [Authors’ analysis at the inspected revision](https://github.com/PLSysGitHub/loop-extrusion_with_replication_analysis/tree/bf35fbb9ae2304af15081bf97afbacb2272f90b4)
