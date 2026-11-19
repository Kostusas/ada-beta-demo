# Compare sister chromosomes in Open OnDemand

Open **Sister segregation** beside the bait-200 map. The included 2024 preview loads eight real trajectories on CPU. Pick 25%, 50% or 75% replication. Click a point in the scatter plot, rotate the two sisters in 3-D, and move **Frame** to inspect the observations behind that point. Cyan and orange lines in the 3-D scene are the two replicated copies. Gray traces the shared unreplicated segment. The label below the scene includes the original HDF5 group and simulation block.

## Connect the capped OnDemand session to Slurm

JupyterLab uses the ordinary four-CPU, 32 GB OnDemand allocation. A new stage or radius submits one short batch job on your compute partition. The job reads the selected stage window, calculates its map and scatter points, writes the result, and exits. Each press of **Compare stage** runs the calculation live by default. Enable **Reuse saved comparisons** to replay a completed measurement, including after a kernel restart. Viewing a trajectory, pinning it and rotating it use the shared coordinates and the browser. [ADA OnDemand](https://rdm.vu.nl/tools/ada/open-ondemand.html) · [ADA Slurm guide](https://rdm.vu.nl/tools/ada/slurm.html)

Use a clone on shared project storage with standard file locking, visible at the same path from both nodes. The submission host and compute node must see the clone, its Python environment, `inputs/` and `results/`. For the dedicated ADA H200, use [Prepare or restore the full ensemble](#prepare-or-restore-the-full-ensemble). To configure another partition manually, run these commands in the shared clone.

```sh
uv sync --locked --extra lab --extra segregation --extra cuda12
.venv/bin/python -m ipykernel install --user --name ada-contacts --display-name "ADA chromosome contacts"
cp -n slurm.example.toml slurm.toml
```

Edit **`slurm.toml`** and set `partition` to the name of your dedicated GPU partition. The default request is one GPU, four CPU cores, 8 GiB of host memory and ten minutes. On a partition with multiple GPU models, set `gres` to the H200 resource string advertised on ADA. Set `account` or `qos` if your partition requires it. The notebook reads the file for each new submission. The H200 handles the pair calculations in batches. The CPU and memory settings are for the separate Slurm job. [Slurm GPU resources](https://slurm.schedmd.com/gres.html)

### When the Jupyter container has no Slurm commands

The container can use SSH to submit, monitor and cancel jobs through an ADA interactive host. Run this in the Jupyter terminal, using an SSH destination authorized for your account:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=8 inter01.labs.vu.nl 'command -v sbatch && command -v squeue && command -v scancel'
```

Three paths confirm noninteractive access from that terminal. If host-key verification fails on first use, connect interactively and verify the displayed fingerprint with ADA before accepting it. An authenticated host shell also lets you run the tiny terminal check directly from the shared clone. The notebook needs noninteractive SSH access for its own submissions. [ADA login guide](https://rdm.vu.nl/tools/ada/login.html)

Add the following top-level setting to `slurm.toml` alongside your partition and GPU request. Restart the notebook kernel after updating the clone. The clone, virtual environment and results must be visible at the same absolute paths on the submission and batch hosts.

```toml
submit_host = "inter01.labs.vu.nl"
```

The preparation command below also uses this SSH route and waits for the indexing job.

### Repair a JupyterLab frontend that reports “Failed to load model class 'AnyModel'”

The kernel can import anywidget while the separate JupyterLab server cannot serve its browser extension. In the Apptainer terminal, from the shared clone, run:

```sh
.venv/bin/python scripts/enable_widget_frontend.py
```

This exposes the packaged browser assets through your Jupyter user data directory. Save your notebook, stop the **OnDemand JupyterLab app**, launch it again, and reload the browser. Select the registered kernel and run all cells. **Eight archived trajectories → Session preview** exercises the charts and linked 3-D viewer entirely within the session.

If your site disables per-user extensions, the app administrator can install the matching anywidget frontend in the **JupyterLab server image**. [Jupyter widget installation with separate environments](https://ipywidgets.readthedocs.io/en/stable/user_install.html)

### Compare four session CPUs with a 60-CPU batch allocation

**Live experiment → Run on → Slurm CPU allocation** uses the same direct or SSH submission route as the segregation tab. Select **CPU ceiling 60** and **Run experiment** to measure the selected resolution immediately at 60 threads. **Compare with 16 threads** runs 60 then 16 in one CPU job. **Thread sweep** measures the full scaling curve. Worker telemetry, the node name and the allocation size stream back to Jupyter. The measurements persist under **Results & replay**.

If the worker reports Julia missing on a batch node, run the following once from the shared clone in the ADA terminal. It downloads the official Linux x86-64 Julia 1.12.6 archive, verifies its published SHA-256 digest, and unpacks it into `runtime/`. A batch worker automatically finds it there. An explicit `julia` setting in `slurm.toml` takes precedence and must point to a working batch-node executable. [Official Julia binary installations](https://julialang.org/downloads/platform/)

```sh
bash scripts/stage-julia.sh
```

The header describes the Jupyter allocation. The `cpus` setting in `slurm.toml` controls segregation requests, including the eight-trajectory GPU smoke test. The Julia benchmark requests the CPU ceiling chosen in the notebook. Its default maximum is 60. The example config exposes optional `benchmark_cpus`, `benchmark_time` and `benchmark_memory` settings.

### Provide a native Slurm client in a site-managed image

The SSH submission route works with the existing image. For native submission, build a new image with ADA-compatible Slurm client utilities and their libraries/plugins. At app launch provide the site's Slurm configuration or configless discovery, network access to the controller, and the site's authentication service. With MUNGE this normally includes access to the host's MUNGE socket as the session user. Keep the daemon's private key on the host. Validate submission, status and cancellation from the running container. [Slurm authentication](https://slurm.schedmd.com/authentication.html) · [Configless Slurm](https://slurm.schedmd.com/configless_slurm.html)

### Try a real submission before downloading the archive

1. In JupyterLab select **ADA chromosome contacts**, restart the kernel, and run all cells in `notebooks/demo.ipynb`. Open **Sister segregation** and expand **Ensemble and compute settings**.
2. Select **Eight archived trajectories**, **Slurm GPU** and **50%** replication at radius **2**. Click **Compare stage**. The notebook shows the job ID and then the worker’s GPU model and hostname. Expect **24 frames and 965,373 eligible pairs**. The included preview also offers **Session preview** for local CPU use.
3. Enable **Reuse saved comparisons**, release the replication slider at **75%**, then return to **50%**. The first change requests another calculation. The second displays **Saved result**. Click a scatter point, rotate its chromosomes and use **Pin trajectory** to compare its history with another point.
4. The dashboard times each interaction in the browser, from the slider or button gesture until both the scatter and contact map show the new result. The kernel also reports its own request-to-update time. The metrics separately report the original worker’s calculation time, throughput, busy CPU equivalents, peak process RAM, and CUDA memory pool reserved at completion. On replay the worker metrics describe the saved calculation.

For a numerical check from the OnDemand terminal, run `.venv/bin/python tests/check_batch_queries.py --slurm-gpu`. This uses the same GPU submission and saved-result path as the notebook, compares the tiny GPU result with an independent CPU run, and checks that the repeat is served from disk. Run the check after the walkthrough above to verify the GPU result you just saw.

Changing the Slurm partition or GPU resource request gives GPU measurements a separate cache, so an A30 result can coexist with an H200 result for the same stage. To rerun the tiny comparison on the currently configured hardware, use `.venv/bin/python tests/check_batch_queries.py --slurm-gpu --fresh`. A successful run replaces that configuration’s saved result. The output includes the original measuring job, device model and node. GPU results from versions that omitted the GPU request from the cache key stay on disk.

**Stop** cancels the active job. A submission or worker error appears in the notebook. Worker errors include the path to the job log under `results/segregation-queries/`. After an unexpected kernel exit, repeat the same query to reconnect to the job or read its result. The shared cache uses a dataset fingerprint, calculation version, backend, stage and radius, plus partition and GPU request for GPU jobs. Concurrent requests for the same calculation share one submission.

## Prepare or restore the full ensemble

From the **Apptainer terminal in the shared clone**, run the following. The last command uses the existing Python environment and submits preparation through SSH. Keep the terminal open until it prints the trajectory and frame counts.

```sh
git pull --ff-only origin main
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked --extra lab --extra segregation --extra cuda12 --extra extract
.venv/bin/python -m demo_storage prepare-ada
```

If the clone has no Slurm settings, the command installs the dedicated ADA preset from `config/ada-h200.toml` as `slurm.toml`. It requests `gpu:nvidia_h200_nvl:1` on `itvo-test` through `inter01.labs.vu.nl`. An existing config takes priority, including settings recovered from the old `results/slurm.toml`. Other sites can copy and edit `slurm.example.toml` before running preparation.

The command recovers available packed tracks, stage index, archive, high-resolution maps and staged Julia from the previous layout. When the full inputs have been deleted it downloads the 10.47 GB source ZIP from Zenodo. Allow room on shared storage for the ZIP, packed tracks and coordinate index. A CPU preparation job verifies the archive MD5, packs 303 no-SMC and 300 SMC trajectories, then indexes their frames by replication fraction. Rerunning preparation resumes an interrupted download or reuses completed steps. To restore the large CPU demonstration as well, append `--with-bait`. [Coordinate archive](https://zenodo.org/records/10877077)

After **Ready** shows **603 trajectories, 89,847 archived frames**, restart the notebook kernel and run all cells. Open **Sister segregation**. It selects **Full matched ensemble**, **Slurm GPU** and **90%**. Click **Compare stage** for a live measurement. Releasing the stage slider at any value from 5% through 95% also runs that window. Selecting a scatter point reads its archived coordinates immediately.

### Clear saved comparisons

```sh
.venv/bin/python -m demo_storage clear-cache
```

This clears completed segregation answers under `results/segregation-queries/`. It preserves run logs, CPU benchmark replays and active requests. Turning off **Reuse saved comparisons** gives a fresh measurement even when a previous answer exists. Deleting `results/` while jobs are idle also removes their logs and all CPU replay sessions. Run `prepare-ada` once before doing this on an older clone to preserve earlier inputs and settings.

| Shared clone path | Contents |
| --- | --- |
| `inputs/` | Source archive, packed trajectories, stage index and prepared bait maps |
| `runtime/` and `slurm.toml` | Staged Julia and the site's Slurm settings |
| `results/` | Disposable stage answers, run logs and CPU replay sessions |

The CUDA extra installs CuPy on Linux x86-64 and needs a compatible driver on the batch node. If the Jupyter server uses a separate environment, make the locked environment’s `.venv/share/jupyter` visible via the server’s `JUPYTER_PATH`, or install anywidget in that server environment. [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) · [Plotly widget documentation](https://plotly.com/python/figurewidget/)

## What is measured

The H200 query streams up to 64 frames at a time from the memory-mapped stage index. The worker reduces the inter-sister distances into contact counts before starting the next batch. One full batch of 32-bit distances is about 40 MiB (64 × 404 × 404 × 4 bytes), so the calculation can process hundreds of millions of pairs using a modest working set. **Peak worker RSS** measures the worker process's host memory. **CuPy pool reserved at finish** counts bytes retained by the CuPy allocator at completion and excludes some GPU allocations, including the CUDA context. The browser timer includes job launch and rendering, while **Analysis time** covers the worker's stage comparison.

- **Stage.** With 404 genomic loci and zero-based fork endpoints `lo, hi`, the archive’s replication-stage coordinate is `1 − (hi − lo)/404`. Queries include frames within four percentage points of the chosen stage. The full ensemble supports 5% through 95% in five-point increments.
- **Horizontal axis.** Use the authors’ absolute segregated-fraction formula on raw coordinates. For replicated loci strictly outside the fork interval, count sister A loci below the cell midpoint and sister B loci above it. Subtract the number of replicated loci, take the absolute value, and divide by that number. The plotted point is the mean over that trajectory’s included frames.
- **Vertical axis.** Our additional observable is the fraction of eligible inter-sister locus pairs closer than the selected 3-D Euclidean radius (simulation monomer lengths). Both loci must have replicated. Each eligible locus on A is compared with every eligible locus on B, including its homolog. Each trajectory contributes one point. Its initial 3-D frame is the one nearest the requested stage.
- **Difference map.** At each locus pair, divide contact counts by the number of eligible frames separately for each condition. Subtract the no-SMC frequency from the SMC frequency. Display tiles average the supported differences over 4 × 4 locus pairs in percentage points. Dark gaps lack comparable observations. White marks zero. The default symmetric color range uses the 95th percentile of absolute tile means for the current query. Colors saturate beyond that range and hover preserves the actual value. When that percentile is zero, the full range is used. **Full range** includes the extremes. Either setting reads the saved comparison. The map pools frames and can weight a trajectory with a longer dwell time more heavily. The scatter averages within each trajectory.
- **Geometry.** The viewer uses the archive’s original locus ordering and long axis, with the authors’ conversion of 0.129 µm per simulation length. Distances and stage comparisons run on the compute node. The browser receives aggregate plots and the selected conformation. These are archived polymer trajectories analyzed live.

[Harju, van Teeseling & Broedersz, Nature Communications (2024)](https://doi.org/10.1038/s41467-024-49039-w) · [Coordinate archive](https://doi.org/10.5281/zenodo.10877077) · [Authors’ analysis at the inspected revision](https://github.com/PLSysGitHub/loop-extrusion_with_replication_analysis/tree/bf35fbb9ae2304af15081bf97afbacb2272f90b4)
