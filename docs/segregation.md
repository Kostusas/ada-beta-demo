# Compare sister chromosomes in Open OnDemand

Open **Sister segregation** beside the bait-200 map. The included 2024 preview loads eight real trajectories on CPU. Pick 25%, 50% or 75% replication. Click a point in the scatter plot, rotate the two sisters in 3-D, and move **Frame** to inspect the observations behind that point. Cyan and orange lines in the 3-D scene are the two replicated copies. Gray traces the shared unreplicated segment. The label below the scene includes the original HDF5 group and simulation block.

## Connect the capped OnDemand session to Slurm

JupyterLab uses the ordinary four-CPU, 32 GB OnDemand allocation. A new stage or radius submits one short batch job on your compute partition. The job reads the selected stage window, calculates its map and scatter points, writes the result, and exits. Repeating a query opens the saved result, including after a kernel restart. Viewing a trajectory, pinning it and rotating it use the shared coordinates and the browser. [ADA OnDemand](https://rdm.vu.nl/tools/ada/open-ondemand.html) · [ADA Slurm guide](https://rdm.vu.nl/tools/ada/slurm.html)

Use a clone on shared project storage with standard file locking, visible at the same path from both nodes. Both nodes must see the Python environment and the `results` directory. Run once from the clone in your OnDemand terminal:

```sh
uv sync --locked --extra lab --extra segregation --extra cuda12
.venv/bin/python -m ipykernel install --user --name ada-contacts --display-name "ADA chromosome contacts"
mkdir -p results
cp -n slurm.example.toml results/slurm.toml
```

Edit **`results/slurm.toml`** and set `partition` to the name of your dedicated GPU partition. The default request is one GPU, four CPU cores, 8 GiB of host memory and ten minutes. On a partition with multiple GPU models, set `gres` to the H200 resource string advertised on ADA. Set `account` or `qos` if your partition requires it. The notebook reads the file for each new submission. The H200 handles the pair calculations in batches. The CPU and memory settings are for the separate Slurm job. [Slurm GPU resources](https://slurm.schedmd.com/gres.html)

### When the Jupyter container has no Slurm commands

The container can use SSH to submit, monitor and cancel jobs through an ADA interactive host. Run this in the Jupyter terminal, using an SSH destination authorized for your account:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=8 inter01.labs.vu.nl 'command -v sbatch && command -v squeue && command -v scancel'
```

Three paths confirm noninteractive access from that terminal. If host-key verification fails on first use, connect interactively and verify the displayed fingerprint with ADA before accepting it. An authenticated host shell also lets you run the tiny terminal check directly from the shared clone. The notebook needs noninteractive SSH access for its own submissions. [ADA login guide](https://rdm.vu.nl/tools/ada/login.html)

Add the following top-level setting to `results/slurm.toml` alongside your partition and GPU request. Restart the notebook kernel after updating the clone. The clone, virtual environment and results must be visible at the same absolute paths on the submission and batch hosts.

```toml
submit_host = "inter01.labs.vu.nl"
```

For the one-time archive preparation, run the `sbatch` command below directly in the interactive host shell.

### Try a real submission before downloading the archive

1. In JupyterLab select **ADA chromosome contacts**, restart the kernel, and run all cells in `notebooks/demo.ipynb`. Open **Sister segregation** and expand **Ensemble and compute settings**.
2. Select **Eight archived trajectories**, **Slurm GPU** and **50%** replication at radius **2**. Click **Compare stage**. The notebook shows the job ID and then the worker’s GPU model and hostname. Expect **24 frames and 965,373 eligible pairs**. The included preview also offers **Session preview** for local CPU use.
3. Release the replication slider at **75%**, then return to **50%**. The first change requests another calculation. The second displays **Saved result**. Click a scatter point, rotate its chromosomes and use **Pin trajectory** to compare its history with another point.
4. The dashboard times each interaction in the browser, from the slider or button gesture until both the scatter and contact map show the new result. The kernel also reports its own request-to-update time. The metrics separately report the original worker’s calculation time, throughput, busy CPU equivalents, peak process RAM, and CUDA memory pool reserved at completion. On replay the worker metrics describe the saved calculation.

For a numerical check from the OnDemand terminal, run `.venv/bin/python tests/check_batch_queries.py --slurm-gpu`. This uses the same GPU submission and saved-result path as the notebook, compares the tiny GPU result with an independent CPU run, and checks that the repeat is served from disk. Run the check after the walkthrough above to verify the GPU result you just saw.

Changing the Slurm partition or GPU resource request gives GPU measurements a separate cache, so an A30 result can coexist with an H200 result for the same stage. To rerun the tiny comparison on the currently configured hardware, use `.venv/bin/python tests/check_batch_queries.py --slurm-gpu --fresh`. A successful run replaces that configuration’s saved result. The output includes the original measuring job, device model and node. GPU results from versions that omitted the GPU request from the cache key stay on disk.

**Stop** cancels the active job. A submission or worker error appears in the notebook. Worker errors include the path to the job log under `results/segregation-queries/`. After an unexpected kernel exit, repeat the same query to reconnect to the job or read its result. The shared cache uses a dataset fingerprint, calculation version, backend, stage and radius, plus partition and GPU request for GPU jobs. Concurrent requests for the same calculation share one submission.

### Prepare the full dataset once

The source ZIP is 10.47 GB. Keep space on shared storage for the ZIP, packed tracks and stage-sorted coordinate index. From the clone:

```sh
curl --fail --location --retry 3 --continue-at - --output results/Simulation_data.zip https://zenodo.org/records/10877077/files/Simulation_data.zip
sbatch --partition=YOUR_PARTITION scripts/prepare-segregation.sbatch
```

Replace `YOUR_PARTITION` with an accessible CPU or dedicated partition. Add `--account` or `--qos` to this preparation submission if your site requires them. If the previous version already created `results/segregation-2024/manifest.json`, skip the download and submit the preparation job alone. The script reuses the packed data. For a fresh download it verifies the source checksum and extracts 303 no-SMC and 300 SMC trajectories. It then orders coordinates by replication fraction on disk. Preparation uses eight CPUs by default, adjustable with `sbatch --cpus-per-task`. It does not calculate any contact maps.

After the preparation job completes, rerun the notebook cell and select **Full matched ensemble** and **Slurm GPU**. Pick a stage from 5% through 95% or press **Compare stage**. Only that stage window’s coordinates enter the distance calculation. Each completed job releases its GPU.

The CUDA extra installs CuPy on Linux x86-64 and needs a compatible driver on the batch node. If the Jupyter server uses a separate environment, make the locked environment’s `.venv/share/jupyter` visible via the server’s `JUPYTER_PATH`, or install anywidget in that server environment. [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) · [Plotly widget documentation](https://plotly.com/python/figurewidget/)

## What is measured

- **Stage.** With 404 genomic loci and zero-based fork endpoints `lo, hi`, the archive’s replication-stage coordinate is `1 − (hi − lo)/404`. Queries include frames within four percentage points of the chosen stage. The full ensemble supports 5% through 95% in five-point increments.
- **Horizontal axis.** Use the authors’ absolute segregated-fraction formula on raw coordinates. For replicated loci strictly outside the fork interval, count sister A loci below the cell midpoint and sister B loci above it. Subtract the number of replicated loci, take the absolute value, and divide by that number. The plotted point is the mean over that trajectory’s included frames.
- **Vertical axis.** Our additional observable is the fraction of eligible inter-sister locus pairs closer than the selected 3-D Euclidean radius (simulation monomer lengths). Both loci must have replicated. Each eligible locus on A is compared with every eligible locus on B, including its homolog. Each trajectory contributes one point. Its initial 3-D frame is the one nearest the requested stage.
- **Difference map.** At each locus pair, divide contact counts by the number of eligible frames separately for each condition. Subtract the no-SMC frequency from the SMC frequency. Empty support stays blank. Display pixels average supported differences over 4 × 4 locus tiles. The map pools frames and can weight a trajectory with a longer dwell time more heavily. The scatter averages within each trajectory.
- **Geometry.** The viewer uses the archive’s original locus ordering and long axis, with the authors’ conversion of 0.129 µm per simulation length. Distances and stage comparisons run on the compute node. The browser receives aggregate plots and the selected conformation. These are archived polymer trajectories analyzed live.

[Harju, van Teeseling & Broedersz, Nature Communications (2024)](https://doi.org/10.1038/s41467-024-49039-w) · [Coordinate archive](https://doi.org/10.5281/zenodo.10877077) · [Authors’ analysis at the inspected revision](https://github.com/PLSysGitHub/loop-extrusion_with_replication_analysis/tree/bf35fbb9ae2304af15081bf97afbacb2272f90b4)
