# Chromosome contacts on ADA

Explore where three regions of a simulated bacterial chromosome meet, then measure how quickly ADA can predict all **65,939,264 triplet entries** at the published resolution. The calculation uses the condensin simulation data and loop-extruder model of Harju, Messelink and Broedersz. [Paper](https://doi.org/10.1103/PhysRevE.111.014403) · [Authors’ Julia repository](https://github.com/PLSysGitHub/chromosomal_multi_contact_data)

![Pair contact frequencies from the condensin simulations and predicted three-point probabilities at bait bin 200](examples/contacts.png)

## Show it in two minutes

1. **Open the offline explorer.** [Download the standalone HTML](https://raw.githubusercontent.com/Kostusas/ada-beta-demo/main/examples/explorer.html) and open the saved file in a browser, or open `examples/explorer.html` from a clone. It works without Julia, Jupyter or a network connection. Move the slider from bait 200 to bait 100. The pair-contact map stays fixed while the predicted triplet pattern moves with the selected locus.
2. **Explain the cost.** The right panel shows one plane of a cube. Halving the genomic bin spacing doubles the number of loci along each of its three axes and multiplies the dense output by eight.
3. **Show what the cores bought.** Open the notebook and select a completed ADA run to show measured speedup, CPU use and peak memory. A recorded laptop smoke test is included for trying the controls before the first cluster job.

## Try the calculation locally

The benchmark uses **Julia 1.12.6** and **Python 3.11+**, with only their standard libraries. With Juliaup, `juliaup add 1.12.6` installs the tested runtime and `juliaup override set 1.12.6` selects it in the clone. An existing matching Julia installation works directly. [Julia installation](https://julialang.org/downloads/)

```sh
git clone https://github.com/Kostusas/ada-beta-demo.git
cd ada-beta-demo
python3 demo.py smoke
```

The smoke test takes every fourth published bin around the entire circular chromosome. It computes the full 101 × 101 × 101 tensor with the authors’ serial code and with up to two Julia threads, and compares SHA-256 digests of **every Float64**. The retained bins have 40 kb spacing. Run `python3 demo.py preview` for just the 404 × 404 plane at the original 10 kb spacing and bait 200. Both commands write a new directory under `results/`.

## Run the first ADA benchmark

Clone onto shared project storage. In the submitting shell, use `module spider Julia` and `module spider Python` to find available modules, or use your own installations. Check `julia --version` and `python3 --version`. The job inherits this environment. You can also set `JULIA` to an absolute executable path.

```sh
/ada-software/ada-info.sh
# Set PARTITION to the actual partition your account can use.
sbatch --partition="$PARTITION" --cpus-per-task=8 slurm/scale.sbatch
```

Submit from the repository root. This first job requests one node, 8 CPUs, 4 GiB and 30 minutes. It tests original serial and 1, 2, 4 and 8 threads on the **same node**. Omit the CPU override for the 1–32 sweep if your partition permits it. Each configuration gets a fresh Julia process, a warm-up and three measured predictions. The full-resolution run also checks the saved bait-200 plane against the bundled reference.

```sh
squeue -j "$JOB_ID"
sacct -j "$JOB_ID" --format=JobID,State,Submit,Start,Elapsed,ReqMem,MaxRSS,TotalCPU
```

Set `JOB_ID` to the number printed by `sbatch`. The Slurm log lists the result directory. `trials.csv` contains raw timings and `run.json` contains the summaries and resource metadata. Use the Submit and Start timestamps for queue delay. The plotted timings cover the predictor allocation and calculation. [ADA hardware and availability](https://rdm.vu.nl/tools/ada/hardware.html) · [ADA Slurm guide](https://rdm.vu.nl/tools/ada/slurm.html)

## Open the notebook

**uv manages the optional Python packages.** The lockfile covers JupyterLab, the plotting stack and the optional trajectory reader. Julia uses its own runtime. The default Python environment for uv is 3.12. [uv installation](https://docs.astral.sh/uv/getting-started/installation/)

```sh
uv run --locked --extra lab jupyter lab notebooks/demo.ipynb
```

On ADA, prepare the kernel once in the clone on shared storage:

```sh
uv sync --locked --extra lab
.venv/bin/python -m ipykernel install --user --name ada-contacts --display-name "ADA chromosome contacts"
sacctmgr show qos ood format=Name,MaxJobsPU,MaxTRESPU%30,MaxWall
```

From campus or eduVPN, launch the portal’s JupyterLab app, request resources within the current `ood` QoS, open `notebooks/demo.ipynb`, select **ADA chromosome contacts**, and run all cells. Select a completed run to explore any bait. Only a rendered 2-D image reaches the browser. The batch launch and monitoring cells are opt-in. [Open OnDemand portal](https://ondemand.labs.vu.nl) · [VU session instructions](https://rdm.vu.nl/tools/ada/open-ondemand.html)

## What has been measured?

| Apple M4 smoke run, Julia 1.12.6 | Median compute time | Peak process RSS |
| --- | ---: | ---: |
| Original serial | 2.27 ms | 330 MiB |
| Threaded code, 1 thread | 2.02 ms | 329 MiB |
| Threaded code, 2 threads | 1.19 ms | 333 MiB |

All nine tensors had identical digests. Two threads achieved **1.70×** the one-thread speed on this small input. These millisecond timings are a functional demonstration. Use the ADA sweep to assess performance at the published resolution and repeat the same profile on a workstation for a procurement comparison. [Raw laptop trials](examples/laptop/trials.csv) · [Run metadata](examples/laptop/run.json)

The bundled bait-200 result reproduces the loop-extruder prediction at the authors’ condensin example settings. The full paper also compares other models with measured triplets. This prototype covers the CPU predictor and finer-resolution memory experiment. ADA timings, simulation ensembles and CUDA simulations are future work.

| Positions | Genomic spacing | One dense Float64 tensor |
| ---: | ---: | ---: |
| 404 | 10 kb | 0.49 GiB |
| 1,010 | 4 kb | 7.68 GiB |
| 2,020 | 2 kb | 61.41 GiB |

For the high-memory job, derive the finer map from the original coordinates first. The extractor, checksums and batch scripts are included. [High-memory runbook](docs/high-memory.md) · [Scientific provenance and measurement details](docs/method.md)

## License

MIT. The original predictor and contact data retain the authors’ notice. [Project license](LICENSE) · [Upstream license](src/upstream/LICENSE)
