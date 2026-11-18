# Chromosome contacts on ADA

Open the [demo notebook](notebooks/demo.ipynb) in your running [Open OnDemand JupyterLab session](https://ondemand.labs.vu.nl) and choose **Run All Cells**. The experiment uses the CPUs and memory of that session. Julia and the scientific Python kernel are assumed to be available.

![Condensin pair-contact frequencies and predicted three-point contacts at bait 200](examples/contacts.png)

## Present the experiment

1. **Chromosome map.** Start at bait 200 and move it to 100. The left panel shows measured pair-contact frequencies from condensin simulations. The right panel predicts where two other chromosome regions meet the bait. Switch between 40, 20 and 10 kb spacing to see how much detail finer sampling reveals.
2. **Live experiment.** Choose **Measure this node**. Leave **Resolution ladder** selected and choose the CPU ceiling and measurement window. The plan shows the number of configurations, largest tensor and minimum compute time before you press **Run experiment**. With eight available CPUs and a ten-second window, the ladder takes **at least two minutes** of timed computation across twelve configurations.
3. **Watch a worker.** One Julia process runs per configuration. Read its PID, requested compute threads, busy CPU equivalents, share of session CPU capacity, resident memory and completed predictions. The plots update continuously. On Linux the busiest OS threads have individual CPU meters. Compare the one-thread and multi-thread phases. **Stop** ends the active worker and retains completed measurements.
4. **Results & replay.** Compare median time per prediction, throughput, speedup, efficiency, average busy CPUs and peak resident memory **on this node**. Scrub the recorded samples to revisit worker activity, or open the saved bait plane to inspect the scientific output. Results persist in shared project storage under `results/session-*/`.

| Loci | Genomic spacing | Entries predicted per pass | Dense output |
| ---: | ---: | ---: | ---: |
| 101 | 40 kb | 1,030,301 | 7.86 MiB |
| 202 | 20 kb | 8,242,408 | 62.88 MiB |
| 404 | 10 kb | 65,939,264 | 503.08 MiB |

All three levels cover the same circular chromosome. Doubling the genomic resolution multiplies the number of triplet entries by eight. Each timed pass fills a real probability tensor. The benchmark repeats complete predictions for the selected window and reports how many it finished.

## Get the repository

From a terminal in your allocated session:

```sh
git clone https://github.com/Kostusas/ada-beta-demo.git
cd ada-beta-demo
```

Open `notebooks/demo.ipynb` in the JupyterLab file browser. The tested runtime is Julia 1.12.6 on PATH, with Python 3.11–3.14, NumPy, Matplotlib, ipywidgets and psutil. To reproduce the Python environment on another installation, the repository includes a uv lockfile. If your existing kernel needs those packages, register the locked kernel once from the clone:

```sh
uv sync --locked --extra lab
.venv/bin/python -m ipykernel install --user --name ada-contacts --display-name "ADA chromosome contacts"
```

Select **ADA chromosome contacts** and run the notebook. The standard OnDemand JupyterLab app is sufficient. [VU Open OnDemand guide](https://rdm.vu.nl/tools/ada/open-ondemand.html)

## Go beyond the published resolution

The notebook also discovers 1,010- and 2,020-locus contact maps prepared from the original trajectory coordinates. Their dense predictions occupy 7.68 and 61.41 GiB respectively, plus process overhead. Prepare the desired map in your sufficiently sized session and rerun the notebook cell to expose it in the controls. [High-memory walkthrough](docs/high-memory.md)

## Scientific source and verification

The predictor and input map come from Harju, Messelink and Broedersz, *Multicontact statistics distinguish models of chromosome organization* (2025). The 101-locus stage compares all entries to the original serial implementation. At 404 loci every configuration checks the published-resolution bait-200 plane. Full-cube hashes check agreement across thread counts for each standard resolution. The larger inputs use a sampled serial check. [Paper](https://doi.org/10.1103/PhysRevE.111.014403) · [Methods and file formats](docs/method.md) · [Upstream repository](https://github.com/PLSysGitHub/chromosomal_multi_contact_data)

MIT. The original predictor and contact data retain the authors’ notice. [Project license](LICENSE) · [Upstream license](src/upstream/LICENSE)
