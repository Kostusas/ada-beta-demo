# Finer resolution on ADA

Stage the 33 condensin trajectory blocks on shared project storage. They contain 3,060 snapshots with 4,040 beads each, about 117 MiB on disk. Run the following from the demo root on an internet-connected host with access to that storage. The partial clone selects only the interacting Brandão coordinates at the source revision recorded in this project.

```sh
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/PLSysGitHub/chromosomal_multi_contact_data trajectories
git -C trajectories fetch --depth 1 origin 3fb7dea343f8426140f38442c47eba000e80b9f2
git -C trajectories restore --source=3fb7dea343f8426140f38442c47eba000e80b9f2 --worktree -- Extract_triplets/Example_Brandao
uv sync --locked --extra extract --extra lab --extra segregation --extra cuda12
```

From the Slurm submission host, change to the same shared clone. This prepares the 2,020-locus input on the ADA demo partition. Use `--stride 4` for the smaller 1,010-locus input. The extractor reads one simulation frame at a time.

```sh
sbatch --partition=itvo-test --cpus-per-task=1 --mem=8G --time=01:00:00 \
  --job-name=bait-map --wrap='.venv/bin/python scripts/extract.py trajectories/Extract_triplets/Example_Brandao --stride 2'
```

The extractor checks each source file against the pinned SHA-256 list. For stride 4 it selects beads 1, 5, 9, … in every frame (1,010 loci). Stride 2 keeps beads 1, 3, 5, … (2,020 loci). It counts strict Euclidean distances below five monomer lengths and normalizes by the frame count. The diagonal is one. Working storage is quadratic. The extractor checks shared physical loci against the published stride-10 map and records the largest difference in frame counts. Its double-precision distance comparison allows one observation per pair to differ near the threshold from the upstream single-precision input calculation.

After the extraction job completes, keep its map under `results/`. Rerun the dashboard cell, open **Live experiment**, select **Slurm CPU allocation**, the prepared resolution and **CPU ceiling 60**, then press **Run experiment**. The 2,020-locus tensor occupies 61.41 GiB of batch-node RAM. The dashboard requests additional runtime memory automatically. An explicit `benchmark_memory` setting overrides that estimate, so use a value such as `"80G"` for this size if you set it manually. The worker records utilization, validates a grid of triplets against the original serial code and saves a bait plane.

Bait 200 at 10 kb has offset 1,990 kb and maps to bin 996 at 2 kb. At 4 kb the closest two bins tie and the driver uses bin 499 (1,992 kb). The predictor applies the original one-bin neighbour exclusion at each resolution, so the excluded physical distance changes. Each session records the input checksum, bin spacing and bait in its manifest. Keep the extracted map alongside the results on shared storage.

The dashboard reports measured process RSS alongside the theoretical tensor size. Finer inputs get one worker at the selected thread count. The warm-up fills the tensor once. Each pass in the measurement window completes before the benchmark stops, even when one pass lasts longer than the selected window.
