# Finer resolution on ADA

Stage the 33 condensin trajectory blocks on shared project storage. They contain 3,060 snapshots with 4,040 beads each, about 117 MiB on disk. Run the following from the demo root on an internet-connected host with access to that storage. The partial clone selects only the interacting Brandão coordinates at the source revision recorded in this project.

```sh
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/PLSysGitHub/chromosomal_multi_contact_data trajectories
git -C trajectories fetch --depth 1 origin 3fb7dea343f8426140f38442c47eba000e80b9f2
git -C trajectories restore --source=3fb7dea343f8426140f38442c47eba000e80b9f2 --worktree -- Extract_triplets/Example_Brandao
uv sync --locked --extra extract --extra lab
sbatch --partition="$PARTITION" slurm/extract.sbatch trajectories/Extract_triplets/Example_Brandao 4
```

The extractor checks each source file against the pinned SHA-256 list. For stride 4 it selects beads 1, 5, 9, … in every frame (1,010 loci). Stride 2 keeps beads 1, 3, 5, … (2,020 loci). It counts strict Euclidean distances below five monomer lengths and normalizes by the frame count. The diagonal is one. Working storage is quadratic. The extractor checks shared physical loci against the published stride-10 map and records the largest difference in frame counts. Its double-precision distance comparison allows one observation per pair to differ near the threshold from the upstream single-precision input calculation.

When extraction finishes, copy the exact `contacts.txt` path from `slurm-JOBID.out` into `MAP`. Submit the 4 kb calculation first.

```sh
sbatch --partition="$PARTITION" --mem=32G slurm/highmem.sbatch "$MAP"
```

For 2 kb, rerun the extraction command with stride `2`, set `MAP` to that output, and submit the large job. Inspect the live node inventory and your account limits before requesting 128 GiB.

```sh
/ada-software/ada-info.sh
sbatch --partition="$PARTITION" --mem=128G slurm/highmem.sbatch "$MAP"
```

The larger job runs one threaded prediction, checks a grid of eligible triplets against the serial implementation and saves one bait plane. Bait 200 at 10 kb has offset 1,990 kb and maps to bin 996 at 2 kb. At 4 kb the closest two bins tie and the driver uses bin 499 (1,992 kb). The predictor applies the original one-bin neighbour exclusion at each resolution, so the excluded physical distance changes. Each run keeps its input path in the manifest for on-demand notebook slices. Keep the extracted map alongside the results on shared storage.

The 0.49, 7.68 and 61.41 GiB figures describe the tensor alone. The 32 and 128 GiB batch requests include headroom. Actual memory and walltime on ADA still require measurement. Inspect the RSS in the result manifest and the `sacct` job steps after completion.
