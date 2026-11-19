# Run the 2,020-locus benchmark on ADA

The 2 kb input comes from 33 trajectory blocks (about 117 MiB) containing 3,060 snapshots of the 4,040-bead chromosome. Preparing the input is a one-time CPU job. After that, the presentation can start directly with 8,242,408,000 output entries on the largest configured CPU allocation.

From the Apptainer terminal in the shared demo clone, run the following. The commands assume the shared clone is at `~/ada-beta-demo` on both hosts.

```sh
cd ~/ada-beta-demo
git pull --ff-only origin main
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked --extra extract --extra lab --extra segregation --extra cuda12
ssh inter01.labs.vu.nl 'cd ~/ada-beta-demo && bash scripts/stage-julia.sh && bash scripts/prepare-bait-map.sh --partition=itvo-test'
```

If you are restoring both demonstrations, use `.venv/bin/python -m demo_storage prepare-ada --with-bait` from the shared clone in Apptainer. It also configures the dedicated H200, recovers older inputs, and stages Julia. [Full recovery instructions](segregation.md#prepare-or-restore-the-full-ensemble)

The bait helper reuses a prepared 2 kb input. Otherwise it locates or fetches the coordinates at the pinned source revision, submits the extraction on the configured command-line partition, and waits. The preparation job requests one CPU and 8 GiB. Its output is in `results/bait-map-<job-id>.log`. You can pass additional site options such as `--account` or `--qos` after the partition. The SSH connection uses the login host's Slurm client.

Restart the notebook kernel and run the dashboard cell. With `slurm.toml` configured, **Live experiment** opens with **Slurm CPU allocation**, the largest prepared map, and the full configured CPU ceiling. For this demo that is **2,020 loci · 2 kb** and **60**. **Selected threads only → Run experiment** starts directly at 60 threads. For a side-by-side measurement, choose **Compare with 16 threads** before starting. One Slurm job runs 60 first, then 16 on the same node with the same map and timing window. **Results & replay** shows both median times, average busy CPUs, peak RSS and speedup relative to 16 threads. The replay initially selects a prediction sample at the highest measured thread count.

The 2,020-locus tensor occupies 61.41 GiB of batch-node RAM. The dashboard requests 65G automatically. If you have an explicit `benchmark_memory` override, allow headroom, for example `benchmark_memory = "80G"`. The `memory` field is for segregation jobs. Julia's compilation and warm-up happen inside the batch job. Each timed pass finishes even if it exceeds the selected measurement window.

The extractor checks each source file against the pinned SHA-256 list. For stride 4 it selects beads 1, 5, 9, … in every frame (1,010 loci). Stride 2 keeps beads 1, 3, 5, … (2,020 loci). It counts strict Euclidean distances below five monomer lengths and normalizes by the frame count. The diagonal is one. Working storage is quadratic. The extractor checks shared physical loci against the published stride-10 map and records the largest difference in frame counts. Its double-precision distance comparison allows one observation per pair to differ near the threshold from the upstream single-precision input calculation.

Bait 200 at 10 kb has offset 1,990 kb and maps to bin 996 at 2 kb. At 4 kb the closest two bins tie and the driver uses bin 499 (1,992 kb). The predictor applies the original one-bin neighbour exclusion at each resolution, so the excluded physical distance changes. Each session records the input checksum, bin spacing and bait in its manifest. Keep the extracted map alongside the results on shared storage.

For the 1,010-locus input, run the same extractor with `--stride 4` on the staged coordinates, leaving its generated map under `inputs/`. Live experiment lists both resolutions once prepared.
