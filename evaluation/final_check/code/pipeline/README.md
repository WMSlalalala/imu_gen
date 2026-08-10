# `pipeline/` — the experiment drivers that were never under version control

Every script here is a **byte-identical copy** of a file in
`/mnt/share/mwang49/data7/code/baselines/`, which is not a git repository and is not
shipped with this repo. Before this directory existed, 48 of the scripts that produced
numbers in the paper were reachable only from that unversioned mount. They are copied
here so that a clone shows every driver, even where the inputs are too large to ship.

Snapshot taken **2026-08-10**. Every copy was verified against its source by sha256
after copying (48/48 identical). The source tree is the live working location — four
workflows were executing out of it during the copy — so nothing was moved or deleted
there, and this directory must be treated as an **archive, not a second working copy**.

## Read this before running anything here

**These scripts run the data7 tree, not this one.** Ten of them contain
`C=/mnt/share/mwang49/data7/code/baselines` followed by `cd "$C"`
(`queues/run_imu_programme.sh`, `run_a7_queue.sh`, `run_everything.sh`,
`run_remaining.sh`, `run_quality_sweep.sh`, `sweep_diffusion_ts.sh`, `sweep_tts_gan.sh`,
`sweep_shot_budget.sh`, `tests/smoke_end_to_end.sh`, and `queues/run_baseline_grid.sh`
with `C=…/data7/code/direct100k`). Three more `source "$C/gpu_slot.sh"`. Launching a
copy from this directory therefore executes the *data7* Python, not the file sitting
next to it. Editing a file here changes nothing about what runs. To actually rerun the
programme you need the data7 tree and the inputs listed in
[`../EXTERNAL_INPUTS.json`](../EXTERNAL_INPUTS.json).

Python throughout is `/home/mwang49/miniconda3/envs/cuhkx/bin/python`; the detector
grids use `…/envs/hml/bin/python` and the generator-quality metrics use
`/home/mwang49/.conda/envs/tsmetric/bin/python` (TF1-style graph code, kept out of the
torch environment on purpose).

Shorthand used in the tables below:

    B  = /mnt/share/mwang49/data7/results/direct100k/baselines     (159 GB, external)
    R  = /mnt/share/mwang49/data7/results/direct100k               (carrier datasets)
    C  = /mnt/share/mwang49/data7/code/baselines                   (the live source tree)

---

## `code/` — analysis and packaging drivers (9 files, Python)

All nine take `argparse` flags; the invocation shown is from the file's own parser.

| file | what it does | invoke | writes |
|---|---|---|---|
| `summarise_far5.py` | FAR at the development-selected **FRR=5%** operating point for any cell grid — the paper's actual criterion, not the EER threshold the cells publish. Refuses to run without `--dataset` or `--source-grid` so declined actions cannot be reported as a baseline's result. | `summarise_far5.py <cells-dir> --dataset <built-dataset> [--source-grid] [--json-out F]` | table on stdout; `--json-out` JSON |
| `build_comparison.py` | Cross-method comparison table: one table per modality, rows action × method, columns the six detectors. Imports `DETECTORS`/`read_grid` from `summarise_far5.py`. | `build_comparison.py --grid LABEL=CELLS[,DATASET] [--grid …] [--json-out F]` | table on stdout; `--json-out` JSON |
| `package_release.py` | Assembles the four baselines into four self-contained rerunnable folders: third-party code, driver, `run.sh` (from `runners/`), the fitting evidence, the grid result, plus a digest of every shared-harness file so drift is detectable. | `package_release.py --out <dir>` | `<out>/00_common/`, `<out>/01…04_*/`, `<out>/MANIFEST.json` |
| `score_generator_quality.py` | Discriminative + predictive scores using the authors' own metric code (`DiffusionTS/Utils/*_metric.py`), always alongside a **real-vs-real control** computed by splitting the genuine windows in half — the floor the data itself allows. | `score_generator_quality.py --real R.npy --generated G.npy --out F.json [--limit 2000] [--iterations 2000] [--repeats 3] [--seed 42]` | `--out` JSON |
| `benchmark_generation_speed.py` | Marginal inference cost per fake event (not training cost), GPU-synchronised, split into **synthesis** and **placement** stages, reported per window and per event at the release's real reuse factor. | `benchmark_generation_speed.py --dataset-dir D --out F.json [--action swipe] [--kind trajectory] [--events 500] [--repeats 3] [--batch 256] [--gpu 0] [--reuse-factor 5.0]` | `--out` JSON |
| `probe_convergence.py` | Finds the step count a five-window Diffusion-TS actually needs, by checkpointing during one training run and scoring each checkpoint on lag-1 autocorrelation, per-channel std ratio, and nearest-neighbour distance to the five training windows. | `probe_convergence.py --dataset-dir D --out F.json --action A --kind {trajectory,imu} [--train-events 5] [--steps 12000] [--checkpoint-every 1000] [--samples 256] [--gpu 0] [--seed 12345]` | `--out` JSON; checkpoints in `<out>.parent/ckpt_probe_<action>_<kind>` |
| `reproduce_diffusion_ts_sines.py` | Environment check behind every Diffusion-TS number: runs the authors' own Sines dataset/config/model through the same driver path and compares against the ICLR-2024 published `discriminative 0.006`, `predictive 0.093`. Scoring is a separate TF env, so it writes the arrays and prints the command. | `reproduce_diffusion_ts_sines.py --output-dir D [--steps 12000] [--gpu 0] [--seed 12345]` | `<D>/sines_generated.npy`, `<D>/sines_real.npy`, `<D>/ckpt_sines/`; prints the `score_generator_quality.py` command |
| `diagnose_placement.py` | Measures what the endpoint binding does to a sampled bank: the similarity **scale** applied (a run with a 10× upper tail is no longer testing the generator) and how often placement **fell back to translation**. Meant to run after a bank is sampled, before its FAR is read. | `diagnose_placement.py --source-dir S --banks B --binding P.pkl --out F.json [--shards 25]` | `--out` JSON |
| `restore_quarantined.py` | Repairs the five-shot quarantine bug: the upstream resume check hardcoded `ref_count == 5`, so any other reference count made every restart quarantine good samples and redraw them. Re-validates each candidate on the way back; never overwrites a redrawn sample. | `restore_quarantined.py --quarantine Q --cache C --expected-refs N [--apply]` (dry-run without `--apply`) | moves files into the cache; counts JSON on stdout |

## `queues/` — programme orchestration (14 files, bash)

Long-running drainers. All are idempotent/resumable by design; most append to
`$B/PROGRESS.txt`.

| file | what it does | invoke | writes |
|---|---|---|---|
| `run_everything.sh` | Drives the whole baseline programme in dependency order; each stage skips if its output exists. | `bash run_everything.sh` (no args) | `$B/PROGRESS.txt`, `<out>.log` per grid |
| `run_remaining.sh` | Finishes every comparison still missing so each paper experiment has a baseline counterpart — chiefly the joint `imu_trajectory_xytime` modality (90 cells vs the 30 each baseline had) and TTS-GAN as a full baseline. Grids never overlap. | `bash run_remaining.sh` (no args) | `$B/PROGRESS.txt`, `$B/verify_diffts_both.txt`, `$B/verify_ttsgan_<kind>.txt`, `<out>.log` |
| `run_imu_programme.sh` | Packs everything GPU-only into the two-card window and leaves the grids (20 of 30 cells are CPU classical detectors) to drain afterwards. Sources `gpu_slot.sh`. | `bash run_imu_programme.sh`; env `SAMPLER_MIB` (2600) | `$B/PROGRESS.txt`, `<out>/log_shard<N>.txt` |
| `scheduler.sh` | Runs remaining GPU work in **priority order rather than in parallel** — both cards were already saturated, and a 30 ms scroll step measured 6.5 s under that load. Order: release ablations, then trained baselines, then ImagenTime last. | `bash scheduler.sh` (no args) | `$B/PROGRESS.txt`, `$B/imagentime_queue.log` |
| `grid_queue.sh` | Drains `$B/GRID_JOBS.txt` (`<method> <modality>` per line) forever, `CONCURRENCY` at a time; finished lines go to `GRID_JOBS_DONE.txt`. Clears stale claims left by a previous instance at startup. | `bash grid_queue.sh`; env `CONCURRENCY` (2) | `$B/GRID_JOBS_DONE.txt`, `$B/gridqueue_<method>.log`, `$B/PROGRESS.txt` |
| `crossscore_queue.sh` | The **second table**: scores every built attack against the release's 90 frozen detectors at their own FRR=5% points, retraining nothing — the deployed-defender view. Kept separate because the two views disagree sharply (one augmentation baseline: 0.000 self-trained vs 0.772 frozen). Pure inference, one job at a time. | `bash crossscore_queue.sh` (no args) | `$B/crossscore/<method>.log`, `$B/PROGRESS.txt` |
| `ablation_pipeline.sh` | Carries each finished ablation cache on to dataset then detector grid. Counts completeness against the release's real requirement (20,000 fake events/action) rather than trusting the sampler's `.complete` marker, which fires at 30,000 samples. Keystroke excluded (its fake IMU never goes through the diffusion generator). | `bash ablation_pipeline.sh` (no args) | `$B/GRID_JOBS.txt`, `$B/final_<method>.log`, `$B/PROGRESS.txt` |
| `fiveshot_repipeline.sh` | Re-draws every cache whose references changed, now that each victim's inertial bank is headed by the five recordings the touch channel was frozen against. A2 and keystroke are skipped (no references consumed). | `bash fiveshot_repipeline.sh [--dry-run]`; env `SHARDS_PER_GPU` (12) | `<out>/sample_<action>_<shard>.log`, `$B/PROGRESS.txt` |
| `run_a7_queue.sh` | Ablation A7: full retrain of the weighted-sum arm for scroll and swipe, one per card, at the released run's own epoch counts (160/145). Resumable. | `bash run_a7_queue.sh` (no args) | `$B/ablations/a7_weighted_sum/queue_<action>.log` |
| `run_baseline_grid.sh` | Runs **one** baseline dataset through the same detector grid the method uses (20 epochs, 10,000 bootstrap replicates, seed 42, both cards). | `bash run_baseline_grid.sh <dataset-dir> <output-dir> <modality>` | `<output-dir>/`, `<output-dir>.log`; success is `completion.json` |
| `run_quality_sweep.sh` | Scores every generated bank with the generator's own literature metrics plus the real-vs-real control, in the separate TF env. | `bash run_quality_sweep.sh <samples-dir> <out-dir> ["trajectory imu"]` | `<out-dir>/` JSONs |
| `sweep_diffusion_ts.sh` | Fits and samples one Diffusion-TS per (action, channel set), five jobs sharing a card. | `bash sweep_diffusion_ts.sh`; env `STEPS` (12000), `SAMPLES` (4000) | `$R/baselines/diffts/`, `log_<action>_<kind>.txt` |
| `sweep_shot_budget.sh` | The attacker's-budget arm: same architecture, same 12,000 steps, only the data changes — `shot5` (5 windows total) and `shot5pu` (5 per training user, 350). Expected to fail; the number has to exist so the paper can measure the failure rather than assert it. | `bash sweep_shot_budget.sh`; env `SAMPLES`, `STEPS`, `KINDS` | `<out>/log_<action>_<kind>.txt` |
| `sweep_tts_gan.sh` | Fits and samples one TTS-GAN per action on the IMU channels; the iteration budget is **per action** because a transformer GAN's step cost scales with `seq_length/patch_size`. | `bash sweep_tts_gan.sh`; env `KIND` (imu), `SAMPLES`, `JOBS` (`action:iters:gpu` list) | `$R/baselines/ttsgan/`, `log_<action>_<KIND>.txt` |

## `infra/` — host babysitters (8 files, bash)

**Not experiment logic.** These keep the machine working; none of them produces a
number. They are archived because the queues above cannot be read without them —
`gpu_slot.sh` in particular is `source`d by three queues.

| file | what it does | invoke | writes |
|---|---|---|---|
| `gpu_slot.sh` | `wait_for_slot <gpu> <needed_mib>` — blocks until a card has the job's footprint *plus the user's reserve* free. Encodes the standing **2026-08-10 two-card policy**: GPU 0 used in full, GPU 1 keeps 10 GB. | `source gpu_slot.sh`; env `RESERVE_MIB` (6000 fallback), `POLL_SECONDS` (20), `GPU_POLICY` | nothing (function library) |
| `gpu_curfew.sh` | Hands GPU 1 back at a cutoff without losing work: rewrites the policy first, then stops queues, then kills stragglers on the reclaimed card. | `bash gpu_curfew.sh`; env `CUTOFF` (17:00), `RECLAIM` (1), `KEEP` (0) | `$B/GPU_POLICY`, `$B/PROGRESS.txt` |
| `supervisor.sh` | Restarts any queue that dies while it still has work. Deliberately never restarts ImagenTime (the scheduler holds it back on purpose) and never restarts a queue whose work is finished. | `bash supervisor.sh` (no args) | `$B/PROGRESS.txt`, `$B/<queue>.log` |
| `watchdog.sh` | Emits one line per milestone or failure — `DONE`/`GONE`/`STALL`/`ERROR`/`OOM` — each fingerprinted so a persistent watch never repeats itself. | `bash watchdog.sh` (no args) | `$SEEN` fingerprint file under `$B` |
| `status_daemon.sh` | Writes a one-glance status every five minutes, forever; launched with `setsid` so it outlives the session. | `bash status_daemon.sh` (no args) | `$B/STATUS.md`, `$B/EVENTS.txt` |
| `pipeline_daemon.sh` | Watches for generators that finished sampling and carries them on: assemble bank → build bundles → verify → grid. Every stage idempotent and marked on disk. | `bash pipeline_daemon.sh` (no args) | `$B/pipeline_<method>.log`, `$B/PROGRESS.txt` |
| `grid_daemon.sh` | Older single-slot grid drainer reading `$B/GRID_QUEUE.txt` (`dataset|…` lines), moving finished lines to `GRID_DONE.txt`. Superseded in practice by `queues/grid_queue.sh`. | `bash grid_daemon.sh` (no args) | `$B/GRID_DONE.txt`, `<out>.log`, `$B/PROGRESS.txt` |
| `spin_check.sh` | Counts grid-queue back-offs in the last N **minutes** rather than the last N lines, so a fixed spin stops re-firing. | `bash spin_check.sh [minutes]` (default 30) | count on stdout |

## `runners/` — per-release reproduction templates (2 files, bash)

These are **not** run from this directory. `code/package_release.py` copies each one
into a release folder as `run.sh`, next to a copy of the shared harness; both resolve
their siblings through `HERE=$(dirname "$0")`, so they only work from inside such a
folder.

| file | what it does | invoke (inside a release folder) | writes |
|---|---|---|---|
| `run_pyclick.sh` | Reproduces baseline 01 end to end: build → verify → score. Genuine events are never touched. | `bash run.sh <carrier-dataset-dir> <work-dir>`; env `PY`, `DETECTOR_PY`, `DETECTOR_CODE` | everything under `<work-dir>` |
| `run_learned.sh` | Reproduces one learned baseline end to end: fit five models (one per action, train users' genuine events only) → sample banks → fill fake carriers → verify harness → detector grid → the generator's own quality metrics against the real-vs-real control. | `GENERATOR=diffusion_ts\|tts_gan KIND=trajectory\|imu bash run.sh <carrier-dataset-dir> <work-dir>`; env `PY`, `METRIC_PY`, `DETECTOR_PY`, `DETECTOR_CODE` | `<work-dir>/samples/`, `<work-dir>/detectors/cells/`, quality JSONs |

## `tests/`

| file | what it does | invoke | writes |
|---|---|---|---|
| `smoke_end_to_end.sh` | Runs every builder on a 3-shard miniature dataset and then verifies it — catching the join between pieces that the unit suite tests in isolation. Built on `replay_dataset_v16`, **not** v12, whose scroll/swipe fake events carry a deterministic `elapsed` column that `verify_harness` check [5] rejects by design. ~1 minute on CPU. | `bash smoke_end_to_end.sh`; env `SOURCE`, `PY`, `WORK`, `SHARDS` (3), `KEEP` | scratch under `$WORK`, removed unless `KEEP=1` |

Note: the checked-in default for `WORK` is a machine-local `/tmp/claude-…/scratchpad/e2e_smoke`
path that will not exist on another host. Pass `WORK=` explicitly.

## `docs/` — the baseline write-ups (6 files, Chinese)

`package_release.py` copies `01_…`, `02_03_…` and `04_…` into the matching release
folder as its `README`.

| file | contents |
|---|---|
| `README_CN.md` | Index of the four third-party baselines (2 trajectory, 2 IMU) and the shared conditions: same carriers, same six detectors, same criterion; only the fake-signal generator changes. |
| `01_traj_pyclick_bezier.md` | Baseline 01 — pyclick Bézier bot (trajectory), MIT, the de-facto anti-bot-evasion implementation. |
| `02_03_diffusion_ts.md` | Baselines 02/03 — Diffusion-TS (trajectory / IMU), ICLR 2024. |
| `04_imu_tts_gan.md` | Baseline 04 — TTS-GAN (IMU). |
| `AUDIT_FIXES_CN.md` | Five independent audits plus adversarial verification of the baseline harness; every finding confirmed real and fixed. Exists so the paper can show a weak baseline is weak on its merits, not because it was run wrong. |
| `RESULTS_CN.md` | Result table, method vs the four baselines, at FAR @ FRR=5% with target ≥ 0.6. Shared carriers, detectors, split, 20 epochs, seed 42; genuine events byte-identical across datasets. |

## `vendor/` — third-party source with no recoverable upstream

> **发布前必读：[`vendor/PROVENANCE.md`](vendor/PROVENANCE.md)。** 这里的文件没有随附许可证文本，
> 而这是一个公开仓库。`ImagenTime_additions/` 是我们自己的代码、可以公开；
> `pyppeteer_ghost_cursor/` 是一份作者从未发布过的第三方移植，**许可状态未知，未决**。

| path | why it is here |
|---|---|
| `pyppeteer_ghost_cursor/` (5 files) | The ghost-cursor Python port used by `build_ghostcursor_baseline.py` (via `ghost_cursor_path.py`). It has **no `.git`** in the source tree and no pinned upstream — this was the only copy on disk. Losing it makes the ghost-cursor baseline unreproducible. |
| `ImagenTime_additions/data/` (3 files) | ImagenTime ships `data/` as an empty package (the authors distribute corpora separately), so nothing in the repo imports until these exist. `long_range.py` routes HMOG windows through their `fred_md` branch with the same per-channel min-max scaling every corpus of theirs gets (EDM hard-codes `sigma_data = 0.5`); `data_provider/data_factory.py` deliberately raises `NotImplementedError` so a wrong dataset name stops the run instead of silently loading ETT. **Not recoverable by cloning ImagenTime at `f372626`** — they are untracked local additions. Restore by copying into `ImagenTime/data/`. |

---

## What was deliberately NOT copied

**41 files in the source tree already have a byte-identical copy in this repo.**
Verified by sha256 on 2026-08-10 (every file in the source tree hashed, compared against
a size-prefiltered hash index of 3,243 repo files). Copying them again would create a
third copy that can drift, which is the problem this directory exists to end.

The three **frozen rule files** named in this lane's brief were checked individually and
are identical — no `rules/` directory was created:

| file | sha256 (both copies) | already tracked at |
|---|---|---|
| `gate_rules.json` | `e06232725a9e7aa3…` | `evaluation/final_check/gate_rules.json` |
| `fairness_rules.json` | `22d1256f43e6bd00…` | `evaluation/final_check/fairness_rules.json` |
| `release_cell_map.json` | `24a2581c73e1b1f0…` | `evaluation/final_check/release_cell_map.json` |

A fourth frozen rule file, the generator manifest `released_generators.json`
(`910d433511…`), is likewise identical to `evaluation/comparison/code/released_generators.json`.

The other 37 identical files are the experiment drivers already archived under
`evaluation/comparison/code/` (27: `assemble_banks`, `bootstrap_far5`,
`build_against_final`, `build_ghostcursor_baseline`, `build_pyclick_baseline`,
`build_sample_bank_baseline`, `check_reference_sync`, `convergence`,
`covered_modalities`, `eer_tables`, `export_real_windows`, `final_release`,
`final_tables`, `ghost_cursor_path`, `grid_job_done`, `hmog_event_builder`,
`released_generators.py`, `run_csdi.py`, `run_csdi_all.sh`, `run_diffusion_ts`,
`run_imagentime.py`, `run_imagentime_all.sh`, `run_tts_gan`, `run_ttsgan_retrain.sh`,
`score_against_fixed_detector`, `summarise_final`, `verify_harness`) and
`evaluation/ablation/code/` (8: `a7_pipeline.sh`, `build_ablation_cache_baseline`,
`critic_pipeline.sh`, `generate_imu_ablation`, `run_a7_weighted_sum`,
`run_ablation_queue.sh`, `run_critic_ablation`, `verify_reconstruction`), plus
`grid_against_final.sh` and `hmog_baseline_common.py`, which are identical in all three
places.

**One further file was excluded on purpose:** `write_baseline_readmes.py`. The source
tree's 23,598 B copy is superseded by the 45,188 B version at
`evaluation/comparison/code/write_baseline_readmes.py`; running the older one would
regenerate baseline notes containing claims the project has since retracted (it still
describes the CSDI five-shot arm as "dropped after measurement" rather than never
completed). The newer copy is already tracked; the older one is not worth a second home.

## Known conflict this directory does not resolve

`infra/gpu_slot.sh` here is the live 4,077 B version carrying the 2026-08-10 two-card
policy. `evaluation/comparison/code/gpu_slot.sh` and `evaluation/ablation/code/gpu_slot.sh`
are a **different, older 2,721 B file** (single `RESERVE_MIB=6000` for both cards) and
they are dead code: every consumer sources `$C/gpu_slot.sh` with `C` hardcoded to the
data7 path, so neither repo copy has ever been executed. Those two files contradict the
standing GPU policy and should be deleted by whoever owns those directories — that is
outside this lane, and nothing here was changed to work around it.
