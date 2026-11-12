# so_data_exploration

Tooling for measuring detector cuts and sample flags through the Simons Observatory
SAT preprocessing pipeline, and for turning the result into yield, efficiency and
noise summaries.

The two scripts at the root of this repository are the entry points. Everything
else is downstream analysis that reads the database they produce.

---

## What the two entry points do

`track_cuts_and_stats.py` replays the preprocessing pipeline for **one**
`(observation, wafer_slot, band)` group and records, at every step, how many
detectors survive and how many samples are flagged. It writes one row per group
into a SQLite database.

`track_cuts_and_stats_parallel.py` is the production driver. It does the same
thing across an observation list, distributing groups over an MPI communicator or
a process pool via `sotodlib.utils.procs_pool.get_exec_env`, and can additionally
aggregate per-detector white noise into HDF5.

Use the single-group script to debug one group or to understand the schema; use
the parallel one for anything real.

---

## Running them

### Environment

These must run against a working `sotodlib` and the soconda environment, not a
bare `python3` — a project virtualenv on `PATH` will shadow it and the import
will fail or pick up a stale `sotodlib`:

```bash
PYTHON=/scratch/gpfs/SIMONSOBS/env/v0.6.24/soconda_v0.6.24/bin/python3
export PYTHONPATH=/home/ms3067/repos/sotodlib
```

Real runs belong on a compute node (`sbatch` or `salloc`), never on a login node.

### One group, interactively

Useful for a single observation or when checking the schema:

```bash
$PYTHON track_cuts_and_stats.py \
    --obs-id   obs_1722543721_satp1_0111111 \
    --wafer    ws1 \
    --band     f090 \
    --init-config /path/to/preprocessing_config_init.yaml \
    --proc-config /path/to/preprocessing_config_proc.yaml \
    --db-path  /path/to/iso_v4_satp1.sqlite \
    --verbosity 1
```

`--proc-config` is optional; without it only the init layer is replayed.

### A full survey, under SLURM

This is how the production databases are made. The pattern below is taken from
`tiger/202601_v4/run_iso_v4_satp1.slurm`, which is a complete working example —
copy it and change the paths rather than writing one from scratch.

```bash
LAUNCH_STR="srun -n ${SLURM_NTASKS} --cpus-per-task=${SLURM_CPUS_PER_TASK} --export=ALL"

${LAUNCH_STR} $PYTHON track_cuts_and_stats_parallel.py \
    --init-config     "${INIT_CONFIG}" \
    --proc-config     "${PROC_CONFIG}" \
    --obs-list        "${OBS_LIST}" \
    --db-path         "${DB_PATH}" \
    --batch-size      100 \
    --verbosity       1 \
    --nprocs          ${SLURM_NTASKS} \
    --aggregate-noise Q \
    --aggregate-output "${AGG_DIR}/noise_aggregation" \
    2>&1 | tee "${LOG_DIR}/processing_${SLURM_JOB_ID}.log"
```

Selecting observations — give exactly one of:

| option | meaning |
|---|---|
| `--obs-list FILE` | text file, one `obs_id` per line (what the production runs use) |
| `--obs-id ID` | a single observation |
| `--query Q` | a context query string |
| `--min-ctime` / `--max-ctime` | timestamp bounds |
| `--update-delay N` | start the list N days in the past |

Other options worth knowing:

| option | meaning |
|---|---|
| `--batch-size N` | commit to SQLite every N groups; `0` commits only at the end |
| `--overwrite` | recompute groups already present instead of skipping them |
| `--aggregate-noise {T,Q,U}` | also aggregate per-detector white noise for that component |
| `--aggregate-output BASE` | path stem for the aggregated HDF5 |
| `--nprocs N` | process-pool size; ignored in MPI mode, which uses the communicator |

Redirect job logs with `>>` to a `${SLURM_JOB_ID}`-stamped path. A fixed path with
`>` truncates earlier logs, which has previously destroyed the per-group failure
attribution for an entire run.

---

## What comes out

One SQLite database, one row per `(obs_id, wafer, band)`, with four column
families:

- `det_<N>_<step>_ndets` — detectors surviving each pipeline step. A step's cut is
  the fractional drop from the preceding `det_` column.
- bare flag names (`noisy_subscan_flags`, `glitches_*`, `jumps_*`, …) — **sample**
  counts, to be divided by `total_samples`.
- `noise_{T,Q,U}_{white_noise,fknee,alpha}_{avg,q10…q90}` and `array_noise`.
- `success`, `failure_step`, `error_message`, and `t2p_*`.

> **Resolve step columns by name, not by number.** The `det_<N>` numbering is not
> aligned across platforms: SATP2's config has no `move` step, so everything after
> it is shifted by one. `noisy_subscan_flags` is `det_44` on SATP1/SATP3 but
> `det_43` on SATP2. Match the step-name substring and take the immediately
> preceding `det_` column as the denominator.

Rows with `total_samples = 0` never produced data and should be excluded from
population statistics.

---

## Downstream analysis

These read the database above and need no pipeline access:

| script | purpose |
|---|---|
| `analyze_cuts_and_stats.py` | main analysis: cut and flag plots and tables |
| `generate_yield_and_efficiency_plots.py` | detector yield, observing and mapping efficiency |
| `generate_cuts_and_flags_summary_tables.py` | per-platform-band summary tables |
| `generate_timeseries_plots.py` | daily cut time-series grids |
| `generate_survey_efficiency_plots.py` | detector-hours per week vs theoretical maximum |
| `build_calibrated_noise.py` | K_CMB copies of the noise aggregation files |
| `generate_report_wrapper.py` | assemble the plots and tables into a PDF |
| `batch_analysis.py`, `run_all_analysis.sh` | run the above across every archive |

`tiger/202601_v4/` holds the SLURM run scripts for the 2026-07-02 processing round,
the attrition tooling (`track_preprocessing_attrition.py` and its table renderer),
and the analysis notebooks.

---

## Note on repository contents

Study write-ups, figures, PDFs, job logs and one-off scripts are **not** kept here.
They live under `/scratch/gpfs/SIMONSOBS/users/ms3067/so_data_exploration_archive/`,
with `MANIFEST_20261112.txt` recording what moved and from where.
