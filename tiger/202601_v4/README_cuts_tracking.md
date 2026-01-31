# SO Preprocessing Cuts and Statistics Tracking

This directory contains scripts for tracking sample cuts, detector cuts, and noise statistics during the SO preprocessing pipeline.

## Core Scripts

### Main Processing Scripts
- **`track_cuts_and_stats.py`**: Main script that runs the preprocessing pipeline step-by-step to track progressive detector cuts, sample cuts, and extract noise statistics. This is the core functionality.
- **`track_cuts_and_stats_parallel.py`**: MPI-parallel version for running on SLURM cluster across many observations.

### Analysis Scripts  
- **`analyze_cuts_and_stats.py`**: Analysis and visualization script for the tracking results.
- **`analyze_job_failures.py`**: Script to analyze failed observations from the jobs database.

### Existing Scripts (for reference)
- **`track_det_counts.py`**: Original detector counting script (existing in repo).
- **`analyze_det_counts.py`**: Analysis script for detector counts (existing in repo).

## SLURM Execution

### SLURM Scripts
- **`run_cuts_stats_satp1.slurm`**: SLURM job script for SATP1 telescope.
- **`run_cuts_stats_satp3.slurm`**: SLURM job script for SATP3 telescope.
- **`submit_cuts_stats_jobs.sh`**: Convenience script to submit both jobs.

### Usage
```bash
# Submit both telescopes
./submit_cuts_stats_jobs.sh

# Submit single telescope
./submit_cuts_stats_jobs.sh satp1
./submit_cuts_stats_jobs.sh satp3
```

## Single Observation Testing

To test the system on a single observation:
```bash
cd /home/ms3067/repos/iso-sat/v4
python /home/ms3067/so_data_exploration/tiger/202601_v4/track_cuts_and_stats.py \
  --obs-id obs_1729315555_satp1_1111111 \
  --wafer ws1 \
  --band f150 \
  --init-config ./preprocessing/satp1/preprocessing_config_20251216_init.yaml \
  --proc-config ./preprocessing/satp1/preprocessing_config_20251216_proc.yaml \
  --db-path /tmp/test_cuts_stats.db \
  --verbosity 2
```

## Output Database Schema

The system creates SQLite databases organized by obs-wafer-band with the following tables:

### detector_counts
Progressive detector counts after each preprocessing step, with columns for each process step (91 columns total for current pipeline).

### sample_cuts  
Sample cuts from various flagging operations:
- smurfgaps_cuts, turnarounds_cuts, jumps_slow_cuts, jumps_2pi_cuts
- glitches_pre_hwpss_cuts, glitches_post_hwpss_cuts, source_moon_cuts

### noise_stats
Noise statistics for T, Q, U polarizations with quantiles (10th, 25th, 50th, 75th, 90th percentiles):
- white_noise statistics  
- fknee statistics (Q, U only)
- alpha statistics (Q, U only)

### t2p_stats
Temperature-to-polarization leakage statistics:
- coeffsQ_avg, coeffsU_avg (T2P coefficients)
- errorsQ_avg, errorsU_avg (fitting errors)
- redchi2sQ_avg, redchi2sU_avg (reduced chi-squared)

### failed_observations
Records observations that failed processing with error messages.

## Key Features

1. **Progressive Detector Tracking**: Runs preprocessing pipeline step-by-step and tracks actual detector count after each selection operation, capturing the true detector elimination process.

2. **Comprehensive Statistics**: Extracts noise parameters (white noise, fknee, alpha) and T2P leakage coefficients with quantile statistics.

3. **Robust Error Handling**: Records failed observations and continues processing, suitable for large-scale production runs.

4. **Database Organization**: Results organized by obs-wafer-band for easy querying and analysis.

5. **SLURM Integration**: Designed for parallel execution on Tiger cluster with 224 processes across 4 nodes.

## Production Deployment

The system outputs databases to:
- `/scratch/gpfs/SIMONSOBS/sat-iso/v4/cuts_stats/cuts_stats_satp1.db`
- `/scratch/gpfs/SIMONSOBS/sat-iso/v4/cuts_stats/cuts_stats_satp3.db`

Job logs are written to:
- `/home/ms3067/so_data_exploration/tiger/202601_v4/logs/`

## Validation Results

The system has been validated against known expected detector counts for observation `obs_1729315555_satp1_1111111` ws1 f150:
- ✅ Tracks progressive detector elimination: 729 → 689 detectors  
- ✅ Matches expected detector cuts at each pipeline step
- ✅ Extracts comprehensive noise and T2P statistics
- ✅ Handles both init and proc pipeline stages