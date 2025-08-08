# Generalized Preprocessing Cuts Analysis Tools

This directory contains tools for analyzing preprocessing cuts and flags from sotodlib preprocessing pipelines in a generalized way that works with any configuration files.

## Files

### 1. `generalized_cuts_analysis_simple.py`
The main analysis script that processes telescope data and generates a database with cuts statistics.

**Key improvements over the original `iso_v1_stats_20250718.py`:**
- Automatically parses configuration files to determine the order of cuts processing
- Removes unused TOD stats fields
- Only includes fields ending in 'nsamps' (sample cuts) or 'cuts' (detector cuts)
- Generalizes to work with any preprocessing configuration files

**Usage:**
```bash
python generalized_cuts_analysis_simple.py config_init.yaml config_proc.yaml [options]
```

**Arguments:**
- `config_file_init`: Initial preprocessing configuration file
- `config_file_proc`: Processing preprocessing configuration file
- `--noise-range N_MIN N_MAX`: Noise range for white noise cuts (default: 18e-6 80e-6)
- `--nproc N`: Number of parallel processes (default: 16)
- `--errlog-ext FILE`: Error log file name (default: generalized_cuts_err.txt)
- `--savename FILE`: Output database name (default: generalized_cuts.sqlite)

**Example:**
```bash
python generalized_cuts_analysis_simple.py \
    /path/to/preprocessing_config_init.yaml \
    /path/to/preprocessing_config_proc.yaml \
    --nproc 32 \
    --savename satp1_cuts_analysis.sqlite
```

### 2. `generate_cuts_tables.py`
Standalone script that generates summary tables from the cuts analysis database, similar to the tables in the attached screenshots.

**Usage:**
```bash
python generate_cuts_tables.py database.sqlite --start-ts START --end-ts END [options]
```

**Arguments:**
- `sqlite_path`: Path to the cuts analysis SQLite database
- `--start-ts`: Start timestamp for analysis period
- `--end-ts`: End timestamp for analysis period
- `--output-csv`: Save tables as CSV files
- `--output-dir`: Directory to save CSV files (default: current directory)

**Example:**
```bash
python generate_cuts_tables.py satp1_cuts_analysis.sqlite \
    --start-ts 1716177600 \
    --end-ts 1734315803 \
    --output-csv \
    --output-dir ./tables/
```

This will generate:
- Formatted tables printed to console
- `detector_cuts_summary.csv` - Summary of detector cuts
- `sample_cuts_summary.csv` - Summary of sample cuts

### 3. `generalized_cuts_analysis.py`
More comprehensive version with additional features (currently has import dependencies).

## Configuration File Analysis

The `generalized_cuts_analysis_simple.py` script automatically analyzes your preprocessing configuration files to determine:

1. **Sample cuts order**: Processes that create sample flags/cuts
   - `turnaround_flags`, `jumps` (slow/2pi), `glitches`, `source_flags`, etc.
   - Results in fields like `turnaround_nsamps`, `jumps_slow_nsamps`, etc.

2. **Detector cuts order**: Processes that create detector cuts
   - `fp_flags`, `trends`, `det_bias_flags`, `ptp_flags`, `noise`, `inv_var_flags`
   - Results in fields like `fp_cuts`, `trend_cuts`, `det_bias_cuts`, etc.

The script processes cuts in the order they appear in your configuration files, ensuring accurate accounting of overlapping cuts.

## Output Database Schema

The output SQLite database contains a `results` table with columns:
- `obsid`: Observation ID
- `ws`: Wafer slot
- `band`: Bandpass
- `nsamps`: Total number of samples
- `ndets`: Total number of detectors
- `*_nsamps`: Sample cut counts for each flagging step
- `*_cuts`: Detector cut counts for each selection step
- `end_yield`: Final number of detectors surviving all cuts

## Sample Cuts vs Detector Cuts

**Sample Cuts (`*_nsamps` fields):**
- Count flagged samples that are excluded from analysis
- Examples: turnaround periods, jump locations, glitch samples, source transits
- Calculated as new samples cut by each step (avoiding double-counting)

**Detector Cuts (`*_cuts` fields):**  
- Count detectors that are completely removed from analysis
- Examples: unlocked detectors, poorly biased detectors, high noise detectors
- Based on thresholds applied to detector-level statistics

## Table Generation

The `generate_cuts_tables.py` script creates tables showing:

1. **Detector Cuts Table:**
   - Fraction of cut detectors contributed by each cut type
   - Fraction of total data volume lost to each cut type

2. **Sample Cuts Table:**
   - Fraction of flagged data contributed by each flag type
   - Fraction of total data volume lost to each flag type

These tables match the format shown in the attached screenshots.

## Dependencies

- `sotodlib`
- `so3g`
- `numpy`
- `pandas`
- `pyyaml`
- `sqlite3` (built-in)

## Notes

- The script automatically handles different jump types (slow vs 2π jumps)
- Edge cuts (first/last 6000 samples) are automatically added for demodulation
- All processing is done in parallel using the sotodlib processing pool
- Error logging captures and reports any processing failures
- The database uses SQLite for efficient storage and querying
