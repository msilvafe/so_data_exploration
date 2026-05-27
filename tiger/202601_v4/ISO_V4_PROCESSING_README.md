# ISO-V4 Multi-Node Processing on Tiger

This directory contains scripts for running full iso-v4 processing on SATPv1 and SATp3 observations across multiple nodes on Tiger.

## Setup

The scripts expect the following directory structure and files to exist:

```
preprocessing/
├── satp1/
│   ├── preprocessing_config_20251216_init.yaml
│   ├── preprocessing_config_20251216_proc.yaml
│   └── listv4.txt
└── satp3/
    ├── preprocessing_config_20251216_init.yaml
    ├── preprocessing_config_20251216_proc.yaml
    └── listv4.txt
```

Output will be created in:
```
/scratch/gpfs/SIMONSOBS/users/ms3067/iso_stats/v4_tests/
├── satp1/
│   ├── databases/       # SQLite database
│   ├── aggregation/     # Aggregated noise HDF5 files
│   └── logs/           # Processing logs
└── satp3/
    ├── databases/
    ├── aggregation/
    └── logs/
```

## Running the Jobs

### Option 1: Submit Both Jobs at Once

```bash
cd /home/ms3067/so_data_exploration/tiger/202601_v4
bash submit_iso_v4_jobs.sh
```

This script will:
1. Submit the SATPv1 job
2. Submit the SATp3 job
3. Print job IDs for reference

### Option 2: Submit Jobs Individually

```bash
# SATPv1 only
sbatch run_iso_v4_satp1.slurm
# Job ID: xxxxxxxx

# SATp3 only
sbatch run_iso_v4_satp3.slurm
# Job ID: xxxxxxxx
```

## Monitoring Jobs

### Check job status
```bash
squeue -u ms3067 -l
```

### View live output
```bash
tail -f /scratch/gpfs/SIMONSOBS/users/ms3067/iso_stats/v4_tests/satp1/logs/processing_*.log
```

### Cancel a job
```bash
scancel JOBID
```

## Job Configuration

Each job runs with:
- **Nodes**: 4
- **Tasks per node**: 8 (32 total MPI ranks)
- **CPUs per task**: 2 (utilizes 64 cores total)
- **Time limit**: 24 hours
- **Batch size**: 100 (database commits every 100 updates to reduce I/O overhead on shared filesystem)

## Output Files

After processing completes, each directory will contain:

### SATPv1 Output
```
satp1/
├── databases/
│   └── iso_v4_satp1.sqlite          # Main results database
├── aggregation/
│   ├── noise_aggregation_f090.h5    # Combined f090 noise statistics
│   ├── noise_aggregation_f150.h5    # Combined f150 noise statistics
│   └── noise_aggregation_*_*_*.h5   # Per-rank intermediate files
└── logs/
    └── processing_JOBID.log          # Complete processing log
```

### SATp3 Output
Same structure with `iso_v4_satp3.sqlite` and appropriate task directories.

## Database Schema

The SQLite database contains the following columns per observation/wafer/band:
- Detector counts at each pipeline step
- Sample counts per flag/cut type
- Noise statistics (white noise, fknee, alpha quantiles)
- T2P statistics (temperature-to-polarization coefficients)
- Success/failure status and error messages

Query example:
```sql
SELECT obs_id, wafer, band, det_0_starting_ndets, det_5_noise_ndets 
FROM obs 
WHERE success=1 
LIMIT 10;
```

## Aggregation Files

The HDF5 aggregation files contain:
- `white_noise`: Detector white noise values
- `det_ids`: Detector names
- `wafer_slots`: Wafer slot assignments
- `obs_ids`: List of observation IDs
- `obs_ranges`: Index ranges for each observation in the detector arrays

These can be read with:
```python
import h5py
with h5py.File('noise_aggregation_f090.h5', 'r') as f:
    wn = f['white_noise'][:]
    dets = f['det_ids'][:]
    ranges = {f['obs_ids'][i]: f['obs_ranges'][i] for i in range(len(f['obs_ids']))}
```

## Troubleshooting

### Job failed to start
Check if config files exist:
```bash
ls -la preprocessing/satp1/preprocessing_config_20251216_*
ls -la preprocessing/satp1/listv4.txt
```

### Slow database commits
If processing is slow due to database I/O:
- Increase `--batch-size` (e.g., 200) to reduce commit frequency
- Or decrease it to (e.g., 50) for more frequent smaller commits

### Memory issues
Reduce `--cpus-per-task` in SLURM script or `--ntasks-per-node` to free up memory.

### Incomplete results
Check the log file for errors:
```bash
tail -100 /scratch/gpfs/SIMONSOBS/users/ms3067/iso_stats/v4_tests/satp1/logs/processing_*.log
```

## Notes

- The batch processing approach (batch_size=100) significantly reduces I/O overhead on Tiger's distributed filesystem
- Aggregation is done on-the-fly during processing and consolidated by band in post-processing
- Each MPI rank handles its own database connection and batches commits independently
- Final commits ensure no data is lost if a rank processes fewer than batch_size items
