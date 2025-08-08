# SLURM Scripts for Generalized Cuts Analysis

This directory contains SLURM job scripts for running the generalized cuts analysis on the supercomputer with job scheduling.

## Scripts Overview

### 1. `run_generalized_cuts_satp1.slurm`
Ready-to-use script for analyzing SATp1 data with the generalized cuts analysis tool.

**Usage:**
```bash
sbatch run_generalized_cuts_satp1.slurm
```

**Key features:**
- Pre-configured for SATp1 telescope
- Uses 4 nodes, 224 tasks total
- 45-minute time limit
- Outputs to organized directory structure
- Includes optional table generation (commented out by default)

### 2. `run_generalized_cuts_satp3.slurm`
Ready-to-use script for analyzing SATp3 data.

**Usage:**
```bash
sbatch run_generalized_cuts_satp3.slurm
```

**Key features:**
- Pre-configured for SATp3 telescope  
- Uses 4 nodes, 224 tasks total
- 75-minute time limit (SATp3 typically takes longer)
- Different noise range (2e-6 to 80e-6) appropriate for SATp3

### 3. `run_generalized_cuts_template.slurm`
General template script that can be customized for any telescope/version combination.

**Usage:**
1. Copy the template: `cp run_generalized_cuts_template.slurm run_my_analysis.slurm`
2. Edit the USER CONFIGURATION section
3. Submit: `sbatch run_my_analysis.slurm`

**Customization points:**
- Telescope name (satp1, satp2, satp3, etc.)
- Version (v1, v2, v3, etc.)
- Configuration file paths
- Noise ranges
- Email address
- Output directories
- Timestamp ranges

### 4. `generate_tables.sh`
Lightweight shell script for generating summary tables from an existing cuts analysis database.

**Usage:**
1. Edit the script to point to your database file and set timestamp range
2. Run directly: `./generate_tables.sh`

This runs as a simple Python script (no SLURM needed) since table generation is computationally light.

### 5. `generate_tables_template.sh`
Template version of the table generation script for easy customization.

## Configuration Before Running

### Essential Updates Required:

1. **Email Address**: Update `--mail-user` in each script
2. **Configuration Paths**: Update paths to your preprocessing config files
3. **Script Paths**: Ensure paths to the Python scripts are correct
4. **Output Directories**: Verify output directory paths exist and are writable

### For the Template Script:
Replace all instances of `USERNAME` with your actual username.

### Common Configuration File Paths:

**v1 configs:**
```
/home/USERNAME/repos/iso-sat/v1/preprocessing/satp1/preprocessing_config_20250108_sat-iso_init.yaml
/home/USERNAME/repos/iso-sat/v1/preprocessing/satp1/preprocessing_config_20250108_sat-iso_proc.yaml
```

**v2 configs:**
```
/home/USERNAME/repos/iso-sat/v2/preprocessing/satp1/preprocessing_config_20250507_isov2_init.yaml
/home/USERNAME/repos/iso-sat/v2/preprocessing/satp1/preprocessing_config_20250507_isov2_proc.yaml
```

**v3 configs:**
```
/home/USERNAME/repos/iso-sat/v3/preprocessing/satp1/preprocessing_config_20250801_init.yaml
/home/USERNAME/repos/iso-sat/v3/preprocessing/satp1/preprocessing_config_20250801_proc.yaml
```

## Resource Allocation

### Standard Configuration:
- **Nodes**: 4
- **Tasks**: 224 total (56 per node)
- **CPUs per task**: 2
- **Memory**: Default allocation should be sufficient

### Time Limits:
- **SATp1**: 45 minutes (typically completes in 20-30 minutes)
- **SATp3**: 75 minutes (typically takes longer due to more data)
- **Table generation**: Run directly, takes seconds to minutes

### Scaling:
If you need to process more/fewer observations:
- Increase `--nodes` and `--ntasks` proportionally for more data
- Decrease for smaller datasets to avoid wasting resources
- Keep `--cpus-per-task=2` for optimal performance

## Output Structure

The scripts create organized output directories:

```
/scratch/gpfs/SIMONSOBS/users/USERNAME/generalized_cuts/
├── satp1/
│   ├── generalized_cuts_satp1_20250807.sqlite
│   ├── generalized_cuts_errlog_satp1_20250807.txt
│   ├── log_generalized_cuts_satp1_20250807.log
│   └── tables_20250807/
│       ├── detector_cuts_summary.csv
│       └── sample_cuts_summary.csv
└── satp3/
    ├── generalized_cuts_satp3_20250807.sqlite
    ├── generalized_cuts_errlog_satp3_20250807.txt
    ├── log_generalized_cuts_satp3_20250807.log
    └── tables_20250807/
        ├── detector_cuts_summary.csv
        └── sample_cuts_summary.csv
```

## Monitoring Jobs

### Check job status:
```bash
squeue -u USERNAME
```

### View job details:
```bash
scontrol show job JOBID
```

### Check log files:
```bash
tail -f /scratch/gpfs/SIMONSOBS/users/USERNAME/generalized_cuts/satp1/log_*.log
```

### Cancel a job:
```bash
scancel JOBID
```

## Troubleshooting

### Common Issues:

1. **Configuration file not found**: Update paths in the script
2. **Permission denied**: Check output directory permissions
3. **Out of memory**: Reduce number of tasks or increase memory request
4. **Time limit exceeded**: Increase `--time` parameter

### Error Logs:
Check these files for detailed error information:
- Main log: `log_generalized_cuts_*.log`
- Error log: `generalized_cuts_errlog_*.txt`
- SLURM output: `slurm-JOBID.out`

### Performance Tips:

1. **For large datasets**: Use more nodes/tasks
2. **For small test runs**: Reduce to 1 node with fewer tasks
3. **For debugging**: Add `--verbose` flag to the Python command
4. **For memory issues**: Add `--mem=XG` to SBATCH directives

## Example Workflow

1. **Run the analysis:**
```bash
sbatch run_generalized_cuts_satp1.slurm
```

2. **Check progress:**
```bash
squeue -u USERNAME
tail -f /scratch/gpfs/SIMONSOBS/users/USERNAME/generalized_cuts/satp1/log_*.log
```

3. **After completion, generate tables:**
```bash
# Edit generate_tables.sh to point to your database and set timestamp range
./generate_tables.sh
```

4. **View results:**
```bash
ls -la /scratch/gpfs/SIMONSOBS/users/USERNAME/generalized_cuts/satp1/tables_*/
cat /scratch/gpfs/SIMONSOBS/users/USERNAME/generalized_cuts/satp1/tables_*/detector_cuts_summary.csv
```
