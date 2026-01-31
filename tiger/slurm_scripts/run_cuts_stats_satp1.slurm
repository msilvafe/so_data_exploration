#!/bin/bash -l

#SBATCH --account=simonsobs
#SBATCH --nodes=4
#SBATCH --ntasks=224  
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --job-name=SATp1-cuts-stats
#SBATCH --mail-user maximiliano.silva-feaver@yale.edu
#SBATCH --mail-type all

set -e

export SOTODLIB_RESOURCES='{"de421.bsp": "file:///scratch/gpfs/SIMONSOBS/planets/de421.bsp"}'

# Paths and configuration
SATELLITE="satp1"
BASE_DIR="/scratch/gpfs/SIMONSOBS/sat-iso/v4/preprocessing"
CONFIG_DIR="/home/ms3067/repos/iso-sat/v4/preprocessing"
SCRIPT_DIR="/home/ms3067/so_data_exploration/tiger/202601_v4"

# Configuration files
CONFIG_INIT="${CONFIG_DIR}/${SATELLITE}/preprocessing_config_20251216_init.yaml"
CONFIG_PROC="${CONFIG_DIR}/${SATELLITE}/preprocessing_config_20251216_proc.yaml"

# Output database
DB_PATH="${BASE_DIR}/${SATELLITE}_20251216_cuts_and_stats.db"

# Log file
LOG_FILE="${BASE_DIR}/log_cuts_stats_${SATELLITE}_$(date +%Y%m%d_%H%M%S)"

#=========== Compute runtime parameters ============

echo "Using 4 node(s) with 224 thread slots total."
echo "Starting 224 process(es) total, each with 2 OpenMP threads."

export OMP_NUM_THREADS=2

launch_str="srun -n 224 --cpus-per-task=2 --export=ALL "

#=========== Run it ============

echo "Configuration files:"
echo "  Init: ${CONFIG_INIT}"
echo "  Proc: ${CONFIG_PROC}"
echo "Output database: ${DB_PATH}"
echo "Log file: ${LOG_FILE}"

# Check that config files exist
if [[ ! -f "$CONFIG_INIT" ]]; then
    echo "Error: Init config file not found: $CONFIG_INIT"
    exit 1
fi

if [[ ! -f "$CONFIG_PROC" ]]; then
    echo "Error: Proc config file not found: $CONFIG_PROC"
    exit 1
fi

# Run the cuts and statistics tracking
com="${launch_str} python3 ${SCRIPT_DIR}/track_cuts_and_stats.py \
${CONFIG_INIT} \
${CONFIG_PROC} \
${DB_PATH} \
--nproc 224 \
--verbosity 2 \
--query 'type == \"obs\" and subtype == \"cmb\" and start_time > 1741669200 and start_time < 1750862552 and hwp_freq_mean > 0.5 and duration > 600'"

echo "Command to run:"
echo ${com}
echo ""
echo "Launching cuts and statistics tracking at $(date)"
eval ${com} > ${LOG_FILE} 2>&1

echo "Cuts and statistics tracking completed at $(date)"
echo "Results stored in: ${DB_PATH}"
echo "Log file: ${LOG_FILE}"

# Run failure analysis if jobs database exists
JOBS_DB="${BASE_DIR}/${SATELLITE}_20251216_jobs.db"
if [[ -f "$JOBS_DB" ]]; then
    echo ""
    echo "Running failure analysis at $(date)"
    FAILURE_LOG="${BASE_DIR}/log_failure_analysis_${SATELLITE}_$(date +%Y%m%d_%H%M%S)"
    
    python3 ${SCRIPT_DIR}/analyze_job_failures.py \
        "$JOBS_DB" \
        --cuts-db "$DB_PATH" \
        --output-csv "${BASE_DIR}/${SATELLITE}_20251216_failures.csv" \
        > ${FAILURE_LOG} 2>&1
    
    echo "Failure analysis completed. Log: ${FAILURE_LOG}"
else
    echo "Warning: Jobs database not found: $JOBS_DB"
fi

echo "Ending batch script at $(date)"