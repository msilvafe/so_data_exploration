#!/bin/bash
#SBATCH --job-name=iso_v4_satp3
#SBATCH --nodes=18
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --account=simonsobs
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maximiliano.silva-feaver@yale.edu

# Run iso-v4 processing for SATp3 across multiple nodes on Tiger
# Full observatory preprocessing for all SATp3 observations with batched database commits

set -e

# Source bashrc to get environment
source ~/.bashrc

# Load necessary modules
module load openmpi/gcc

# Set OpenMP threads to match cpus-per-task to prevent oversubscription
export OMP_NUM_THREADS=2

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="/scratch/gpfs/SIMONSOBS/users/ms3067/iso_stats/v4_tests/satp3"
DB_DIR="${OUTPUT_BASE}/databases"
AGG_DIR="${OUTPUT_BASE}/aggregation"
LOG_DIR="${OUTPUT_BASE}/logs"

# Config and observation list paths
INIT_CONFIG="preprocessing/satp3/preprocessing_config_20251216_init.yaml"
PROC_CONFIG="preprocessing/satp3/preprocessing_config_20251216_proc.yaml"
OBS_LIST="preprocessing/satp3/listv4.txt"
DB_PATH="${DB_DIR}/iso_v4_satp3.sqlite"

# Create output directories
mkdir -p "${DB_DIR}" "${AGG_DIR}" "${LOG_DIR}"

echo "=========================================="
echo "ISO-V4 Processing for SATp3"
echo "=========================================="
echo "Output base: ${OUTPUT_BASE}"
echo "Database: ${DB_PATH}"
echo "Aggregation: ${AGG_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}, Tasks: ${SLURM_NTASKS}, CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "=========================================="

# Start processing with MPI
echo "[$(date)] Starting MPI-based parallel processing..."

mpirun -np ${SLURM_NTASKS} python3 "${SCRIPT_DIR}/track_cuts_and_stats_parallel.py" \
    --init-config "${INIT_CONFIG}" \
    --proc-config "${PROC_CONFIG}" \
    --obs-list "${OBS_LIST}" \
    --db-path "${DB_PATH}" \
    --batch-size 100 \
    --verbosity 1 \
    --aggregate-noise T \
    --aggregate-output "${AGG_DIR}/noise_aggregation" \
    2>&1 | tee "${LOG_DIR}/processing_${SLURM_JOB_ID}.log"

PROCESS_STATUS=$?

if [ $PROCESS_STATUS -eq 0 ]; then
    echo "[$(date)] Main processing completed successfully"
    
    # Post-processing: combine aggregation files by band
    echo "[$(date)] Starting post-processing..."
    python3 "${SCRIPT_DIR}/combine_aggregation_files.py" \
        --aggregation-dir "${AGG_DIR}" \
        --output-dir "${AGG_DIR}" \
        --bands f090,f150 \
        2>&1 | tee -a "${LOG_DIR}/processing_${SLURM_JOB_ID}.log"
    
    echo "[$(date)] Post-processing completed"
else
    echo "[$(date)] Main processing failed with status $PROCESS_STATUS"
    exit $PROCESS_STATUS
fi

# Final summary
echo "=========================================="
echo "ISO-V4 Processing Complete"
echo "=========================================="
echo "Database: ${DB_PATH}"
ls -lh "${DB_PATH}"
echo ""
echo "Aggregation files:"
ls -lh "${AGG_DIR}"/*.h5 2>/dev/null || echo "No HDF5 files generated"
echo ""
echo "Log: ${LOG_DIR}/processing_${SLURM_JOB_ID}.log"
echo "[$(date)] Job finished"
echo "=========================================="
