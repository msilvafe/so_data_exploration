#!/bin/bash
#SBATCH --job-name=iso_v4_satp3
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
#SBATCH --time=01:15:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --account=simonsobs
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maximiliano.silva-feaver@yale.edu

# Run iso-v4 processing for SATp3 using sotodlib get_exec_env (MPI/process-pool)
# Full observatory preprocessing for all SATp3 observations with batched database commits

# Set OpenMP threads to match cpus-per-task to prevent oversubscription
export OMP_NUM_THREADS=1

# Configuration
BASE_DIR="${HOME}/repos/iso-sat/v4"
SCRIPT_PATH="/home/ms3067/so_data_exploration/tiger/202601_v4/track_cuts_and_stats_parallel.py"
SAT_TAG="satp3"

# Config and observation list paths
INIT_CONFIG="${BASE_DIR}/preprocessing/satp3/preprocessing_config_20260417_init.yaml"
PROC_CONFIG="${BASE_DIR}/preprocessing/satp3/preprocessing_config_20260417_proc.yaml"
OBS_LIST="${BASE_DIR}/preprocessing/satp3/listv4.txt"

# Derive version number from proc config name, e.g. preprocessing_config_20260417_proc.yaml -> satp3_20260417
PROC_BASENAME="$(basename "${PROC_CONFIG}")"
DATE_TAG="${PROC_BASENAME#preprocessing_config_}"
DATE_TAG="${DATE_TAG%_proc.yaml}"
VERSION_NUMBER="${SAT_TAG}_${DATE_TAG}"

OUTPUT_BASE="/scratch/gpfs/SIMONSOBS/sat-iso/v4/cuts_stats/${VERSION_NUMBER}"
DB_DIR="${OUTPUT_BASE}/databases"
AGG_DIR="${OUTPUT_BASE}/aggregation"
LOG_DIR="${OUTPUT_BASE}/logs"
DB_PATH="${DB_DIR}/iso_v4_${VERSION_NUMBER}.sqlite"

# Create output directories
mkdir -p "${DB_DIR}" "${AGG_DIR}" "${LOG_DIR}"

if ! cd "${BASE_DIR}"; then
    echo "[$(date)] Failed to cd to base directory: ${BASE_DIR}"
    exit 1
fi

echo "=========================================="
echo "ISO-V4 Processing for SATp3"
echo "=========================================="
echo "Version number: ${VERSION_NUMBER}"
echo "Run base directory: ${BASE_DIR}"
echo "Output base: ${OUTPUT_BASE}"
echo "Database: ${DB_PATH}"
echo "Aggregation: ${AGG_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}, Tasks: ${SLURM_NTASKS}, CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "=========================================="

# Start processing
echo "[$(date)] Starting get_exec_env-based parallel processing..."
echo "Allocated: ${SLURM_NNODES} node(s), ${SLURM_CPUS_ON_NODE} CPUs"

LAUNCH_STR="srun -n ${SLURM_NTASKS} --cpus-per-task=${SLURM_CPUS_PER_TASK} --export=ALL"

echo "Launch string: ${LAUNCH_STR}"

${LAUNCH_STR} python3 "${SCRIPT_PATH}" \
    --init-config "${INIT_CONFIG}" \
    --proc-config "${PROC_CONFIG}" \
    --obs-list "${OBS_LIST}" \
    --db-path "${DB_PATH}" \
    --batch-size 100 \
    --verbosity 1 \
    --nprocs ${SLURM_NTASKS} \
    --aggregate-noise Q \
    --aggregate-output "${AGG_DIR}/noise_aggregation" \
    2>&1 | tee "${LOG_DIR}/processing_${SLURM_JOB_ID}.log"

PROCESS_STATUS=$?

if [ $PROCESS_STATUS -ne 0 ]; then
    echo "[$(date)] Main processing failed with status $PROCESS_STATUS"
    exit $PROCESS_STATUS
fi

echo "[$(date)] Processing completed successfully"

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
