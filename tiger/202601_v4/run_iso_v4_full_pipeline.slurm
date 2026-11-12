#!/bin/bash
#SBATCH --job-name=iso_v4_full
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --account=simonsobs
#SBATCH --reservation=simonsobs
#SBATCH --mail-type all
#SBATCH --mail-user maximiliano.silva-feaver@yale.edu

# Full ISO-V4 Pipeline: Generate sqlite+hdf5 from scratch, then create analysis plots+PDF
# Usage: sbatch run_iso_v4_full_pipeline.slurm <init_config> <proc_config> [obs_list] [sat_tag]
#
# Example (explicit obs list):
#   sbatch run_iso_v4_full_pipeline.slurm \
#     /path/to/preprocessing_config_20260720_init.yaml \
#     /path/to/preprocessing_config_20260720_proc.yaml \
#     /path/to/listv4.txt \
#     satp1
#
# Example (all observations in the context/obsdb referenced by the configs):
#   sbatch run_iso_v4_full_pipeline.slurm \
#     /path/to/preprocessing_config_20260720_init.yaml \
#     /path/to/preprocessing_config_20260720_proc.yaml \
#     satp1
#
# obs_list is optional. If the 3rd argument is an existing file it is treated as
# an obs_list; otherwise it (and any 4th argument) is treated as sat_tag, and
# ALL observations in the config's context/obsdb are processed.
#
# If sat_tag is not provided, it will be derived from the config file paths or defaults to 'sat'

set -e  # Exit on first error
set -o pipefail  # Make `cmd | tee ...` fail if cmd fails, not just tee

# Force the soconda env's python3 (with sotodlib/yaml/etc.) first on PATH.
# Needed because --export=ALL forwards whatever PATH the submitting shell had,
# which can be polluted by an unrelated project virtualenv shadowing python3.
export PATH="/scratch/gpfs/SIMONSOBS/env/v0.6.18/soconda_v0.6.18/bin:${PATH}"

# Set OpenMP threads to match cpus-per-task to prevent oversubscription
export OMP_NUM_THREADS=1

# ============================================================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================================================

INIT_CONFIG="${1:-}"
PROC_CONFIG="${2:-}"
ARG3="${3:-}"
ARG4="${4:-}"

# 3rd arg is obs_list if it's an existing file, otherwise it's sat_tag and
# obs_list is left empty (meaning: process all observations in the context/obsdb)
if [[ -n "${ARG3}" ]] && [[ -f "${ARG3}" ]]; then
    OBS_LIST="${ARG3}"
    SAT_TAG="${ARG4}"
else
    OBS_LIST=""
    SAT_TAG="${ARG3}"
fi

# Validate required arguments
if [[ -z "${INIT_CONFIG}" ]] || [[ -z "${PROC_CONFIG}" ]]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: sbatch run_iso_v4_full_pipeline.slurm <init_config> <proc_config> [obs_list] [sat_tag]"
    exit 1
fi

# Verify config files exist
if [[ ! -f "${INIT_CONFIG}" ]]; then
    echo "ERROR: Init config not found: ${INIT_CONFIG}"
    exit 1
fi

if [[ ! -f "${PROC_CONFIG}" ]]; then
    echo "ERROR: Proc config not found: ${PROC_CONFIG}"
    exit 1
fi

if [[ -n "${OBS_LIST}" ]]; then
    echo "[$(date)] Using observation list: ${OBS_LIST}"
else
    echo "[$(date)] No observation list provided - processing ALL observations in the context/obsdb"
fi

# ============================================================================
# EXTRACT METADATA FROM CONFIG FILES
# ============================================================================

# Derive version number from proc config name, e.g. preprocessing_config_20260417_proc.yaml -> 20260417
PROC_BASENAME="$(basename "${PROC_CONFIG}")"
DATE_TAG="${PROC_BASENAME#preprocessing_config_}"
DATE_TAG="${DATE_TAG%_proc.yaml}"

# Auto-detect SAT_TAG if not provided
if [[ -z "${SAT_TAG}" ]]; then
    # Try to extract from config file paths
    INIT_DIR="$(dirname "${INIT_CONFIG}")"
    if [[ "${INIT_DIR}" == *"satp1"* ]] || [[ "${INIT_DIR}" == *"SATp1"* ]]; then
        SAT_TAG="satp1"
    elif [[ "${INIT_DIR}" == *"satp3"* ]] || [[ "${INIT_DIR}" == *"SATp3"* ]]; then
        SAT_TAG="satp3"
    else
        echo "WARNING: Could not auto-detect SAT_TAG from config path: ${INIT_DIR}"
        echo "         Deriving from config file contents..."
        # Try to extract from config file
        if grep -q "satp1\|SATp1" "${INIT_CONFIG}" 2>/dev/null; then
            SAT_TAG="satp1"
        elif grep -q "satp3\|SATp3" "${INIT_CONFIG}" 2>/dev/null; then
            SAT_TAG="satp3"
        else
            SAT_TAG="sat"  # Default fallback
            echo "WARNING: Using default SAT_TAG='sat'"
        fi
    fi
fi

VERSION_NUMBER="${SAT_TAG}_${DATE_TAG}"

# ============================================================================
# SET UP PATHS
# ============================================================================

BASE_DIR="$(cd "$(dirname "${INIT_CONFIG}")/../.." && pwd)"

# Determine script directory - handle SLURM working directory issue
# When run via sbatch, BASH_SOURCE might point to temp directory
# So we also check common locations
if [[ -f "$(dirname "${BASH_SOURCE[0]}")/track_cuts_and_stats_parallel.py" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
elif [[ -f "/home/ms3067/so_data_exploration/tiger/202601_v4/track_cuts_and_stats_parallel.py" ]]; then
    SCRIPT_DIR="/home/ms3067/so_data_exploration/tiger/202601_v4"
else
    echo "ERROR: Could not find script directory. Tried:"
    echo "  1. $(dirname "${BASH_SOURCE[0]}")"
    echo "  2. /home/ms3067/so_data_exploration/tiger/202601_v4"
    exit 1
fi

TRACK_SCRIPT="${SCRIPT_DIR}/track_cuts_and_stats_parallel.py"
REPORT_WRAPPER="${SCRIPT_DIR}/generate_report_wrapper.py"

OUTPUT_BASE="/scratch/gpfs/SIMONSOBS/sat-iso/v4/cuts_stats/${VERSION_NUMBER}"
DB_DIR="${OUTPUT_BASE}/databases"
AGG_DIR="${OUTPUT_BASE}/aggregation"
LOG_DIR="${OUTPUT_BASE}/logs"
ANALYSIS_DIR="${OUTPUT_BASE}/analysis"

DB_PATH="${DB_DIR}/iso_v4_${VERSION_NUMBER}.sqlite"
HDF5_OUTPUT="${AGG_DIR}/noise_aggregation"
REPORT_OUTPUT="${ANALYSIS_DIR}/iso_v4_${VERSION_NUMBER}_report.pdf"

# ============================================================================
# CREATE DIRECTORY STRUCTURE
# ============================================================================

echo "[$(date)] Creating output directory structure..."
mkdir -p "${DB_DIR}" "${AGG_DIR}" "${LOG_DIR}" "${ANALYSIS_DIR}"

if [[ ! -d "${DB_DIR}" ]]; then
    echo "ERROR: Failed to create database directory: ${DB_DIR}"
    exit 1
fi

# ============================================================================
# VERIFY SCRIPTS EXIST
# ============================================================================

if [[ ! -f "${TRACK_SCRIPT}" ]]; then
    echo "ERROR: track_cuts_and_stats_parallel.py not found: ${TRACK_SCRIPT}"
    exit 1
fi

if [[ ! -f "${REPORT_WRAPPER}" ]]; then
    echo "ERROR: generate_report_wrapper.py not found: ${REPORT_WRAPPER}"
    exit 1
fi

# ============================================================================
# CHANGE TO BASE DIRECTORY
# ============================================================================

if ! cd "${BASE_DIR}"; then
    echo "ERROR: Failed to cd to base directory: ${BASE_DIR}"
    exit 1
fi

# ============================================================================
# PRINT JOB CONFIGURATION
# ============================================================================

echo "=========================================="
echo "ISO-V4 FULL PIPELINE"
echo "=========================================="
echo "Job ID:                ${SLURM_JOB_ID}"
echo "SAT Tag:               ${SAT_TAG}"
echo "Date Tag:              ${DATE_TAG}"
echo "Version Number:        ${VERSION_NUMBER}"
echo ""
echo "Inputs:"
echo "  Init Config:         ${INIT_CONFIG}"
echo "  Proc Config:         ${PROC_CONFIG}"
echo "  Observation List:    ${OBS_LIST:-<none - processing all observations in context/obsdb>}"
echo ""
echo "Scripts:"
echo "  Track Script:        ${TRACK_SCRIPT}"
echo "  Report Script:       ${REPORT_SCRIPT}"
echo ""
echo "Base Directory:        ${BASE_DIR}"
echo "Output Base:           ${OUTPUT_BASE}"
echo ""
echo "Outputs:"
echo "  Database Dir:        ${DB_DIR}"
echo "  Aggregation Dir:     ${AGG_DIR}"
echo "  Analysis Dir:        ${ANALYSIS_DIR}"
echo "  Log Dir:             ${LOG_DIR}"
echo ""
echo "  Database Path:       ${DB_PATH}"
echo "  HDF5 Output:         ${HDF5_OUTPUT}_*.h5"
echo "  Report PDF:          ${REPORT_OUTPUT}"
echo ""
echo "Compute:"
echo "  Nodes:               ${SLURM_NNODES}"
echo "  Tasks:               ${SLURM_NTASKS}"
echo "  CPUs per task:       ${SLURM_CPUS_PER_TASK}"
echo "  Time limit:          ${SLURM_TIME} minutes"
echo "=========================================="

# ============================================================================
# PHASE 1: GENERATE SQLITE + HDF5 FILES
# ============================================================================

echo ""
echo "[$(date)] ========== PHASE 1: DATABASE & AGGREGATION GENERATION =========="
echo "[$(date)] Starting parallel processing with get_exec_env..."
echo "[$(date)] Allocated: ${SLURM_NNODES} node(s), ${SLURM_NTASKS} task(s)"

LAUNCH_STR="srun -n ${SLURM_NTASKS} --cpus-per-task=${SLURM_CPUS_PER_TASK} --export=ALL"

# Only pass --obs-list when one was provided; otherwise pass --query "1" so
# sp_util.get_obslist() matches every observation in the context/obsdb.
# (Passing no query at all crashes get_obslist in the installed sotodlib: it does
# `open(query, "r")` unconditionally, which raises TypeError -- not the caught
# FileNotFoundError -- when query is None.)
# Set QUERY_FILTER to override the default "1" (=match everything) fallback,
# e.g. QUERY_FILTER='type=="obs" and subtype=="cmb" and start_time>1716177000'
OBS_LIST_ARGS=()
if [[ -n "${OBS_LIST}" ]]; then
    OBS_LIST_ARGS=(--obs-list "${OBS_LIST}")
else
    OBS_LIST_ARGS=(--query "${QUERY_FILTER:-1}")
fi

# Set OVERWRITE=1 to pass --overwrite through to track_cuts_and_stats_parallel.py
# (forces reprocessing of obs_ids already present in the sqlite db -- needed
# e.g. to regenerate the noise_aggregation HDF5 files, which are truncated and
# rewritten from scratch each run using only *this* run's processed set, unlike
# the sqlite db which accumulates incrementally).
OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
    OVERWRITE_ARGS=(--overwrite)
fi

# Disable errexit around this pipeline so a failure falls through to the
# explicit status check below (with `set -e` alone, a failing piped command
# that isn't in a tested context aborts the script immediately, skipping our
# own error message).
set +e
${LAUNCH_STR} python3 "${TRACK_SCRIPT}" \
    --init-config "${INIT_CONFIG}" \
    --proc-config "${PROC_CONFIG}" \
    "${OBS_LIST_ARGS[@]}" \
    "${OVERWRITE_ARGS[@]}" \
    --db-path "${DB_PATH}" \
    --batch-size 100 \
    --verbosity 1 \
    --nprocs ${SLURM_NTASKS} \
    --aggregate-noise Q \
    --aggregate-output "${HDF5_OUTPUT}" \
    2>&1 | tee "${LOG_DIR}/phase1_tracking_${SLURM_JOB_ID}.log"

PHASE1_STATUS=$?
set -e

if [ $PHASE1_STATUS -ne 0 ]; then
    echo "[$(date)] ERROR: Phase 1 (database generation) failed with status $PHASE1_STATUS"
    exit $PHASE1_STATUS
fi

echo "[$(date)] Phase 1 completed successfully"

# Verify critical output files exist
if [[ ! -f "${DB_PATH}" ]]; then
    echo "ERROR: Database file not created: ${DB_PATH}"
    exit 1
fi

echo "[$(date)] Database created: $(ls -lh "${DB_PATH}" | awk '{print $5, $9}')"

# ============================================================================
# PHASE 2: GENERATE ANALYSIS PLOTS AND PDF REPORT
# ============================================================================

echo ""
echo "[$(date)] ========== PHASE 2: ANALYSIS & REPORT GENERATION =========="
echo "[$(date)] Generating analysis plots and PDF report..."
echo "[$(date)] Using database: ${DB_PATH}"

# Run report generation (use single process for this phase).
# This phase is allowed to fail without killing the job (see below), so disable
# errexit around it too -- otherwise `set -e` would abort here before we ever
# reach the "continue anyway" logic.
set +e
python3 "${REPORT_WRAPPER}" \
    --db-path "${DB_PATH}" \
    --hdf5-dir "${AGG_DIR}" \
    --output-dir "${ANALYSIS_DIR}" \
    --version-number "${VERSION_NUMBER}" \
    --pdf-output "${REPORT_OUTPUT}" \
    2>&1 | tee "${LOG_DIR}/phase2_report_${SLURM_JOB_ID}.log"

PHASE2_STATUS=$?
set -e

if [ $PHASE2_STATUS -ne 0 ]; then
    echo "[$(date)] WARNING: Phase 2 (report generation) exited with status $PHASE2_STATUS"
    echo "[$(date)] Continuing despite report generation issues..."
    # Don't exit here - the database was successfully created
fi

# ============================================================================
# PHASE 3: VERIFY AND SUMMARIZE OUTPUTS
# ============================================================================

echo ""
echo "[$(date)] ========== PHASE 3: OUTPUT VERIFICATION =========="

# Check database
if [[ -f "${DB_PATH}" ]]; then
    DB_SIZE=$(ls -lh "${DB_PATH}" | awk '{print $5}')
    DB_ROWS=$(sqlite3 "${DB_PATH}" "SELECT COUNT(*) FROM obs;" 2>/dev/null || echo "unknown")
    echo "[$(date)] ✓ Database: ${DB_PATH} (${DB_SIZE}, ${DB_ROWS} rows)"
else
    echo "[$(date)] ✗ Database: NOT FOUND"
fi

# Check aggregation files
echo "[$(date)] Aggregation files:"
if ls "${AGG_DIR}"/*.h5 >/dev/null 2>&1; then
    ls -lh "${AGG_DIR}"/*.h5 | awk '{print "       ", $5, $9}'
else
    echo "         No HDF5 files found"
fi

# Check analysis files
echo "[$(date)] Analysis files:"
if ls "${ANALYSIS_DIR}"/* >/dev/null 2>&1; then
    ls -lh "${ANALYSIS_DIR}" | tail -n +2 | awk '{print "       ", $5, $9}'
else
    echo "         No analysis files found"
fi

# Check PDF report
if [[ -f "${REPORT_OUTPUT}" ]]; then
    PDF_SIZE=$(ls -lh "${REPORT_OUTPUT}" | awk '{print $5}')
    echo "[$(date)] ✓ PDF Report: ${REPORT_OUTPUT} (${PDF_SIZE})"
else
    echo "[$(date)] ✗ PDF Report: NOT FOUND (may have failed to generate)"
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo "=========================================="
echo "ISO-V4 FULL PIPELINE COMPLETED"
echo "=========================================="
echo "Summary:"
echo "  Version: ${VERSION_NUMBER}"
echo "  Output:  ${OUTPUT_BASE}"
echo ""
echo "Output structure:"
echo "  ${OUTPUT_BASE}/"
echo "  ├── databases/"
echo "  │   └── iso_v4_${VERSION_NUMBER}.sqlite"
echo "  ├── aggregation/"
echo "  │   ├── noise_aggregation_*.h5"
echo "  │   └── [other intermediate files]"
echo "  ├── analysis/"
echo "  │   ├── iso_v4_${VERSION_NUMBER}_report.pdf"
echo "  │   ├── [analysis plots in various formats]"
echo "  │   └── [analysis metadata]"
echo "  └── logs/"
echo "      ├── phase1_tracking_${SLURM_JOB_ID}.log"
echo "      └── phase2_report_${SLURM_JOB_ID}.log"
echo ""
echo "To view the PDF report:"
echo "  open ${REPORT_OUTPUT}"
echo "  # or on a remote system:"
echo "  # scp tiger:${REPORT_OUTPUT} ."
echo ""
echo "Database query example:"
echo "  sqlite3 ${DB_PATH} \"SELECT COUNT(*) FROM obs WHERE success=1;\""
echo ""
echo "Timestamps:"
echo "  Job started: ${SLURM_JOB_START_TIME:-unknown}"
echo "  Job ended:   $(date)"
echo "=========================================="

exit 0
