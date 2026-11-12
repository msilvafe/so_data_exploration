#!/bin/bash
#
# Quick reference submission script for ISO-V4 full pipeline
#
# Usage: ./submit_pipeline.sh <init_config> <proc_config> <obs_list> [sat_tag]
#
# Example:
#   ./submit_pipeline.sh \
#     /path/to/preprocessing_config_20260720_init.yaml \
#     /path/to/preprocessing_config_20260720_proc.yaml \
#     /path/to/listv4.txt \
#     satp1

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <init_config> <proc_config> <obs_list> [sat_tag]"
    echo ""
    echo "Example:"
    echo "  $0 init.yaml proc.yaml obs_list.txt satp1"
    echo ""
    echo "If sat_tag is omitted, it will be auto-detected from config paths."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/run_iso_v4_full_pipeline.slurm"

if [[ ! -f "${SLURM_SCRIPT}" ]]; then
    echo "ERROR: SLURM script not found: ${SLURM_SCRIPT}"
    exit 1
fi

echo "=========================================="
echo "ISO-V4 Full Pipeline Job Submission"
echo "=========================================="
echo ""
echo "SLURM script: ${SLURM_SCRIPT}"
echo "Init config: $1"
echo "Proc config: $2"
echo "Obs list:    $3"
echo "SAT tag:     ${4:-(auto-detect)}"
echo ""

# Submit the job
sbatch "${SLURM_SCRIPT}" "$@"

echo ""
echo "Job submitted! Check status with: squeue -u $(whoami) -l"
echo "=========================================="
