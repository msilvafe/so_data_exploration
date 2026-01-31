#!/bin/bash

# Submission script for cuts and statistics tracking jobs
# Usage: ./submit_cuts_stats_jobs.sh [satellite]
# where satellite is satp1, satp3, or both (default)

SATELLITE=${1:-both}
SCRIPT_DIR="/home/ms3067/so_data_exploration/tiger/202601_v4"

case $SATELLITE in
    satp1)
        echo "Submitting SATP1 cuts and statistics tracking job..."
        sbatch ${SCRIPT_DIR}/run_cuts_stats_satp1.slurm
        ;;
    satp3)
        echo "Submitting SATP3 cuts and statistics tracking job..."
        sbatch ${SCRIPT_DIR}/run_cuts_stats_satp3.slurm
        ;;
    both)
        echo "Submitting both SATP1 and SATP3 cuts and statistics tracking jobs..."
        sbatch ${SCRIPT_DIR}/run_cuts_stats_satp1.slurm
        sbatch ${SCRIPT_DIR}/run_cuts_stats_satp3.slurm
        ;;
    *)
        echo "Usage: $0 [satp1|satp3|both]"
        echo "  satp1: Submit SATP1 job only"
        echo "  satp3: Submit SATP3 job only"
        echo "  both:  Submit both jobs (default)"
        exit 1
        ;;
esac

echo ""
echo "Check job status with: squeue -u $USER"
echo "Monitor logs in: /scratch/gpfs/SIMONSOBS/sat-iso/v4/preprocessing/"