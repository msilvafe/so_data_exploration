#!/bin/bash -l
#SBATCH --job-name=cuts_stats_satp1
#SBATCH --account=simonsobs
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=224
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=compute
#SBATCH --output=/home/ms3067/so_data_exploration/tiger/202601_v4/logs/cuts_stats_satp1_%j.out
#SBATCH --error=/home/ms3067/so_data_exploration/tiger/202601_v4/logs/cuts_stats_satp1_%j.err

set -e

# Create logs directory if it doesn't exist
mkdir -p /home/ms3067/so_data_exploration/tiger/202601_v4/logs

# Set environment variables
export OMP_NUM_THREADS=1

# Change to correct working directory where contexts are defined
cd /home/ms3067/repos/iso-sat/v4

# Set output database path
DB_PATH="/scratch/gpfs/SIMONSOBS/sat-iso/v4/cuts_stats/cuts_stats_satp1.db"
mkdir -p $(dirname $DB_PATH)

# Configuration files
INIT_CONFIG="./preprocessing/satp1/preprocessing_config_20251216_init.yaml"
PROC_CONFIG="./preprocessing/satp1/preprocessing_config_20251216_proc.yaml"

# Run with simple parallel distribution across tasks
srun python /home/ms3067/so_data_exploration/tiger/202601_v4/track_cuts_and_stats_simple.py \
    --init-config $INIT_CONFIG \
    --proc-config $PROC_CONFIG \
    --db-path $DB_PATH \
    --obs-list ./preprocessing/satp1/listv4.txt \
    --verbosity 1