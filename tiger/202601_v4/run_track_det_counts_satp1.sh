#!/bin/bash -l

#SBATCH --account=simonsobs
#SBATCH --nodes=4
#SBATCH --ntasks=224
#SBATCH --cpus-per-task=2
#SBATCH --time=16:00:00
#SBATCH --job-name=SATp1-track-det-counts
#SBATCH --mail-user maximiliano.silva-feaver@yale.edu
#SBATCH --mail-type all

set -e

export SOTODLIB_RESOURCES='{"de421.bsp": "file:///scratch/gpfs/SIMONSOBS/planets/de421.bsp"}'

# Log file
log="/scratch/gpfs/SIMONSOBS/users/ms3067/iso_stats/202601_v4/track_det_counts_satp1/log_track_det_counts_satp1"

#=========== Compute runtime parameters ============

echo "Using ${NODES} node(s), which have ${NODE_SLOTS} thread slots each."
echo "Starting ${NODE_PROC} process(es) per node (${NPROC} total), each with ${PROC_THREADS} OpenMP threads."

export OMP_NUM_THREADS=2

launch_str="srun -n 224 --cpus-per-task=2 --export=ALL "

#=========== Run it ============

PWG_SCRIPTS_PATH=/home/ms3067/repos/iso-sat

# Output database path
DB_PATH="/scratch/gpfs/SIMONSOBS/users/ms3067/iso_stats/202601_v4/track_det_counts_satp1/det_counts_satp1.db"

# Config files (modify these paths as needed)
CONFIG_INIT="${PWG_SCRIPTS_PATH}/v4/preprocessing/satp1/preprocessing_config_20251216_init.yaml"
CONFIG_PROC="${PWG_SCRIPTS_PATH}/v4/preprocessing/satp1/preprocessing_config_20251216_proc.yaml"

com="${launch_str} python3 /home/ms3067/repos/so_data_exploration/tiger/202601_v4/track_det_counts.py \
${CONFIG_INIT} \
${CONFIG_PROC} \
${DB_PATH} \
--nproc 224 \
--verbosity 2 \
--query ${PWG_SCRIPTS_PATH}/v2/preprocessing/satp1/list200obs_junna.txt"

echo ${com}
echo "Launching detector count tracking at $(date)"
eval ${com} > ${log} 2>&1

echo "Ending batch script at $(date)"