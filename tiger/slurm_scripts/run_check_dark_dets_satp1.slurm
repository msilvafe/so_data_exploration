#!/bin/bash -l

#SBATCH --account=simonsobs
#SBATCH --nodes=4
#SBATCH --ntasks=224
#SBATCH --cpus-per-task=2
#SBATCH --time=16:00:00
#SBATCH --job-name=SATp1-dark-dets
#SBATCH --mail-user maximiliano.silva-feaver@yale.edu
#SBATCH --mail-type all

set -e

export SOTODLIB_RESOURCES='{"de421.bsp": "file:///scratch/gpfs/SIMONSOBS/planets/de421.bsp"}'

# Log file
log="/scratch/gpfs/SIMONSOBS/users/ms3067/dark_dets/log_dark_dets_check_satp1"

#=========== Compute runtime parameters ============

echo "Using ${NODES} node(s), which have ${NODE_SLOTS} thread slots each."
echo "Starting ${NODE_PROC} process(es) per node (${NPROC} total), each with ${PROC_THREADS} OpenMP threads."

export OMP_NUM_THREADS=2

launch_str="srun -n 224 --cpus-per-task=2 --export=ALL "

#=========== Run it ============

PWG_SCRIPTS_PATH=/home/ms3067/repos/iso-sat/v2

com="${launch_str} python3 /home/ms3067/scripts/check_dark_dets.py \
${PWG_SCRIPTS_PATH}/contexts/satp1/use_this_local_250516.yaml \
/scratch/gpfs/SIMONSOBS/users/ms3067/dark_dets/ \
--nproc 224 \
--query 'type == \"obs\" and subtype == \"cmb\" and start_time > 1741669200 and start_time < 1750862552 and hwp_freq_mean > 0.5 and duration > 600'"

echo ${com}
echo "Launching pipeline at $(date)"
eval ${com} > ${log} 2>&1

echo "Ending batch script at $(date)"

