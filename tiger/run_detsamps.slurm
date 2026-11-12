#!/bin/bash -l

#SBATCH --account=simonsobs
#SBATCH --nodes=4
#SBATCH --ntasks=224
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --job-name=SATp1-getsamps
#SBATCH --mail-user maximiliano.silva-feaver@yale.edu
#SBATCH --mail-type all

set -e

export SOTODLIB_RESOURCES='{"de421.bsp": "file:///scratch/gpfs/SIMONSOBS/planets/de421.bsp"}'

# Log file
log="/home/ms3067/repos/iso-sat/v4/log_getsamps_satp1"

#=========== Compute runtime parameters ============

echo "Using ${NODES} node(s), which have ${NODE_SLOTS} thread slots each."
echo "Starting ${NODE_PROC} process(es) per node (${NPROC} total), each with ${PROC_THREADS} OpenMP threads."

export OMP_NUM_THREADS=2

launch_str="srun --cpus-per-task=2 --export=ALL "

#=========== Run it ============

SOTODLIB_PATH=/home/ms3067/repos/sotodlib
PWG_SCRIPTS_PATH=.

com="${launch_str} python3 /home/ms3067/so_data_exploration/tiger/check_detsamps.py \
${PWG_SCRIPTS_PATH}/contexts/satp1/use_this_local_260115.yaml \
--obslist ${PWG_SCRIPTS_PATH}/preprocessing/satp1/listv4.txt
--nproc 224"

echo ${com}
echo "Launching pipeline at $(date)"
eval ${com} > ${log} 2>&1

echo "Ending batch script at $(date)"
