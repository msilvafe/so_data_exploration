#!/bin/bash -l

#SBATCH --account=simonsobs
#SBATCH --nodes=4
#SBATCH --ntasks=224
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --job-name=SATp3-iso-stats
#SBATCH --mail-user maximiliano.silva-feaver@yale.edu
#SBATCH --mail-type all

set -e

export SOTODLIB_RESOURCES='{"de421.bsp": "file:///scratch/gpfs/SIMONSOBS/planets/de421.bsp"}'

# Log file
log="/scratch/gpfs/SIMONSOBS/users/ms3067/iso_stats/v1/log_iso_stats_satp3_20250718.log"

#=========== Compute runtime parameters ============

echo "Using ${NODES} node(s), which have ${NODE_SLOTS} thread slots each."
echo "Starting ${NODE_PROC} process(es) per node (${NPROC} total), each with ${PROC_THREADS} OpenMP threads."

export OMP_NUM_THREADS=2

launch_str="srun -n 224 --cpus-per-task=2 --export=ALL "

#=========== Run it ============


com="${launch_str} python3 /home/ms3067/scripts/iso_v1_stats_20250718.py \
/home/ms3067/repos/iso-sat/v1/preprocessing/satp3/preprocessing_config_20250108_sat-iso_init.yaml \
/home/ms3067/repos/iso-sat/v1/preprocessing/satp3/preprocessing_config_20250108_sat-iso_proc.yaml \
--noise-range 2e-6 80e-6 \
--nproc 224 \
--savename /scratch/gpfs/SIMONSOBS/users/ms3067/iso_stats/v1/20250718_cut_stats_satp3.h5 \
--errlog-ext /scratch/gpfs/SIMONSOBS/users/ms3067/iso_stats/v1/20250718_errlog_satp3"

echo ${com}
echo "Launching pipeline at $(date)"
eval ${com} > ${log} 2>&1

echo "Ending batch script at $(date)"
