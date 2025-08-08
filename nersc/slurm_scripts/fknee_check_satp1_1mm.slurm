#!/bin/bash -l

#SBATCH --qos=premium
#SBATCH --constraint=cpu
#SBATCH --account=mp107b
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --job-name=preprocess
#SBATCH --mail-user maximiliano.silva-feaver@yale.edu
#SBATCH --mail-type all

set -e

# Number or processes per node
NODE_PROC=128

# Log file
log="/global/homes/m/msilvafe/so_home/shared_files/fknee_check/fknee_log_satp1"

#=========== Compute runtime parameters ============

# Set TMPDIR to be on the ramdisk
export TMPDIR=/dev/shm

# nodes used by this job
NODES=${SLURM_JOB_NUM_NODES}

# set procs and threads
NODE_SLOTS=256
NODE_CORES=128
PROC_THREADS=$(( NODE_CORES / NODE_PROC ))
PROC_DEPTH=$(( NODE_SLOTS / NODE_PROC ))

# total number of processes on all nodes
NPROC=$(( NODES * NODE_PROC - 1))

echo "Using ${NODES} node(s), which have ${NODE_SLOTS} thread slots each."
echo "Starting ${NODE_PROC} process(es) per node (${NPROC} total), each with ${PROC_THREADS} OpenMP threads."

export OMP_NUM_THREADS=${PROC_THREADS}

# Because we are using multiprocessing, we do NOT want to specify any pinning
# of processes / threads to specific cores.  This is because the processes are
# spawned dynamically.  If using MPI we would change to these lines.
#
# export OMP_PROC_BIND=spread
# export OMP_PLACES=threads
#launch_str="srun -n ${NPROC} -N ${NODES} -c ${PROC_DEPTH} --cpu_bind=cores"

# The launching command and options.  We are just launching a single process
# (the starting python process) with no binding of processes to cores.
launch_str="srun -n 1"

#=========== Run it ============

com="${launch_str} python3 /global/homes/m/msilvafe/so_home/scripts/check_fknee.py \
/global/homes/m/msilvafe/so_home/preprocess/configs/20240819_satp1_cmb_preproc_cfg.yaml \
--errlog-ext satp1_net_errlog.txt \
--savename satp1_net_check.npy \
--nproc ${NPROC}"

echo ${com}
echo "Launching pipeline at $(date)"
eval ${com} > ${log} 2>&1

echo "Ending batch script at $(date)"
