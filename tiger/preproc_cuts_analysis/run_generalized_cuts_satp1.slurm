#!/bin/bash -l

#SBATCH --account=simonsobs
#SBATCH --nodes=4
#SBATCH --ntasks=224
#SBATCH --cpus-per-task=2
#SBATCH --time=00:45:00
#SBATCH --job-name=SATp1-generalized-cuts
#SBATCH --mail-user maximiliano.silva-feaver@yale.edu
#SBATCH --mail-type all

set -e

export SOTODLIB_RESOURCES='{"de421.bsp": "file:///scratch/gpfs/SIMONSOBS/planets/de421.bsp"}'

# Output directory and log file
output_dir="/scratch/gpfs/SIMONSOBS/users/ms3067/iso_stats/generalized_script/v1/satp1"
log="${output_dir}/log_generalized_cuts_satp1_$(date +%Y%m%d).log"

# Create output directory if it doesn't exist
mkdir -p ${output_dir}

#=========== Compute runtime parameters ============

echo "Using ${SLURM_JOB_NUM_NODES} node(s), each with ${SLURM_CPUS_PER_TASK} CPUs per task."
echo "Starting ${SLURM_NTASKS} total tasks."

export OMP_NUM_THREADS=2

launch_str="srun -n ${SLURM_NTASKS} --cpus-per-task=${SLURM_CPUS_PER_TASK} --export=ALL "

#=========== Run generalized cuts analysis ============

# Define configuration files - update these paths as needed
config_init="/home/ms3067/repos/iso-sat/v1/preprocessing/satp1/preprocessing_config_20250108_sat-iso_init.yaml"
config_proc="/home/ms3067/repos/iso-sat/v1/preprocessing/satp1/preprocessing_config_20250108_sat-iso_proc.yaml"

# For newer configs, use v3 configs:
# config_init="/home/ms3067/repos/iso-sat/v3/preprocessing/satp1/preprocessing_config_20250801_init.yaml"
# config_proc="/home/ms3067/repos/iso-sat/v3/preprocessing/satp1/preprocessing_config_20250801_proc.yaml"

output_file="${output_dir}/generalized_cuts_satp1_$(date +%Y%m%d).sqlite"
errlog_file="${output_dir}/generalized_cuts_errlog_satp1_$(date +%Y%m%d).txt"

com="${launch_str} python3 /home/ms3067/so_data_exploration/tiger/generalized_cuts_analysis_simple.py \
${config_init} \
${config_proc} \
--noise-range 18e-6 80e-6 \
--nproc ${SLURM_NTASKS} \
--savename ${output_file} \
--errlog-ext ${errlog_file}"

echo "Command to run:"
echo ${com}
echo ""
echo "Launching generalized cuts analysis at $(date)"
echo "Config init: ${config_init}"
echo "Config proc: ${config_proc}"
echo "Output file: ${output_file}"
echo "Error log: ${errlog_file}"
echo ""

eval ${com} > ${log} 2>&1

echo "Analysis completed at $(date)"
echo "Log file: ${log}"
echo "Output database: ${output_file}"

#=========== Optional: Generate summary tables ============

# Table generation is now done separately for efficiency
# To generate tables after analysis completes, use the lightweight script:
# 
# ./generate_tables.sh
# 
# (after updating the database path and timestamp range in the script)

# Uncomment the following section to automatically generate summary tables
# after the analysis completes (not recommended - use separate script instead)

# echo "Generating summary tables at $(date)"
# 
# # Define timestamp range for table generation
# start_ts=1716177600  # Update this to your analysis start timestamp
# end_ts=1734315803    # Update this to your analysis end timestamp
# 
# table_dir="${output_dir}/tables_$(date +%Y%m%d)"
# mkdir -p ${table_dir}
# 
# python3 /home/ms3067/repos/so_data_exploration/tiger/generate_cuts_tables.py \
#     ${output_file} \
#     --start-ts ${start_ts} \
#     --end-ts ${end_ts} \
#     --output-csv \
#     --output-dir ${table_dir} >> ${log} 2>&1
# 
# echo "Summary tables generated in: ${table_dir}"

echo ""
echo "To generate summary tables, edit and run:"
echo "./generate_tables.sh"

echo "Script completed at $(date)"
