#!/bin/bash -l

#SBATCH --account=simonsobs
#SBATCH --nodes=4
#SBATCH --ntasks=224
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --job-name=generalized-cuts-template
#SBATCH --mail-user YOUR_EMAIL@DOMAIN.COM
#SBATCH --mail-type all

set -e

export SOTODLIB_RESOURCES='{"de421.bsp": "file:///scratch/gpfs/SIMONSOBS/planets/de421.bsp"}'

#=========== USER CONFIGURATION ============

# Update these paths for your specific analysis:
TELESCOPE="satp1"  # Change to satp1, satp2, satp3, etc.
VERSION="v2"       # Change to v1, v2, v3, etc.
DATE_TAG="$(date +%Y%m%d)"

# Configuration files - UPDATE THESE PATHS
config_init="/home/ms3067/repos/iso-sat/${VERSION}/preprocessing/${TELESCOPE}/preprocessing_config_20250507_isov2_init_bs.yaml"
config_proc="/home/ms3067/repos/iso-sat/${VERSION}/preprocessing/${TELESCOPE}/preprocessing_config_20250507_isov2_proc_bs.yaml"

# Noise range - adjust based on telescope
if [[ "${TELESCOPE}" == "satp1" ]]; then
    noise_min="18e-6"
    noise_max="80e-6"
elif [[ "${TELESCOPE}" == "satp3" ]]; then
    noise_min="2e-6"
    noise_max="80e-6"
else
    noise_min="18e-6"  # default
    noise_max="80e-6"
fi

# Output directory and files
output_dir="/scratch/gpfs/SIMONSOBS/users/ms3067/iso_stats/generalized_script/v2_bs/satp1"
output_file="${output_dir}/generalized_cuts_${TELESCOPE}_${DATE_TAG}.sqlite"
errlog_file="${output_dir}/generalized_cuts_errlog_${TELESCOPE}_${DATE_TAG}.txt"
log="${output_dir}/log_generalized_cuts_${TELESCOPE}_${DATE_TAG}.log"

# Timestamp range for optional table generation
start_ts=1716177600  # UPDATE: Start timestamp for your analysis period
end_ts=1734315803    # UPDATE: End timestamp for your analysis period

#=========== SLURM SETUP ============

# Create output directory if it doesn't exist
mkdir -p ${output_dir}

echo "Using ${SLURM_JOB_NUM_NODES} node(s), each with ${SLURM_CPUS_PER_TASK} CPUs per task."
echo "Starting ${SLURM_NTASKS} total tasks."

export OMP_NUM_THREADS=2

launch_str="srun -n ${SLURM_NTASKS} --cpus-per-task=${SLURM_CPUS_PER_TASK} --export=ALL "

#=========== RUN GENERALIZED CUTS ANALYSIS ============

com="${launch_str} python3 /home/ms3067/so_data_exploration/tiger/generalized_cuts_analysis_simple.py \
${config_init} \
${config_proc} \
--noise-range ${noise_min} ${noise_max} \
--nproc ${SLURM_NTASKS} \
--savename ${output_file} \
--errlog-ext ${errlog_file}"

echo "=================================================="
echo "GENERALIZED CUTS ANALYSIS"
echo "=================================================="
echo "Telescope: ${TELESCOPE}"
echo "Version: ${VERSION}"
echo "Date: ${DATE_TAG}"
echo ""
echo "Config init: ${config_init}"
echo "Config proc: ${config_proc}"
echo "Noise range: ${noise_min} to ${noise_max}"
echo ""
echo "Output file: ${output_file}"
echo "Error log: ${errlog_file}"
echo "Main log: ${log}"
echo ""
echo "Command to run:"
echo ${com}
echo "=================================================="
echo ""

echo "Launching generalized cuts analysis at $(date)"
eval ${com} > ${log} 2>&1

if [ $? -eq 0 ]; then
    echo "Analysis completed successfully at $(date)"
else
    echo "Analysis failed at $(date)"
    exit 1
fi

#=========== GENERATE SUMMARY TABLES ============

# Table generation is now done separately using generate_tables.sh
# To generate tables after analysis completes, run:
# 
# ./generate_tables.sh
# 
# Or customize and run:
# python3 /home/USERNAME/repos/so_data_exploration/tiger/generate_cuts_tables.py \
#     ${output_file} \
#     --start-ts ${start_ts} \
#     --end-ts ${end_ts} \
#     --output-csv \
#     --output-dir ${output_dir}/tables_${DATE_TAG}

echo ""
echo "To generate summary tables, run:"
echo "./generate_tables.sh"
echo "(after updating the database path in the script)"

echo ""
echo "=================================================="
echo "JOB SUMMARY"
echo "=================================================="
echo "Job completed at $(date)"
echo "Main log: ${log}"
echo "Output database: ${output_file}"
echo "Summary tables: ${table_dir}"
echo "=================================================="
