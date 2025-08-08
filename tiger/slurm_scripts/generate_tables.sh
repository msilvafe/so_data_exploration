#!/bin/bash

set -e

#=========== USER CONFIGURATION ============

# Input database file - UPDATE THIS PATH
input_db="/scratch/gpfs/SIMONSOBS/users/ms3067/generalized_cuts/satp1/generalized_cuts_satp1_20250807.sqlite"

# Timestamp range for analysis - UPDATE THESE
start_ts=1716177600  # Start timestamp
end_ts=1734315803    # End timestamp

# Output directory - will be created if it doesn't exist
output_dir="/scratch/gpfs/SIMONSOBS/users/ms3067/generalized_cuts/tables_$(date +%Y%m%d)"

#=========== SETUP ============

mkdir -p ${output_dir}

echo "=================================================="
echo "CUTS ANALYSIS TABLE GENERATION"
echo "=================================================="
echo "Input database: ${input_db}"
echo "Start timestamp: ${start_ts}"
echo "End timestamp: ${end_ts}"
echo "Output directory: ${output_dir}"
echo "=================================================="
echo ""

#=========== GENERATE TABLES ============

echo "Generating summary tables at $(date)"

python3 /home/ms3067/repos/so_data_exploration/tiger/generate_cuts_tables.py \
    ${input_db} \
    --start-ts ${start_ts} \
    --end-ts ${end_ts} \
    --output-csv \
    --output-dir ${output_dir}

if [ $? -eq 0 ]; then
    echo ""
    echo "Tables generated successfully!"
    echo "Output files:"
    ls -la ${output_dir}/*.csv
    
    # Display the tables
    echo ""
    echo "=================================================="
    echo "DETECTOR CUTS SUMMARY"
    echo "=================================================="
    if [ -f "${output_dir}/detector_cuts_summary.csv" ]; then
        cat ${output_dir}/detector_cuts_summary.csv
    fi
    
    echo ""
    echo "=================================================="
    echo "SAMPLE CUTS SUMMARY"
    echo "=================================================="
    if [ -f "${output_dir}/sample_cuts_summary.csv" ]; then
        cat ${output_dir}/sample_cuts_summary.csv
    fi
    
else
    echo "Table generation failed!"
    exit 1
fi

echo ""
echo "=================================================="
echo "SUMMARY"
echo "=================================================="
echo "Table generation completed at $(date)"
echo "Output directory: ${output_dir}"
echo "Files created:"
ls -la ${output_dir}/
echo "=================================================="
