#!/bin/bash

set -e

#=========== USER CONFIGURATION ============

# Input database file - UPDATE THIS PATH
input_db="PATH_TO_YOUR_DATABASE.sqlite"

# Timestamp range for analysis - UPDATE THESE
start_ts=START_TIMESTAMP  # Start timestamp for your analysis period
end_ts=END_TIMESTAMP      # End timestamp for your analysis period

# Output directory - will be created if it doesn't exist
output_dir="PATH_TO_OUTPUT_DIR/tables_$(date +%Y%m%d)"

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

# Check if input database exists
if [ ! -f "${input_db}" ]; then
    echo "ERROR: Input database file not found: ${input_db}"
    echo "Please update the input_db variable with the correct path."
    exit 1
fi

#=========== GENERATE TABLES ============

echo "Generating summary tables at $(date)"

python3 PATH_TO_SCRIPTS/generate_cuts_tables.py \
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
