#!/bin/bash

# Batch analysis runner for ISO-V4 cuts and stats data
# Runs analysis and report generation on all folders

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/scratch/gpfs/SIMONSOBS/sat-iso/v4/cuts_stats"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Tracking
TOTAL=0
SUCCESS=0
FAILED=0
RESULTS_FILE="/tmp/analysis_results_$$.txt"

echo "=========================================="
echo "ISO-V4 Batch Analysis Runner"
echo "=========================================="
echo "Data directory: $DATA_DIR"
echo "Script directory: $SCRIPT_DIR"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}Error: Data directory not found: $DATA_DIR${NC}"
    exit 1
fi

# Get list of folders to process
FOLDERS=(satp1_20260417 satp1_20260520 satp1_260702 satp3_20260417 satp3_20260520 satp3_260702)

echo "Processing ${#FOLDERS[@]} folders..."
echo ""

for folder in "${FOLDERS[@]}"; do
    FOLDER_PATH="$DATA_DIR/$folder"
    
    if [ ! -d "$FOLDER_PATH" ]; then
        echo -e "${YELLOW}[SKIP]${NC} Folder not found: $folder"
        continue
    fi
    
    TOTAL=$((TOTAL + 1))
    
    # Check for database
    DB_FILE="$FOLDER_PATH/databases/iso_v4_${folder}.sqlite"
    if [ ! -f "$DB_FILE" ]; then
        echo -e "${RED}[FAIL]${NC} $folder - Database not found: $DB_FILE"
        FAILED=$((FAILED + 1))
        echo "$folder: FAILED (missing database)" >> "$RESULTS_FILE"
        continue
    fi
    
    # Create analysis output directory
    ANALYSIS_DIR="$FOLDER_PATH/analysis"
    mkdir -p "$ANALYSIS_DIR"
    
    echo -e "${YELLOW}[ANALYZING]${NC} $folder"
    
    # Run analyze_cuts_and_stats.py
    if python3 "$SCRIPT_DIR/analyze_cuts_and_stats.py" \
        --db-path "$DB_FILE" \
        --output-dir "$ANALYSIS_DIR" 2>&1 | tail -5; then
        
        echo -e "${YELLOW}[REPORT]${NC} $folder - Generating PDF report"
        
        # Run generate_report_wrapper.py
        if python3 "$SCRIPT_DIR/generate_report_wrapper.py" \
            --analysis-dir "$ANALYSIS_DIR" \
            --output-dir "$ANALYSIS_DIR" 2>&1 | tail -3; then
            
            echo -e "${GREEN}[SUCCESS]${NC} $folder"
            SUCCESS=$((SUCCESS + 1))
            echo "$folder: SUCCESS" >> "$RESULTS_FILE"
        else
            echo -e "${RED}[FAIL]${NC} $folder - Report generation failed"
            FAILED=$((FAILED + 1))
            echo "$folder: FAILED (report generation)" >> "$RESULTS_FILE"
        fi
    else
        echo -e "${RED}[FAIL]${NC} $folder - Analysis failed"
        FAILED=$((FAILED + 1))
        echo "$folder: FAILED (analysis)" >> "$RESULTS_FILE"
    fi
    
    echo ""
done

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total folders:    $TOTAL"
echo -e "Successful:       ${GREEN}$SUCCESS${NC}"
echo -e "Failed:           ${RED}$FAILED${NC}"
echo ""

# Print results
if [ -f "$RESULTS_FILE" ]; then
    echo "Details:"
    cat "$RESULTS_FILE"
    rm "$RESULTS_FILE"
fi

# Verify outputs
echo ""
echo "=========================================="
echo "Output Summary"
echo "=========================================="

for folder in "${FOLDERS[@]}"; do
    ANALYSIS_DIR="$DATA_DIR/$folder/analysis"
    
    if [ ! -d "$ANALYSIS_DIR" ]; then
        continue
    fi
    
    # Count files
    PNG_COUNT=$(ls -1 "$ANALYSIS_DIR"/*.png 2>/dev/null | wc -l)
    PDF_COUNT=$(ls -1 "$ANALYSIS_DIR"/*.pdf 2>/dev/null | wc -l)
    CSV_COUNT=$(ls -1 "$ANALYSIS_DIR"/*.csv 2>/dev/null | wc -l)
    
    echo "$folder:"
    echo "  PNG files:  $PNG_COUNT"
    echo "  PDF files:  $PDF_COUNT"
    echo "  CSV files:  $CSV_COUNT"
    echo "  Total:      $((PNG_COUNT + PDF_COUNT + CSV_COUNT))"
done

echo ""
echo "=========================================="
echo "Expected outputs per folder:"
echo "  Time series grid plots:        2 (f090, f150)"
echo "  Time series by-wafer plots:   ~30 (15 cut types × 2 bands)"
echo "  Detector cuts/yield plots:     6"
echo "  Sample flags plots:            3"
echo "  CSV tables:                    4+"
echo "  PDF report:                    1"
echo "=========================================="
