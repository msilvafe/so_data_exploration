#!/bin/bash

# Base path provided as first argument
BASE_PATH=$(realpath "${1:-.}")
USER_ID="ms3067"

if [ ! -d "$BASE_PATH" ]; then
    echo "Error: $BASE_PATH is not a directory."
    exit 1
fi

# Initialize total size accumulator (in bytes)
TOTAL_BYTES=0

echo "Size (GB)   Directory Structure (User: $USER_ID)"
echo "------------------------------------------------------------"

# Calculate the depth of the base path to determine relative indentation
BASE_DEPTH=$(echo "$BASE_PATH" | tr -cd '/' | wc -c)

# 1. Find directories from depth 1 to 3
# 2. Sort them alphabetically so parents appear before children
while read -r dir; do
    # Calculate relative depth for indentation
    CURRENT_DEPTH=$(echo "$dir" | tr -cd '/' | wc -c)
    RELATIVE_DEPTH=$((CURRENT_DEPTH - BASE_DEPTH))
    
    # Create indentation (3 spaces per level)
    INDENT=$(printf '%*s' "$((RELATIVE_DEPTH * 3))" "")

    # Get total bytes for files owned by ms3067 in this specific directory branch
    dir_bytes=$(find "$dir" -user "$USER_ID" -type f -printf "%s\n" 2>/dev/null | awk '{s+=$1} END {print s+0}')

    if [ "$dir_bytes" -gt 0 ]; then
        # Convert to GB for the display
        size_gb=$(echo "scale=2; $dir_bytes / 1024^3" | bc -l)
        
        # Strip the base path from the display for cleaner indented reading
        DIR_NAME=$(basename "$dir")
        if [ "$RELATIVE_DEPTH" -eq 1 ]; then
            printf "%-11s %s%s/\n" "${size_gb} GB" "$INDENT" "$DIR_NAME"
            # ONLY add to Grand Total if it is a top-level directory (Depth 1)
            TOTAL_BYTES=$((TOTAL_BYTES + dir_bytes))
        else
            printf "%-11s %s%s/\n" "${size_gb} GB" "$INDENT" "$DIR_NAME"
        fi
    fi
done < <(find "$BASE_PATH" -mindepth 1 -maxdepth 3 -type d | sort)

# Calculate and report final total in TB
echo "------------------------------------------------------------"
TOTAL_TB=$(echo "scale=4; $TOTAL_BYTES / 1024^4" | bc -l)
echo "GRAND TOTAL (Sum of Depth 1): $TOTAL_TB TB"

