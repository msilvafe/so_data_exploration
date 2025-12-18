#!/bin/bash
#SBATCH --account=simonsobs
#SBATCH --nodes=1
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --job-name=max-parallel-disk-scan
#SBATCH --mail-user maximiliano.silva-feaver@yale.edu
#SBATCH --mail-type all

# MAXIMUM SINGLE-NODE UTILIZATION - 112 processes on Tiger

set -e

SEARCH_DIR="${1:-/scratch/gpfs/SIMONSOBS/sat-iso}"
USERNAME="${2:-ms3067}"
OUTPUT_DIR="$SLURM_SUBMIT_DIR"
FINAL_OUTPUT="${OUTPUT_DIR}/disk_usage_${USERNAME}_${SLURM_JOB_ID}.txt"

echo "=== MAXIMUM CPU UTILIZATION DISK SCANNER ==="
echo "Node: $SLURMD_NODENAME"
echo "Total CPUs: $SLURM_NTASKS (all 112 CPUs on Tiger node)"
echo "Target: $SEARCH_DIR (user: $USERNAME)"
echo ""

START_TIME=$(date +%s)

echo "Launching 112 parallel processes..."

# Use all 112 CPUs with MPI-style work distribution
srun -n "$SLURM_NTASKS" bash -c '
    # Each of the 112 processes gets a unique subset
    PROC_ID=$SLURM_PROCID
    TOTAL_PROCS=$SLURM_NTASKS
    
    # Distribute directory scanning across all 112 processes
    find "'"$SEARCH_DIR"'" -type d 2>/dev/null | \
    awk "NR % $TOTAL_PROCS == $PROC_ID" | \
    while read dir; do
        if [ -d "$dir" ]; then
            find "$dir" -maxdepth 1 -user "'"$USERNAME"'" -type f -printf "%s %h\n" 2>/dev/null
        fi
    done
' | \
awk '
BEGIN {
    print "=== 112-CPU PARALLEL SCAN RESULTS ==="
    print "Generated: " strftime("%Y-%m-%d %H:%M:%S")
    print "CPUs used: '"$SLURM_NTASKS"'"
    print "Node: '"$SLURMD_NODENAME"'"
    print "Directory: '"$SEARCH_DIR"'"
    print "Username: '"$USERNAME"'"
    print ""
}
{
    if (NF >= 2) {
        dir_size[$2] += $1
        total += $1
        count++
        
        if (count % 50000 == 0) {
            print "Processed " count " files..." > "/dev/stderr"
        }
    }
}
END {
    print "DIRECTORY USAGE SUMMARY:"
    print "================================================================="
    
    PROCINFO["sorted_in"] = "@val_num_desc"
    
    for(dir in dir_size) {
        size = dir_size[dir]
        if(size >= 1073741824) {
            human = sprintf("%.2f GB", size/1073741824)
        } else if(size >= 1048576) {
            human = sprintf("%.2f MB", size/1048576)
        } else if(size >= 1024) {
            human = sprintf("%.2f KB", size/1024)
        } else {
            human = sprintf("%d B", size)
        }
        printf "%-65s %15s (%d bytes)\n", dir, human, size
    }
    
    print "================================================================="
    if(total >= 1073741824) {
        total_human = sprintf("%.2f GB", total/1073741824)
    } else if(total >= 1048576) {
        total_human = sprintf("%.2f MB", total/1048576)
    } else {
        total_human = sprintf("%.2f KB", total/1024)
    }
    printf "%-65s %15s (%d bytes)\n", "TOTAL USAGE:", total_human, total
    printf "%-65s %15d files\n", "TOTAL FILES:", count
}
' > "$FINAL_OUTPUT"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=== SCAN COMPLETED ==="
echo "Duration: ${DURATION} seconds"
echo "CPUs used: 112/112 (100% utilization)"
echo "Theoretical speedup: ~112x vs single-threaded"
echo "Results: $FINAL_OUTPUT"

echo ""
echo "Quick preview:"
head -20 "$FINAL_OUTPUT"
echo "..."
tail -10 "$FINAL_OUTPUT"
