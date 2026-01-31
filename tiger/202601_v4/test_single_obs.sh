#!/bin/bash

# Test script for single observation cuts tracking
# Run from /home/ms3067/repos/iso-sat/v4

cd /home/ms3067/repos/iso-sat/v4

echo "Testing single observation cuts tracking..."
echo "Observation: obs_1729315555_satp1_1111111 ws1 f150"

python /home/ms3067/so_data_exploration/tiger/202601_v4/track_cuts_and_stats.py \
  --obs-id obs_1729315555_satp1_1111111 \
  --wafer ws1 \
  --band f150 \
  --init-config ./preprocessing/satp1/preprocessing_config_20251216_init.yaml \
  --proc-config ./preprocessing/satp1/preprocessing_config_20251216_proc.yaml \
  --db-path /tmp/test_cuts_single.db \
  --verbosity 2

if [ $? -eq 0 ]; then
    echo "SUCCESS: Single observation test completed!"
    echo "Checking database contents..."
    sqlite3 /tmp/test_cuts_single.db "SELECT obs_id, wafer, band FROM detector_counts;"
else
    echo "FAILED: Single observation test failed!"
fi