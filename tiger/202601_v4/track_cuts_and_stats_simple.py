#!/usr/bin/env python3
"""
Simple parallel version that distributes observations across SLURM tasks.
Each task processes a subset of observations without MPI complexity.
"""

import os
import yaml
import sys
import argparse
import sqlite3
from track_cuts_and_stats import track_cuts_and_stats, create_cuts_stats_tables, build_process_names, save_results_to_db

def main():
    parser = argparse.ArgumentParser(description='Track cuts and statistics in parallel via SLURM task array')
    parser.add_argument('--init-config', required=True, help='Init pipeline config file')
    parser.add_argument('--proc-config', required=True, help='Proc pipeline config file')
    parser.add_argument('--db-path', required=True, help='SQLite database path')
    parser.add_argument('--obs-list', required=True, help='Text file with list of observation IDs')
    parser.add_argument('--task-id', type=int, help='Task ID (0-based), uses SLURM_PROCID if not specified')
    parser.add_argument('--num-tasks', type=int, help='Total number of tasks, uses SLURM_NPROCS if not specified')
    parser.add_argument('--verbosity', type=int, default=1, help='Verbosity level (0-3)')
    
    args = parser.parse_args()
    
    # Get task ID and number of tasks from SLURM environment if not specified
    task_id = args.task_id if args.task_id is not None else int(os.environ.get('SLURM_PROCID', 0))
    num_tasks = args.num_tasks if args.num_tasks is not None else int(os.environ.get('SLURM_NPROCS', 1))
    
    print(f"Task {task_id}/{num_tasks} starting...")
    
    # Load configs
    with open(args.init_config, 'r') as f:
        configs_init = yaml.safe_load(f)
    
    with open(args.proc_config, 'r') as f:
        configs_proc = yaml.safe_load(f)
    
    # Build process names for table creation
    process_names = build_process_names(configs_init, configs_proc)
    
    # Load observation list
    with open(args.obs_list, 'r') as f:
        obs_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Task {task_id}: Loaded {len(obs_ids)} total observations")
    
    # Create tables (only task 0)
    if task_id == 0:
        print("Task 0: Creating database tables...")
        create_cuts_stats_tables(args.db_path, process_names)
        print("Task 0: Tables created successfully")
    
    # Create work items (obs_id, wafer, band) and distribute across tasks
    work_items = []
    for obs_id in obs_ids:
        for ws_idx in range(7):
            wafer = f'ws{ws_idx}'
            for band in ['f090', 'f150']:
                work_items.append((obs_id, wafer, band))
    
    # Distribute work items to this task
    my_work_items = [work_items[i] for i in range(task_id, len(work_items), num_tasks)]
    print(f"Task {task_id}: Processing {len(my_work_items)} work items")
    
    # Process work items
    n_processed = 0
    n_failed = 0
    
    for i, (obs_id, wafer, band) in enumerate(my_work_items):
        if args.verbosity >= 2:
            print(f"Task {task_id}: Processing {i+1}/{len(my_work_items)}: {obs_id} {wafer} {band}")
        
        try:
            # Track cuts and statistics
            results = track_cuts_and_stats(obs_id, wafer, band, configs_init, configs_proc,
                                           process_names, args.verbosity)
            
            # Save results to database
            save_results_to_db(args.db_path, obs_id, wafer, band, results, process_names)
            
            if results['success']:
                n_processed += 1
                if args.verbosity >= 1:
                    print(f"Task {task_id}: SUCCESS {obs_id} {wafer} {band}")
            else:
                n_failed += 1
                print(f"Task {task_id}: FAILED {obs_id} {wafer} {band}: {results['error_message']}")
                
        except Exception as e:
            n_failed += 1
            print(f"Task {task_id}: EXCEPTION {obs_id} {wafer} {band}: {e}")
    
    print(f"Task {task_id} complete: {n_processed} processed, {n_failed} failed")

if __name__ == "__main__":
    main()