#!/usr/bin/env python3
"""
Simple parallel version that distributes observations across SLURM tasks.
Each task processes a subset of observations without MPI complexity.
"""

import os
import yaml
import sys
import argparse
import numpy as np
import h5py
from sotodlib.preprocess import preprocess_util as pp_util
from sotodlib.preprocess import Pipeline
from sotodlib.core.metadata import ObsDb
from track_cuts_and_stats import (
    track_cuts_and_stats,
    create_cuts_stats_tables,
    build_process_names,
    save_results_to_db,
    get_sorted_flag_labs,
    write_band_noise_to_hdf5,
)

def main():
    parser = argparse.ArgumentParser(description='Track cuts and statistics in parallel via SLURM task array')
    parser.add_argument('--init-config', required=True, help='Init pipeline config file')
    parser.add_argument('--proc-config', required=True, help='Proc pipeline config file')
    parser.add_argument('--db-path', required=True, help='SQLite database path')
    parser.add_argument('--obs-list', required=True, help='Text file with list of observation IDs')
    parser.add_argument('--task-id', type=int, help='Task ID (0-based), uses SLURM_PROCID if not specified')
    parser.add_argument('--num-tasks', type=int, help='Total number of tasks, uses SLURM_NPROCS if not specified')
    parser.add_argument('--verbosity', type=int, default=1, help='Verbosity level (0-3)')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for database commits (set to 0 to commit at end only)')
    parser.add_argument('--aggregate-noise', choices=['T', 'Q', 'U'], help='Aggregate noise data for specified polarization')
    parser.add_argument('--aggregate-output', help='HDF5 output file for aggregated data (per band, will append _f090.h5 and _f150.h5)')
    
    args = parser.parse_args()
    
    # Get task ID and number of tasks from SLURM environment if not specified
    task_id = args.task_id if args.task_id is not None else int(os.environ.get('SLURM_PROCID', 0))
    num_tasks = args.num_tasks if args.num_tasks is not None else int(os.environ.get('SLURM_NPROCS', 1))
    
    print(f"Task {task_id}/{num_tasks} starting...")
    
    # Load configs and get preprocessing context
    with open(args.init_config, 'r') as f:
        configs_init = yaml.safe_load(f)
    
    with open(args.proc_config, 'r') as f:
        configs_proc = yaml.safe_load(f)
    
    # Build process names and flag labels for table creation
    process_names = build_process_names(configs_init, configs_proc)
    cfg_init, ctx_init = pp_util.get_preprocess_context(configs_init)
    pipe_init = Pipeline(cfg_init["process_pipe"])
    full_flag_labels = get_sorted_flag_labs(pipe_init)
    base_flag_labels = [label.split('.')[0] for label in full_flag_labels]
    
    # Load observation list
    with open(args.obs_list, 'r') as f:
        obs_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Task {task_id}: Loaded {len(obs_ids)} total observations")
    
    # Query unique bands from obsdb using tube_flavor
    bands = []
    tube_flavor_to_bands = {
        'lf': ['f030', 'f040'],
        'mf': ['f090', 'f150'],
        'uf': ['f220', 'f280'],
    }
    try:
        if ctx_init.obsdb and obs_ids:
            # Query tube_flavor from first observation
            rs = ctx_init.obsdb.query(f"obs_id = '{obs_ids[0]}'")
            # Convert ResultSet to list to check if results exist
            results = list(rs)
            if len(results) > 0:
                # Access tube_flavor from the first result row
                if 'tube_flavor' in results[0]:
                    tube_flavor = results[0]['tube_flavor']
                    bands = tube_flavor_to_bands.get(tube_flavor, [])
                    if args.verbosity >= 2:
                        print(f"Task {task_id}: Found tube_flavor='{tube_flavor}' for {obs_ids[0]}")
                else:
                    if args.verbosity >= 2:
                        print(f"Task {task_id}: 'tube_flavor' field not in obsdb results. Available fields: {list(results[0].keys())}")
            else:
                if args.verbosity >= 2:
                    print(f"Task {task_id}: No results from obsdb query for {obs_ids[0]}")
        elif not ctx_init.obsdb:
            if args.verbosity >= 2:
                print(f"Task {task_id}: ctx_init.obsdb is not available")
        elif not obs_ids:
            if args.verbosity >= 2:
                print(f"Task {task_id}: No observation IDs loaded")
    except Exception as e:
        if args.verbosity >= 2:
            print(f"Task {task_id}: Exception during band detection: {e}")
        import traceback
        if args.verbosity >= 3:
            traceback.print_exc()
    
    # Fall back to defaults if band detection fails
    if not bands:
        bands = ['f090', 'f150']
        print(f"Task {task_id}: Using default bands: {bands}")
    else:
        print(f"Task {task_id}: Detected bands: {bands}")
        print(f"Task {task_id}: Detected bands: {bands}")
    
    # Create tables (only task 0)
    if task_id == 0:
        print("Task 0: Creating database tables...")
        create_cuts_stats_tables(args.db_path, process_names, base_flag_labels)
        print("Task 0: Tables created successfully")
    
    # Create work items (obs_id, wafer, band) and distribute across tasks
    work_items = []
    for obs_id in obs_ids:
        for ws_idx in range(7):
            wafer = f'ws{ws_idx}'
            for band in bands:
                work_items.append((obs_id, wafer, band))
    
    # Distribute work items to this task
    my_work_items = [work_items[i] for i in range(task_id, len(work_items), num_tasks)]
    print(f"Task {task_id}: Processing {len(my_work_items)} work items")
    
    # Initialize aggregation storage if requested
    aggregate_data = {band: {} for band in bands} if args.aggregate_noise else None
    if args.aggregate_noise:
        for band in bands:
            aggregate_data[band] = {
                'white_noise': [],
                'det_ids': [],
                'wafer_slots': [],
                'obs_ranges': {},
                'current_idx': 0,
            }
    
    # Process work items
    n_processed = 0
    n_failed = 0
    
    # Use batch processing for database operations to reduce I/O overhead on distributed filesystems
    # Batch size: commit every N updates (set to 0 to commit at end only)
    batch_size = args.batch_size
    update_count = 0
    obsdb = None
    
    try:
        # Open database connection once for all work items in this task
        if len(my_work_items) > 0:
            obsdb = ObsDb(map_file=args.db_path, init_db=False)
        
        for i, (obs_id, wafer, band) in enumerate(my_work_items):
            if args.verbosity >= 2:
                print(f"Task {task_id}: Processing {i+1}/{len(my_work_items)}: {obs_id} {wafer} {band}")
            
            try:
                # Track cuts and statistics
                results = track_cuts_and_stats(obs_id, wafer, band, configs_init, configs_proc,
                                               process_names, full_flag_labels, args.verbosity,
                                               aggregate_noise=args.aggregate_noise)
                
                # Save results to database without committing (for batched operations)
                save_results_to_db(args.db_path, obs_id, wafer, band, results,
                                   process_names, base_flag_labels, commit=False)
                update_count += 1
                
                # Commit periodically to reduce I/O overhead on distributed filesystems
                if batch_size > 0 and update_count % batch_size == 0:
                    if args.verbosity >= 2:
                        print(f"Task {task_id}: Committing batch of {batch_size} database updates")
                    obsdb.conn.commit()
                
                # Collect aggregation data if requested and available
                if args.aggregate_noise and results.get('aggregate_data') is not None:
                    agg = results['aggregate_data']
                    band_agg = aggregate_data[band]
                    
                    # Track range for this observation
                    n_dets = len(agg['det_ids'])
                    band_agg['obs_ranges'][obs_id] = (band_agg['current_idx'], band_agg['current_idx'] + n_dets)
                    
                    # Append data
                    band_agg['white_noise'].append(agg['white_noise'])
                    band_agg['det_ids'].append(agg['det_ids'])
                    band_agg['wafer_slots'].append(agg['wafer_slots'])
                    band_agg['current_idx'] += n_dets
                
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
    
    finally:
        # Final commit for any remaining database updates
        if obsdb is not None and update_count > 0:
            # Only commit if we haven't already committed the last batch
            remaining = update_count % batch_size if batch_size > 0 else update_count
            if remaining > 0:
                if args.verbosity >= 2:
                    print(f"Task {task_id}: Final commit of {remaining} database updates")
                obsdb.conn.commit()
        
        # Close the database connection
        if obsdb is not None:
            obsdb.conn.close()
    
    print(f"Task {task_id} complete: {n_processed} processed, {n_failed} failed")
    
    # Write aggregated data if requested  
    if args.aggregate_noise and args.aggregate_output:
        for band in bands:
            band_agg = aggregate_data[band]
            if band_agg['white_noise']:
                # Combine arrays
                band_data = {
                    'white_noise': np.concatenate(band_agg['white_noise']),
                    'det_ids': np.concatenate(band_agg['det_ids']),
                    'wafer_slots': np.concatenate(band_agg['wafer_slots']),
                    'obs_ranges': band_agg['obs_ranges'],
                    'obs_ids': list(band_agg['obs_ranges'].keys()),
                }
                
                # Write to HDF5 with task ID in filename
                output_file = f"{args.aggregate_output}_task{task_id}_{band}.h5"
                print(f"Task {task_id}: Writing {len(band_data['det_ids'])} detectors to {output_file}")
                write_band_noise_to_hdf5(band_data, band, output_file)
            else:
                print(f"Task {task_id}: No aggregation data for {band}")

if __name__ == "__main__":
    main()