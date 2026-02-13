#!/usr/bin/env python3
"""
Parallel version of track_cuts_and_stats.py for MPI execution on SLURM.
Tracks detector cuts, sample cuts, and noise statistics during SO preprocessing pipeline.
"""

import os
import yaml
import time
import logging
from typing import Optional, Union, Callable, List, Dict, Any
import numpy as np
import argparse
import traceback
from sotodlib.utils.procs_pool import get_exec_env
from sotodlib import core
from sotodlib.core.metadata.obsdb import ObsDb
import sotodlib.site_pipeline.util as sp_util
from sotodlib.preprocess import preprocess_util as pp_util
from sotodlib.preprocess import Pipeline

# Import the single observation tracking function
from track_cuts_and_stats import (
    track_cuts_and_stats,
    create_cuts_stats_tables,
    build_process_names,
    save_results_to_db,
    get_sorted_flag_labs,
)

logger = sp_util.init_logger("track_cuts_stats_parallel")

def main():
    parser = argparse.ArgumentParser(description='Track cuts and statistics in parallel via MPI')
    parser.add_argument('--init-config', required=True, help='Init pipeline config file')
    parser.add_argument('--proc-config', required=True, help='Proc pipeline config file') 
    parser.add_argument('--db-path', required=True, help='SQLite database path')
    parser.add_argument('--query', help='Query to pass to observation list')
    parser.add_argument('--obs-list', help='Text file with list of observation IDs')
    parser.add_argument('--obs-id', help='Single obs-id to process')
    parser.add_argument('--min-ctime', help='Minimum timestamp for observation list')
    parser.add_argument('--max-ctime', help='Maximum timestamp for observation list') 
    parser.add_argument('--update-delay', type=int, help='Days in past to start observation list')
    parser.add_argument('--tags', nargs='*', help='Observation tags')
    parser.add_argument('--verbosity', type=int, default=1, help='Verbosity level (0-3)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing entries')
    
    args = parser.parse_args()
    
    # Load configs
    with open(args.init_config, 'r') as f:
        configs_init = yaml.safe_load(f)
    
    with open(args.proc_config, 'r') as f:
        configs_proc = yaml.safe_load(f)
    
    # Build process names and flag labels for table creation
    process_names = build_process_names(configs_init, configs_proc)
    cfg_init, _ = pp_util.get_preprocess_context(configs_init)
    pipe_init = Pipeline(cfg_init["process_pipe"])
    full_flag_labels = get_sorted_flag_labs(pipe_init)
    base_flag_labels = [label.split('.')[0] for label in full_flag_labels]
    
    # Get observation list
    cfg_init, context = pp_util.get_preprocess_context(args.init_config)
    
    if args.obs_list:
        # Load from observation list file
        with open(args.obs_list, 'r') as f:
            obs_ids = [line.strip() for line in f if line.strip()]
        obs_list = [{'obs_id': obs_id} for obs_id in obs_ids]
        logger.info(f'Loaded {len(obs_list)} observations from {args.obs_list}')
    else:
        # Use database query
        obs_list = sp_util.get_obslist(context, query=args.query, obs_id=args.obs_id, 
                                       min_ctime=args.min_ctime, max_ctime=args.max_ctime,
                                       update_delay=args.update_delay, tags=args.tags)
    
    if len(obs_list) == 0:
        source = args.obs_list if args.obs_list else f"query: {args.query}"
        logger.warning(f"No observations returned from {source}")
        return
    
    logger.info(f'Processing {len(obs_list)} observations')
    
    # Filter existing observations if not overwriting
    if not args.overwrite and os.path.exists(args.db_path):
        try:
            obsdb = ObsDb(map_file=args.db_path, init_db=False)
            existing_obsids = set([row['obs_id'] for row in obsdb.query()])
            obs_list = [obs for obs in obs_list if obs['obs_id'] not in existing_obsids]
            logger.info(f'Filtered to {len(obs_list)} new observations')
        except Exception as e:
            logger.warning(f"Could not filter existing observations: {e}")
    
    # Create tables (only on rank 0)
    try:
        exec_env = get_exec_env()
        if hasattr(exec_env, 'comm'):
            comm = exec_env.comm
            rank = comm.rank if comm else 0
            size = comm.size if comm else 1
        else:
            # get_exec_env returns (comm, rank, size) tuple in some versions
            comm, rank, size = exec_env if isinstance(exec_env, tuple) else (None, 0, 1)
    except Exception as e:
        logger.warning(f"Could not get MPI environment: {e}")
        comm, rank, size = None, 0, 1
    
    if rank == 0:
        create_cuts_stats_tables(args.db_path, process_names, base_flag_labels)
    
    # Wait for table creation
    if comm:
        comm.barrier()
    
    # Create work list - each item is (obs_id, wafer, band)
    work_items = []
    for obs in obs_list:
        for ws_idx in range(7):
            wafer = f'ws{ws_idx}'
            for band in ['f090', 'f150']:
                work_items.append((obs['obs_id'], wafer, band))
    
    logger.info(f'Total work items: {len(work_items)}')
    
    # Process work items in parallel
    n_processed = 0
    n_failed = 0
    
    for item_idx in range(rank, len(work_items), size):
        obs_id, wafer, band = work_items[item_idx]
        
        try:
            if args.verbosity >= 2:
                logger.info(f"Rank {rank} processing {obs_id} {wafer} {band}")
            
            # Track cuts and statistics for this observation
            results = track_cuts_and_stats(obs_id, wafer, band, configs_init, configs_proc,
                                           process_names, full_flag_labels, args.verbosity)
            
            # Save results to database
            save_results_to_db(args.db_path, obs_id, wafer, band, results,
                               process_names, base_flag_labels)
            
            if results['success']:
                n_processed += 1
                if args.verbosity >= 1:
                    logger.info(f"Rank {rank}: Processed {obs_id} {wafer} {band}")
            else:
                n_failed += 1
                logger.warning(f"Rank {rank}: Failed {obs_id} {wafer} {band}: {results['error_message']}")
                
        except Exception as e:
            n_failed += 1
            logger.error(f"Rank {rank}: Exception processing {obs_id} {wafer} {band}: {e}")
            if args.verbosity >= 2:
                logger.error(traceback.format_exc())
    
    # Gather statistics
    if comm:
        all_processed = comm.gather(n_processed, root=0)
        all_failed = comm.gather(n_failed, root=0)
        
        if rank == 0:
            total_processed = sum(all_processed)
            total_failed = sum(all_failed)
            logger.info(f"Final results: {total_processed} processed, {total_failed} failed")
    else:
        logger.info(f"Single process results: {n_processed} processed, {n_failed} failed")

if __name__ == "__main__":
    main()