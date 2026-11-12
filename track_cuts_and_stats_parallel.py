#!/usr/bin/env python3
"""
Parallel version of track_cuts_and_stats.py using sotodlib get_exec_env.
Tracks detector cuts, sample cuts, and noise statistics during SO preprocessing pipeline.
"""

import os
import yaml
import time
import logging
from typing import Optional, Union, Callable, List, Dict, Any
import numpy as np
import h5py
import argparse
import traceback
from sotodlib import core
from sotodlib.core.metadata.obsdb import ObsDb
from sotodlib.site_pipeline.utils.logging import init_logger
from sotodlib.site_pipeline.utils.obsdb import get_obslist
from sotodlib.preprocess import preprocess_util as pp_util
from sotodlib.preprocess import Pipeline
from sotodlib.utils.procs_pool import get_exec_env

from track_cuts_and_stats import (
    track_cuts_and_stats,
    create_cuts_stats_tables,
    build_process_names,
    save_results_to_db,
    get_sorted_flag_labs,
    write_band_noise_to_hdf5,
)

logger = init_logger("track_cuts_stats_parallel")

def process_work_item(obs_id, wafer, band, configs_init, configs_proc, 
                      process_names, full_flag_labels, verbosity, aggregate_noise):
    """Worker function to process a single work item.
    
    Parameters
    ----------
    obs_id : str
        Observation ID
    wafer : str
        Wafer slot
    band : str
        Band name
    configs_init : dict
        Init pipeline config
    configs_proc : dict
        Proc pipeline config
    process_names : list
        Process names list
    full_flag_labels : list
        Flag labels list
    verbosity : int
        Verbosity level
    aggregate_noise : str or None
        Polarization to aggregate
    
    Returns
    -------
    dict
        Result dictionary with keys: obs_id, wafer, band, results, aggregate_data
    """
    start_wall = time.time()
    try:
        results = track_cuts_and_stats(obs_id, wafer, band, configs_init, configs_proc,
                                      process_names, full_flag_labels, verbosity,
                                      aggregate_noise=aggregate_noise)
        worker_walltime_s = time.time() - start_wall
        return {
            'obs_id': obs_id,
            'wafer': wafer,
            'band': band,
            'results': results,
            'aggregate_data': results.get('aggregate_data'),
            'worker_walltime_s': worker_walltime_s,
        }
    except Exception as e:
        worker_walltime_s = time.time() - start_wall
        logger.error(f"Exception processing {obs_id} {wafer} {band}: {e}")
        if verbosity >= 2:
            logger.error(traceback.format_exc())
        return {
            'obs_id': obs_id,
            'wafer': wafer,
            'band': band,
            'results': {
                'success': False,
                'error_message': str(e),
                'aggregate_data': None,
            },
            'aggregate_data': None,
            'worker_walltime_s': worker_walltime_s,
        }

def _main(executor, as_completed_callable, args):
    # Load configs
    with open(args.init_config, 'r') as f:
        configs_init = yaml.safe_load(f)
    
    with open(args.proc_config, 'r') as f:
        configs_proc = yaml.safe_load(f)
    
    # Build process names and flag labels for table creation
    process_names = build_process_names(configs_init, configs_proc)
    cfg_init, context = pp_util.get_preprocess_context(configs_init)
    pipe_init = Pipeline(cfg_init["process_pipe"])
    full_flag_labels = get_sorted_flag_labs(pipe_init)
    base_flag_labels = [label.split('.')[0] for label in full_flag_labels]
    
    # Get observation list first
    if args.obs_list:
        # Load from observation list file
        with open(args.obs_list, 'r') as f:
            obs_ids = [line.strip() for line in f if line.strip()]
        obs_list = [{'obs_id': obs_id} for obs_id in obs_ids]
        logger.info(f'Loaded {len(obs_list)} observations from {args.obs_list}')
    else:
        # Use database query
        obs_list = get_obslist(context, query=args.query, obs_id=args.obs_id, 
                                       min_ctime=args.min_ctime, max_ctime=args.max_ctime,
                                       update_delay=args.update_delay, tags=args.tags)
    
    if len(obs_list) == 0:
        source = args.obs_list if args.obs_list else f"query: {args.query}"
        logger.warning(f"No observations returned from {source}")
        return
    
    logger.info(f'Processing {len(obs_list)} observations')
    
    # Query unique bands from obsdb using tube_flavor
    bands = []
    tube_flavor_to_bands = {
        'lf': ['f030', 'f040'],
        'mf': ['f090', 'f150'],
        'uhf': ['f220', 'f280'],
    }
    try:
        if context.obsdb and obs_list:
            # Query tube_flavor from first observation
            rs = context.obsdb.query(f"obs_id = '{obs_list[0]['obs_id']}'")
            # Convert ResultSet to list to check if results exist
            results = list(rs)
            if len(results) > 0:
                # Access tube_flavor from the first result row
                if 'tube_flavor' in results[0]:
                    tube_flavor = results[0]['tube_flavor']
                    bands = tube_flavor_to_bands.get(tube_flavor, [])
                    if args.verbosity >= 2:
                        logger.info(f"Found tube_flavor='{tube_flavor}' for {obs_list[0]['obs_id']}")
                else:
                    if args.verbosity >= 2:
                        logger.info(f"'tube_flavor' field not in obsdb results. Available fields: {list(results[0].keys())}")
            else:
                if args.verbosity >= 2:
                    logger.info(f"No results from obsdb query for {obs_list[0]['obs_id']}")
        elif not context.obsdb:
            if args.verbosity >= 2:
                logger.info("context.obsdb is not available")
        elif not obs_list:
            if args.verbosity >= 2:
                logger.info("No observation IDs loaded")
    except Exception as e:
        if args.verbosity >= 2:
            logger.error(f"Exception during band detection: {e}")
        if args.verbosity >= 3:
            import traceback
            logger.error(traceback.format_exc())
    
    # Fall back to defaults if band detection fails
    if not bands:
        bands = ['f090', 'f150']
        logger.info(f'Using default bands: {bands}')
    else:
        logger.info(f'Detected bands: {bands}')
    
    # Filter existing observations if not overwriting
    if not args.overwrite and os.path.exists(args.db_path):
        filter_obsdb = None
        try:
            filter_obsdb = ObsDb(map_file=args.db_path, init_db=False)
            existing_obsids = set([row['obs_id'] for row in filter_obsdb.query()])
            obs_list = [obs for obs in obs_list if obs['obs_id'] not in existing_obsids]
            logger.info(f'Filtered to {len(obs_list)} new observations')
        except Exception as e:
            logger.warning(f"Could not filter existing observations: {e}")
        finally:
            if filter_obsdb is not None:
                filter_obsdb.conn.close()
    
    # Create tables and get database connection for reuse
    obsdb = create_cuts_stats_tables(args.db_path, process_names, base_flag_labels)
    
    # Create work list - each item is (obs_id, wafer, band)
    work_items = []
    for obs in obs_list:
        for ws_idx in range(7):
            wafer = f'ws{ws_idx}'
            for band in bands:
                work_items.append((obs['obs_id'], wafer, band))
    
    logger.info(f'Total work items: {len(work_items)}')
    
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
    
    # Process work items in parallel using executor from get_exec_env
    n_processed = 0
    n_failed = 0
    
    try:
        # Submit all work items
        futures = {}
        for obs_id, wafer, band in work_items:
            future = executor.submit(process_work_item, obs_id, wafer, band, 
                                    configs_init, configs_proc, process_names, 
                                    full_flag_labels, args.verbosity, args.aggregate_noise)
            futures[future] = (obs_id, wafer, band)
        
        # Use batch processing for database operations
        batch_size = args.batch_size
        update_count = 0
        
        # Process results as they complete
        for future in as_completed_callable(futures):
            obs_id, wafer, band = futures[future]
            
            try:
                result = future.result()
                results = result['results']
                worker_walltime_s = result.get('worker_walltime_s')
                
                # Save results to database without committing (for batched operations)
                save_results_to_db(args.db_path, obs_id, wafer, band, results,
                                   process_names, base_flag_labels, commit=False,
                                   obsdb=obsdb)
                update_count += 1
                
                # Commit periodically to reduce I/O overhead on distributed filesystems
                if batch_size > 0 and update_count % batch_size == 0:
                    if args.verbosity >= 2:
                        logger.info(f"Committing batch of {batch_size} database updates")
                    obsdb.conn.commit()
                
                # Collect aggregation data if requested and available
                if args.aggregate_noise and result.get('aggregate_data') is not None:
                    agg = result['aggregate_data']
                    band_agg = aggregate_data[band]
                    
                    # Track the range for this GROUP, not this observation.
                    # Work items are (obs_id, wafer, band), so keying on
                    # obs_id alone made each wafer overwrite the previous
                    # wafer's range: the arrays below still received every
                    # wafer, but the index could only find the last one
                    # written, leaving ~80% of the aggregated detectors
                    # unreachable and making every per-observation statistic
                    # a single-wafer sample.
                    n_dets = len(agg['det_ids'])
                    band_agg['obs_ranges'][(obs_id, wafer)] = (
                        band_agg['current_idx'], band_agg['current_idx'] + n_dets)
                    
                    # Append data
                    band_agg['white_noise'].append(agg['white_noise'])
                    band_agg['det_ids'].append(agg['det_ids'])
                    band_agg['wafer_slots'].append(agg['wafer_slots'])
                    band_agg['current_idx'] += n_dets
                
                if results.get('success', True):
                    n_processed += 1
                    if args.verbosity >= 1:
                        if worker_walltime_s is not None:
                            logger.info(f"Processed {obs_id} {wafer} {band} worker={worker_walltime_s:.3f}s")
                        else:
                            logger.info(f"Processed {obs_id} {wafer} {band}")
                else:
                    n_failed += 1
                    error_msg = results.get('error_message', 'Unknown error')
                    if worker_walltime_s is not None:
                        logger.warning(f"Failed {obs_id} {wafer} {band} worker={worker_walltime_s:.3f}s: {error_msg}")
                    else:
                        logger.warning(f"Failed {obs_id} {wafer} {band}: {error_msg}")
                    
            except Exception as e:
                n_failed += 1
                logger.error(f"Exception processing {obs_id} {wafer} {band}: {e}")
                if args.verbosity >= 2:
                    logger.error(traceback.format_exc())
        
        # Final commit for any remaining database updates
        if obsdb is not None and update_count > 0:
            remaining = update_count % batch_size if batch_size > 0 else update_count
            if remaining > 0:
                if args.verbosity >= 2:
                    logger.info(f"Final commit of {remaining} database updates")
                obsdb.conn.commit()
        
        # Log final statistics
        total_processed = n_processed
        total_failed = n_failed
        logger.info(f"Final results: {total_processed} processed, {total_failed} failed")
        
    finally:
        # Close the database connection
        if obsdb is not None:
            obsdb.conn.close()

        # Shutdown process pool if using process-based executor
        if hasattr(executor, 'shutdown'):
            executor.shutdown(wait=True)
    
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
                    # Keys are (obs_id, wafer); the writer emits one entry per
                    # group, so obs_ids repeats once per wafer.
                    'obs_ranges': band_agg['obs_ranges'],
                    'group_keys': list(band_agg['obs_ranges'].keys()),
                }
                
                # Write to HDF5
                output_file = f"{args.aggregate_output}_{band}.h5"
                logger.info(f"Writing {len(band_data['det_ids'])} detectors to {output_file}")
                write_band_noise_to_hdf5(band_data, band, output_file)
            else:
                logger.info(f"No aggregation data for {band}")


def main():
    parser = argparse.ArgumentParser(description='Track cuts and statistics in parallel via sotodlib get_exec_env')
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
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for database commits (set to 0 to commit at end only)')
    parser.add_argument('--nprocs', type=int, help='Number of processes for process-pool mode (MPI mode uses communicator size)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing entries')
    parser.add_argument('--aggregate-noise', choices=['T', 'Q', 'U'], help='Aggregate white noise data for polarization')
    parser.add_argument('--aggregate-output', help='Base path for aggregated HDF5 output (e.g., /path/to/noise)')

    args = parser.parse_args()

    rank, executor, as_completed_callable = get_exec_env(args.nprocs)
    if rank == 0:
        _main(executor=executor, as_completed_callable=as_completed_callable, args=args)

if __name__ == "__main__":
    main()