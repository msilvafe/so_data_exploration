import os
import yaml
import time
import logging
from typing import Optional, Union, Callable, List
import numpy as np
import argparse
import traceback
import sqlite3
from typing import Optional
from sotodlib.utils.procs_pool import get_exec_env
import copy
from tqdm import tqdm
from sotodlib import core
from sotodlib.core import AxisManager
import sotodlib.site_pipeline.util as sp_util
from sotodlib.preprocess import preprocess_util as pp_util
from sotodlib.preprocess import _Preprocess, Pipeline, processes
import sotodlib.preprocess.processes as pcore

logger = sp_util.init_logger("track_det_counts")


def create_counts_table(db_path: str, count_names: List[str]):
    """Create the detector counts table with dynamic columns based on process names."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create columns string for all count names
    count_columns = ', '.join([f'"{name}" INTEGER' for name in count_names])
    
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS detector_counts (
        obsid TEXT,
        wafer TEXT,
        band TEXT,
        {count_columns},
        PRIMARY KEY (obsid, wafer, band)
    )
    """
    
    cursor.execute(create_table_sql)
    conn.commit()
    conn.close()


def insert_counts_to_db(db_path: str, obsid: str, wafer: str, band: str, 
                       counts: List[int], count_names: List[str]):
    """Insert or update detector counts for a specific obsid/wafer/band combination."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the INSERT OR REPLACE query
    placeholders = ', '.join(['?' for _ in count_names])
    columns = ', '.join([f'"{name}"' for name in count_names])
    
    insert_sql = f"""
    INSERT OR REPLACE INTO detector_counts 
    (obsid, wafer, band, {columns})
    VALUES (?, ?, ?, {placeholders})
    """
    
    values = [obsid, wafer, band] + counts
    cursor.execute(insert_sql, values)
    conn.commit()
    conn.close()


def track_det_counts_single(obsid: str,
                           wafer: str,
                           band: str,
                           configs_init: dict,
                           configs_proc: dict,
                           logger,
                           db_path: str,
                           count_names: List[str] = None,
                           run_parallel: bool = False):
    """Track detector counts for a single wafer/band combination."""
    
    error = None
    counts_result = None
    
    try:
        # Initialize contexts
        cfg_init, ctx_init = pp_util.get_preprocess_context(configs_init)
        cfg_proc, ctx_proc = pp_util.get_preprocess_context(configs_proc)
        
        dets = {'wafer_slot': wafer, 'wafer.bandpass': band}
        counts = []
        first_time = (count_names is None)
        
        if first_time:
            names = ['starting']
        else:
            names = count_names[:]
            
        try:
            # First attempt with get_meta (preferred method)
            aman = ctx_init.get_meta(obsid, dets=dets)
            counts.append(aman.dets.count)
            
            proc_aman = aman.preprocess.copy()
            pipe_init = Pipeline(configs_init["process_pipe"])
            
            for i, process in enumerate(pipe_init):
                process.select(aman, proc_aman)
                proc_aman.restrict('dets', aman.dets.vals)
                if first_time:
                    names.append(process.name)
                counts.append(aman.dets.count)
            
            meta_proc = ctx_proc.get_meta(obsid, dets=dets)
            proc_aman.move('valid_data', None)
            proc_aman.merge(meta_proc.preprocess)
            pipe_proc = Pipeline(configs_proc["process_pipe"])
            
            for i, process in enumerate(pipe_proc):
                process.select(aman, proc_aman)
                proc_aman.restrict('dets', aman.dets.vals)
                if first_time:
                    names.append(process.name)
                counts.append(aman.dets.count)
                
            logger.info(f"{wafer} {band} final count: {aman.dets.count}")
            
        except Exception as e:
            # Fallback to get_obs method with full processing
            logger.warning(f"{wafer} {band} get_meta failed, trying get_obs: {e}")
            
            # Reset contexts and modify metadata
            cfg_init, ctx_init = pp_util.get_preprocess_context(configs_init)
            cfg_proc, ctx_proc = pp_util.get_preprocess_context(configs_proc)
            ctx_init['metadata'] = ctx_init['metadata'][:-1]
            ctx_proc['metadata'] = ctx_proc['metadata'][:-1]
            
            try:
                aman = ctx_init.get_obs(obsid, dets=dets)
                counts = [aman.dets.count]
                proc_aman = AxisManager(aman.dets, aman.samps)
                full = AxisManager(aman.dets, aman.samps)
                
                # First Layer (init pipeline)
                pipe_init = Pipeline(configs_init["process_pipe"])
                for i, process in enumerate(pipe_init):
                    aman, proc_aman = process.process(aman, proc_aman)
                    aman, proc_aman = process.calc_and_save(aman, proc_aman)
                    pcore.update_full_aman(proc_aman, full, True)
                    process.select(aman, proc_aman)
                    proc_aman.restrict('dets', aman.dets.vals)
                    if aman.dets.count != 0:
                        counts.append(aman.dets.count)
                
                # Second Layer (proc pipeline)
                pipe_proc = Pipeline(configs_proc["process_pipe"])
                for i, process in enumerate(pipe_proc):
                    logger.debug(f"{i} {process.name}")
                    aman, proc_aman = process.process(aman, proc_aman)
                    aman, proc_aman = process.calc_and_save(aman, proc_aman)
                    pcore.update_full_aman(proc_aman, full, True)
                    process.select(aman, proc_aman)
                    proc_aman.restrict('dets', aman.dets.vals)
                    if aman.dets.count != 0:
                        counts.append(aman.dets.count)
                        
                logger.info(f"{wafer} {band} final count (get_obs): {aman.dets.count}")
                
            except Exception as e2:
                logger.error(f"{wafer} {band} failed completely: {e2}")
                error = f"Both get_meta and get_obs failed for {wafer} {band}"
                if run_parallel:
                    return error, None, None
                else:
                    return
        
        if first_time:
            count_names = [f'{i}_{name}' for i, name in enumerate(names)]
        
        counts_result = {
            'obsid': obsid,
            'wafer': wafer,
            'band': band,
            'counts': counts,
            'count_names': count_names
        }
        
        if not run_parallel:
            # Store directly in database
            insert_counts_to_db(db_path, obsid, wafer, band, counts, count_names)
        
    except Exception as e:
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        logger.error(f"ERROR: {obsid} {wafer} {band}\n{errmsg}\n{tb}")
        error = errmsg
        if run_parallel:
            return error, counts_result, count_names
        else:
            return
    
    if run_parallel:
        return error, counts_result, count_names


def track_det_counts_obs(obsid: str,
                        configs_init,
                        configs_proc,
                        db_path: str,
                        verbosity: int = 2,
                        run_parallel: bool = False):
    """Track detector counts for all wafer/band combinations for a single observation."""
    
    logger = sp_util.init_logger("track_det_counts", verbosity=verbosity)
    
    if isinstance(configs_init, str):
        configs_init = yaml.safe_load(open(configs_init, "r"))
    if isinstance(configs_proc, str):
        configs_proc = yaml.safe_load(open(configs_proc, "r"))
    
    outputs = []
    count_names = None
    
    # Process all wafer/band combinations
    missing_ws_bands = []
    
    for wsi in range(7):
        ws = f'ws{wsi}'
        for band in ['f090', 'f150']:
            try:
                error, result, names = track_det_counts_single(
                    obsid, ws, band, configs_init, configs_proc, 
                    logger, db_path, count_names, run_parallel
                )
                
                if count_names is None and names is not None:
                    count_names = names
                    if not run_parallel:
                        # Create table on first successful run
                        create_counts_table(db_path, count_names)
                
                if result is not None:
                    if run_parallel:
                        outputs.append(result)
                else:
                    missing_ws_bands.append((ws, band))
                    
            except Exception as e:
                logger.error(f"Failed to process {ws} {band}: {e}")
                missing_ws_bands.append((ws, band))
    
    if missing_ws_bands:
        logger.warning(f"Missing wafer/band combinations for {obsid}: {missing_ws_bands}")
    
    if run_parallel:
        return None, outputs, count_names
    

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('config_init', help="Initial Preprocessing Configuration File")
    parser.add_argument('config_proc', help="Processing Configuration File")
    parser.add_argument('db_path', help="Path to SQLite database file")
    parser.add_argument(
        '--query',
        help="Query to pass to the observation list. Use \\'string\\' to "
             "pass in strings within the query.",
        type=str
    )
    parser.add_argument(
        '--obs-id',
        help="obs-id of particular observation if we want to run on just one"
    )
    parser.add_argument(
        '--overwrite',
        help="If true, overwrites existing entries in the database",
        action='store_true',
    )
    parser.add_argument(
        '--min-ctime',
        help="Minimum timestamp for the beginning of an observation list",
    )
    parser.add_argument(
        '--max-ctime',
        help="Maximum timestamp for the beginning of an observation list",
    )
    parser.add_argument(
        '--update-delay',
        help="Number of days (unit is days) in the past to start observation list.",
        type=int
    )
    parser.add_argument(
        '--tags',
        help="Observation tags. Ex: --tags 'jupiter' 'setting'",
        nargs='*',
        type=str
    )
    parser.add_argument(
        '--verbosity',
        help="increase output verbosity. 0:Error, 1:Warning, 2:Info(default), 3:Debug",
        default=2,
        type=int
    )
    parser.add_argument(
        '--nproc',
        help="Number of parallel processes to run on.",
        type=int,
        default=4
    )
    parser.add_argument(
        '--raise-error',
        help="Raise an error upon completion if any obsids fail.",
        type=bool,
        default=False
    )
    return parser


def _main(executor: Union["MPICommExecutor", "ProcessPoolExecutor"],
          as_completed_callable: Callable,
          config_init: str,
          config_proc: str,
          db_path: str,
          query: Optional[str] = None,
          obs_id: Optional[str] = None,
          overwrite: bool = False,
          min_ctime: Optional[int] = None,
          max_ctime: Optional[int] = None,
          update_delay: Optional[int] = None,
          tags: Optional[List[str]] = None,
          verbosity: Optional[int] = None,
          nproc: Optional[int] = 4,
          raise_error: Optional[bool] = False):

    configs_init, context = pp_util.get_preprocess_context(config_init)
    configs_proc, _ = pp_util.get_preprocess_context(config_proc)
    logger = sp_util.init_logger("track_det_counts", verbosity=verbosity)

    errlog = os.path.join(os.path.dirname(db_path), 'det_counts_errlog.txt')

    obs_list = sp_util.get_obslist(context, query=query, obs_id=obs_id, min_ctime=min_ctime,
                                   max_ctime=max_ctime, update_delay=update_delay, tags=tags)

    if len(obs_list) == 0:
        logger.warning(f"No observations returned from query: {query}")
        return

    logger.info(f'Processing {len(obs_list)} observations')

    # Check if database exists and handle overwrite
    if not overwrite and os.path.exists(db_path):
        # Filter out observations that already exist in database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT DISTINCT obsid FROM detector_counts")
            existing_obsids = set([row[0] for row in cursor.fetchall()])
            obs_list = [obs for obs in obs_list if obs['obs_id'] not in existing_obsids]
            logger.info(f'Filtered to {len(obs_list)} new observations')
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            pass
        finally:
            conn.close()

    run_list = [obs for obs in obs_list]
    n_fail = 0
    count_names = None

    # Run observations in parallel
    futures = [executor.submit(track_det_counts_obs, obs['obs_id'],
                              configs_init=configs_init,
                              configs_proc=configs_proc,
                              db_path=db_path,
                              verbosity=verbosity,
                              run_parallel=True) for obs in run_list]
    
    for future in as_completed_callable(futures):
        logger.info('New future completed')
        try:
            err, results, names = future.result()
            
            if count_names is None and names is not None:
                count_names = names
                create_counts_table(db_path, count_names)
            
            if err is not None:
                n_fail += 1
                logger.error(f"Future failed: {err}")
            elif results:
                # Store results in database
                for result in results:
                    insert_counts_to_db(db_path, result['obsid'], result['wafer'], 
                                      result['band'], result['counts'], count_names)
                logger.info(f"Stored {len(results)} wafer/band results")
                
        except Exception as e:
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            logger.error(f"ERROR: future.result()\n{errmsg}\n{tb}")
            
            with open(errlog, 'a') as f:
                f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
            n_fail += 1
            continue
        
        futures.remove(future)

    if raise_error and n_fail > 0:
        raise RuntimeError(f"track_det_counts: {n_fail}/{len(run_list)} obs_ids failed")

    logger.info(f"Detector count tracking completed. Failed: {n_fail}/{len(run_list)}")


def main(config_init: str,
         config_proc: str,
         db_path: str,
         query: Optional[str] = None,
         obs_id: Optional[str] = None,
         overwrite: bool = False,
         min_ctime: Optional[int] = None,
         max_ctime: Optional[int] = None,
         update_delay: Optional[int] = None,
         tags: Optional[List[str]] = None,
         verbosity: Optional[int] = None,
         nproc: Optional[int] = 4,
         raise_error: Optional[bool] = False):

    rank, executor, as_completed_callable = get_exec_env(nproc)
    if rank == 0:
        _main(executor=executor,
              as_completed_callable=as_completed_callable,
              config_init=config_init,
              config_proc=config_proc,
              db_path=db_path,
              query=query,
              obs_id=obs_id,
              overwrite=overwrite,
              min_ctime=min_ctime,
              max_ctime=max_ctime,
              update_delay=update_delay,
              tags=tags,
              verbosity=verbosity,
              nproc=nproc,
              raise_error=raise_error)


if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)