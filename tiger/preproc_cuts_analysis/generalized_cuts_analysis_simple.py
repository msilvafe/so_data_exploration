#!/usr/bin/env python3
"""
Generalized Preprocessing Cuts Analysis Tool (Simplified)

This script analyzes preprocessing cuts and flags from sotodlib preprocessing pipelines
by automatically extracting cuts information from configuration files and generating
a database with cuts statistics.

Key improvements over the original iso_v1_stats_20250718.py:
1. Automatically parses configuration files to determine cut order
2. Removes unused TOD stats fields 
3. Only includes fields ending in 'nsamps' (sample cuts) or 'cuts' (detector cuts)
4. Generalizes to work with any preprocessing configuration
"""

import numpy as np
import os
import yaml
import argparse
import traceback
import time
import sqlite3
from typing import Dict, List, Tuple, Optional, Any

from sotodlib import core
from sotodlib.core.flagman import has_all_cut, has_any_cuts, count_cuts
import sotodlib.site_pipeline.util as sp_util
from sotodlib.preprocess import preprocess_util, pcore
from sotodlib.utils.procs_pool import get_exec_env
from so3g.proj import Ranges, RangesMatrix


def get_wrap_name_from_process_name(process_name: str, process_config: dict = None) -> str:
    """
    Map from the process name in the config to the actual wrap name used in proc_aman.wrap().
    This is necessary because the process class 'name' attribute doesn't always match
    the string passed to proc_aman.wrap() in the save method.
    
    Args:
        process_name: The 'name' field from the config process
        process_config: The full process config dict (for customizable wrap names)
        
    Returns:
        The actual wrap name used in proc_aman.wrap()
    """
    if process_config is None:
        process_config = {}
    
    # Handle processes with customizable wrap names based on sotodlib/preprocess/processes.py
    if process_name == 'glitches':
        # GlitchDetection: uses self.glitch_name = step_cfgs.get('glitch_name', 'glitches')
        return process_config.get('glitch_name', 'glitches')
        
    elif process_name == 'jumps':
        # Jumps: uses self.save_cfgs.get('jumps_name', 'jumps')
        save_cfg = process_config.get('save', {})
        if isinstance(save_cfg, dict):
            return save_cfg.get('jumps_name', 'jumps')
        else:
            return 'jumps'
            
    elif process_name == 'source_flags':
        # SourceFlags: uses self.source_flags_name = step_cfgs.get('source_flags_name', 'source_flags')
        return process_config.get('source_flags_name', 'source_flags')
        
    elif process_name == 'noise':
        # CalcNoise: uses self.save_cfgs['wrap_name'] if specified, otherwise 'noise'
        save_cfg = process_config.get('save', {})
        if isinstance(save_cfg, dict):
            return save_cfg.get('wrap_name', 'noise')
        else:
            return 'noise'
    
    # Static mapping for processes where name == wrap_name
    static_mapping = {
        # Flag processes that use their name directly
        'split_flags': 'split_flags',
        'ptp_flags': 'ptp_flags', 
        'inv_var_flags': 'inv_var_flags',
        
        # Processes with name != wrap_name but no customization
        'flag_turnarounds': 'turnaround_flags',
        'det_bias_cuts': 'det_bias_flags',
        'trends': 'trends',
        
        # Other processes that use name directly
        'hwp_angle': 'hwp_angle',
        'darks': 'darks',
        'sso_footprint': 'sso_footprint',
    }
    
    # Return mapped name or fall back to original name
    return static_mapping.get(process_name, process_name)


def parse_config_for_cuts(config_file: str) -> Tuple[List[str], List[str], str]:
    """
    Parse configuration file to extract the order of sample and detector cuts
    and the name of the last process.
    
    Args:
        config_file: Path to preprocessing configuration file
        
    Returns:
        Tuple of (sample_cuts_order, detector_cuts_order, last_process_name)
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    sample_cuts = []
    detector_cuts = []
    last_process_wrap_name = None
    
    for process in config.get('process_pipe', []):
        name = process.get('name', '')
        # Get the actual wrap name that will be used in proc_aman.wrap()
        wrap_name = get_wrap_name_from_process_name(name, process)
        last_process_wrap_name = wrap_name  # Track the last process wrap name
        
        # Processes that primarily create sample cuts/flags
        if any(flag_type in name for flag_type in [
            'turnaround', 'jump', 'glitch', 'source', 'smurf', 'subscan'
        ]):
            # Generate the nsamps field name
            if 'turnaround' in name:
                cut_name = 'turnaround_nsamps'
                if cut_name not in sample_cuts:
                    sample_cuts.append(cut_name)
            elif 'jump' in name:
                # Handle different jump types
                if 'slow' in name or process.get('calc', {}).get('function') == 'slow_jumps':
                    cut_name = 'jumps_slow_nsamps'
                elif '2pi' in name or process.get('calc', {}).get('function') == 'twopi_jumps':
                    cut_name = 'jumps_2pi_nsamps'
                else:
                    cut_name = 'jumps_nsamps'
                if cut_name not in sample_cuts:
                    sample_cuts.append(cut_name)
            elif 'glitch' in name:
                cut_name = 'glitch_nsamps'
                if cut_name not in sample_cuts:
                    sample_cuts.append(cut_name)
            elif 'source' in name:
                cut_name = 'source_flags_nsamps'
                if cut_name not in sample_cuts:
                    sample_cuts.append(cut_name)
            elif 'subscan' in name:
                cut_name = 'noisy_subscans_nsamps'
                if cut_name not in sample_cuts:
                    sample_cuts.append(cut_name)
        
        # Processes that primarily create detector cuts
        elif any(flag_type in name for flag_type in [
            'fp_', 'trend', 'bias', 'ptp', 'noise', 'inv_var'
        ]):
            if 'fp_' in name or 'fp' == name:
                cut_name = 'fp_cuts'
            elif 'trend' in name:
                cut_name = 'trend_cuts'
            elif 'bias' in name:
                cut_name = 'det_bias_cuts'
            elif 'ptp' in name:
                cut_name = 'ptp_cuts'
            elif 'noise' in name:
                cut_name = 'white_noise_cuts'
            elif 'inv_var' in name:
                cut_name = 'inv_var_cuts'
            else:
                continue
            
            if cut_name not in detector_cuts:
                detector_cuts.append(cut_name)
    
    return sample_cuts, detector_cuts, last_process_wrap_name


def count_new_cuts(ranges_matrix, survivors_mask, already_cut):
    """
    Count new sample cuts from a ranges matrix.
    (Same as original function)
    """
    n_dets = ranges_matrix.shape[0]
    new_cut_count = 0
    updated_already_cut = []
    
    for det_idx in range(n_dets):
        if not survivors_mask[det_idx]:
            updated_already_cut.append(already_cut[det_idx])
            continue
            
        op_ranges = ranges_matrix.ranges[det_idx]
        prev_ranges = already_cut[det_idx]
        
        # Compute new cuts: samples in op_ranges but not in prev_ranges
        det_cut_count = np.sum(np.ptp((op_ranges * ~prev_ranges).ranges(), axis=1))
        
        if det_cut_count != 0:
            new_cut_count += det_cut_count
            updated_already_cut.append(prev_ranges + op_ranges)
        else:
            updated_already_cut.append(prev_ranges)
            
    return new_cut_count, updated_already_cut


def get_dict_entry(entry, config_file_init, config_file_proc, noise_range, 
                   sample_cuts_order, detector_cuts_order, last_process_name):
    """
    Generalized version of get_dict_entry that uses configuration-derived order
    and dynamically determines proc_survivors_mask from the last process wrap name.
    
    Args:
        last_process_name: The actual wrap name used in proc_aman.wrap() for the final process
    """
    try:
        logger = preprocess_util.init_logger('subproc_logger')
        logger.info(f'Processing entry for {entry["dataset"]}')
        
        _, context_init = preprocess_util.get_preprocess_context(config_file_init)
        _, context_proc = preprocess_util.get_preprocess_context(config_file_proc)
        
        dets = {'wafer_slot': entry['dets:wafer_slot'],
               'wafer.bandpass': entry['dets:wafer.bandpass']}
        
        mdata_init = context_init.get_meta(entry['obs:obs_id'], dets=dets)
        mdata_proc = context_proc.get_meta(entry['obs:obs_id'], dets=dets)
        
        del context_proc
        del context_init
        
        x = mdata_init.preprocess
        ndet = mdata_init.dets.count
        nsamp = mdata_init.samps.count
        
        # Find detectors that survive all cuts
        # Get proc_survivors_mask from the last process in the pipeline
        if not last_process_name:
            raise ValueError("No last process name found in configuration file")
            
        if not hasattr(mdata_proc.preprocess, last_process_name):
            raise AttributeError(f"Last process '{last_process_name}' not found in mdata_proc.preprocess. "
                                f"Available attributes: {list(mdata_proc.preprocess._fields.keys())}")
            
        last_process_data = getattr(mdata_proc.preprocess, last_process_name)
        
        if not hasattr(last_process_data, 'valid'):
            raise AttributeError(f"Last process '{last_process_name}' has no 'valid' attribute. "
                                f"Available attributes: {list(last_process_data._fields.keys())}")
            
        proc_survivors_mask = has_all_cut(last_process_data.valid)
        
        survivors_mask = np.zeros(mdata_init.dets.count, dtype=bool)
        _, ind_init, _ = np.intersect1d(mdata_init.dets.vals, 
                                      mdata_proc.dets.vals[proc_survivors_mask], 
                                      return_indices=True)
        survivors_mask[ind_init] = True
        
        # Initialize tracking
        already_cut = RangesMatrix.zeros((ndet, nsamp))
        
        keys = []
        vals = []
        
        # Base counts
        keys.extend(['nsamps', 'ndets'])
        vals.extend([nsamp, ndet])
        
        # Process detector cuts from init config (in order)
        for cut_name in detector_cuts_order:
            if cut_name == 'fp_cuts' and hasattr(x, 'fp_flags'):
                m = has_all_cut(x.fp_flags.valid)
                keys.append('fp_cuts')
                vals.append(np.sum(has_all_cut(x.fp_flags.fp_nans)[m]))
                
            elif cut_name == 'trend_cuts' and hasattr(x, 'trends'):
                m = has_all_cut(x.trends.valid)
                keys.append('trend_cuts')
                vals.append(np.sum(has_all_cut(x.trends.trend_flags)[m]))
                
            elif cut_name == 'det_bias_cuts' and hasattr(x, 'det_bias_flags'):
                m = has_all_cut(x.det_bias_flags.valid)
                keys.append('det_bias_cuts')
                vals.append(np.sum(has_all_cut(x.det_bias_flags.det_bias_flags)[m]))
                
            elif cut_name == 'ptp_cuts' and hasattr(x, 'ptp_flags'):
                m = has_all_cut(x.ptp_flags.valid)
                keys.append('ptp_cuts')
                vals.append(np.sum(has_all_cut(x.ptp_flags.ptp_flags)[m]))
                
            elif cut_name == 'white_noise_cuts' and hasattr(x, 'white_noise_nofit'):
                m = has_all_cut(x.white_noise_nofit.valid)
                n_min, n_max = noise_range
                keys.append('white_noise_cuts')
                vals.append(np.sum(((x.white_noise_nofit.white_noise)[m] < n_min) | 
                                  ((x.white_noise_nofit.white_noise)[m] > n_max)))
        
        # Process sample cuts from init config (in order)
        for cut_name in sample_cuts_order:
            if cut_name == 'turnaround_nsamps' and hasattr(x, 'turnaround_flags'):
                n_cuts, already_cut = count_new_cuts(x.turnaround_flags.turnarounds, 
                                                    survivors_mask, already_cut)
                keys.append('turnaround_nsamps')
                vals.append(int(n_cuts))
                
            elif cut_name == 'jumps_slow_nsamps' and hasattr(x, 'jumps_slow'):
                n_cuts, already_cut = count_new_cuts(x.jumps_slow.jump_flag, 
                                                    survivors_mask, already_cut)
                keys.append('jumps_slow_nsamps')
                vals.append(int(n_cuts))
                
            elif cut_name == 'jumps_2pi_nsamps' and hasattr(x, 'jumps_2pi'):
                n_cuts, already_cut = count_new_cuts(x.jumps_2pi.jump_flag, 
                                                    survivors_mask, already_cut)
                keys.append('jumps_2pi_nsamps')
                vals.append(int(n_cuts))
                
            elif cut_name == 'jumps_nsamps' and hasattr(x, 'jumps'):
                # Generic jumps (if not handled by specific types above)
                n_cuts, already_cut = count_new_cuts(x.jumps.jump_flag, 
                                                    survivors_mask, already_cut)
                keys.append('jumps_nsamps')
                vals.append(int(n_cuts))
                
            elif cut_name == 'glitch_nsamps' and hasattr(x, 'glitches'):
                n_cuts, already_cut = count_new_cuts(x.glitches.glitch_flags, 
                                                    survivors_mask, already_cut)
                keys.append('glitch_nsamps')
                vals.append(int(n_cuts))
                
            elif cut_name == 'source_flags_nsamps' and hasattr(x, 'source_flags'):
                n_cuts, already_cut = count_new_cuts(x.source_flags.moon, 
                                                    survivors_mask, already_cut)
                keys.append('source_flags_nsamps')
                vals.append(int(n_cuts))
        
        # Add corresponding detector cuts for sample cuts that also cut detectors
        if hasattr(x, 'jumps_slow'):
            m = has_all_cut(x.jumps_slow.valid)
            keys.append('jumps_slow_cuts')
            vals.append(np.sum(count_cuts(x.jumps_slow.jump_flag)[m] > 5))
            
        if hasattr(x, 'jumps_2pi'):
            m = has_all_cut(x.jumps_2pi.valid)
            keys.append('jumps_2pi_cuts')
            vals.append(np.sum(count_cuts(x.jumps_2pi.jump_flag)[m] > 20))
            
        if hasattr(x, 'glitches'):
            m = has_all_cut(x.glitches.valid)
            keys.append('glitch_cuts')
            vals.append(np.sum(count_cuts(x.glitches.glitch_flags)[m] > 50000))
            
        if hasattr(x, 'source_flags'):
            m = has_all_cut(x.source_flags.valid)
            keys.append('source_flags_cuts')
            vals.append(np.sum(has_all_cut(x.source_flags.moon)[m]))
        
        # Edge cuts (common in demodulation)
        edge_mask = np.zeros((ndet, nsamp), dtype=bool)
        edge_mask[:, :6000] = True
        edge_mask[:, -6000:] = True
        edge_ranges = RangesMatrix.from_mask(edge_mask)
        n_edge_cuts, already_cut = count_new_cuts(edge_ranges, survivors_mask, already_cut)
        keys.append('edge_nsamps')
        vals.append(int(n_edge_cuts))
        
        # Processing cuts
        x_proc = mdata_proc.preprocess
        
        # Inv var cuts
        if hasattr(x_proc, 'inv_var_flags'):
            m = has_all_cut(x_proc.inv_var_flags.valid)
            keys.append('inv_var_cuts')
            vals.append(np.sum(has_all_cut(x_proc.inv_var_flags.inv_var_flags)[m]))
        
        # Noisy subscan cuts
        if hasattr(x_proc, 'noisy_subscan_flags'):
            n_cuts, already_cut = count_new_cuts(x_proc.noisy_subscan_flags.valid_subscans, 
                                                survivors_mask, already_cut)
            keys.append('noisy_subscans_nsamps')
            vals.append(int(n_cuts))
            
            # Noisy subscan detector cuts
            keys.append('noisy_subscans_cuts')
            vals.append(np.sum(~x_proc.noisy_dets_flags.valid_dets))
        
        # End yield - use the last process dynamically
        # Note: We already validated that last_process_name exists and has 'valid' attribute above
        last_process_data = getattr(x_proc, last_process_name)
        m = has_all_cut(last_process_data.valid)
        
        keys.append('end_yield')
        vals.append(np.sum(m))
        
        return entry['obs:obs_id'], entry['dets:wafer_slot'], entry['dets:wafer.bandpass'], keys, vals
        
    except Exception as e:
        logger.info(f"Error in process for {entry['dataset']}")
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        return None, None, None, errmsg, tb


def get_parser(parser=None):
    """Create argument parser."""
    if parser is None:
        parser = argparse.ArgumentParser(
            description='Generalized preprocessing cuts analysis'
        )
    
    parser.add_argument('config_file_init',
                       help="Preprocessing init configuration file")
    parser.add_argument('config_file_proc',
                       help="Preprocessing proc configuration file")
    parser.add_argument('--noise-range',
                       nargs=2,
                       type=float,
                       metavar=('N_MIN', 'N_MAX'),
                       default=[18e-6, 80e-6],
                       help="Noise range for white_noise cuts: [N_MIN N_MAX] (default: 18e-6 80e-6)")
    parser.add_argument('--nproc',
                       help="Number of parallel processes to run on.",
                       type=int,
                       default=16)
    parser.add_argument('--errlog-ext',
                       help="Error log file name.",
                       default='generalized_cuts_err.txt')
    parser.add_argument('--savename',
                       help="Output database save name.",
                       default='generalized_cuts.sqlite')
    
    return parser


def main(executor, as_completed_callable, config_file_init, config_file_proc,
         errlog_ext, savename, noise_range, nproc):
    """Main processing function."""
    
    logger = preprocess_util.init_logger('main_proc')
    configs_proc, _ = preprocess_util.get_preprocess_context(config_file_proc)
    base_dir = os.path.dirname(configs_proc['archive']['index'])
    errlog = errlog_ext
    
    # Parse configurations to get cut order
    logger.info('Parsing configuration files for cuts order...')
    # Parse configuration files to get cuts ordering and last process wrap names
    sample_cuts_init, detector_cuts_init, last_wrap_name_init = parse_config_for_cuts(config_file_init)
    sample_cuts_proc, detector_cuts_proc, last_wrap_name_proc = parse_config_for_cuts(config_file_proc)
    
    # Combine orders (init first, then proc)
    sample_cuts_order = sample_cuts_init + sample_cuts_proc
    detector_cuts_order = detector_cuts_init + detector_cuts_proc
    
    logger.info(f'Found sample cuts order: {sample_cuts_order}')
    logger.info(f'Found detector cuts order: {detector_cuts_order}')
    
    logger.info('Connect to database')
    proc = core.metadata.ManifestDb(os.path.join(base_dir, 'process_archive.sqlite'))
    
    sqlite_path = savename.replace('.h5', '.sqlite')
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    
    run_list = proc.inspect()
    logger.info('Run list created')
    
    # Get first entry to build schema
    first_entry = run_list[0]
    obsid, ws, band, keys, vals = get_dict_entry(
        entry=first_entry, 
        config_file_init=config_file_init,
        config_file_proc=config_file_proc, 
        noise_range=noise_range,
        sample_cuts_order=sample_cuts_order,
        detector_cuts_order=detector_cuts_order,
        last_process_name=last_wrap_name_proc
    )
    
    # Build table schema
    columns = ', '.join([f'"{k}" INTEGER' for k in keys])
    create_stmt = f'''
        CREATE TABLE IF NOT EXISTS results (
            obsid TEXT,
            ws TEXT,
            band TEXT,
            {columns},
            PRIMARY KEY (obsid, ws, band)
        )
    '''
    cur.execute(create_stmt)
    conn.commit()
    
    del proc
    logger.info('Deleted database connection')
    
    n = 0
    ntot = len(run_list)
    
    logger.info(f'Writing to sqlite file at {sqlite_path}')
    futures = [executor.submit(get_dict_entry, 
                              entry=entry,
                              config_file_init=config_file_init,
                              config_file_proc=config_file_proc,
                              noise_range=noise_range,
                              sample_cuts_order=sample_cuts_order,
                              detector_cuts_order=detector_cuts_order,
                              last_process_name=last_wrap_name_proc) 
               for entry in run_list]
    
    for future in as_completed_callable(futures):
        try:
            obsid, ws, band, keys, vals = future.result()
            logger.info(f'{n}/{ntot}: Unpacked future for {ws}, {band}')
        except Exception as e:
            logger.info(f'{n}/{ntot}: Future unpack error.')
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            with open(errlog, 'a') as f:
                f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
            continue
            
        futures.remove(future)
        
        if obsid is None:
            logger.info('Writing error to log.')
            with open(errlog, 'a') as f:
                f.write(f'\n{time.time()}, error\n{keys}\n{vals}\n')
        else:
            try:
                col_names = ['obsid', 'ws', 'band'] + list(keys)
                placeholders = ','.join(['?'] * len(col_names))
                vals_to_store = [int(v) if isinstance(v, (np.integer, np.int64, np.int32)) else v for v in vals]
                row_values = [obsid, ws, band] + vals_to_store
                insert_stmt = f'INSERT OR REPLACE INTO results ({",".join(col_names)}) VALUES ({placeholders})'
                cur.execute(insert_stmt, row_values)
                conn.commit()
                logger.info(f'{n}/{ntot}: Finished with {obsid} {ws} {band}.')
            except Exception as e:
                logger.info('Packaging and saving error.')
                errmsg = f'{type(e)}: {e}'
                tb = ''.join(traceback.format_tb(e.__traceback__))
                with open(errlog, 'a') as f:
                    f.write(f'\n{time.time()}, save error\n{errmsg}\n{tb}\n')
                continue
        n += 1
        
    logger.info(f'All entries written to sqlite file at {sqlite_path}')
    conn.close()


if __name__ == '__main__':
    args = get_parser().parse_args()
    rank, executor, as_completed_callable = get_exec_env(args.nproc)
    if rank == 0:
        main(executor=executor, as_completed_callable=as_completed_callable, **vars(args))
