#!/usr/bin/env python3
"""
Track detector and sample cuts along with noise statistics during SO preprocessing pipeline.
This script runs the pipeline step-by-step to track actual detector selection like track_det_counts.py.
"""

import os
import yaml
import time
import logging
from typing import Optional, Union, Callable, List, Dict, Any
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
from sotodlib.preprocess import _Preprocess, Pipeline, processes, pcore

logger = sp_util.init_logger("track_cuts_and_stats")

def get_quantiles(data, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
    """Calculate quantiles of data, handling NaN values."""
    data_clean = data[~np.isnan(data)]
    if len(data_clean) == 0:
        return [np.nan] * len(quantiles)
    return np.quantile(data_clean, quantiles)

def get_samp_cuts_from_ranges(ranges_matrix, total_samples):
    """Calculate number of samples cut from RangesMatrix object."""
    if ranges_matrix is None:
        return 0
    try:
        # Count flagged samples
        flagged_samples = ranges_matrix.ranges
        return len(flagged_samples) if flagged_samples is not None else 0
    except Exception as e:
        logger.warning(f"Error getting sample cuts: {e}")
        return 0

def build_process_names(configs_init, configs_proc):
    """Derive process names from the configured pipelines."""
    cfg_init, _ = pp_util.get_preprocess_context(configs_init)
    cfg_proc, _ = pp_util.get_preprocess_context(configs_proc)

    names = ["starting"]

    pipe_init = Pipeline(cfg_init["process_pipe"])
    for process in pipe_init:
        names.append(process.name)

    if configs_proc:
        pipe_proc = Pipeline(cfg_proc["process_pipe"])
        for process in pipe_proc:
            names.append(process.name)

    return names

def create_cuts_stats_tables(db_path: str, process_names: List[str]):
    """Create tables for cuts and statistics tracking."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create detector counts table with columns for each process step
    # Make sure column names are unique by adding indices
    det_count_columns = ', '.join([f'"{i}_{name}" INTEGER' for i, name in enumerate(process_names)])
    
    create_det_counts_sql = f"""
    CREATE TABLE IF NOT EXISTS detector_counts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        obs_id TEXT NOT NULL,
        wafer TEXT NOT NULL,
        band TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        {det_count_columns},
        UNIQUE(obs_id, wafer, band)
    )"""
    
    cursor.execute(create_det_counts_sql)
    
    # Create sample cuts table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sample_cuts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        obs_id TEXT NOT NULL,
        wafer TEXT NOT NULL,
        band TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        total_samples INTEGER,
        smurfgaps_cuts INTEGER DEFAULT 0,
        turnarounds_cuts INTEGER DEFAULT 0,
        jumps_slow_cuts INTEGER DEFAULT 0,
        jumps_2pi_cuts INTEGER DEFAULT 0,
        glitches_pre_hwpss_cuts INTEGER DEFAULT 0,
        glitches_post_hwpss_cuts INTEGER DEFAULT 0,
        source_moon_cuts INTEGER DEFAULT 0,
        UNIQUE(obs_id, wafer, band)
    )""")
    
    # Create noise statistics table  
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS noise_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        obs_id TEXT NOT NULL,
        wafer TEXT NOT NULL,
        band TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        polarization TEXT NOT NULL,
        white_noise_avg REAL,
        white_noise_q10 REAL,
        white_noise_q25 REAL,
        white_noise_q50 REAL,
        white_noise_q75 REAL,
        white_noise_q90 REAL,
        fknee_avg REAL,
        fknee_q10 REAL,
        fknee_q25 REAL,
        fknee_q50 REAL,
        fknee_q75 REAL,
        fknee_q90 REAL,
        alpha_avg REAL,
        alpha_q10 REAL,
        alpha_q25 REAL,
        alpha_q50 REAL,
        alpha_q75 REAL,
        alpha_q90 REAL,
        UNIQUE(obs_id, wafer, band, polarization)
    )""")
    
    # Create T2P statistics table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS t2p_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        obs_id TEXT NOT NULL,
        wafer TEXT NOT NULL,
        band TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        coeffsQ_avg REAL,
        coeffsU_avg REAL,
        errorsQ_avg REAL,
        errorsU_avg REAL,
        redchi2sQ_avg REAL,
        redchi2sU_avg REAL,
        UNIQUE(obs_id, wafer, band)
    )""")
    
    # Create failed observations table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS failed_observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        obs_id TEXT NOT NULL,
        wafer TEXT NOT NULL,
        band TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        failure_step TEXT,
        error_message TEXT,
        UNIQUE(obs_id, wafer, band)
    )""")
    
    conn.commit()
    conn.close()

def extract_statistics(aman_init, aman_proc):
    """Extract noise and T2P statistics from processed data."""
    noise_stats = {}
    t2p_stats = {}
    
    # Extract T noise statistics from init pipeline
    if hasattr(aman_init, 'noiseT') and hasattr(aman_init.noiseT, 'white_noise'):
        wn_quantiles = get_quantiles(aman_init.noiseT.white_noise)
        noise_stats['T'] = {
            'white_noise_avg': np.nanmean(aman_init.noiseT.white_noise),
            'white_noise_q10': wn_quantiles[0],
            'white_noise_q25': wn_quantiles[1], 
            'white_noise_q50': wn_quantiles[2],
            'white_noise_q75': wn_quantiles[3],
            'white_noise_q90': wn_quantiles[4],
            'fknee_avg': np.nan,  # No fknee in T noise
            'fknee_q10': np.nan,
            'fknee_q25': np.nan,
            'fknee_q50': np.nan,
            'fknee_q75': np.nan,
            'fknee_q90': np.nan,
            'alpha_avg': np.nan,  # No alpha in T noise
            'alpha_q10': np.nan,
            'alpha_q25': np.nan,
            'alpha_q50': np.nan,
            'alpha_q75': np.nan,
            'alpha_q90': np.nan,
        }
    
    # Extract Q and U noise statistics from proc pipeline
    if aman_proc:
        for pol in ['Q', 'U']:
            noise_field = f'noise{pol}'
            if hasattr(aman_proc, noise_field):
                noise_obj = getattr(aman_proc, noise_field)
                noise_data = {}
                
                if hasattr(noise_obj, 'white_noise'):
                    wn_quantiles = get_quantiles(noise_obj.white_noise)
                    noise_data.update({
                        'white_noise_avg': np.nanmean(noise_obj.white_noise),
                        'white_noise_q10': wn_quantiles[0],
                        'white_noise_q25': wn_quantiles[1],
                        'white_noise_q50': wn_quantiles[2],
                        'white_noise_q75': wn_quantiles[3],
                        'white_noise_q90': wn_quantiles[4],
                    })
                
                if hasattr(noise_obj, 'fit'):
                    # fit array has [white_noise, fknee, alpha] for each detector
                    if noise_obj.fit.ndim == 2 and noise_obj.fit.shape[1] >= 3:
                        # fknee is column 1
                        fknee_quantiles = get_quantiles(noise_obj.fit[:, 1])
                        # alpha is column 2 
                        alpha_quantiles = get_quantiles(noise_obj.fit[:, 2])
                        
                        noise_data.update({
                            'fknee_avg': np.nanmean(noise_obj.fit[:, 1]),
                            'fknee_q10': fknee_quantiles[0],
                            'fknee_q25': fknee_quantiles[1],
                            'fknee_q50': fknee_quantiles[2],
                            'fknee_q75': fknee_quantiles[3],
                            'fknee_q90': fknee_quantiles[4],
                            'alpha_avg': np.nanmean(noise_obj.fit[:, 2]),
                            'alpha_q10': alpha_quantiles[0],
                            'alpha_q25': alpha_quantiles[1],
                            'alpha_q50': alpha_quantiles[2],
                            'alpha_q75': alpha_quantiles[3],
                            'alpha_q90': alpha_quantiles[4],
                        })
                
                # Fill missing values with NaN
                for key in ['white_noise_avg', 'white_noise_q10', 'white_noise_q25', 'white_noise_q50', 
                           'white_noise_q75', 'white_noise_q90', 'fknee_avg', 'fknee_q10', 'fknee_q25',
                           'fknee_q50', 'fknee_q75', 'fknee_q90', 'alpha_avg', 'alpha_q10', 'alpha_q25',
                           'alpha_q50', 'alpha_q75', 'alpha_q90']:
                    if key not in noise_data:
                        noise_data[key] = np.nan
                
                noise_stats[pol] = noise_data
        
        # Extract T2P statistics
        if hasattr(aman_proc, 't2p'):
            if hasattr(aman_proc.t2p, 'coeffsQ'):
                t2p_stats['coeffsQ_avg'] = np.nanmean(aman_proc.t2p.coeffsQ)
            if hasattr(aman_proc.t2p, 'coeffsU'):
                t2p_stats['coeffsU_avg'] = np.nanmean(aman_proc.t2p.coeffsU)
            if hasattr(aman_proc.t2p, 'errorsQ'):
                t2p_stats['errorsQ_avg'] = np.nanmean(aman_proc.t2p.errorsQ)
            if hasattr(aman_proc.t2p, 'errorsU'):
                t2p_stats['errorsU_avg'] = np.nanmean(aman_proc.t2p.errorsU)
            if hasattr(aman_proc.t2p, 'redchi2sQ'):
                t2p_stats['redchi2sQ_avg'] = np.nanmean(aman_proc.t2p.redchi2sQ)
            if hasattr(aman_proc.t2p, 'redchi2sU'):
                t2p_stats['redchi2sU_avg'] = np.nanmean(aman_proc.t2p.redchi2sU)
    
    return noise_stats, t2p_stats

def extract_sample_cuts(aman_init, total_samples):
    """Extract sample cuts from init pipeline metadata."""
    sample_cuts = {'total_samples': total_samples}
    
    # Extract sample cuts from various flagging steps
    if hasattr(aman_init, 'smurfgaps') and hasattr(aman_init.smurfgaps, 'smurfgaps'):
        sample_cuts['smurfgaps_cuts'] = get_samp_cuts_from_ranges(
            aman_init.smurfgaps.smurfgaps, total_samples)
    
    if hasattr(aman_init, 'turnaround_flags') and hasattr(aman_init.turnaround_flags, 'turnarounds'):
        sample_cuts['turnarounds_cuts'] = get_samp_cuts_from_ranges(
            aman_init.turnaround_flags.turnarounds, total_samples)
    
    if hasattr(aman_init, 'jumps_slow') and hasattr(aman_init.jumps_slow, 'jump_flag'):
        sample_cuts['jumps_slow_cuts'] = get_samp_cuts_from_ranges(
            aman_init.jumps_slow.jump_flag, total_samples)
    
    if hasattr(aman_init, 'jumps_2pi') and hasattr(aman_init.jumps_2pi, 'jump_flag'):
        sample_cuts['jumps_2pi_cuts'] = get_samp_cuts_from_ranges(
            aman_init.jumps_2pi.jump_flag, total_samples)
    
    if hasattr(aman_init, 'glitches_pre_hwpss') and hasattr(aman_init.glitches_pre_hwpss, 'glitch_flags'):
        sample_cuts['glitches_pre_hwpss_cuts'] = get_samp_cuts_from_ranges(
            aman_init.glitches_pre_hwpss.glitch_flags, total_samples)
    
    if hasattr(aman_init, 'glitches_post_hwpss') and hasattr(aman_init.glitches_post_hwpss, 'glitch_flags'):
        sample_cuts['glitches_post_hwpss_cuts'] = get_samp_cuts_from_ranges(
            aman_init.glitches_post_hwpss.glitch_flags, total_samples)
    
    if hasattr(aman_init, 'source_flags') and hasattr(aman_init.source_flags, 'moon'):
        sample_cuts['source_moon_cuts'] = get_samp_cuts_from_ranges(
            aman_init.source_flags.moon, total_samples)
    
    return sample_cuts

def track_cuts_and_stats(obsid: str, wafer: str, band: str, configs_init: dict, configs_proc: dict, 
                        process_names: Optional[List[str]] = None, verbosity: int = 0):
    """
    Track detector cuts, sample cuts, and statistics for a single observation.
    This follows the same approach as track_det_counts.py but also extracts statistics.
    """
    
    if verbosity >= 1:
        logger.info(f"Processing {obsid} {wafer} {band}")
    
    try:
        # Initialize contexts
        cfg_init, ctx_init = pp_util.get_preprocess_context(configs_init)
        cfg_proc, ctx_proc = pp_util.get_preprocess_context(configs_proc) if configs_proc else (None, None)
        
        dets = {'wafer_slot': wafer, 'wafer.bandpass': band}
        det_counts = []
        first_time = (process_names is None)
        
        if first_time:
            names = ['starting']
        else:
            names = process_names[:]
            
        try:
            # First attempt with get_meta (preferred method)
            aman = ctx_init.get_meta(obsid, dets=dets)
            initial_det_count = aman.dets.count
            initial_sample_count = aman.samps.count if hasattr(aman, 'samps') else 0
            det_counts.append(initial_det_count)
            
            if verbosity >= 2:
                logger.info(f"Initial: {initial_det_count} dets, {initial_sample_count} samples")
            
            # Run init pipeline step by step
            proc_aman_init = aman.preprocess.copy()
            pipe_init = Pipeline(configs_init["process_pipe"])
            
            for i, process in enumerate(pipe_init):
                process.select(aman, proc_aman_init)
                proc_aman_init.restrict('dets', aman.dets.vals)
                if first_time:
                    names.append(process.name)
                det_counts.append(aman.dets.count)
                
                if verbosity >= 2:
                    logger.info(f"After {process.name}: {aman.dets.count} dets")
            
            # Store references to the processed data for statistics extraction
            aman_init_final = proc_aman_init
            
            # Run proc pipeline if configured
            aman_proc_final = None
            if cfg_proc:
                try:
                    meta_proc = ctx_proc.get_meta(obsid, dets=dets)
                    proc_aman_init.move('valid_data', None)
                    proc_aman_init.merge(meta_proc.preprocess)
                    pipe_proc = Pipeline(configs_proc["process_pipe"])
                    
                    for i, process in enumerate(pipe_proc):
                        process.select(aman, proc_aman_init)
                        proc_aman_init.restrict('dets', aman.dets.vals)
                        if first_time:
                            names.append(process.name)
                        det_counts.append(aman.dets.count)
                        
                        if verbosity >= 2:
                            logger.info(f"After {process.name}: {aman.dets.count} dets")
                    
                    aman_proc_final = proc_aman_init
                    
                except Exception as e:
                    logger.warning(f"Proc pipeline failed for {obsid} {wafer} {band}: {e}")
                    # Fill remaining det_counts with final init count
                    if configs_proc and first_time:
                        pipe_proc = Pipeline(configs_proc["process_pipe"])
                        for process in pipe_proc:
                            names.append(process.name)
                            det_counts.append(aman.dets.count)
            
            if verbosity >= 1:
                logger.info(f"{wafer} {band} final count: {aman.dets.count}")
            
            # Extract sample cuts from init pipeline
            sample_cuts = extract_sample_cuts(aman_init_final, initial_sample_count)
            
            # Extract statistics
            noise_stats, t2p_stats = extract_statistics(aman_init_final, aman_proc_final)
            
            return {
                'success': True,
                'process_names': names if first_time else None,
                'det_counts': det_counts,
                'sample_cuts': sample_cuts,
                'noise_stats': noise_stats,
                't2p_stats': t2p_stats,
                'failure_step': None,
                'error_message': None
            }
            
        except Exception as e:
            # Fallback to get_obs method with full processing
            logger.warning(f"{wafer} {band} get_meta failed, trying get_obs: {e}")
            
            # Reset contexts and modify metadata
            cfg_init, ctx_init = pp_util.get_preprocess_context(configs_init)
            cfg_proc, ctx_proc = pp_util.get_preprocess_context(configs_proc) if configs_proc else (None, None)
            ctx_init['metadata'] = ctx_init['metadata'][:-1]
            
            aman = ctx_init.get_obs(obsid, dets=dets, no_signal=True)
            initial_det_count = aman.dets.count
            initial_sample_count = aman.samps.count if hasattr(aman, 'samps') else 0
            det_counts.append(initial_det_count)
            
            # Run pipelines with full processing
            proc_aman_init = aman.preprocess.copy()
            pipe_init = Pipeline(configs_init["process_pipe"])
            
            for i, process in enumerate(pipe_init):
                process.select(aman, proc_aman_init)
                if first_time:
                    names.append(process.name)
                det_counts.append(aman.dets.count)
            
            # Store init results and continue with proc if configured
            aman_init_final = proc_aman_init
            aman_proc_final = None
            
            if cfg_proc:
                try:
                    ctx_proc['metadata'] = ctx_proc['metadata'][:-1]
                    meta_proc = ctx_proc.get_obs(obsid, dets=dets, no_signal=True)
                    proc_aman_init.move('valid_data', None)
                    proc_aman_init.merge(meta_proc.preprocess)
                    pipe_proc = Pipeline(configs_proc["process_pipe"])
                    
                    for i, process in enumerate(pipe_proc):
                        process.select(aman, proc_aman_init)
                        if first_time:
                            names.append(process.name)
                        det_counts.append(aman.dets.count)
                    
                    aman_proc_final = proc_aman_init
                    
                except Exception as e2:
                    logger.warning(f"Proc pipeline failed in fallback for {obsid} {wafer} {band}: {e2}")
                    # Fill remaining det_counts
                    if configs_proc and first_time:
                        pipe_proc = Pipeline(configs_proc["process_pipe"])
                        for process in pipe_proc:
                            names.append(process.name)
                            det_counts.append(aman.dets.count)
            
            # Extract sample cuts and statistics
            sample_cuts = extract_sample_cuts(aman_init_final, initial_sample_count)
            noise_stats, t2p_stats = extract_statistics(aman_init_final, aman_proc_final)
            
            return {
                'success': True,
                'process_names': names if first_time else None,
                'det_counts': det_counts,
                'sample_cuts': sample_cuts,
                'noise_stats': noise_stats,
                't2p_stats': t2p_stats,
                'failure_step': None,
                'error_message': None
            }
                
    except Exception as e:
        error_msg = f"Failed to process {obsid} {wafer} {band}: {str(e)}"
        logger.error(error_msg)
        if verbosity >= 2:
            logger.error(traceback.format_exc())
        
        return {
            'success': False,
            'process_names': None,
            'det_counts': [],
            'sample_cuts': {},
            'noise_stats': {},
            't2p_stats': {},
            'failure_step': 'unknown',
            'error_message': error_msg
        }

def save_results_to_db(db_path: str, obs_id: str, wafer: str, band: str, results: dict, process_names: List[str]):
    """Save results to SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if not results['success']:
        # Insert failed observation
        cursor.execute("""
            INSERT OR REPLACE INTO failed_observations 
            (obs_id, wafer, band, failure_step, error_message)
            VALUES (?, ?, ?, ?, ?)
        """, (obs_id, wafer, band, results['failure_step'], results['error_message']))
        conn.commit()
        conn.close()
        return
    
    # Insert detector counts
    if results['det_counts']:
        det_count_values = [obs_id, wafer, band] + results['det_counts']
        # Pad with nulls if we have fewer counts than expected
        while len(det_count_values) < len(process_names) + 3:
            det_count_values.append(None)
        
        placeholders = ', '.join(['?'] * len(det_count_values))
        det_count_columns = ['obs_id', 'wafer', 'band'] + [f'"{i}_{name}"' for i, name in enumerate(process_names)]
        columns_str = ', '.join(det_count_columns)
        
        cursor.execute(f"""
            INSERT OR REPLACE INTO detector_counts ({columns_str})
            VALUES ({placeholders})
        """, det_count_values)
    
    # Insert sample cuts
    if results['sample_cuts']:
        sample_cuts = results['sample_cuts']
        cursor.execute("""
            INSERT OR REPLACE INTO sample_cuts 
            (obs_id, wafer, band, total_samples, smurfgaps_cuts, turnarounds_cuts, 
             jumps_slow_cuts, jumps_2pi_cuts, glitches_pre_hwpss_cuts, 
             glitches_post_hwpss_cuts, source_moon_cuts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (obs_id, wafer, band,
              sample_cuts.get('total_samples', 0),
              sample_cuts.get('smurfgaps_cuts', 0),
              sample_cuts.get('turnarounds_cuts', 0),
              sample_cuts.get('jumps_slow_cuts', 0),
              sample_cuts.get('jumps_2pi_cuts', 0),
              sample_cuts.get('glitches_pre_hwpss_cuts', 0),
              sample_cuts.get('glitches_post_hwpss_cuts', 0),
              sample_cuts.get('source_moon_cuts', 0)))
    
    # Insert noise statistics
    for pol, stats in results['noise_stats'].items():
        cursor.execute("""
            INSERT OR REPLACE INTO noise_stats 
            (obs_id, wafer, band, polarization, white_noise_avg, white_noise_q10, white_noise_q25,
             white_noise_q50, white_noise_q75, white_noise_q90, fknee_avg, fknee_q10, fknee_q25,
             fknee_q50, fknee_q75, fknee_q90, alpha_avg, alpha_q10, alpha_q25, alpha_q50, 
             alpha_q75, alpha_q90)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (obs_id, wafer, band, pol,
              stats.get('white_noise_avg'), stats.get('white_noise_q10'), stats.get('white_noise_q25'),
              stats.get('white_noise_q50'), stats.get('white_noise_q75'), stats.get('white_noise_q90'),
              stats.get('fknee_avg'), stats.get('fknee_q10'), stats.get('fknee_q25'),
              stats.get('fknee_q50'), stats.get('fknee_q75'), stats.get('fknee_q90'),
              stats.get('alpha_avg'), stats.get('alpha_q10'), stats.get('alpha_q25'),
              stats.get('alpha_q50'), stats.get('alpha_q75'), stats.get('alpha_q90')))
    
    # Insert T2P statistics
    if results['t2p_stats']:
        t2p_stats = results['t2p_stats']
        cursor.execute("""
            INSERT OR REPLACE INTO t2p_stats 
            (obs_id, wafer, band, coeffsQ_avg, coeffsU_avg, errorsQ_avg, errorsU_avg, 
             redchi2sQ_avg, redchi2sU_avg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (obs_id, wafer, band,
              t2p_stats.get('coeffsQ_avg'), t2p_stats.get('coeffsU_avg'),
              t2p_stats.get('errorsQ_avg'), t2p_stats.get('errorsU_avg'),
              t2p_stats.get('redchi2sQ_avg'), t2p_stats.get('redchi2sU_avg')))
    
    conn.commit()
    conn.close()

def main():
    parser = argparse.ArgumentParser(description='Track cuts and statistics during SO preprocessing')
    parser.add_argument('--obs-id', required=True, help='Observation ID')
    parser.add_argument('--wafer', required=True, help='Wafer name')  
    parser.add_argument('--band', required=True, help='Band name')
    parser.add_argument('--init-config', default='satp1_v4.yaml', help='Init pipeline config file')
    parser.add_argument('--proc-config', help='Proc pipeline config file (optional)')
    parser.add_argument('--db-path', required=True, help='SQLite database path')
    parser.add_argument('--verbosity', type=int, default=0, help='Verbosity level (0-2)')
    
    args = parser.parse_args()
    
    # Load configs
    with open(args.init_config, 'r') as f:
        configs_init = yaml.safe_load(f)
    
    configs_proc = None
    if args.proc_config:
        with open(args.proc_config, 'r') as f:
            configs_proc = yaml.safe_load(f)
    
    # Build process names for table creation
    process_names = build_process_names(configs_init, configs_proc)
    
    # Create tables
    create_cuts_stats_tables(args.db_path, process_names)
    
    # Track cuts and statistics
    results = track_cuts_and_stats(args.obs_id, args.wafer, args.band, configs_init, configs_proc, 
                                   process_names, args.verbosity)
    
    # Save to database
    save_results_to_db(args.db_path, args.obs_id, args.wafer, args.band, results, process_names)
    
    if args.verbosity >= 1:
        if results['success']:
            print(f"Successfully processed {args.obs_id} {args.wafer} {args.band}")
            print(f"Detector counts: {results['det_counts']}")
        else:
            print(f"Failed to process {args.obs_id} {args.wafer} {args.band}: {results['error_message']}")

if __name__ == "__main__":
    main()