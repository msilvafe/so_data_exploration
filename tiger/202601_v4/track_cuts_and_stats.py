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
from sotodlib.utils.procs_pool import get_exec_env
import copy
from tqdm import tqdm
from sotodlib import core
from sotodlib.core import AxisManager
from so3g.proj import RangesMatrix
from sotodlib.core.flagman import count_cuts
from sotodlib.core.metadata.obsdb import ObsDb
import sotodlib.site_pipeline.util as sp_util
from sotodlib.preprocess import preprocess_util as pp_util
from sotodlib.preprocess import _Preprocess, Pipeline, processes, pcore

logger = sp_util.init_logger("track_cuts_and_stats")

def get_quantiles(data, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
    """Compute quantiles of data while ignoring NaNs.

    Parameters
    ----------
    data : array-like
        Input data values.
    quantiles : sequence of float, optional
        Quantiles to compute in the range [0, 1].

    Returns
    -------
    list
        Quantile values; all NaN if no finite samples exist.
    """
    data_clean = data[~np.isnan(data)]
    if len(data_clean) == 0:
        return [np.nan] * len(quantiles)
    return np.quantile(data_clean, quantiles)

def get_sorted_flag_labs(pipe_init):
    """Return combine_flags labels ordered by their pipeline position.

    Parameters
    ----------
    pipe_init : sotodlib.preprocess.Pipeline
        Initialized pipeline for the init preprocess config.

    Returns
    -------
    list
        Ordered list of flag labels used to construct glitch flags.
    """
    gf_idx = None
    save_names = []
    for i, process in enumerate(pipe_init):
        save_names.append(process.save_name)
        if process.name == 'combine_flags':
            if process.process_cfgs.get('total_flags_label') == 'glitch_flags':
                gf_idx = i
    if gf_idx is None:
        return []
    flag_labs = pipe_init[gf_idx].process_cfgs.get('flag_labels')
    if not flag_labs:
        return []
    flat_list = [
        item
        for x in save_names
        if x is not None
        for item in (x if isinstance(x, list) else [x])
    ]
    order = {name: i for i, name in enumerate(flat_list)}
    return sorted(flag_labs, key=lambda fl: order[fl.split('.')[0]])

def count_ranges_matrix_samples(ranges_matrix):
    """Count flagged samples in a RangesMatrix across all detectors.
    """
    return int(np.sum([np.sum(np.diff(rng.ranges())) for rng in ranges_matrix]))

def build_padded_valid(meta_init, meta_proc):
    """Build a valid-data mask aligned to the init detector axis.

    Parameters
    ----------
    meta_init : AxisManager
        Metadata for the init pipeline.
    meta_proc : AxisManager or None
        Metadata for the proc pipeline, if present.

    Returns
    -------
    so3g.proj.RangesMatrix or None
        Valid-data ranges aligned to the init detectors, or None if absent.
    """
    if not hasattr(meta_init.preprocess, 'valid_data'):
        return None
    data_valid_init = meta_init.preprocess.valid_data.valid_data
    if meta_proc is None:
        return data_valid_init
    if not hasattr(meta_proc.preprocess, 'valid_data'):
        return data_valid_init
    data_valid_proc = meta_proc.preprocess.valid_data.valid_data
    if meta_init.dets.count == meta_proc.dets.count:
        return data_valid_proc

    dets_with_valid = np.where(count_cuts(data_valid_init))[0]
    if dets_with_valid.size == 0:
        return RangesMatrix.zeros((meta_init.dets.count, meta_init.samps.count))
    fill_rng = data_valid_init[dets_with_valid[0]].ranges()[0]

    where_valid = count_cuts(data_valid_proc).astype(bool)
    valid_det_names_proc = meta_proc.dets.vals[where_valid]
    _, init_indices, _ = np.intersect1d(
        meta_init.dets.vals,
        valid_det_names_proc,
        return_indices=True
    )

    padded_valid = RangesMatrix.zeros((meta_init.dets.count, meta_init.samps.count))
    for init_idx in init_indices:
        padded_valid[init_idx].add_interval(fill_rng[0], fill_rng[1])
    return padded_valid

def extract_sample_cuts(meta_init, padded_valid, flag_labs):
    """Count sample cuts per flag, excluding previous flags in order.

    Parameters
    ----------
    meta_init : AxisManager
        Metadata containing preprocess flag fields.
    padded_valid : so3g.proj.RangesMatrix or None
        Valid-data ranges aligned to the init detector axis.
    flag_labs : list
        Ordered flag labels used to compute sample cuts.

    Returns
    -------
    dict
        Mapping of flag labels to sample counts, including total samples.
    """
    sample_cuts = {}
    if padded_valid is None:
        sample_cuts['total_samples'] = 0
        return sample_cuts
    total_count = count_ranges_matrix_samples(padded_valid)
    sample_cuts['total_samples'] = total_count
    if not flag_labs:
        return sample_cuts

    prev_flags = None
    for fl in flag_labs:
        base_label = fl.split('.')[0]
        if fl not in meta_init.preprocess:
            sample_cuts[base_label] = 0
            continue
        if prev_flags is None:
            cur_flag = meta_init.preprocess[fl]
        else:
            cur_flag = meta_init.preprocess[fl] * ~prev_flags
        cur_count = count_ranges_matrix_samples(cur_flag * padded_valid)
        sample_cuts[base_label] = cur_count
        if prev_flags is None:
            prev_flags = copy.deepcopy(cur_flag)
        else:
            prev_flags += cur_flag
    return sample_cuts

def build_process_names(configs_init, configs_proc):
    """Derive process names from the configured pipelines.

    Parameters
    ----------
    configs_init : dict
        Init pipeline config dictionary.
    configs_proc : dict or None
        Proc pipeline config dictionary.

    Returns
    -------
    list
        Ordered list of process names including the initial state.
    """
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

def create_cuts_stats_tables(db_path: str, process_names: List[str], flag_labels: List[str]):
    """Create tables for cuts and statistics tracking.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    process_names : list
        Ordered list of process names for detector count columns.
    flag_labels : list
        Ordered list of sample-flag labels for sample cut columns.
    """
    obsdb = ObsDb(map_file=db_path, init_db=True, wafer_info=['wafer', 'band'])

    column_defs = []
    for i, name in enumerate(process_names):
        column_defs.append((f"{i}_{name}_ndets", int))

    column_defs.append(("total_samples", int))
    column_defs.extend([(label, int) for label in flag_labels])

    noise_metrics = [
        'white_noise_avg',
        'white_noise_q10',
        'white_noise_q25',
        'white_noise_q50',
        'white_noise_q75',
        'white_noise_q90',
        'array_noise',
        'n_array_noise',
        'fknee_avg',
        'fknee_q10',
        'fknee_q25',
        'fknee_q50',
        'fknee_q75',
        'fknee_q90',
        'alpha_avg',
        'alpha_q10',
        'alpha_q25',
        'alpha_q50',
        'alpha_q75',
        'alpha_q90',
    ]
    for pol in ['T', 'Q', 'U']:
        for metric in noise_metrics:
            column_defs.append((f"noise_{pol}_{metric}", float))

    t2p_metrics = [
        'coeffsQ_avg',
        'coeffsU_avg',
        'errorsQ_avg',
        'errorsU_avg',
        'redchi2sQ_avg',
        'redchi2sU_avg',
    ]
    for metric in t2p_metrics:
        column_defs.append((f"t2p_{metric}", float))

    column_defs.append(("success", int))
    column_defs.append(("failure_step", str))
    column_defs.append(("error_message", str))

    obsdb.add_obs_columns(column_defs, ignore_duplicates=True, commit=True)

def pick_noise_save_names(pipe_init, pipe_proc):
    """Infer which noise process outputs to use for T, Q, U statistics.

    Traverses the combined init+proc pipeline to identify all PSD and noise processes,
    tracking whether each occurs before or after demodulation. For each noise process,
    infers the signal type (signal, dsT, demodQ, demodU) by matching to the most recent
    preceding PSD process by name. Then selects the best noise source for each polarization:

    - T: Requires either dsT post-demod or signal pre-demod. Prefers earliest noise with
      fit=True; if none have fit=True, uses the last (most processed) candidate.
    - Q: Requires demodQ signal. Prefers earliest noise with fit=True; if none have fit=True,
      uses the last (most processed) candidate.
    - U: Requires demodU signal. Prefers earliest noise with fit=True; if none have fit=True,
      uses the last (most processed) candidate.

    Parameters
    ----------
    pipe_init : sotodlib.preprocess.Pipeline
        Init pipeline.
    pipe_proc : sotodlib.preprocess.Pipeline or None
        Proc pipeline.

    Returns
    -------
    dict
        Map from polarization ('T', 'Q', 'U') to (save_name, has_fit, where).
    """
    from bisect import bisect_right

    pipeline = list(pipe_init) + (list(pipe_proc) if pipe_proc else [])

    pre_demod = True
    PSD = []    # (global_i, psd_name, signal, pre_demod)
    NOISE = []  # (global_i, psd_name, fit, save_name, where, process)

    n_init = len(pipe_init)

    for gi, p in enumerate(pipeline):
        if p.name == "demodulate":
            pre_demod = False
            continue

        if p.name == "psd":
            PSD.append((gi, p.wrap, p.signal, pre_demod))
            continue

        if p.name == "noise":
            where = "init" if gi < n_init else "proc"
            NOISE.append((gi, p.psd, bool(p.fit), p.save_name, where, p))
            continue

    # Index PSD steps by name
    psd_by_name = {}
    for gi, psd_name, sig, pd in PSD:
        psd_by_name.setdefault(psd_name, []).append((gi, sig, pd))
    for name in psd_by_name:
        psd_by_name[name].sort(key=lambda x: x[0])

    def infer_signal(noise_gi, noise_psd_name):
        """Find the most recent PSD process before this noise process by matching name.
        Returns the signal type and pre_demod status from that PSD.
        """
        arr = psd_by_name.get(noise_psd_name, [])
        if not arr:
            return None, None
        idx = bisect_right([x[0] for x in arr], noise_gi) - 1
        if idx < 0:
            return None, None
        _, sig, pd = arr[idx]
        return sig, pd

    # Decorate noise records
    recs = []
    for gi, psd_name, fit, save_name, where, process in NOISE:
        sig, psd_pre_demod = infer_signal(gi, psd_name)
        recs.append(dict(
            gi=gi,
            fit=fit,
            save_name=save_name,
            where=where,
            signal=sig,
            psd_pre_demod=psd_pre_demod,
            process=process,
        ))

    def choose(cands):
        """Select the best noise candidate from a list.
        Prefers the earliest (lowest global index) candidate with fit=True.
        If none have fit=True, returns the last (most processed) candidate.
        """
        if not cands:
            return None
        fit_true = [c for c in cands if c["fit"]]
        if fit_true:
            return min(fit_true, key=lambda c: c["gi"])
        return max(cands, key=lambda c: c["gi"])

    # T
    T_cands = [
        r for r in recs
        if (r["signal"] == "dsT" and r["psd_pre_demod"] is False)
        or (r["signal"] == "signal" and r["psd_pre_demod"] is True)
    ]
    T = choose(T_cands)

    # Q/U
    Q = choose([r for r in recs if r["signal"] == "demodQ"])
    U = choose([r for r in recs if r["signal"] == "demodU"])

    def pack(r):
        """Convert a noise record to (save_name, has_fit, where) tuple.
        If fit=True, appends the merge_name to the save_name to access the fit statistics.
        """
        if r is None:
            return (None, False, None)

        if r["fit"]:
            merge_name = r["process"].calc_cfgs.get("merge_name", "noise_fit_stats")
            save = f"{r['save_name']}.{merge_name}"
        else:
            save = r["save_name"]

        return (save, r["fit"], r["where"])

    return {
        "T": pack(T),
        "Q": pack(Q),
        "U": pack(U),
    }

def extract_statistics(aman_init, aman_proc, pipe_init, pipe_proc):
    """Extract noise and T2P statistics from processed data.

    Parameters
    ----------
    aman_init : AxisManager
        Init pipeline results containing temperature noise statistics.
    aman_proc : AxisManager or None
        Proc pipeline results containing polarization noise and T2P.
    pipe_init : sotodlib.preprocess.Pipeline
        Init pipeline used to infer noise locations.
    pipe_proc : sotodlib.preprocess.Pipeline or None
        Proc pipeline used to infer noise locations.

    Returns
    -------
    tuple of dict
        Noise statistics and T2P statistics dictionaries.
    """
    noise_stats = {}
    t2p_stats = {}

    # Determine which noise outputs to use
    noise_map = pick_noise_save_names(pipe_init, pipe_proc)

    for pol in ['T', 'Q', 'U']:
        save_name, has_fit, where = noise_map[pol]
        if save_name is None:
            continue

        # Select the appropriate aman
        aman = aman_init if where == 'init' else aman_proc
        if aman is None:
            continue

        # Get the noise object
        noise_obj = aman[save_name]

        noise_data = {}
        
        # Always extract white noise
        wn_quantiles = get_quantiles(noise_obj.white_noise)
        noise_data.update({
            'white_noise_avg': np.nanmean(noise_obj.white_noise),
            'white_noise_q10': wn_quantiles[0],
            'white_noise_q25': wn_quantiles[1],
            'white_noise_q50': wn_quantiles[2],
            'white_noise_q75': wn_quantiles[3],
            'white_noise_q90': wn_quantiles[4],
            'array_noise': 1/np.sqrt(np.nansum(1/np.array(noise_obj.white_noise)**2)),
            'n_array_noise': len(noise_obj.white_noise),
        })

        # Extract fit parameters if available
        if has_fit:
            fknee_quantiles = get_quantiles(noise_obj.fit[:, 1])
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
        else:
            noise_data.update({
                'fknee_avg': np.nan,
                'fknee_q10': np.nan,
                'fknee_q25': np.nan,
                'fknee_q50': np.nan,
                'fknee_q75': np.nan,
                'fknee_q90': np.nan,
                'alpha_avg': np.nan,
                'alpha_q10': np.nan,
                'alpha_q25': np.nan,
                'alpha_q50': np.nan,
                'alpha_q75': np.nan,
                'alpha_q90': np.nan,
            })

        noise_stats[pol] = noise_data

    # Extract T2P statistics
    if aman_proc and hasattr(aman_proc, 't2p'):
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

def track_cuts_and_stats(obsid: str, wafer: str, band: str, configs_init: dict, configs_proc: dict,
                        process_names: Optional[List[str]] = None, flag_labels: Optional[List[str]] = None,
                        verbosity: int = 0):
    """Track detector counts, sample cuts, and statistics for one observation.

    Parameters
    ----------
    obsid : str
        Observation id to process.
    wafer : str
        Wafer slot name.
    band : str
        Band name.
    configs_init : dict
        Init pipeline config dictionary.
    configs_proc : dict or None
        Proc pipeline config dictionary.
    process_names : list or None
        Process name list for detector count columns.
    flag_labels : list or None
        Ordered full sample-flag labels for cut counting.
    verbosity : int, optional
        Verbosity level.

    Returns
    -------
    dict
        Results including counts, statistics, and failure metadata.
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

        if flag_labels is None:
            flag_labels = []
            
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
            pipe_init = Pipeline(cfg_init["process_pipe"])
            if not flag_labels:
                flag_labels = get_sorted_flag_labs(pipe_init)
            
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
            meta_proc = None
            if cfg_proc:
                try:
                    meta_proc = ctx_proc.get_meta(obsid, dets=dets)
                    proc_aman_init.move('valid_data', None)
                    proc_aman_init.merge(meta_proc.preprocess)
                    pipe_proc = Pipeline(cfg_proc["process_pipe"])
                    
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
                        pipe_proc = Pipeline(cfg_proc["process_pipe"])
                        for process in pipe_proc:
                            names.append(process.name)
                            det_counts.append(aman.dets.count)
            
            if verbosity >= 1:
                logger.info(f"{wafer} {band} final count: {aman.dets.count}")
            
            # Extract sample cuts based on combine_flags ordering
            padded_valid = build_padded_valid(aman, meta_proc)
            sample_cuts = extract_sample_cuts(aman, padded_valid, flag_labels)
            
            # Extract statistics
            pipe_proc_for_stats = pipe_proc if cfg_proc else None
            noise_stats, t2p_stats = extract_statistics(aman_init_final, aman_proc_final, pipe_init, pipe_proc_for_stats)
            
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
            pipe_init = Pipeline(cfg_init["process_pipe"])
            if not flag_labels:
                flag_labels = get_sorted_flag_labs(pipe_init)
            
            for i, process in enumerate(pipe_init):
                process.select(aman, proc_aman_init)
                if first_time:
                    names.append(process.name)
                det_counts.append(aman.dets.count)
            
            # Store init results and continue with proc if configured
            aman_init_final = proc_aman_init
            aman_proc_final = None
            meta_proc = None
            
            if cfg_proc:
                try:
                    ctx_proc['metadata'] = ctx_proc['metadata'][:-1]
                    meta_proc = ctx_proc.get_obs(obsid, dets=dets, no_signal=True)
                    proc_aman_init.move('valid_data', None)
                    proc_aman_init.merge(meta_proc.preprocess)
                    pipe_proc = Pipeline(cfg_proc["process_pipe"])
                    
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
                        pipe_proc = Pipeline(cfg_proc["process_pipe"])
                        for process in pipe_proc:
                            names.append(process.name)
                            det_counts.append(aman.dets.count)
            
            # Extract sample cuts and statistics
            padded_valid = build_padded_valid(aman, meta_proc)
            sample_cuts = extract_sample_cuts(aman, padded_valid, flag_labels)
            pipe_proc_for_stats = pipe_proc if cfg_proc else None
            noise_stats, t2p_stats = extract_statistics(aman_init_final, aman_proc_final, pipe_init, pipe_proc_for_stats)
            
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

def save_results_to_db(db_path: str, obs_id: str, wafer: str, band: str, results: dict,
                       process_names: List[str], flag_labels: List[str]):
    """Save results to the SQLite database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    obs_id : str
        Observation id to store.
    wafer : str
        Wafer slot name.
    band : str
        Band name.
    results : dict
        Result dictionary from `track_cuts_and_stats`.
    process_names : list
        Process name list for detector count columns.
    flag_labels : list
        Ordered base sample-flag labels for sample cut columns.
    """
    obsdb = ObsDb(map_file=db_path, init_db=False)

    data = {
        'success': int(results['success']),
        'failure_step': results.get('failure_step'),
        'error_message': results.get('error_message'),
    }

    det_counts = results.get('det_counts', [])
    for i, name in enumerate(process_names):
        key = f"det_{i}_{name}"
        data[key] = det_counts[i] if i < len(det_counts) else None

    sample_cuts = results.get('sample_cuts', {})
    data['total_samples'] = sample_cuts.get('total_samples', 0)
    for label in flag_labels:
        data[label] = sample_cuts.get(label, 0)

    noise_metrics = [
        'white_noise_avg',
        'white_noise_q10',
        'white_noise_q25',
        'white_noise_q50',
        'white_noise_q75',
        'white_noise_q90',
        'array_noise',
        'n_array_noise',
        'fknee_avg',
        'fknee_q10',
        'fknee_q25',
        'fknee_q50',
        'fknee_q75',
        'fknee_q90',
        'alpha_avg',
        'alpha_q10',
        'alpha_q25',
        'alpha_q50',
        'alpha_q75',
        'alpha_q90',
    ]
    noise_stats = results.get('noise_stats', {})
    for pol in ['T', 'Q', 'U']:
        pol_stats = noise_stats.get(pol, {})
        for metric in noise_metrics:
            data[f"noise_{pol}_{metric}"] = pol_stats.get(metric)

    t2p_stats = results.get('t2p_stats', {})
    for metric in [
        'coeffsQ_avg',
        'coeffsU_avg',
        'errorsQ_avg',
        'errorsU_avg',
        'redchi2sQ_avg',
        'redchi2sU_avg',
    ]:
        data[f"t2p_{metric}"] = t2p_stats.get(metric)

    obsdb.update_obs(obs_id, data=data, wafer_info={'wafer': wafer, 'band': band}, commit=True)

def main():
    """CLI entry point."""
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
    pipe_init = Pipeline(pp_util.get_preprocess_context(configs_init)[0]["process_pipe"])
    full_flag_labels = get_sorted_flag_labs(pipe_init)
    base_flag_labels = [label.split('.')[0] for label in full_flag_labels]
    
    # Create tables
    create_cuts_stats_tables(args.db_path, process_names, base_flag_labels)
    
    # Track cuts and statistics
    results = track_cuts_and_stats(args.obs_id, args.wafer, args.band, configs_init, configs_proc,
                                   process_names, full_flag_labels, args.verbosity)
    
    # Save to database
    save_results_to_db(args.db_path, args.obs_id, args.wafer, args.band, results,
                       process_names, base_flag_labels)
    
    if args.verbosity >= 1:
        if results['success']:
            print(f"Successfully processed {args.obs_id} {args.wafer} {args.band}")
            print(f"Detector counts: {results['det_counts']}")
        else:
            print(f"Failed to process {args.obs_id} {args.wafer} {args.band}: {results['error_message']}")

if __name__ == "__main__":
    main()