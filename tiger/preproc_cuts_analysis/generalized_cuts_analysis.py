#!/usr/bin/env python3
"""
Generalized Preprocessing Cuts Analysis Tool

This script analyzes preprocessing cuts and flags from sotodlib preprocessing pipelines
by extracting the cuts information from configuration files and automatically generating
summaries of sample and detector cuts.
"""

import numpy as np
import os
import yaml
import argparse
import traceback
import time
import sqlite3
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from sotodlib import core
from sotodlib.core.flagman import has_all_cut, has_any_cuts, count_cuts
import sotodlib.site_pipeline.util as sp_util
from sotodlib.preprocess import preprocess_util, pcore
from sotodlib.utils.procs_pool import get_exec_env
from so3g.proj import Ranges, RangesMatrix


class CutsAnalyzer:
    """Class to analyze preprocessing cuts and generate statistics."""
    
    def __init__(self, config_file_init: str, config_file_proc: str, 
                 noise_range: Tuple[float, float] = (18e-6, 80e-6)):
        """
        Initialize the cuts analyzer.
        
        Args:
            config_file_init: Initial preprocessing configuration file
            config_file_proc: Processing preprocessing configuration file  
            noise_range: Range for white noise cuts [min, max]
        """
        self.config_file_init = config_file_init
        self.config_file_proc = config_file_proc
        self.noise_range = noise_range
        
        # Parse configuration files to understand cut structure
        self.init_cuts_config = self._parse_config(config_file_init)
        self.proc_cuts_config = self._parse_config(config_file_proc)
        
        # Organize cuts by type
        self.sample_cuts = self._identify_sample_cuts()
        self.detector_cuts = self._identify_detector_cuts()
        
    def _parse_config(self, config_file: str) -> Dict[str, Any]:
        """Parse configuration file and extract cut-related processes."""
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        cuts_config = {}
        for process in config.get('process_pipe', []):
            name = process.get('name', '')
            
            # Identify processes that create cuts
            if self._is_cut_process(name, process):
                cuts_config[name] = process
                
        return cuts_config
    
    def _is_cut_process(self, name: str, process: Dict[str, Any]) -> bool:
        """Determine if a process creates cuts or flags."""
        # Processes that create sample cuts (flags)
        sample_cut_names = [
            'flag_turnarounds', 'turnaround_flags', 'turnarounds',
            'jumps', 'glitches', 'source_flags', 'smurfgaps_flags',
            'noisy_subscan_flags', 'combine_flags'
        ]
        
        # Processes that create detector cuts
        detector_cut_names = [
            'fp_flags', 'trends', 'det_bias_flags', 'ptp_flags', 
            'noise', 'white_noise', 'inv_var_flags'
        ]
        
        # Check if this is a cut-related process
        if any(cut_name in name for cut_name in sample_cut_names + detector_cut_names):
            return True
            
        # Also check if it has select configuration (indicates it creates cuts)
        if process.get('select') is not None:
            return True
            
        return False
    
    def _identify_sample_cuts(self) -> List[str]:
        """Identify processes that create sample cuts."""
        sample_cuts = []
        
        # Add base sample count
        sample_cuts.append('nsamps')
        
        # Look for processes that create sample flags
        for config in [self.init_cuts_config, self.proc_cuts_config]:
            for name, process in config.items():
                if any(flag_type in name for flag_type in [
                    'turnaround', 'jump', 'glitch', 'source', 'subscan', 'smurf'
                ]):
                    cut_name = f"{name}_nsamps"
                    if cut_name not in sample_cuts:
                        sample_cuts.append(cut_name)
        
        # Add edge cuts (common in demodulation)
        if 'edge_nsamps' not in sample_cuts:
            sample_cuts.append('edge_nsamps')
            
        return sample_cuts
    
    def _identify_detector_cuts(self) -> List[str]:
        """Identify processes that create detector cuts."""
        detector_cuts = []
        
        # Add base detector count
        detector_cuts.append('ndets')
        
        # Look for processes that create detector flags
        for config in [self.init_cuts_config, self.proc_cuts_config]:
            for name, process in config.items():
                if any(flag_type in name for flag_type in [
                    'fp_', 'trend', 'bias', 'ptp', 'noise', 'inv_var'
                ]):
                    cut_name = f"{name}_cuts"
                    if cut_name not in detector_cuts:
                        detector_cuts.append(cut_name)
        
        # Add end yield
        if 'end_yield' not in detector_cuts:
            detector_cuts.append('end_yield')
            
        return detector_cuts

    def count_new_cuts(self, ranges_matrix: RangesMatrix, survivors_mask: np.ndarray, 
                      already_cut: RangesMatrix) -> Tuple[int, RangesMatrix]:
        """
        Count new sample cuts from a ranges matrix.
        
        Args:
            ranges_matrix: RangesMatrix for this operation
            survivors_mask: 1D bool array, True for detectors that survive all cuts  
            already_cut: RangesMatrix representing samples already cut
            
        Returns:
            Tuple of (new_cut_count, updated_already_cut)
        """
        n_dets = ranges_matrix.shape[0]
        new_cut_count = 0
        updated_already_cut = []
        
        for det_idx in range(n_dets):
            if not survivors_mask[det_idx]:
                # This detector doesn't survive all cuts, skip
                updated_already_cut.append(already_cut.ranges[det_idx])
                continue
                
            op_ranges = ranges_matrix.ranges[det_idx]
            prev_ranges = already_cut.ranges[det_idx]
            
            # Compute new cuts: samples in op_ranges but not in prev_ranges
            det_cut_count = np.sum(np.ptp((op_ranges * ~prev_ranges).ranges(), axis=1))
            
            if det_cut_count != 0:
                new_cut_count += det_cut_count
                # Update already_cut for this detector
                updated_already_cut.append(prev_ranges + op_ranges)
            else:
                updated_already_cut.append(prev_ranges)
                
        return new_cut_count, RangesMatrix.from_ranges_list(updated_already_cut)

    def get_dict_entry(self, entry: Dict[str, Any]) -> Tuple[str, str, str, List[str], List[int]]:
        """
        Process a single observation entry and extract cuts statistics.
        
        Args:
            entry: Database entry containing observation information
            
        Returns:
            Tuple of (obs_id, wafer_slot, bandpass, keys, values)
        """
        try:
            logger = preprocess_util.init_logger('subproc_logger')
            logger.info(f'Processing entry for {entry["dataset"]}')
            
            # Load contexts
            _, context_init = preprocess_util.get_preprocess_context(self.config_file_init)
            _, context_proc = preprocess_util.get_preprocess_context(self.config_file_proc)
            
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
            proc_survivors_mask = has_all_cut(mdata_proc.preprocess.split_flags.valid)
            survivors_mask = np.zeros(mdata_init.dets.count, dtype=bool)
            _, ind_init, _ = np.intersect1d(mdata_init.dets.vals, 
                                          mdata_proc.dets.vals[proc_survivors_mask], 
                                          return_indices=True)
            survivors_mask[ind_init] = True
            
            # Initialize tracking for sample cuts
            already_cut = RangesMatrix.zeros((ndet, nsamp))
            
            keys = []
            vals = []
            
            # Base counts
            keys.extend(['nsamps', 'ndets'])
            vals.extend([nsamp, ndet])
            
            # Process cuts in order based on configuration
            self._extract_init_cuts(x, keys, vals, survivors_mask, already_cut)
            
            # Edge cuts (common in demodulation)
            edge_mask = np.zeros((ndet, nsamp), dtype=bool)
            edge_mask[:, :6000] = True
            edge_mask[:, -6000:] = True
            edge_ranges = RangesMatrix.from_mask(edge_mask)
            n_edge_cuts, already_cut = self.count_new_cuts(edge_ranges, survivors_mask, already_cut)
            keys.append('edge_nsamps')
            vals.append(int(n_edge_cuts))
            
            # Processing cuts
            x_proc = mdata_proc.preprocess
            self._extract_proc_cuts(x_proc, keys, vals)
            
            # End yield
            m = has_all_cut(x_proc.split_flags.valid)
            keys.append('end_yield')
            vals.append(np.sum(m))
            
            return entry['obs:obs_id'], entry['dets:wafer_slot'], entry['dets:wafer.bandpass'], keys, vals
            
        except Exception as e:
            logger.info(f"Error in process for {entry['dataset']}")
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            return None, None, None, errmsg, tb

    def _extract_init_cuts(self, x: Any, keys: List[str], vals: List[int], 
                          survivors_mask: np.ndarray, already_cut: RangesMatrix) -> None:
        """Extract cuts from initial preprocessing."""
        
        # FP flags
        if hasattr(x, 'fp_flags'):
            m = has_all_cut(x.fp_flags.valid)
            keys.append('fp_cuts')
            vals.append(np.sum(has_all_cut(x.fp_flags.fp_nans)[m]))
        
        # Trends  
        if hasattr(x, 'trends'):
            m = has_all_cut(x.trends.valid)
            keys.append('trend_cuts')
            vals.append(np.sum(has_all_cut(x.trends.trend_flags)[m]))
        
        # Turnarounds
        if hasattr(x, 'turnaround_flags'):
            m = has_all_cut(x.turnaround_flags.valid)
            n_turnaround_cuts, already_cut = self.count_new_cuts(
                x.turnaround_flags.turnarounds, survivors_mask, already_cut)
            keys.append('turnaround_nsamps')
            vals.append(int(n_turnaround_cuts))
        
        # Jumps
        for jump_type in ['jumps_slow', 'jumps_2pi']:
            if hasattr(x, jump_type):
                jump_obj = getattr(x, jump_type)
                m = has_all_cut(jump_obj.valid)
                n_jump_cuts, already_cut = self.count_new_cuts(
                    jump_obj.jump_flag, survivors_mask, already_cut)
                keys.extend([f'{jump_type}_nsamps', f'{jump_type}_cuts'])
                vals.extend([int(n_jump_cuts), 
                           np.sum(count_cuts(jump_obj.jump_flag)[m] > 5)])
        
        # Glitches
        if hasattr(x, 'glitches'):
            m = has_all_cut(x.glitches.valid)
            n_glitch_nsamps, already_cut = self.count_new_cuts(
                x.glitches.glitch_flags, survivors_mask, already_cut)
            keys.extend(['glitch_nsamps', 'glitch_cuts'])
            vals.extend([int(n_glitch_nsamps),
                        np.sum(count_cuts(x.glitches.glitch_flags)[m] > 50000)])
        
        # Det bias flags
        if hasattr(x, 'det_bias_flags'):
            m = has_all_cut(x.det_bias_flags.valid)
            keys.append('det_bias_cuts')
            vals.append(np.sum(has_all_cut(x.det_bias_flags.det_bias_flags)[m]))
        
        # PTP flags
        if hasattr(x, 'ptp_flags'):
            m = has_all_cut(x.ptp_flags.valid)
            keys.append('ptp_cuts')
            vals.append(np.sum(has_all_cut(x.ptp_flags.ptp_flags)[m]))
        
        # White noise
        if hasattr(x, 'white_noise_nofit'):
            m = has_all_cut(x.white_noise_nofit.valid)
            n_min, n_max = self.noise_range
            keys.append('white_noise_cuts')
            vals.append(np.sum(((x.white_noise_nofit.white_noise)[m] < n_min) | 
                              ((x.white_noise_nofit.white_noise)[m] > n_max)))

    def _extract_proc_cuts(self, x_proc: Any, keys: List[str], vals: List[int]) -> None:
        """Extract cuts from processing step."""
        
        # Inv var cuts
        if hasattr(x_proc, 'inv_var_flags'):
            m = has_all_cut(x_proc.inv_var_flags.valid)
            keys.append('inv_var_cuts')
            vals.append(np.sum(has_all_cut(x_proc.inv_var_flags.inv_var_flags)[m]))
        
        # Noisy subscan cuts
        if hasattr(x_proc, 'noisy_subscan_flags'):
            m = has_all_cut(x_proc.noisy_subscan_flags.valid)
            keys.append('noisy_subscans_cuts')
            vals.append(np.sum(~x_proc.noisy_dets_flags.valid_dets))


def generate_summary_table(sqlite_path: str, start_ts: int, end_ts: int, 
                          cut_type: str = 'detector') -> pd.DataFrame:
    """
    Generate summary table from the cuts database.
    
    Args:
        sqlite_path: Path to SQLite database
        start_ts: Start timestamp
        end_ts: End timestamp  
        cut_type: Either 'detector' or 'sample'
        
    Returns:
        DataFrame with cut statistics
    """
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    
    # Get column information
    cur.execute('PRAGMA table_info(results)')
    columns = [row[1] for row in cur.fetchall()]
    
    if cut_type == 'detector':
        cut_keys = [k for k in columns if k.endswith('cuts')]
        query = f"""
        SELECT {', '.join(cut_keys)}, ndets FROM results
        WHERE CAST(substr(obsid, 5, 10) AS INTEGER) BETWEEN ? AND ?
        """
        cur.execute(query, (start_ts, end_ts))
        
        # Transpose and sum
        columns = list(zip(*cur))
        cut_sums = [sum(x for x in col) for col in columns]
        *cut_sums, total_dets = cut_sums
        
        # Calculate fractions
        sum_of_all_cuts = sum(cut_sums)
        total_fractions = [cut_sum / total_dets if total_dets else 0 for cut_sum in cut_sums]
        cut_fractions = [cut_sum / sum_of_all_cuts if sum_of_all_cuts else 0 for cut_sum in cut_sums]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Cut Type': [key.replace('_cuts', '').replace('_', ' ').title() for key in cut_keys],
            'Total Count': cut_sums,
            'Fraction of Total Data': [f"{frac*100:.2f}%" for frac in total_fractions],
            'Fraction of Cut Data': [f"{frac*100:.2f}%" for frac in cut_fractions]
        })
        
    elif cut_type == 'sample':
        cut_keys = [k for k in columns if k.endswith('nsamps') and k != 'nsamps']
        query = f"""
        SELECT {', '.join(cut_keys)}, nsamps, end_yield FROM results
        WHERE CAST(substr(obsid, 5, 10) AS INTEGER) BETWEEN ? AND ?
        """
        cur.execute(query, (start_ts, end_ts))
        
        # Fetch all rows and process
        rows = list(cur)
        cut_cols = list(zip(*rows))
        cut_sums = [sum(int(x) for x in col) for col in cut_cols[:-2]]
        nsamps_col = cut_cols[-2]
        end_yield_col = cut_cols[-1]
        
        # Total data volume
        total_data = sum(int(n) * int(e) for n, e in zip(nsamps_col, end_yield_col))
        sum_of_all_cuts = sum(cut_sums)
        
        # Calculate fractions
        total_fractions = [cut_sum / total_data if total_data else 0 for cut_sum in cut_sums]
        cut_fractions = [cut_sum / sum_of_all_cuts if sum_of_all_cuts else 0 for cut_sum in cut_sums]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Cut Type': [key.replace('_nsamps', '').replace('_', ' ').title() for key in cut_keys],
            'Total Count': cut_sums,
            'Fraction of Total Data': [f"{frac*100:.2f}%" for frac in total_fractions],
            'Fraction of Cut Data': [f"{frac*100:.2f}%" for frac in cut_fractions]
        })
    
    conn.close()
    return df


def get_parser(parser=None):
    """Create argument parser."""
    if parser is None:
        parser = argparse.ArgumentParser(description='Generalized preprocessing cuts analysis')
    
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
                       default='cuts_analysis_err.txt')
    parser.add_argument('--savename',
                       help="Output database save name.",
                       default='cuts_analysis.sqlite')
    parser.add_argument('--generate-tables',
                       help="Generate summary tables after analysis",
                       action='store_true')
    parser.add_argument('--start-ts',
                       help="Start timestamp for table generation",
                       type=int)
    parser.add_argument('--end-ts', 
                       help="End timestamp for table generation",
                       type=int)
    
    return parser


def main(executor, as_completed_callable, config_file_init, config_file_proc,
         errlog_ext, savename, noise_range, nproc, generate_tables=False,
         start_ts=None, end_ts=None):
    """Main processing function."""
    
    logger = preprocess_util.init_logger('main_proc')
    configs_proc, _ = preprocess_util.get_preprocess_context(config_file_proc)
    base_dir = os.path.dirname(configs_proc['archive']['index'])
    errlog = errlog_ext
    
    # Initialize cuts analyzer
    analyzer = CutsAnalyzer(config_file_init, config_file_proc, noise_range)
    
    logger.info('Connect to database')
    proc = core.metadata.ManifestDb(os.path.join(base_dir, 'process_archive.sqlite'))
    
    # Setup output database
    sqlite_path = savename.replace('.h5', '.sqlite')
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    
    run_list = proc.inspect()
    logger.info('Run list created')
    
    # Get first entry to build schema
    first_entry = run_list[0]
    obsid, ws, band, keys, vals = analyzer.get_dict_entry(entry=first_entry)
    
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
    futures = [executor.submit(analyzer.get_dict_entry, entry=entry) for entry in run_list]
    
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
    
    # Generate summary tables if requested
    if generate_tables and start_ts is not None and end_ts is not None:
        logger.info('Generating summary tables...')
        
        # Generate detector cuts table
        det_table = generate_summary_table(sqlite_path, start_ts, end_ts, 'detector')
        det_output = sqlite_path.replace('.sqlite', '_detector_cuts_table.csv')
        det_table.to_csv(det_output, index=False)
        logger.info(f'Detector cuts table saved to {det_output}')
        
        # Generate sample cuts table  
        samp_table = generate_summary_table(sqlite_path, start_ts, end_ts, 'sample')
        samp_output = sqlite_path.replace('.sqlite', '_sample_cuts_table.csv')
        samp_table.to_csv(samp_output, index=False)
        logger.info(f'Sample cuts table saved to {samp_output}')


if __name__ == '__main__':
    args = get_parser().parse_args()
    rank, executor, as_completed_callable = get_exec_env(args.nproc)
    if rank == 0:
        main(executor=executor, as_completed_callable=as_completed_callable, **vars(args))
