import numpy as np
import os
import argparse
import traceback
import time
import sqlite3

from sotodlib import core
from sotodlib.core.flagman import has_all_cut, has_any_cuts, count_cuts
import sotodlib.site_pipeline.util as sp_util
from sotodlib.preprocess import preprocess_util, pcore
from sotodlib.utils.procs_pool import get_exec_env
from so3g.proj import Ranges, RangesMatrix

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('config_file_init',
                        help="Preprocessing init configuration file")
    parser.add_argument('config_file_proc',
                        help="Preprocessing proc configuration file")
    parser.add_argument(
        '--noise-range',
        nargs=2,
        type=float,
        metavar=('N_MIN', 'N_MAX'),
        default=[18e-6, 80e-6],
        help="Noise range for white_noise cuts: [N_MIN N_MAX] (default: 18e-6 80e-6)"
    )
    parser.add_argument(
        '--nproc',
        help="Number of parallel processes to run on.",
        type=int,
        default=16
    )
    parser.add_argument(
        '--errlog-ext',
        help="Error log file name.",
        default='iso_noise_check_err.txt'
    )
    parser.add_argument(
        '--savename',
        help="Output dictionary save name.",
        default='iso_cuts_check.npy'
    )
    return parser

def ranges_difference(a, b):
    """
    Returns a new Ranges object with intervals in `a` that are not in `b`.
    """
    a_intervals = a.ranges()
    b_intervals = b.ranges()
    if len(a_intervals) == 0:
        return 0
    if len(b_intervals) == 0:
        return 0
    # Build a mask for all samples in a, then remove those in b
    # This assumes all intervals are within [0, N)
    max_stop = max(a_intervals[:,1].max(), b_intervals[:,1].max())
    mask = np.zeros(max_stop, dtype=bool)
    for start, stop in a_intervals:
        mask[start:stop] = True
    for start, stop in b_intervals:
        mask[start:stop] = False
    return mask

def count_new_cuts(ranges_matrix, survivors_mask, already_cut):
    """
    ranges_matrix: RangesMatrix for this operation (e.g., x.fp_flags.fp_nans)
    survivors_mask: 1D bool array, True for detectors that survive all cuts
    already_cut: list of Ranges, one per detector, representing samples already cut
    Returns: total number of new samples cut for surviving detectors, updated already_cut
    """
    n_dets = ranges_matrix.shape[0]
    new_cut_count = 0
    updated_already_cut = []
    for det_idx in range(n_dets):
        if not survivors_mask[det_idx]:
            # This detector doesn't survive all cuts, skip
            updated_already_cut.append(already_cut[det_idx])
            continue
        op_ranges = ranges_matrix.ranges[det_idx]
        prev_ranges = already_cut[det_idx]
        # Compute new cuts: samples in op_ranges but not in prev_ranges
        # This is set subtraction: op_ranges - prev_ranges
        det_cut_count = np.sum(np.ptp((op_ranges * ~prev_ranges).ranges(),axis=1))
        # Count new samples cut
        if det_cut_count != 0:
            new_cut_count += det_cut_count
            # Update already_cut for this detector
            updated_already_cut.append(prev_ranges + op_ranges)
        else:
            updated_already_cut.append(prev_ranges)
    return new_cut_count, updated_already_cut

def get_dict_entry(entry, config_file_init, config_file_proc, noise_range):
    try:
        logger = preprocess_util.init_logger('subproc_logger')
        logger.info(f'Processing entry for {entry["dataset"]}')
        logger.info(f'Getting context for {entry["dataset"]}')
        _, context_init = preprocess_util.get_preprocess_context(config_file_init)
        _, context_proc = preprocess_util.get_preprocess_context(config_file_proc)
        dets = {'wafer_slot':entry['dets:wafer_slot'],
                'wafer.bandpass':entry['dets:wafer.bandpass']}
        mdata_init = context_init.get_meta(entry['obs:obs_id'], dets=dets)
        mdata_proc = context_proc.get_meta(entry['obs:obs_id'], dets=dets)
        del context_proc
        del context_init    
        x = mdata_init.preprocess

        ndet = mdata_init.dets.count
        nsamp = mdata_init.samps.count

        # Find which detectors survive all cuts (end_yield)
        proc_survivors_mask = has_all_cut(mdata_proc.preprocess.split_flags.valid)  # shape: (ndet,)
        survivors_mask = np.zeros(mdata_init.dets.count, dtype=bool)
        _, ind_init, _ = np.intersect1d(mdata_init.dets.vals, mdata_proc.dets.vals[proc_survivors_mask], return_indices=True)
        survivors_mask[ind_init] = True

        # Initialize already_cut as empty Ranges for each detector
        already_cut = RangesMatrix.zeros((ndet, nsamp))

        keys = []
        vals = []

        # First layer
        keys.append('nsamps')
        vals.append(nsamp)
        keys.append('ndets')
        vals.append(ndet)
        # fp flags
        m = has_all_cut(x.fp_flags.valid)
        keys.append('fp_cuts')
        vals.append(np.sum(has_all_cut(x.fp_flags.fp_nans)[m]))
        # trends
        m = has_all_cut(x.trends.valid)
        keys.append('trend_cuts')
        vals.append(np.sum(has_all_cut(x.trends.trend_flags)[m]))
        # turnarounds
        m = has_all_cut(x.turnaround_flags.valid)
        n_turnaround_cuts, already_cut = count_new_cuts(x.turnaround_flags.turnarounds, survivors_mask, already_cut)
        keys.append('turnaround_nsamps')
        vals.append(int(n_turnaround_cuts))
        # jumps and glitches
        m = has_all_cut(x.jumps_slow.valid)
        n_jumps_slow_cuts, already_cut = count_new_cuts(x.jumps_slow.jump_flag, survivors_mask, already_cut)
        keys.append('jumps_slow_nsamps')
        vals.append(int(n_jumps_slow_cuts))
        keys.append('jumps_slow_cuts')
        vals.append(np.sum(count_cuts(x.jumps_slow.jump_flag)[m] > 5))
        m = has_all_cut(x.jumps_2pi.valid)
        n_jumps_2pi_cuts, already_cut = count_new_cuts(x.jumps_2pi.jump_flag, survivors_mask, already_cut)
        keys.append('jumps_2pi_nsamps')
        vals.append(int(n_jumps_2pi_cuts))
        keys.append('jumps_2pi_cuts')
        vals.append(np.sum(count_cuts(x.jumps_2pi.jump_flag)[m] > 20))
        m = has_all_cut(x.glitches.valid)
        n_glitch_nsamps, already_cut = count_new_cuts(x.glitches.glitch_flags, survivors_mask, already_cut)
        keys.append('glitch_nsamps')
        vals.append(int(n_glitch_nsamps))
        keys.append('glitch_cuts')
        vals.append(np.sum(count_cuts(x.glitches.glitch_flags)[m] > 50000))
        # det bias flags
        m = has_all_cut(x.det_bias_flags.valid)
        keys.append('det_bias_cuts')
        vals.append(np.sum(has_all_cut(x.det_bias_flags.det_bias_flags)[m]))
        # ptp cut
        m = has_all_cut(x.ptp_flags.valid)
        keys.append('ptp_cuts')
        vals.append(np.sum(has_all_cut(x.ptp_flags.ptp_flags)[m]))
        # noise
        m = has_all_cut(x.white_noise_nofit.valid)
        n_min, n_max = noise_range
        keys.append('white_noise_cuts')
        vals.append(np.sum(((x.white_noise_nofit.white_noise)[m] < n_min) | ((x.white_noise_nofit.white_noise)[m] > n_max)))

        # edge cuts from demodulation
        edge_mask = np.zeros((ndet,nsamp), dtype=bool)
        edge_mask[:, :6000] = True
        edge_mask[:,-6000:] = True
        edge_ranges = RangesMatrix.from_mask(edge_mask)
        n_edge_cuts, already_cut = count_new_cuts(edge_ranges, survivors_mask, already_cut)
        keys.append('edge_nsamps')
        vals.append(int(n_edge_cuts))


        x = mdata_proc.preprocess
        # Inv Var Cuts
        m = has_all_cut(x.inv_var_flags.valid)
        keys.append('inv_var_cuts')
        vals.append(np.sum(has_all_cut(x.inv_var_flags.inv_var_flags)[m]))

        # TOD stats
        # T
        m = has_all_cut(x.tod_stats_T.valid)
        noisy_subscan_indicator = np.zeros_like(x.tod_stats_T["std"][m], dtype=bool)
        keys.append('TOD_stats_T_ptp')
        y = x.tod_stats_T.ptp[m] > 0.8
        vals.append(np.sum(y, -1))
        keys.append('TOD_stats_T_std')
        median_std = np.median(x.tod_stats_T["std"][m], axis=1)[:, np.newaxis]
        y = x.tod_stats_T["std"][m] > median_std * 3.0
        vals.append(np.sum(y, -1))   
        keys.append('TOD_stats_T_det_cut')
        vals.append(np.sum(np.sum(noisy_subscan_indicator, -1) >= mdata_init.subscans.count//2))
        # Q
        m = has_all_cut(x.tod_stats_Q.valid)
        noisy_subscan_indicator = np.zeros_like(x.tod_stats_Q["std"][m], dtype=bool)
        keys.append('TOD_stats_Q_ptp')
        y = x.tod_stats_Q.ptp[m] > 0.8
        vals.append(np.sum(y, -1))
        keys.append('TOD_stats_Q_std')
        median_std = np.median(x.tod_stats_Q["std"][m], axis=1)[:, np.newaxis]
        y = x.tod_stats_Q["std"][m] > median_std * 3.0
        vals.append(np.sum(y, -1))   
        keys.append('TOD_stats_Q_kurt')
        y = (np.abs(x.tod_stats_Q['kurtosis']) > 0.5)
        vals.append(np.sum(y, -1))
        keys.append('TOD_stats_Q_skew')
        y = (np.abs(x.tod_stats_Q['skew']) > 0.5)
        vals.append(np.sum(y, -1))
        keys.append('TOD_stats_Q_det_cut')
        vals.append(np.sum(np.sum(noisy_subscan_indicator, -1) >= mdata_init.subscans.count//2))
        # U
        m = has_all_cut(x.tod_stats_U.valid)
        keys.append('TOD_stats_U_ptp')
        y = x.tod_stats_U.ptp[m] > 0.8
        vals.append(np.sum(y, -1))
        keys.append('TOD_stats_U_std')
        median_std = np.median(x.tod_stats_U["std"][m], axis=1)[:, np.newaxis]
        y = x.tod_stats_U["std"][m] > median_std * 3.0
        vals.append(np.sum(y, -1))
        keys.append('TOD_stats_U_kurt')
        y = (np.abs(x.tod_stats_U['kurtosis']) > 0.5)
        vals.append(np.sum(y, -1))
        keys.append('TOD_stats_U_skew')
        y = (np.abs(x.tod_stats_U['skew']) > 0.5)
        vals.append(np.sum(y, -1))
        keys.append('TOD_stats_U_det_cut')
        vals.append(np.sum(np.sum(noisy_subscan_indicator, -1) >= mdata_init.subscans.count//2))
        # noisy subscans
        m = has_all_cut(x.noisy_subscan_flags.valid)
        n_noisy_subscans_cuts, already_cut = count_new_cuts(x.noisy_subscan_flags.valid_subscans, survivors_mask, already_cut)
        keys.append('noisy_subscans_nsamps')
        vals.append(int(n_noisy_subscans_cuts))
        # noisy subscan dets
        keys.append('noisy_subscans_cuts')
        vals.append(np.sum(~x.noisy_dets_flags.valid_dets))
        # moon flags
        m = has_all_cut(x.source_flags.valid)
        n_source_cuts, already_cut = count_new_cuts(x.source_flags.moon, survivors_mask, already_cut)
        keys.append('source_flags_nsamps')
        vals.append(int(n_source_cuts))
        keys.append('source_flags_cuts')
        vals.append(np.sum(has_all_cut(x.source_flags.moon)[m]))
        
        # End yield
        m = has_all_cut(x.split_flags.valid)
        keys.append('end_yield')
        vals.append(np.sum(m))
        return entry['obs:obs_id'], entry['dets:wafer_slot'], entry['dets:wafer.bandpass'], keys, vals
    except Exception as e:
        # Collects errors if this fails.
        logger.info(f"Error in process for {entry['dataset']}")
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        return None, None, None, errmsg, tb

def main(executor, as_completed_callable, config_file_init,
         config_file_proc, errlog_ext, savename, noise_range, nproc):
    # Prepares the error log file and logger to write info and errors out to
    logger = preprocess_util.init_logger('main_proc')
    configs_proc, _ = preprocess_util.get_preprocess_context(config_file_proc)
    base_dir = os.path.dirname(configs_proc['archive']['index'])
    errlog = errlog_ext

    logger.info('connect to database')
    # Connects to the archive database as described in the previous section
    proc = core.metadata.ManifestDb(os.path.join(base_dir, 'process_archive.sqlite'))

    sqlite_path = savename.replace('.h5', '.sqlite')
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()

    run_list = proc.inspect()
    logger.info('run list created')

    first_entry = run_list[0]
    obsid, ws, band, keys, vals = get_dict_entry(entry=first_entry, config_file_init=config_file_init,
                                                 config_file_proc=config_file_proc, noise_range=args.noise_range)
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
    logger.info('deleted database connection')
    n = 0
    ntot = len(run_list)

    logger.info(f'Writing to sqlite file at {savename.replace('.h5', '.sqlite')}')
    futures = [executor.submit(get_dict_entry, entry=entry,
                                config_file_init=config_file_init,
                                config_file_proc=config_file_proc,
                                noise_range=args.noise_range) for entry in run_list]
    for future in as_completed_callable(futures):
        try:
            obsid, ws, band, keys, vals = future.result()
            logger.info(f'{n}/{ntot}: Unpacked future for {ws}, {band}')
        except Exception as e:
            logger.info('{n}/{ntot}: Future unpack error.')
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
                    f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
                continue
        n+=1
    logger.info(f'All entries written to sqlite file at {savename.replace('.h5', '.sqlite')}')
    conn.close()
if __name__ == '__main__':
    args = get_parser().parse_args()
    rank, executor, as_completed_callable = get_exec_env(args.nproc)
    if rank == 0:
        main(executor=executor, as_completed_callable=as_completed_callable, **vars(args))
