import numpy as np
import os
import argparse
import traceback
import time
import sqlite3

from sotodlib import core
from sotodlib.core.flagman import has_all_cut, has_any_cuts, count_cuts
import sotodlib.site_pipeline.util as sp_util
from sotodlib.preprocess import preprocess_util
from sotodlib.utils.procs_pool import get_exec_env

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('config_file_init',
                        help="Preprocessing init configuration file")
    parser.add_argument('config_file_proc',
                        help="Preprocessing proc configuration file")
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
def get_dict_entry(entry, config_file_init, config_file_proc):
    try:
        logger = preprocess_util.init_logger('subproc_logger')
        logger.info(f'Processing entry for {entry["dataset"]}')
        logger.info(f'Getting context for {entry["dataset"]}')
        _, context_init = preprocess_util.get_preprocess_context(config_file_init)
        _, context_proc = preprocess_util.get_preprocess_context(config_file_proc)
        dets = {'wafer_slot':entry['dets:wafer_slot'],
                'wafer.bandpass':entry['dets:wafer.bandpass']}
        mdata_init = context_init.get_meta(entry['obs:obs_id'], dets=dets)
        del context_init    
        x = mdata_init.preprocess

        keys = []
        vals = []

        # First layer
        keys.append('nsamps')
        vals.append(mdata_init.samps.count)
        keys.append('ndets')
        vals.append(mdata_init.dets.count)
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
        keys.append('turnaround_nsamps')
        vals.append(np.sum([np.sum(np.ptp(r.ranges(), axis=1)) for r, mask in zip(x.turnaround_flags.turnarounds.ranges, m) if mask]))
        # jumps and glitches
        m = has_all_cut(x.jumps_slow.valid)
        keys.append('jumps_slow_nsamps')
        vals.append(np.sum([np.sum(np.ptp(r.ranges(), axis=1)) for r, mask in zip(x.jumps_slow.jump_flag.ranges, m) if mask]))
        keys.append('jumps_slow_cuts')
        vals.append(np.sum(count_cuts(x.jumps_slow.jump_flag)[m] > 5))
        m = has_all_cut(x.jumps_2pi.valid)
        keys.append('jumps_2pi_nsamps')
        vals.append(np.sum([np.sum(np.ptp(r.ranges(), axis=1)) for r, mask in zip(x.jumps_2pi.jump_flag.ranges, m) if mask]))
        keys.append('jumps_2pi_cuts')
        vals.append(np.sum(count_cuts(x.jumps_2pi.jump_flag)[m] > 20))
        m = has_all_cut(x.glitches.valid)
        keys.append('glitch_nsamps')
        vals.append(np.sum([np.sum(np.ptp(r.ranges(), axis=1)) for r, mask in zip(x.glitches.glitch_flags.ranges, m) if mask]))
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
        keys.append('white_noise_cuts')
        vals.append(np.sum(((x.white_noise_nofit.white_noise)[m] < 2e-6) | ((x.white_noise_nofit.white_noise)[m] > 60e-6)))

        # Second layer
        mdata_proc = context_proc.get_meta(entry['obs:obs_id'], dets=dets)
        del context_proc
        
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
        keys.append('noisy_subscans_nsamps')
        vals.append(np.sum([np.sum(np.ptp(r.ranges(), axis=1)) for r, mask in zip(x.noisy_subscan_flags.valid_subscans.ranges, m) if mask]))
        # noisy subscan dets
        keys.append('noisy_subscans_cuts')
        vals.append(np.sum(~x.noisy_dets_flags.valid_dets))
        # moon flags
        m = has_all_cut(x.source_flags.valid)
        keys.append('source_flags_nsamps')
        vals.append(np.sum([np.sum(np.ptp(r.ranges(), axis=1)) for r, mask in zip(x.source_flags.moon.ranges, m) if mask]))
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
         config_file_proc, errlog_ext, savename, nproc):
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
    
    n_attempts = 0
    for first_entry in run_list[::-14]:
        obsid, ws, band, keys, vals = get_dict_entry(entry=first_entry, config_file_init=config_file_init,
                                                     config_file_proc=config_file_proc)
        if obsid is not None:
            break
        n_attempts += 1
        logger.info(f"N_attempts = {n_attempts}")
        logger.info(f"error: {keys}, tb: {vals}")

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
                                config_file_proc=config_file_proc) for entry in run_list]
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
    logger.info(f"All entries written to sqlite file at {savename.replace('.h5', '.sqlite')}")
    conn.close()
if __name__ == '__main__':
    args = get_parser().parse_args()
    rank, executor, as_completed_callable = get_exec_env(args.nproc)
    if rank == 0:
        main(executor=executor, as_completed_callable=as_completed_callable, **vars(args))
