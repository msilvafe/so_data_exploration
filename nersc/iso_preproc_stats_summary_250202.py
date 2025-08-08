import numpy as np
import os
import argparse
import traceback
import time

from sotodlib import core
from sotodlib.core.flagman import has_all_cut, has_any_cuts, count_cuts
import sotodlib.site_pipeline.util as sp_util
from sotodlib.preprocess import preprocess_util

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('arxiv', help="Preprocessing arxiv filepath")

    parser.add_argument('configs', help="Preprocessing Configuration File")
    
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
        default='iso_noise_check.npy'
    )
    return parser

def get_dict_entry(base_dir, entry, config):
    t0 = time.time()
    try:
        logger = preprocess_util.init_logger('subproc_logger')
        logger.info(f'Processing entry for {entry["dataset"]}')
        path = os.path.join( base_dir, entry['filename'])
        logger.info(f'Getting context for {entry["dataset"]}')
        configs, context = preprocess_util.get_preprocess_context(config)
        dets = {'wafer_slot':entry['dets:wafer_slot'],
                'wafer.bandpass':entry['dets:wafer.bandpass']}
        #logger.info(f'Calling get meta {entry["dataset"]}')
        mdata = context.get_meta(entry['obs:obs_id'], dets=dets)
        #t1 = time.time()
        #logger.info(f'{t1-t0} sec to run get meta {entry["dataset"]}')
        del context
        del configs
        x = mdata.preprocess
        keys = []
        vals = []

        # det bias and fp cuts
        m = has_all_cut(x.det_bias_flags.valid)
        keys.append('det_bias_cuts_total')
        vals.append(np.sum(has_all_cut(x.det_bias_flags.det_bias_flags)[m]))
        keys.append('det_bias_cuts_bg')
        vals.append(np.sum(has_all_cut(x.det_bias_flags.bg_flags)[m]))
        keys.append('det_bias_cuts_rtes')
        vals.append(np.sum(has_all_cut(x.det_bias_flags.r_tes_flags)[m]))
        keys.append('det_bias_cuts_gt_rfrac')
        vals.append(np.sum(has_all_cut(x.det_bias_flags.r_frac_gt_flags)[m]))
        keys.append('det_bias_cuts_lt_rfrac')
        vals.append(np.sum(has_all_cut(x.det_bias_flags.r_frac_lt_flags)[m]))
        m = has_all_cut(x.fp_flags.valid)
        keys.append('fp_cuts')
        vals.append(np.sum(has_all_cut(x.fp_flags.fp_nans)[m]))
        #t2 = time.time()
        #logger.info(f'{t2-t1} sec to run det_bias_flags {entry["dataset"]}')

        # trends
        m = has_all_cut(x.trends.valid)
        keys.append('trend_cuts_total')
        vals.append(np.sum(has_all_cut(x.trends.trend_flags)[m]))
        #t3 = time.time()
        #logger.info(f'{t3-t2} sec to run trend {entry["dataset"]}')

        # jumps and glitches
        m = has_all_cut(x.jumps_slow.valid)
        keys.append('jumps_slow_count')
        vals.append(count_cuts(x.jumps_slow.jump_flag)[m])
        m = has_all_cut(x.jumps_2pi.valid)
        keys.append('jumps_2pi_count')
        vals.append(count_cuts(x.jumps_2pi.jump_flag)[m])
        m = has_all_cut(x.glitches.valid)
        keys.append('jumps_2pi_count')
        vals.append(count_cuts(x.glitches.glitch_flags)[m])
        #t4 = time.time()
        #logger.info(f'{t4-t3} sec to run jumps and glitches {entry["dataset"]}')

        # noise
        # fit
        m = has_all_cut(x.noise_signal_fit.valid)
        keys.append('noise_fit_knee')
        vals.append(x.noise_signal_fit.fit[m,0])
        keys.append('noise_fit_white')
        vals.append(x.noise_signal_fit.fit[m,1])
        keys.append('noise_fit_alpha')
        vals.append(x.noise_signal_fit.fit[m,2])
        # no fit
        m = has_all_cut(x.white_noise_nofit.valid)
        keys.append('noise_nofit_white')
        vals.append(x.white_noise_nofit.white_noise[m])
        #t5 = time.time()
        #logger.info(f'{t5-t4} sec to run white noise {entry["dataset"]}')

        # ptp cut
        m = has_all_cut(x.ptp_flags.valid)
        keys.append('ptp_cuts')
        vals.append(np.sum(has_all_cut(x.ptp_flags.ptp_flags)[m]))

        # Total yield
        m = has_all_cut(x.noiseU_nofit.valid)
        keys.append('total_yield')
        vals.append(np.sum(m))
        t6 = time.time()
        logger.info(f'{np.round(t6-t0,2)}s to process {entry["dataset"]}')

        return entry['obs:obs_id'], entry['dets:wafer_slot'], entry['dets:wafer.bandpass'], keys, vals
    except Exception as e:
        logger.info(f"Error in process for {entry['dataset']}")
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        return None, None, None, errmsg, tb

def main(arxiv, configs, nproc, errlog_ext, savename):
    logger = preprocess_util.init_logger('main_proc')
    base_dir = os.path.dirname(arxiv)
    errlog = os.path.join('/global/homes/m/msilvafe/so_home/shared_files/iso_noise_check', errlog_ext)
    logger.info('connect to database')
    logger = preprocess_util.init_logger('main_proc')
    proc = core.metadata.ManifestDb(os.path.join(base_dir, 'process_archive.sqlite'))

    outdict = {}
    noise_dict = {}
    fit_wn = []
    nofit_wn = []
    fit_knee = []
    ###lowf_wn = []
    
    logger.info('launch multiprocessing')
    multiprocessing.set_start_method('spawn')
    logger.info('multiprocess pool spawned')
    
    ###############################
    run_list = proc.inspect()
    logger.info('run list created')
    del proc
    logger.info('deleted database connection')

    n = 0
    ntot = len(run_list)

    with ProcessPoolExecutor(nproc) as exe:
        futures = [exe.submit(get_dict_entry, base_dir=base_dir, entry=entry, config=configs) for entry in run_list]
        for future in as_completed(futures):
            try:
                obsid, ws, band, keys, vals = future.result()
                logger.info(f'{n}/{ntot}: Unpacked future for {ws}, {band}')
            except Exception as e:
                logger.info('{n}/{ntot}: Future unpack error.')
                errmsg = f'{type(e)}: {e}'
                tb = ''.join(traceback.format_tb(e.__traceback__))
                f = open(errlog, 'a')
                f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
                f.close()
                continue
            futures.remove(future)
            
            if obsid is None:
                logger.info('Writing error to log.')
                f = open(errlog, 'a')
                f.write(f'\n{time.time()}, error\n{keys}\n{vals}\n')
                f.close()
            else:
                try:
                    if obsid in outdict.keys():
                        if ws in outdict[obsid].keys():
                            if not band in outdict[obsid][ws].keys():
                                outdict[obsid][ws][band]={}
                        else:
                            outdict[obsid][ws]={}
                            outdict[obsid][ws][band]={}
                    else:
                        outdict[obsid] = {}
                        outdict[obsid][ws] = {}
                        outdict[obsid][ws][band] = {}

                    for k, v in zip(keys, vals):
                        outdict[obsid][ws][band][k] = v
                    
                    logger.info(f'{n}/{ntot}: Finished with {obsid} {ws} {band}.')
                except Exception as e:
                    logger.info('Packaging and saving error.')
                    errmsg = f'{type(e)}: {e}'
                    tb = ''.join(traceback.format_tb(e.__traceback__))
                    f = open(errlog, 'a')
                    f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
                    f.close()
                    continue
            n+=1
    logger.info(f'Saving to npy file.')
    np.save(os.path.join('/global/homes/m/msilvafe/so_home/shared_files/iso_noise_check', savename), outdict)

if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)
