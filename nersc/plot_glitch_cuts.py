import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import traceback
import yaml
import time

from sotodlib import core, preprocess
from operator import attrgetter
from sotodlib.core.flagman import sparse_to_ranges_matrix, count_cuts, has_any_cuts, has_all_cut
from sotodlib.preprocess.pcore import update_full_aman
from pixell.utils import block_expand, block_reduce
from sotodlib.tod_ops import filters
from scipy.stats import iqr
import sotodlib.site_pipeline.util as sp_util

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('arxiv', help="Preprocessing arxiv filepath")

    parser.add_argument('configs', help="Preprocess config file path")
    
    parser.add_argument(
        '--nproc',
        help="Number of parallel processes to run on.",
        type=int,
        default=4
    )

    parser.add_argument(
        '--errlog-ext',
        help="Error log file name.",
        default='check_opt_yield_err.txt'
    )

    parser.add_argument(
        '--plotdir',
        help="Output dictionary save name.",
        default='./'
    )
    return parser

def make_cuts_plots(rl,plotdir,configs):
    configs = yaml.safe_load(open(configs, "r"))
    ctx = core.Context(configs['context_file'])
    pipe = preprocess.Pipeline(configs["process_pipe"])

    aman = ctx.get_obs(rl['obs:obs_id'], dets={'wafer_slot': rl['dets:wafer_slot'],
                                           'wafer.bandpass': rl['dets:wafer.bandpass']})
    proc_aman = core.AxisManager( aman.dets, aman.samps)
    full = core.AxisManager( aman.dets, aman.samps)

    for i, process in enumerate(pipe):
        process.process(aman, proc_aman)
        process.calc_and_save(aman, proc_aman)
        update_full_aman(proc_aman, full, True)
        if process.name == 'glitches':
            break
        process.select(aman, proc_aman)
        proc_aman.restrict('dets', aman.dets.vals)

    flag = sparse_to_ranges_matrix(
            proc_aman.glitches.glitch_detection > 10
        )
    ncuts = count_cuts(flag)

    hp_fc = 1
    t_glitch = .00001
    n_sig = 10
    filt = filters.high_pass_sine2(cutoff=hp_fc) * filters.gaussian_filter(t_sigma=t_glitch)
    
    mos = '''
    AAAAA
    BBBBB
    CDEFG
    HIJKL
    '''
    plabs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    buff = 400
    fvec, iqr_range = get_glitch_crit(aman, filt)
    for idx in np.where(ncuts >= 10)[0]:
        fig, axs = plt.subplot_mosaic(mos, figsize=(12,8), layout='constrained')
        plt.suptitle(f'obsid: {rl["obs:obs_id"]}, wafer: {rl["dets:wafer_slot"]}, band {rl["dets:wafer.bandpass"]}\ndetid: {aman.dets.vals[idx]}')
        axs['A'].plot(aman.timestamps, aman.signal[idx])
        axs['B'].plot(aman.timestamps[2000:-2000], fvec[idx,2000:-2000])
        axs['B'].axhline(iqr_range[idx, None] * n_sig, c='r', ls=':')
        for i, [r0, r1] in enumerate(flag[idx].ranges()):
            axs['A'].axvspan(aman.timestamps[r0], aman.timestamps[r1], alpha=0.5, color='grey')
            if i < 10:
                axs[plabs[i+2]].plot(aman.timestamps[r0-buff:r1+buff], aman.signal[idx, r0-buff:r1+buff])
                axs[plabs[i+2]].axvspan(aman.timestamps[r0], aman.timestamps[r1], alpha=0.5, color='grey')
        plt.savefig(os.path.join(plotdir, f'{rl["obs:obs_id"]}_{aman.dets.vals[idx]}'))
        plt.close()

def get_dict_entry(base_dir, entry, plotdir, configs):
    cut_names = ['det_bias_flags', 'trends', 'jumps_2pi', 'glitches', 'ptp_flags', 'inv_var_flags']
    try:
        path = os.path.join(base_dir, entry['filename'])
        test = core.AxisManager.load(path, entry['dataset'])
        i = -1000
        cuts = []
        names = []
        cuts.append(test.dets.count)
        names.append('start')
        for fld in test._fields:
            if i == 1:
                cuts.append(np.sum(has_all_cut(test[fld].valid)))
                names.append(name)
            if fld in cut_names:
                i = 0
                name = fld
            i += 1
        cuts.append(np.sum(has_all_cut(test.inv_var_flags.inv_var_flags)[has_all_cut(test.inv_var_flags.valid)]))
        names.append('inv_var_flags')
        if (cuts[4]/cuts[1] < 0.4) and (cuts[1] > 100):
            make_cuts_plots(entry, plotdir, configs)
            return 'success', entry['obs:obs_id'], entry['dets:wafer_slot'], entry['dets:wafer.bandpass'], entry['_id']
        else:
            return 'skip', None, None, None, None
    except Exception as e:
        #print(f"Error in process for {entry['dataset']}")
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        return errmsg, tb, None, None, None

def get_glitch_crit(aman, filt):
    fvec = filters.fourier_filter(
    aman, filt, detrend=None, signal_name='signal', resize="zero_pad"
    )
    # get the threshods based on n_sig x nlev = n_sig x iqu x 0.741
    fvec = np.abs(fvec)
    if fvec.shape[1] > 50000:
        ds = int(fvec.shape[1]/20000)
    else: 
        ds = 1
    iqr_range = 0.741 * iqr(fvec[:,::ds], axis=1)
    return fvec, iqr_range

def main(arxiv, nproc, errlog_ext, plotdir,configs):
    base_dir = os.path.dirname(arxiv)
    errlog = os.path.join(plotdir, errlog_ext)
    proc = core.metadata.ManifestDb(os.path.join(base_dir, 'process_archive.sqlite'))

    multiprocessing.set_start_method('spawn')
    run_list = proc.inspect()
    nruns = len(run_list)
    flagged_ids = []
    with ProcessPoolExecutor(nproc) as exe:
        futures = [exe.submit(get_dict_entry, base_dir=base_dir, entry=entry, plotdir=plotdir, configs=configs) for entry in run_list]
        for future in as_completed(futures):
            try:
                err, obsid, ws, band, nr = future.result()
            except Exception as e:
                errmsg = f'{type(e)}: {e}'
                tb = ''.join(traceback.format_tb(e.__traceback__))
                f = open(errlog, 'a')
                f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
                f.close()
                continue
            futures.remove(future)
            
            if err == 'success':
                print(f'{nr}/{nruns} completed = {int(100*nr/nruns)}%')
                flagged_ids.append([obsid, ws, band])
                continue 
            elif err == 'skip':
                continue
            else:
                f = open(errlog, 'a')
                f.write(f'\n{time.time()}, error\n{err}\n{obsid}\n')
                f.close()

if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)
