import numpy as np
import os
import argparse
import traceback
import time
from scipy import stats

from sotodlib import core
from sotodlib.core.flagman import has_all_cut, has_any_cuts
import sotodlib.site_pipeline.util as sp_util
from sotodlib.site_pipeline import preprocess_tod as pt

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Preprocessing arxiv filepath")
    
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
        '--savename',
        help="Output dictionary save name.",
        default='opt_yield.npy'
    )

    parser.add_argument(
        '--min-ctime',
        help="Minimum ctime to include in runlist",
        type=int, default=None
    )
    return parser

def get_dict_entry(base_dir, entry, config):
    nperseg = 65536 #2**16
    try:
        #print(f"Starting process for {entry['dataset']}")
        configs,ctx = pt._get_preprocess_context(config)
        path = os.path.join( base_dir, entry['filename'])
        meta = ctx.get_meta(entry['obs:obs_id'], dets={'wafer_slot':entry['dets:wafer_slot'], 
                            'wafer.bandpass': entry['dets:wafer.bandpass']})
        test = meta.preprocess
        nseg = test.samps.count//nperseg
        biasfact = 2*nseg/stats.chi2.ppf(0.5, 2*nseg)
        outdict = {}
        for fld in test._fields:
            outdict['fit_noiseT'] = np.sqrt(test.noiseT_fit.fit[:,1]*biasfact)
            outdict['nofit_noiseT'] = test.noiseT_nofit.white_noise*np.sqrt(biasfact)
            outdict['fit_noiseQ'] = np.sqrt(test.noiseQ_fit.fit[:,1]*biasfact)
            outdict['nofit_noiseQ'] = test.noiseQ_nofit.white_noise*np.sqrt(biasfact)
            outdict['fit_noiseU'] = np.sqrt(test.noiseU_fit.fit[:,1]*biasfact)
            outdict['nofit_noiseU'] = test.noiseU_nofit.white_noise*np.sqrt(biasfact)
            outdict['el'] = round(meta.obs_info.el_center)
            outdict['xi'] = meta.focal_plane.xi
            outdict['eta'] = meta.focal_plane.eta
            outdict['s2'] = test.hwpss_stats.coeffs[:,2]
            outdict['c2'] = test.hwpss_stats.coeffs[:,3]
            outdict['biasfact'] = biasfact
        return meta.det_info.det_id, entry['obs:obs_id'], entry['dets:wafer_slot'], entry['dets:wafer.bandpass'], outdict
    except Exception as e:
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        return None, None, None, errmsg, tb

def main(config, nproc, errlog_ext, savename, min_ctime):
    configs,ctx = pt._get_preprocess_context(config)
    arxiv = configs['archive']['index']
    base_dir = os.path.dirname(arxiv)
    errlog = os.path.join('/global/homes/m/msilvafe/so_home/shared_files/net_check', errlog_ext)
    proc = core.metadata.ManifestDb(os.path.join(base_dir, 'process_archive.sqlite'))
    outdict = {'ws0': {'f090': {'oid' : [], 'did' : [], 'noise' : []}, 'f150': {'oid' : [], 'did' : [], 'noise' : []}},
               'ws1': {'f090': {'oid' : [], 'did' : [], 'noise' : []}, 'f150': {'oid' : [], 'did' : [], 'noise' : []}},
               'ws2': {'f090': {'oid' : [], 'did' : [], 'noise' : []}, 'f150': {'oid' : [], 'did' : [], 'noise' : []}},
               'ws3': {'f090': {'oid' : [], 'did' : [], 'noise' : []}, 'f150': {'oid' : [], 'did' : [], 'noise' : []}},
               'ws4': {'f090': {'oid' : [], 'did' : [], 'noise' : []}, 'f150': {'oid' : [], 'did' : [], 'noise' : []}},
               'ws5': {'f090': {'oid' : [], 'did' : [], 'noise' : []}, 'f150': {'oid' : [], 'did' : [], 'noise' : []}},
               'ws6': {'f090': {'oid' : [], 'did' : [], 'noise' : []}, 'f150': {'oid' : [], 'did' : [], 'noise' : []}}}

    multiprocessing.set_start_method('spawn')
    entries = proc.inspect()
    if min_ctime is None:
        run_list = np.asarray(entries)
    else:
        ctimes = np.asarray([int(e['obs:obs_id'].split('_')[1]) for e in entries])
        m = ctimes > min_ctime #1722402000 # SATp3 Run 13
        run_list = np.asarray(entries)[m]

    nruns = len(run_list)
    print(f'Total number of runs: {nruns}')
    i = 0

    with ProcessPoolExecutor(nproc) as exe:
        futures = [exe.submit(get_dict_entry, base_dir=base_dir, entry=entry, config=config) for entry in run_list]
        for future in as_completed(futures):
            try:
                did, oid, ws, band, output = future.result()
            except Exception as e:
                print('Writing to error log')
                errmsg = f'{type(e)}: {e}'
                tb = ''.join(traceback.format_tb(e.__traceback__))
                f = open(errlog, 'a')
                f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
                f.close()
                continue
            futures.remove(future)
            
            if did is None:
                f = open(errlog, 'a')
                f.write(f'\n{time.time()}, error\n{band}\n{output}\n')
                f.close()
            else:
                print(f'Saving {oid}, {ws}, {band}, run {i}/{nruns}')
                outdict[ws][band]['did'].append(did)
                outdict[ws][band]['oid'].append(oid)
                outdict[ws][band]['noise'].append(output)
                np.save(os.path.join('/global/homes/m/msilvafe/so_home/shared_files/net_check', savename), outdict)
            i += 1

if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)
