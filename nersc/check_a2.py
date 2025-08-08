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
        default='check_a2_yield_err.txt'
    )

    parser.add_argument(
        '--savename',
        help="Output dictionary save name.",
        default='check_a2.npy'
    )

    parser.add_argument(
        '--min-ctime',
        help="Minimum ctime to include in runlist",
        type=int, default=None
    )
    
    parser.add_argument(
        '--max-ctime',
        help="Maximum ctime to include in runlist",
        type=int, default=None
    )
    
    parser.add_argument(
        '--output-dir',
        help="Path to output directory.",
        default='/global/cfs/cdirs/sobs/users/davidvng/preprocess/outputs'
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
        outdict = {}
        hwp = False
        pwv = False
        # Check that HWP was spinning
        if not(np.isclose(np.mean(meta.hwp_solution.hwp_rate_1), 0)):
            hwp = True
        # Check that PWV data is present
        if not(np.all(np.isnan(meta.get('pwv_class', np.nan)))):
            pwv = True
        if hwp:
            for fld in test._fields:
                outdict['a2sin'] = meta.preprocess.hwpss_stats.coeffs[:,2]
                outdict['a2cos'] = meta.preprocess.hwpss_stats.coeffs[:,3]
                outdict['a2f'] = np.sqrt((meta.preprocess.hwpss_stats.coeffs[:,2]*meta.det_cal.phase_to_pW)**2 + (meta.preprocess.hwpss_stats.coeffs[:,3]*meta.det_cal.phase_to_pW)**2)
                outdict['phase2pW'] = meta.det_cal.phase_to_pW
                outdict['bandpass'] = meta.det_info.wafer.bandpass
                outdict['rfrac'] = meta.det_cal.r_frac
                outdict['el'] = round(meta.obs_info.el_center)
                # outdict['hwp_angle'] = meta.hwp_angle
                outdict['valid'] = meta.preprocess.hwpss_stats.valid
                outdict['redchi2s'] = meta.preprocess.hwpss_stats.redchi2s
                if pwv:
                    outdict['pwv'] = np.nanmean(meta.pwv_class)
                else:
                    outdict['pwv'] = None
        return meta.det_info.det_id, entry['obs:obs_id'], entry['dets:wafer_slot'], outdict
    except Exception as e:
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        return None, None, errmsg, tb

def main(config, nproc, errlog_ext, savename, min_ctime, max_ctime, output_dir):
    configs,ctx = pt._get_preprocess_context(config)
    arxiv = configs['archive']['index']
    base_dir = os.path.dirname(arxiv)
    errlog = os.path.join(output_dir, errlog_ext)
    proc = core.metadata.ManifestDb(os.path.join(base_dir, 'process_archive.sqlite'))
    outdict = {'ws0': {'oid' : [], 'did' : [], 'output' : []},
               'ws1': {'oid' : [], 'did' : [], 'output' : []},
               'ws2': {'oid' : [], 'did' : [], 'output' : []},
               'ws3': {'oid' : [], 'did' : [], 'output' : []},
               'ws4': {'oid' : [], 'did' : [], 'output' : []},
               'ws5': {'oid' : [], 'did' : [], 'output' : []},
               'ws6': {'oid' : [], 'did' : [], 'output' : []}}

    multiprocessing.set_start_method('spawn')
    entries = proc.inspect()
    if min_ctime is None:
        run_list = np.asarray(entries)
    else:
        ctimes = np.asarray([int(e['obs:obs_id'].split('_')[1]) for e in entries])
        if max_ctime is not None:
            m = np.logical_and(ctimes > min_ctime, ctimes < max_ctime)
        else:
            m = ctimes > min_ctime
        run_list = np.asarray(entries)[m]

    nruns = len(run_list)
    print(f'Total number of runs: {nruns}')
    i = 0

    with ProcessPoolExecutor(nproc) as exe:
        futures = [exe.submit(get_dict_entry, base_dir=base_dir, entry=entry, config=config) for entry in run_list]
        for future in as_completed(futures):
            try:
                did, oid, ws, output = future.result()
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
                f.write(f'\n{time.time()}, error\n{output}\n')
                f.close()
            else:
                print(f'Saving {oid}, {ws}, run {i}/{nruns}')
                outdict[ws]['did'].append(did)
                outdict[ws]['oid'].append(oid)
                outdict[ws]['output'].append(output)
                np.save(os.path.join(output_dir, savename), outdict)
            i += 1

if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)
