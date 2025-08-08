import numpy as np
import os
import argparse
import traceback

from sotodlib import core
from sotodlib.core.flagman import has_all_cut, has_any_cuts
import sotodlib.site_pipeline.util as sp_util

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('arxiv', help="Preprocessing arxiv filepath")
    
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
    return parser

def get_dict_entry(base_dir, entry):
    cut_names = ['det_bias_flags', 'trends', 'jumps_2pi', 'glitches', 'ptp_flags', 'inv_var_flags']
    try:
        #print(f"Starting process for {entry['dataset']}")
        path = os.path.join( base_dir, entry['filename'])
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
        #print(f"Finished process for {entry['dataset']}")
        return entry['_id'], entry['dets:wafer_slot'], entry['dets:wafer.bandpass'], cuts, names
    except Exception as e:
        #print(f"Error in process for {entry['dataset']}")
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        return None, None, None, errmsg, tb

def main(arxiv, nproc, errlog_ext, savename):
    base_dir = os.path.dirname(arxiv)
    errlog = os.path.join('/global/homes/m/msilvafe/so_home/shared_files/yield_check', errlog_ext)
    proc = core.metadata.ManifestDb(os.path.join(base_dir, 'process_archive.sqlite'))
    cut_names = ['det_bias_flags', 'trends', 'jumps_2pi', 'glitches', 'ptp_flags', 'inv_var_flags']
    all_cuts = []
    all_names = []
    outdict = {'ws0': {'f090': {'all_cuts': [], 'all_names' : []},
                       'f150': {'all_cuts': [], 'all_names' : []}},
               'ws1': {'f090': {'all_cuts': [], 'all_names' : []},
                       'f150': {'all_cuts': [], 'all_names' : []}},
               'ws2': {'f090': {'all_cuts': [], 'all_names' : []},
                       'f150': {'all_cuts': [], 'all_names' : []}},
               'ws3': {'f090': {'all_cuts': [], 'all_names' : []},
                       'f150': {'all_cuts': [], 'all_names' : []}},
               'ws4': {'f090': {'all_cuts': [], 'all_names' : []},
                       'f150': {'all_cuts': [], 'all_names' : []}},
               'ws5': {'f090': {'all_cuts': [], 'all_names' : []},
                       'f150': {'all_cuts': [], 'all_names' : []}},
               'ws6': {'f090': {'all_cuts': [], 'all_names' : []},
                      'f150': {'all_cuts': [], 'all_names' : []}}}

    multiprocessing.set_start_method('spawn')
    run_list = proc.inspect()
    nruns = len(run_list)
    with ProcessPoolExecutor(nproc) as exe:
        futures = [exe.submit(get_dict_entry, base_dir=base_dir, entry=entry) for entry in run_list]
        for future in as_completed(futures):
            try:
                nr, ws, band, cuts, names = future.result()
                #print(f'Unpacked future for {ws}, {band}')
            except Exception as e:
                #print('Future unpack error.')
                errmsg = f'{type(e)}: {e}'
                tb = ''.join(traceback.format_tb(e.__traceback__))
                f = open(errlog, 'a')
                f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
                f.close()
                continue
            futures.remove(future)
            
            if nr is None:
                #print('Writing error to log.')
                f = open(errlog, 'a')
                f.write(f'\n{time.time()}, error\n{cuts}\n{names}\n')
                f.close()
            else:
                print(f'{nr}/{nruns} complete = {int(100*nr/nruns)}%')
                outdict[ws][band]['all_cuts'].append(cuts)
                outdict[ws][band]['all_names'].append(names)
                #print('Saving to npy file.')
                np.save(os.path.join('/global/homes/m/msilvafe/so_home/shared_files/yield_check', savename), outdict)

if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)
