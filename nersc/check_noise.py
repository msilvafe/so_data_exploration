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

    parser.add_argument('configs', help="Preprocessing Configuration File")
    
    parser.add_argument(
        '--nproc',
        help="Number of parallel processes to run on.",
        type=int,
        default=4
    )

    parser.add_argument(
        '--errlog-ext',
        help="Error log file name.",
        default='check_noise_err.txt'
    )

    parser.add_argument(
        '--savename',
        help="Output dictionary save name.",
        default='noise_levels.npy'
    )
    return parser

def get_dict_entry(base_dir, entry):
    try:
        path = os.path.join( base_dir, entry['filename'])
        configs, context = pt._get_preprocess_context(config)
        dets = {'wafer_slot':entry['dets:wafer_slot'],
                'wafer.bandpass':entry['dets:wafer.bandpass']}
        mdata = context.get_meta(entry['obs:obs_id'], dets=dets)
        m = has_any_cuts(mdata.preprocess.pca_relcal.valid)
        rcal_scale_fact = np.median(mdata.preprocess.pca_relcal.medians[0]/mdata.preprocess.pca_relcal.relcal[m])/mdata.preprocess.pca_relcal.medians[0]

        mQf = has_all_cut(mdata['preprocess']['noiseQ_fit']['valid'])
        Qfit_NET = np.sqrt(mdata['preprocess']['noiseQ_fit']['fit'][mQf,1])*rcal_scale_fact
        Qfit_NEP = np.sqrt(mdata['preprocess']['noiseQ_fit']['fit'][mQf,1])/mdata['abscal']['abscal_factor'][mQf]/mdata.preprocess.pca_relcal.relcal[mQf]
        Qnofit_NET = mdata['preprocess']['noiseQ_nofit']['white_noise'][mQf]*rcal_scale_fact
        Qnofit_NEP = mdata['preprocess']['noiseQ_nofit']['white_noise'][mQf]/mdata['abscal']['abscal_factor'][mQf]/mdata.preprocess.pca_relcal.relcal[mQf]
        mTf = has_all_cut(mdata['preprocess']['noiseT_fit']['valid'])
        Tfit_NET = np.sqrt(mdata['preprocess']['noiseT_fit']['fit'][mTf,1])*rcal_scale_fact
        Tfit_NEP = np.sqrt(mdata['preprocess']['noiseT_fit']['fit'][mTf,1])/mdata['abscal']['abscal_factor'][mTf]/mdata.preprocess.pca_relcal.relcal[mTf]
        Tnofit_NET = mdata['preprocess']['noiseT_nofit']['white_noise'][mTf]*rcal_scale_fact
        Tnofit_NEP = mdata['preprocess']['noiseT_nofit']['white_noise'][mTf]/mdata['abscal']['abscal_factor'][mTf]/mdata.preprocess.pca_relcal.relcal[mTf]
        return entry['_id'], entry['dets:wafer_slot'], entry['dets:wafer.bandpass'], Qfit_NET, Qnofit_NET, Tfit_NET, Tnofit_NET, Qfit_NEP, Qnofit_NEP, Tfit_NET, T_nofit_NET
    except Exception as e:
        #print(f"Error in process for {entry['dataset']}")
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        return None, None, None, None, None, None, None, None, None, errmsg, tb

def main(arxiv, configs, nproc, errlog_ext, savename):
    base_dir = os.path.dirname(arxiv)
    errlog = os.path.join('/global/homes/m/msilvafe/so_home/shared_files/noise_check', errlog_ext)
    proc = core.metadata.ManifestDb(os.path.join(base_dir, 'process_archive.sqlite'))
    all_cuts = []
    all_names = []
    outdict = {'ws0': {'f090': {'Q': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []},
                                'T': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []}},
                       'f150': {'Q': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []},
                                'T': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []}}},
               'ws1': {'f090': {'Q': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []},
                                'T': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []}},
                       'f150': {'Q': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []},
                                'T': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []}}},
               'ws2': {'f090': {'Q': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []},
                                'T': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []}},
                       'f150': {'Q': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []},
                                'T': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []}}},
               'ws3': {'f090': {'Q': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []},
                                'T': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []}},
                       'f150': {'Q': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []},
                                'T': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []}}},
               'ws4': {'f090': {'Q': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []},
                                'T': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []}},
                       'f150': {'Q': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []},
                                'T': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []}}},
               'ws5': {'f090': {'Q': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []},
                                'T': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []}},
                       'f150': {'Q': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []},
                                'T': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []}}},
               'ws6': {'f090': {'Q': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []},
                                'T': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []}},
                       'f150': {'Q': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []},
                                'T': {'NET_nofit': [], 'NET_fit' : [], 'NEP_nofit': [], 'NEP_fit': []}}}}
    multiprocessing.set_start_method('spawn')
    run_list = proc.inspect()
    nruns = len(run_list)
    with ProcessPoolExecutor(nproc) as exe:
        futures = [exe.submit(get_dict_entry, base_dir=base_dir, entry=entry, configs=configs) for entry in run_list]
        for future in as_completed(futures):
            try:
                nr, ws, band, Qfit_NET, Qnofit_NET, Tfit_NET, Tnofit_NET, Qfit_NEP, Qnofit_NEP, Tfit_NET, T_nofit_NET = future.result()
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
                outdict[ws][band]['T']['NET_nofit'].append(Tnofit_NET)
                outdict[ws][band]['T']['NET_fit'].append(Tfit_NET)
                outdict[ws][band]['T']['NEP_nofit'].append(Tnofit_NEP)
                outdict[ws][band]['T']['NEP_fit'].append(Tfit_NEP)
                outdict[ws][band]['Q']['NET_nofit'].append(Tnofit_NET)
                outdict[ws][band]['Q']['NET_fit'].append(Tfit_NET)
                outdict[ws][band]['Q']['NEP_nofit'].append(Tnofit_NEP)
                outdict[ws][band]['Q']['NEP_fit'].append(Tfit_NEP)
                #print('Saving to npy file.')
                np.save(os.path.join('/global/homes/m/msilvafe/so_home/shared_files/noise_check', savename), outdict)

if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)
