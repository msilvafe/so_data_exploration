import yaml
import argparse
import os
import traceback
import numpy as np

from sotodlib import core
import sotodlib.site_pipeline.util as sp_util
from sotodlib.preprocess import preprocess_util as pp_util
from sotodlib.core.flagman import has_all_cut

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_dict_entry(ppcfg, oid, ws, band):
    try:
        _,ctx = pp_util.get_preprocess_context(ppcfg)
        meta = ctx.get_meta(oid, dets={'wafer_slot':ws, 'wafer.bandpass': band})  
        m = has_all_cut(meta.preprocess.noiseQ_fit.valid)
        fknees = meta.preprocess.noiseQ_fit.fit[m,0]
        return band, ws, fknees
    except Exception as e:
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        return None, tb, errmsg

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('ppcfg', help="Preprocessing arxiv filepath")
    
    parser.add_argument(
        '--nproc',
        help="Number of parallel processes to run on.",
        type=int,
        default=4
    )

    parser.add_argument(
        '--errlog-ext',
        help="Error log file name.",
        default='check_fknee_err.txt'
    )

    parser.add_argument(
        '--savename',
        help="Output dictionary save name.",
        default='fknee_data.npy'
    )
    return parser

def main(ppcfg, nproc, errlog_ext, savename):
    configs,ctx = pp_util.get_preprocess_context(ppcfg)
    errlog = os.path.join('/global/homes/m/msilvafe/so_home/shared_files/fknee_check', errlog_ext)

    # Load in NETs and PWVs
    nets = np.load('/global/homes/m/msilvafe/so_home/shared_files/net_check/satp1_net_check.npy',
                   allow_pickle=True).item()
    pwv = np.load('/global/homes/m/msilvafe/so_home/shared_files/pwv_20240601_to_20240731.npy',
                   allow_pickle=True).item()

    netcmb = {'f090':{'ws0': [], 'ws1': [], 'ws2': [], 'ws3': [], 'ws4': [], 'ws5': [], 'ws6': []},
            'f150':{'ws0': [], 'ws1': [], 'ws2': [], 'ws3': [], 'ws4': [], 'ws5': [], 'ws6': []}}
    ndets = {'f090':{'ws0': [], 'ws1': [], 'ws2': [], 'ws3': [], 'ws4': [], 'ws5': [], 'ws6': []},
            'f150':{'ws0': [], 'ws1': [], 'ws2': [], 'ws3': [], 'ws4': [], 'ws5': [], 'ws6': []}}
    pwvscmb = {'f090':{'ws0': [], 'ws1': [], 'ws2': [], 'ws3': [], 'ws4': [], 'ws5': [], 'ws6': []},
            'f150':{'ws0': [], 'ws1': [], 'ws2': [], 'ws3': [], 'ws4': [], 'ws5': [], 'ws6': []}}
    oidsanal = {'f090':{'ws0': [], 'ws1': [], 'ws2': [], 'ws3': [], 'ws4': [], 'ws5': [], 'ws6': []},
                'f150':{'ws0': [], 'ws1': [], 'ws2': [], 'ws3': [], 'ws4': [], 'ws5': [], 'ws6': []}}
    all_oids = []
    for ws in nets.keys():
        for band in nets[ws].keys():
            for oid in nets[ws][band]['oid']:
                if oid in all_oids:
                    continue
                else:
                    all_oids.append(oid)

    for oid in all_oids:
        for ws in nets.keys():
            for band in nets[ws].keys():
                idx = np.where(np.asarray(nets[ws][band]['oid']) == oid)[0]
                if len(idx) == 0:
                    continue
                if round(np.min(np.abs(int(nets[ws][band]['oid'][idx[0]].split('_')[1])-np.asarray(pwv['timestamps']))),1) > 60*5:
                    continue
                noise = np.asarray(nets[ws][band]['noise'][idx[0]]['fit_noiseQ'])
                m = (noise != 0) & (np.isfinite(noise))
                netcmb[band][ws].append(noise[m])
                ndets[band][ws].append(len(np.asarray(noise[m])))
                el = np.deg2rad(nets[ws][band]['noise'][idx[0]]['el'])
                pwvscmb[band][ws].append(np.interp(int(oid.split('_')[1]), pwv['timestamps'], (0.84*pwv['pwv'])+0.03)/np.sin(el))
                oidsanal[band][ws].append(oid)
    
    ws = 'ws4'
    band = 'f090'
    m = (np.asarray(pwvscmb[band][ws]) > 0.9) & (np.asarray(pwvscmb[band][ws]) < 1.1) 
    oids_ws4 = np.asarray(oidsanal[band][ws])[m]

    arxiv = configs['archive']['index']
    base_dir = os.path.dirname(arxiv)
    proc = core.metadata.ManifestDb(os.path.join(base_dir, 'process_archive.sqlite'))
    fknees =  {'f090':{'ws0': [], 'ws1': [], 'ws2': [], 'ws3': [], 'ws4': [], 'ws5': [], 'ws6': []},
               'f150':{'ws0': [], 'ws1': [], 'ws2': [], 'ws3': [], 'ws4': [], 'ws5': [], 'ws6': []}}
    
    runlist = []
    for oid in oids_ws4:
        for wsi in range(7):
            if wsi == 5:
                continue
            ws = f'ws{wsi}'
            for band in ['f090', 'f150']:
                entries = proc.inspect({'obs:obs_id':oid, 'dets:wafer_slot': ws, 'dets:wafer.bandpass': band})
                if len(entries) == 0:
                    continue
                runlist.append((ppcfg, oid, ws, band))

    multiprocessing.set_start_method('spawn')
    with ProcessPoolExecutor(nproc) as exe:
        futures = [exe.submit(get_dict_entry, ppcfg=rn[0], oid=rn[1], ws=rn[2], band=rn[3]) for rn in runlist]
        for future in as_completed(futures):
            try:
                band, ws, fknees = future.result()
            except Exception as e:
                print('Writing to error log')
                errmsg = f'{type(e)}: {e}'
                tb = ''.join(traceback.format_tb(e.__traceback__))
                f = open(errlog, 'a')
                f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
                f.close()
                continue
            futures.remove(future)
            
            if band is None:
                f = open(errlog, 'a')
                f.write(f'\n{time.time()}, error\n{ws}\n{fknees}\n')
                f.close()
            else:
                print(f'Saving {ws}, {band}, run {i}/{nruns}')
                fknees[band][ws].append(fknees)
            i += 1
    np.save(os.path.join('/global/homes/m/msilvafe/so_home/shared_files/fknee_check', savename), fknees)

if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)
