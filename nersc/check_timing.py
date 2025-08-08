import numpy as np
import os
import argparse
import traceback
import time
import copy

from sotodlib import core
from sotodlib.preprocess import preprocess_util
from sotodlib.site_pipeline import util as sp_util


import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--context', help="path to context file")
    parser.add_argument('--query', help="query for context.get_obs")
    parser.add_argument('--ufm-list', nargs='+',
                        help="provide a list of ufm's instead of getting from first obs")
    
    parser.add_argument(
        '--nproc',
        help="Number of parallel processes to run on.",
        type=int,
        default=16
    )

    parser.add_argument(
        '--errlog-ext',
        help="Error log file name.",
        default='timing_check_err.txt'
    )

    parser.add_argument(
        '--savename',
        help="Output dictionary save name.",
        default='timing_check.npy'
    )
    return parser

def split_ts_bits(c):
    """Split up 64 bit to 2x32 bit"""
    NUM_BITS_PER_INT = 32
    MAXINT = (1 << NUM_BITS_PER_INT) - 1
    a = (c >> NUM_BITS_PER_INT) & MAXINT
    b = c & MAXINT
    return a, b

def get_timing_info(obs, context):
    try:
        logger = preprocess_util.init_logger('subproc_logger')
        logger.info(f'Processing entry for {obs}')
        ctx = core.Context(context)
        aman = ctx.get_obs(obs, no_signal=True)
        outdict = {}
        for fld in aman.primary._fields:
            a, b = split_ts_bits(aman.primary[fld].Counter2)
            c0_pls = aman.primary[fld].Counter0
            c0diff = np.diff(c0_pls)
            idxs = np.where(c0diff < -4e5)[0]
            c2_ts = a + b/1e9
            c2_ts = c2_ts[:-1]
            c2_secs = c2_ts[idxs][1:]-c2_ts[idxs][:-1]

            outdict[fld] = {}
            outdict[fld]['ctime'] = float(aman.timestamps[0])
            outdict[fld]['obsid'] = str(obs)
            # Counter 0
            outdict[fld]['c0mean'] = float(np.mean(c0diff[idxs]))
            outdict[fld]['c0median'] = float(np.median(c0diff[idxs]))
            outdict[fld]['c0min'] = float(np.min(c0diff[idxs]))
            outdict[fld]['c0max'] = float(np.max(c0diff[idxs]))
            outdict[fld]['c0std'] = float(np.std(c0diff[idxs]))

            # Counter 2 timestamps
            outdict[fld]['c2mean'] = float(np.mean(c2_secs))
            outdict[fld]['c2median'] = float(np.median(c2_secs))
            outdict[fld]['c2min'] = float(np.min(c2_secs))
            outdict[fld]['c2max'] = float(np.max(c2_secs))
            outdict[fld]['c2std'] = float(np.std(c2_secs))
        return None, outdict
    except Exception as e:
        logger.info(f"Error in process")
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        return errmsg, tb

def main(context, query, ufm_list, nproc, errlog_ext, savename):
    logger = preprocess_util.init_logger('main_proc')
    #context = '/global/cfs/cdirs/sobs/metadata/satp1/contexts/use_this_local.yaml'
    ctx = core.Context(context)
    obslist = ctx.obsdb.query(query)
    if ufm_list is None:
        aman = ctx.get_obs(obslist[0], no_signal=True)
        ufm_list = [fld for fld in aman.primary._fields]
    logger.info(f'UFM list: {ufm_list}')
    empty_dict = {'obsids':[],'ctimes':[], 'c0mean':[], 'c0median':[], 'c0min':[], 'c0max':[], 'c0std':[],
              'c2mean':[], 'c2median':[], 'c2min':[], 'c2max':[], 'c2std':[]}
    outdict = {fld: copy.deepcopy(empty_dict) for fld in ufm_list}

    errlog = os.path.join('/global/homes/m/msilvafe/so_home/shared_files/timing_check', errlog_ext)
    
    logger.info('launch multiprocessing')
    multiprocessing.set_start_method('spawn')
    logger.info('multiprocess pool spawned')

    n = 0
    ntot = len(obslist)

    with ProcessPoolExecutor(nproc) as exe:
        futures = [exe.submit(get_timing_info, obs=obs['obs_id'], context=context) for obs in obslist]
        for future in as_completed(futures):
            try:
                errmsg, data_dict = future.result()
                logger.info(f'{n}/{ntot}: Unpacked future.')
            except Exception as e:
                logger.info(f'{n}/{ntot}: Future unpack error.')
                errmsg = f'{type(e)}: {e}'
                tb = ''.join(traceback.format_tb(e.__traceback__))
                f = open(errlog, 'a')
                f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
                f.close()
                continue
            futures.remove(future)
            
            if errmsg:
                logger.info('Writing error to log.')
                f = open(errlog, 'a')
                f.write(f'\n{time.time()}, error\n{errmsg}\n{data_dict}\n')
                f.close()
            else:
                try:
                    for fld in data_dict.keys():
                        outdict[fld]['ctimes'].append(data_dict[fld]['ctime'])
                        outdict[fld]['obsids'].append(data_dict[fld]['obsid'])
                        outdict[fld]['c0mean'].append(data_dict[fld]['c0mean'])
                        outdict[fld]['c0median'].append(data_dict[fld]['c0median'])
                        outdict[fld]['c0min'].append(data_dict[fld]['c0min'])
                        outdict[fld]['c0max'].append(data_dict[fld]['c0max'])
                        outdict[fld]['c0std'].append(data_dict[fld]['c0std'])
                        outdict[fld]['c2mean'].append(data_dict[fld]['c2mean'])
                        outdict[fld]['c2median'].append(data_dict[fld]['c2median'])
                        outdict[fld]['c2min'].append(data_dict[fld]['c2min'])
                        outdict[fld]['c2max'].append(data_dict[fld]['c2max'])
                        outdict[fld]['c2std'].append(data_dict[fld]['c2std'])
                except Exception as e:
                    logger.info('Packaging and saving error.')
                    errmsg = f'{type(e)}: {e}'
                    tb = ''.join(traceback.format_tb(e.__traceback__))
                    f = open(errlog, 'a')
                    f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{data_dict}\n')
                    f.close()
                    continue
            n+=1
    logger.info(f'Saving to npy file.')
    np.save(os.path.join('/global/homes/m/msilvafe/so_home/shared_files/timing_check', savename), outdict)

if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)
