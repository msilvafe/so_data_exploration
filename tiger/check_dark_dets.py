from sotodlib.utils.procs_pool import get_exec_env
from sotodlib import core
import sotodlib.site_pipeline.util as sp_util

import argparse
import traceback
import copy
import os
from typing import Optional, Union, Callable
import numpy as np
from scipy.signal import filtfilt, butter, freqz
import matplotlib.pyplot as plt

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
    y = filtfilt(b, a, data)
    w, h = freqz(b, a, worN=8000, fs=fs)
    passband_gain = np.abs(h[w < cutoff]).mean()
    y /= passband_gain
    return y

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('context_file', help="Preprocessing context File")
    parser.add_argument('save_path', help="Save path")
    parser.add_argument(
        '--query',
        help="Query to pass to the observation list. Use \\'string\\' to "
             "pass in strings within the query.",
        type=str
    )
    parser.add_argument(
        '--min-ctime',
        help="Minimum timestamp for the beginning of an observation list",
    )
    parser.add_argument(
        '--max-ctime',
        help="Maximum timestamp for the beginning of an observation list",
    )
    parser.add_argument(
        '--nproc',
        help="Number of parallel processes to run on.",
        type=int,
        default=4
    )
    return parser

def check_dark_dets(context_file: str, obsid: str, save_path: str):
    """
    """
    try:
        ctx = core.Context(context_file)
        obs = ctx.get_obs(obsid, dets = {'wafer.type':'DARK'})
        wafer_slots = np.unique(obs.det_info.wafer_slot)
        output_dict = {'obs_id': obsid, 'wafer_slots': {}}
    except Exception as e:
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        print(f"ERROR: loading data\n{errmsg}\n{tb}")
        return True, None, {}
    nfail = 0
    _, axes = plt.subplots(4,4, figsize=(12,12))
    ax = axes.flatten()
    for pi, a in enumerate(ax):
        if pi%2 == 0:
            a.set_ylabel(f'Wafer ws{pi//2}\nSignal [pW]')
        if pi < 14:
            a.set_xlabel('Samples in Middle of TOD')
    plt.suptitle(f'{obsid} Dark Dets Signal')
    for wafer_slot in wafer_slots:
        try:
            wsi = int(wafer_slot[2:])
            mwf = (obs.det_info.wafer_slot == wafer_slot)
            m = (np.ptp(obs.signal[mwf], axis=1) < 5) & \
                (obs.det_cal.r_frac[mwf] > 0.1) & \
                (obs.det_cal.r_frac[mwf] < 0.8)
            if len(obs.dets.vals[mwf][m]) == 0:
                output_dict['wafer_slots'][wafer_slot] = {'ndets':0, 'data': []}
                continue
            data = copy.deepcopy(obs.signal[mwf][m])
            data = np.subtract(data.T, np.mean(data, axis=-1)).T
            data = np.multiply(data.T, obs.det_cal.phase_to_pW[mwf][m]).T
            pad_samps = 30*200
            pad_width = ((0, 0), (pad_samps, pad_samps))  # (dets axis, samps axis)
            padded_data = np.pad(data, pad_width, mode='reflect')
            # Apply filter, then trim
            filtered = butter_lowpass_filter(padded_data, 1, 200)
            data = filtered[:, pad_samps:-pad_samps]
            output_dict['wafer_slots'][wafer_slot] = {'ndets': len(data), 
                                                      'data': np.median(data, axis=0)[::200]}
            ax[2*wsi].plot(np.median(data, axis=0)[::200], label=f'Ndets: {len(data)}')
            six = np.shape(data)[1]//2
            eix = six + 60000
            ax[2*wsi+1].plot(data[:,six:eix:200].T, alpha=0.5)
            ax[2*wsi+1].plot(np.median(data, axis=0)[six:eix:200], 'k--')
            ax[2*wsi].legend()
            ax2 = ax[2*wsi+1].twinx()
            ax2.plot(np.rad2deg(obs.boresight.az[six:eix:200]),'C1--', alpha=0.8)
            ax2.set_ylabel('Azimuth', color='C1')
        except Exception as e:
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            print(f"ERROR: processing wafer {wafer_slot}\n{errmsg}\n{tb}")
            nfail +=1
            if nfail == len(wafer_slots):
                return True, None, {}
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'plots', f'{obsid}_dark_dets_common_mode.png'))
    # outpath = os.path.join(save_path, 'temp', f'{obsid}_dark_dets_data')
    # np.save(outpath, output_dict)
    outpath = ''
    return False, outpath, output_dict
    
def _main(executor: Union["MPICommExecutor", "ProcessPoolExecutor"],
          as_completed_callable: Callable,
          context_file: str,
          save_path: str,
          query: Optional[str] = None,
          min_ctime: Optional[int] = None,
          max_ctime: Optional[int] = None,
          nproc: Optional[int] = 4):
    ctx = core.Context(context_file)
    obs_list = sp_util.get_obslist(ctx, query=query, min_ctime=min_ctime,
                                   max_ctime=max_ctime)
    futures = [executor.submit(check_dark_dets, obsid=r['obs_id'],
                               context_file=context_file,
                               save_path=save_path) for r in obs_list]
    nobs = len(obs_list)
    obnum = 0
    out_dict = {}
    for future in as_completed_callable(futures):
        obnum += 1
        print(f'{obnum}/{nobs}: New future as_completed result')
        try:
            err, outpath, outdata = future.result()
        except Exception as e:
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            print(f"ERROR: future.result()\n{errmsg}\n{tb}")
            continue
        futures.remove(future)
        if ~err:
            out_dict[outdata['obs_id']] = outdata['wafer_slots']
    np.save(os.path.join(save_path, f'dark_dets_data'), out_dict)
    
def main(context_file: str,
         save_path: str,
         query: Optional[str] = None,
         min_ctime: Optional[int] = None,
         max_ctime: Optional[int] = None,
         nproc: Optional[int] = 4):
    rank, executor, as_completed_callable = get_exec_env(nproc)
    if rank == 0:
        _main(executor=executor,
              as_completed_callable=as_completed_callable,
              context_file=context_file,
              save_path=save_path,
              query=query,
              min_ctime=min_ctime,
              max_ctime=max_ctime,
              nproc=nproc)

if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)


