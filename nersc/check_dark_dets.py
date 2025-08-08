from sotodlib.utils.procs_pool import get_exec_env
from sotodlib import core
import sotodlib.site_pipeline.util as sp_util

import argparse
import traceback
from typing import Optional, Union, Callable
import numpy as np
from scipy.signal import firwin2, filtfilt, butter, freqz
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
    parser.add_argument('context_file', help="Preprocessin context File")
    parser.add_argument(
        '--query',
        help="Query to pass to the observation list. Use \\'string\\' to "
             "pass in strings within the query.",
        type=str
    )
    parser.add_argument(
        '--obs-id',
        help="obs-id of particular observation if we want to run on just one"
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
        '--verbosity',
        help="increase output verbosity. 0:Error, 1:Warning, 2:Info(default), 3:Debug",
        default=2,
        type=int
    )
    parser.add_argument(
        '--nproc',
        help="Number of parallel processes to run on.",
        type=int,
        default=4
    )
    return parser

def check_dark_dets(context_file: str, obsid: str):
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
    nfail = 0
    fig, axes = plt.subplots(1,2, figsize=(6,3))
    ax = axes.flatten()
    for a in ax:
        a.set_xlabel('Samples')
        a.set_ylabel('Signal [pW]')
    plt.suptitle(f'{obsid} Dark Dets Signal')
    for wafer_slot in wafer_slots:
        try:
            mwf = (obs.det_info.wafer_slot == wafer_slot)
            m = (np.ptp(obs.signal[mwf], axis=1) < 5) & \
                (obs.det_cal.r_frac[mwf] > 0.1) & \
                (obs.det_cal.r_frac[mwf] < 0.8)
            if len(aman.dets.vals[mwf][m]) == 0:
                output_dict['obs_id']['wafer_slots'][wafer_slot] = {'ndets':0, 'data': []}
                continue
            data = copy.deepcopy(aman.signal[mwf][m])
            data = np.subtract(data.T, np.mean(data, axis=-1)).T
            data = np.multiply(data.T, aman.det_cal.phase_to_pW[mwf][m]).T
            pad_samps = 30*200
            pad_width = ((0, 0), (pad_samps, pad_samps))  # (dets axis, samps axis)
            padded_data = np.pad(data, pad_width, mode='reflect')
            # Apply filter, then trim
            filtered = butter_lowpass_filter(padded_data, 1, 200)
            data = filtered[:, pad_samps:-pad_samps]
            output_dict['obs_id']['wafer_slots'][wafer_slot]['ndets'] = len(data)
            output_dict['obs_id']['wafer_slots'][wafer_slot]['data'] = np.median(data, axis=0)
        except Exception as e:
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            print(f"ERROR: processing wafer {wafer_slot}\n{errmsg}\n{tb}")
            nfail +=1
            if nfail == len(wafer_slots):
                return True, {}

        return False, output_dict

def _main(executor: Union["MPICommExecutor", "ProcessPoolExecutor"],
          as_completed_callable: Callable,
          context_file: str,
          query: Optional[str] = None,
          min_ctime: Optional[int] = None,
          max_ctime: Optional[int] = None,
          verbosity: Optional[int] = None,
          nproc: Optional[int] = 4):
    logger = sp_util.init_logger("preprocess", verbosity=verbosity)
    obs_list = sp_util.get_obslist(context, query=query, obs_id=obs_id, min_ctime=min_ctime, max_ctime=max_ctime)
    futures = [executor.submit(check_dark_dets, obs_id=r['obs_id'],
         max_ctime: Optional[int] = None,
         verbosity: Optional[int] = None,
         nproc: Optional[int] = 4):

    rank, executor, as_completed_callable = get_exec_env(nproc)
    if rank == 0:
        _main(executor=executor,
              as_completed_callable=as_completed_callable,
              context_file=context_file,
              query=query,
              min_ctime=min_ctime,
              max_ctime=max_ctime,
              verbosity=verbosity,
              nproc=nproc)

if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)
from sotodlib.utils.procs_pool import get_exec_env
from sotodlib import core

import argparse
from typing import Optional, Union, Callable
import numpy as np
from scipy.signal import firwin2, filtfilt, butter, freqz
import matplotlib.pyplot as plt

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
    y = filtfilt(b, a, data)
    w, h = freqz(b, a, worN=8000, fs=fs)
    passband_gain = np.abs(h[w < cutoff]).mean()
    y /= passband_gain
    return y

