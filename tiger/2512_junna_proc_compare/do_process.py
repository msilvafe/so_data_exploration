import os
import sys
sys.path.insert(0, '/scratch/gpfs/SIMONSOBS/users/js7893/common_scripts')
from gallery.gallery import make_gallery
from time import time
from tqdm import tqdm
import argparse
import numpy as np
import scipy
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from lmfit import Model
import matplotlib.pyplot as plt
import sqlite3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pickle
from sotodlib.io.load_smurf import Observations, Files, TuneSets, Tunes
from sotodlib import coords, core, tod_ops, hwp
from sotodlib.core import metadata
from sotodlib.io.metadata import write_dataset
from sotodlib.core.metadata import ResultSet
from sotodlib.hwp.g3thwp import G3tHWP
from sotodlib import hwp
#import demod as demod_mm
from sotodlib.hwp.hwp_angle_model import apply_hwp_angle_model
from so3g.proj import quat
from pixell import enmap
import h5py
from sotodlib import tod_ops
from sotodlib.tod_ops import pca
import filters
from sotodlib.tod_ops.fft_ops import calc_psd, calc_wn
from sotodlib.tod_ops import sub_polyf
from so3g.proj import Ranges, RangesMatrix
from sotodlib.io.metadata import write_dataset, read_dataset
from pixell import enmap, enplot
import cmath
import scipy.stats as stats
from sotodlib.tod_ops.flags import get_ptp_flags
from sotodlib.tod_ops.flags import get_turnaround_flags
from sotodlib.tod_ops import t2pleakage as t2p
from sotodlib.tod_ops import azss, apodize
from sotodlib.tod_ops.sub_polyf import subscan_polyfilter
import so3g
import warnings
import glob
from scipy.fft import rfft, irfft, rfftfreq
from scipy.signal import butter, filtfilt, welch
from scipy.optimize import minimize
from scipy.odr import ODR, Model, RealData

# PLOT PATH
plot_path = "/home/ms3067/shared_files/preprocess/2512_compare_junna_plots/"
plot_ws = "ws1"

warnings.filterwarnings('ignore')
ds_factor = 50 # 4Hz

fft_filter = filters.FilterFunc.deco

def circular_mean(angles, deg=False):
    """
    Circular mean of angle data
    """
    a = np.deg2rad(angles) if deg else np.array(angles)
    angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
    mean = cmath.phase(angles_complex.sum()) # -pi < mean < pi
    return np.rad2deg(mean) if deg else mean

def circular_std(angles, deg=False):
    """
    Circular standard deviation of angle data
    """
    a = np.deg2rad(angles) if deg else np.array(angles)
    angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
    r = abs(angles_complex.sum()) / len(angles)
    std = np.sqrt(-2 * np.log(r))
    return np.rad2deg(std) if deg else std

def wrap_relcal(aman, bandpass):
    relcal_hdf_dir = '/scratch/gpfs/SIMONSOBS/users/js7893/common_scripts/wg_relcal/wg_amps_combined'
    # relcal_file = get_relcal_file(aman, bandpass, relcal_hdf_dir)
    relcal_file = os.path.join(relcal_hdf_dir, f'relcal_{bandpass}.hdf')
    relcal_aman = core.AxisManager.load(relcal_file)
    aman.wrap_new('rel_factor', ('dets',))
    aman.rel_factor *= np.nan
    for di, det in enumerate(aman.dets.vals):
        di_relcal = np.where(relcal_aman.dets.vals == det)[0]
        if len(di_relcal) == 1:
            if relcal_aman.rel_factor[di_relcal] > 0.1:
                aman.rel_factor[di] = relcal_aman.rel_factor[di_relcal]
    return

def calc_wrap_psd(aman, signal_name, nperseg=200*1000, 
                  nusamps_suffix=None,
                  merge=False, merge_wn=False, merge_suffix=None, low_f=2.0, high_f=5.0):
    freqs, Pxx = calc_psd(aman, signal=aman[signal_name], nperseg=nperseg, merge=False)
    if merge:
        assert merge_suffix is not None
        if nusamps_suffix is None:
            nusamps_name = 'nusamps'
            freqs_name = 'freqs'
        else:
            nusamps_name = 'nusamps_' + nusamps_suffix
            freqs_name = 'freqs_' + nusamps_suffix
        if nusamps_name not in list(aman._axes.keys()):
            nusamps = core.OffsetAxis(nusamps_name, len(freqs))
            aman.wrap(freqs_name, freqs, [(0, nusamps, )])
        aman.wrap(f'Pxx_{merge_suffix}', Pxx, [(0, 'dets'), (1, nusamps_name)])
    if merge_wn:
        wn = calc_wn(aman, pxx=Pxx, freqs=freqs, low_f=low_f, high_f=high_f)
        _ = aman.wrap(f'wn_{merge_suffix}', wn, [(0, 'dets')])
    return  freqs, Pxx

def wrap_pwv_data(aman, archive_path='/scratch/gpfs/SIMONSOBS/so/tracked/data/site/hk'):
    keys=['site.env-radiometer-class.feeds.pwvs.pwv']
    alias=[key.split('.')[-1] for key in keys]
    start = int(aman.timestamps[0])
    end = int(aman.timestamps[-1])
    data = so3g.hk.load_range(start, end, fields=keys, alias=alias, data_dir=archive_path)
    if len(data)==0:
        aman.wrap('pwv', np.nan)
        aman.wrap('dpwv', np.nan)
        return
    pwv = data['pwv'][1]
    pwv = pwv[(0.3<pwv)&(pwv<3)]
    if len(pwv)==0:
        aman.wrap('pwv', np.nan)
        aman.wrap('dpwv', np.nan)
    else:
        aman.wrap('pwv', np.nanmedian(pwv))
        aman.wrap('dpwv', np.nanmax(pwv) - np.nanmin(pwv))
    return

def data_cuts_and_report_first(aman, bandpass):
    num_dets_perfect = 910 * 7
    num_dets_streamed = aman.dets.count
    
    # survive biasstep 
    # stricter p_sat threshold (from 0.1pW to 2.0 pW)
    flag_valid_biasstep = (aman.det_cal.r_tes > 0) & \
                        (0.05 < aman.det_cal.r_frac) & (aman.det_cal.r_frac < 0.95) & \
                        (2.0 < aman.det_cal.p_sat*1e12) & (aman.det_cal.p_sat*1e12 < 20) & \
                        (aman.det_cal.s_i < 0)
    num_dets_valid_biasstep = np.count_nonzero(flag_valid_biasstep)

    # avaiable wiregrid
    if "rel_factor" in aman._fields:
        flag_wg_available = np.logical_and(flag_valid_biasstep, ~np.isnan(aman.rel_factor))
        num_dets_wg_available = np.count_nonzero(flag_wg_available)
    else:
        flag_wg_available = num_dets_valid_biasstep
        num_dets_wg_available = np.count_nonzero(flag_wg_available)

    # available focalplane
    flag_valid_fp = np.sum(np.isnan([aman.focal_plane.xi, aman.focal_plane.eta, aman.focal_plane.gamma]).astype(int), axis=0) == 0
    flag_valid_fp = np.logical_and(flag_wg_available, flag_valid_fp)
    num_dets_valid_fp = np.count_nonzero(flag_valid_fp)

    # white noise level
    flag_valid_wnl = (20<aman.wn_signal*1e6) & (aman.wn_signal*1e6<200)
    flag_valid_wnl = np.logical_and(flag_valid_fp, flag_valid_wnl)
    num_dets_valid_wnl = np.count_nonzero(flag_valid_wnl)
    
    # restrict to only accepted wafer
    if bandpass == 'f090':
        accepted_wafers = ['ws0', 'ws1', 'ws2', 'ws3', 'ws4', 'ws5', 'ws6']
    elif bandpass == 'f150':
        accepted_wafers = ['ws0', 'ws1', 'ws2', 'ws3', 'ws4', 'ws5', 'ws6']
    flag_valid_wafers = np.in1d(aman.det_info.wafer_slot, accepted_wafers)
    flag_valid_wafers = np.logical_and(flag_valid_wnl, flag_valid_wafers)
    num_dets_valid_wafers = np.count_nonzero(flag_valid_wafers)
    
    num_dets_dict = {
                    'num_dets_perfect':num_dets_perfect,
                    'num_dets_streamed':num_dets_streamed,
                    'num_dets_valid_biasstep': num_dets_valid_biasstep,
                    'num_dets_wg_available': num_dets_wg_available,
                    'num_dets_valid_fp': num_dets_valid_fp,
                    'num_dets_valid_wnl': num_dets_valid_wnl,
                    'num_dets_valid_wafers': num_dets_valid_wafers,
                    }
    
    flag_valid_total = flag_valid_wafers
    return flag_valid_total, num_dets_dict

def load_data(ctx_file, obs_id, bandpass, debug=False, debug_style='one_ws', 
              do_calibration=True, 
              do_relcal=True,
              do_get_wn_signal=True,
              do_first_data_cut=True,
              remove_useless=True,
              ws=None
             ):
    ctx = core.Context(ctx_file)
    print(f'loading meta: {obs_id}, {bandpass}')
    if ws is not None:
        meta = ctx.get_meta(obs_id, dets={'wafer.bandpass': bandpass, 'wafer_slot': ws})
    else:
        meta = ctx.get_meta(obs_id, dets={'wafer.bandpass': bandpass})
    if debug:
        assert debug_style in ['one_ws', 'sparse']
        if debug_style == 'one_ws':
            meta.restrict('dets', meta.dets.vals[:debug])
        elif debug_style == 'sparse':
            meta.restrict('dets', meta.dets.vals[::int(meta.dets.count/debug)])
    
    print(f'loading obs: {obs_id}, {bandpass}')
    aman = ctx.get_obs(meta)
    tod_ops.detrend_tod(aman, method='median')
    
    print('apply biasstep cal')
    if do_calibration:
        if 'det_cal' in aman._fields.keys():
            aman.signal *= aman.det_cal.phase_to_pW[:, np.newaxis]
        else:
            raise ValueError

    print('apply hwp model')
    apply_hwp_angle_model(aman)
    fhwp = (np.sum(np.diff(np.unwrap(aman.hwp_angle)))) / (aman.timestamps[-1] - aman.timestamps[0]) / (2 * np.pi)
    aman.wrap('fhwp', fhwp)
    print('apply pointing model')
    coords.pointing_model.apply_pointing_model(aman)
    
    # wrap misc params
    aman.wrap('ctime', int(aman.timestamps[0]))
    scanspeed = 180/np.pi * np.median(np.abs(np.diff(aman.boresight.az))/np.diff(aman.timestamps))
    aman.wrap('scanspeed', scanspeed)
    aman.wrap('az_min', np.rad2deg(np.min(aman.boresight.az[180:-180])))
    aman.wrap('az_max', np.rad2deg(np.max(aman.boresight.az[180:-180])))
    wrap_pwv_data(aman)
    wrap_relcal(aman, bandpass)
    
    if do_relcal:
        print('apply relative gain cal')
        aman.signal /= aman.rel_factor[:, np.newaxis]
        # if aman.hwp_solution.pid_direction>0: #ccw
        #     if bandpass == 'f150':
        #         offset = -2.81*np.pi/180/2
        #     else:
        #         offset = 0.76*np.pi/180/2
        # elif aman.hwp_solution.pid_direction<0: #cw
        #     if bandpass == 'f150':
        #         offset = -1.62*np.pi/180/2
        #     else:
        #         offset = 2.13*np.pi/180/2
        # aman.hwp_angle = aman.hwp_angle + offset
    if do_get_wn_signal:
        print('get white noise level')
        freqs_raw, Pxx_raw = calc_wrap_psd(aman, signal_name='signal', 
                                              merge=False, merge_wn=False, 
                                             nperseg=200*10)
        wn_signal = calc_wn(aman, pxx=Pxx_raw, freqs=freqs_raw, low_f=5, high_f=20)
        _ = aman.wrap(f'wn_signal', wn_signal, [(0, 'dets')])
        

    if remove_useless:
        fields = ['ancil', 'primary', 'biases', 'hwp_solution',]
        for field in fields:
            if field in aman:
                aman.move(field, None)
    if do_first_data_cut:
        print('apply first data cuts')
        flag_valid_first, num_dets_dict =  data_cuts_and_report_first(aman, bandpass)
        aman.restrict('dets', aman.dets.vals[flag_valid_first])
        
    print('correcting iir')
    tod_ops.apodize_cosine(aman, apodize_samps=200*5) #5sec
    iir_filt = tod_ops.filters.iir_filter(invert=True)
    aman.signal = tod_ops.fourier_filter(aman, iir_filt)
    
    if do_first_data_cut:
        return aman, num_dets_dict
    else:
        return aman
    
def flag_filt_buffer(aman):
    filt_buffer_flags = np.zeros(aman.samps.count, dtype=bool)
    filt_buffer_flags[:200*30] = True
    filt_buffer_flags[-200*30:] = True
    filt_buffer_flags = Ranges.from_bitmask(filt_buffer_flags)
    filt_buffer_flags = RangesMatrix([filt_buffer_flags for di in range(aman.dets.count)])
    aman.flags.wrap('filt_buffer', filt_buffer_flags, [(0, 'dets'), (1, 'samps')])
    return

def get_good_distribution_flags(aman, param_name='wn_signal', 
                                outlier_range=(0.5, 2.), kurtosis_threshold=2., 
                               blame_max=False, blame_min=False):
    det_mask = np.full(aman.dets.count, True, dtype=bool)
    ratio = aman[param_name]/np.median(aman[param_name])
    outlier_mask = (ratio<outlier_range[0]) | (outlier_range[1]<ratio)

    det_mask[outlier_mask] = False
    while True:
        if len(aman.dets.vals[det_mask]) > 0:
            distributions = aman[param_name][det_mask]
        else:
            break
        kurtosis_wn = stats.kurtosis(distributions)
        if np.abs(kurtosis_wn) < kurtosis_threshold:
            break
        else:
            assert (blame_max==False) or (blame_min==False)
            if blame_max:
                det_mask[aman[param_name] >= np.max(distributions)] = False
            elif blame_min:
                det_mask[aman[param_name] <= np.min(distributions)] = False
            else:
                max_is_bad_factor = np.max(distributions)/np.median(distributions)
                min_is_bad_factor = np.median(distributions)/np.min(distributions)
                if max_is_bad_factor > min_is_bad_factor:
                    det_mask[aman[param_name] >= np.max(distributions)] = False
                else:
                    det_mask[aman[param_name] <= np.min(distributions)] = False
    return Ranges.from_bitmask(det_mask)

def my_get_ptp_flags(aman, signal_name='signal', kurtosis_threshold=5, mask=None,
                  merge=False, overwrite=False, ptp_flag_name='ptp_flag',
                  outlier_range=(0.5, 2.)):
    det_mask = np.full(aman.dets.count, True, dtype=bool)
    if mask is None:
        ptps_full = np.ptp(aman[signal_name], axis=1)
    else:
        ptps_full = np.ma.ptp(np.ma.masked_array(aman[signal_name], mask=mask.mask()), axis=1).data
    
    ratio = ptps_full/np.median(ptps_full)
    outlier_mask = (ratio<outlier_range[0]) | (outlier_range[1]<ratio)
    det_mask[outlier_mask] = False
    if np.any(np.logical_not(np.isfinite(aman[signal_name][det_mask]))):
        raise ValueError(f"There is a nan in {signal_name} in aman {aman.obs_info['obs_id']} !!!")
    while True:
        if len(aman.dets.vals[det_mask]) > 0:
            ptps = ptps_full[det_mask]
        else:
            break
        kurtosis_ptp = stats.kurtosis(ptps)
        if np.abs(kurtosis_ptp) < kurtosis_threshold:
            break
        else:
            max_is_bad_factor = np.max(ptps)/np.median(ptps)
            min_is_bad_factor = np.median(ptps)/np.min(ptps)
            if max_is_bad_factor > min_is_bad_factor:
                det_mask[ptps_full >= np.max(ptps)] = False
            else:
                det_mask[ptps_full <= np.min(ptps)] = False
    x = Ranges(aman.samps.count)
    mskptps = RangesMatrix([Ranges.zeros_like(x) if Y
                             else Ranges.ones_like(x) for Y in det_mask])
    if merge:
        if ptp_flag_name in aman.flags and not overwrite:
            raise ValueError(f"Flag name {ptp_flag_name} already exists in aman.flags")
        if ptp_flag_name in aman.flags:
            aman.flags[ptp_flag_name] = mskptps
        else:
            aman.flags.wrap(ptp_flag_name, mskptps, [(0, 'dets'), (1, 'samps')])

    return mskptps

def apply_wnl_data_cuts(aman, wn_name='wn_signal'):
    wafer_slots = np.unique(aman.det_info.wafer_slot)
    good_dets = np.array([])
    for ws in wafer_slots[:]:
        mask_ws = aman.det_info.wafer_slot == ws
        aman_ws = aman.restrict('dets', aman.dets.vals[mask_ws], in_place=False)
        flag_valid_wnl = get_good_distribution_flags(aman_ws, param_name=wn_name, 
                                outlier_range=(0.5, 2.), kurtosis_threshold=2., )
        flag_valid_wnl = flag_valid_wnl.mask()
        good_dets = np.append(good_dets, aman_ws.dets.vals[flag_valid_wnl])
        
    aman.restrict('dets', good_dets, )
    return

def apply_ptp_data_cuts(aman, ptp_signal_name='dsT'):
    wafer_slots = np.unique(aman.det_info.wafer_slot)
    good_dets = np.array([])
    for ws in wafer_slots[:]:
        mask_ws = aman.det_info.wafer_slot == ws
        aman_ws = aman.restrict('dets', aman.dets.vals[mask_ws], in_place=False)
        ptp_flags = my_get_ptp_flags(aman_ws, signal_name=ptp_signal_name, mask=aman_ws.flags.filt_buffer, kurtosis_threshold=1, merge=False)
        ptp_flags = ptp_flags.mask()[:, 0]
        good_dets = np.append(good_dets, aman_ws.dets.vals[~ptp_flags])        
    aman.restrict('dets', good_dets, )
    return

def resample_cuts(flag, t0, t1):
    """ Resample flags from one time range to another. Follows the 'any'
    convention. Flags are by convention [s,e) intervals. The flagged time range
    is defined as [t0[s], t0[e]) and flags are transfered to t1 if any samples
    on an interval are flagged. 

    Will raise an Assertion error if any part of t1 is outside the range of t0.

    Args
    ----
    flag: Ranges or 2D RangesMatrix
        flag to resample, resamples along the last axis
    t0: 1D ndarray
        array of timestamps that align with the last axis of the flag
    t1: 1D ndarray
        array of timestamps that align with the new flag

    Returns
    -------
    new_flag: type(flag)
        resampled flag
    """

    assert flag.__class__ in [Ranges, RangesMatrix]
    assert len(flag.shape) <= 2
    
    ## resampling will not extrapolate
    assert t0[0] <= t1[0]
    assert t0[-1] >= t1[-1]
    
    def transfer( rng, new_rng):
        for (s,e) in rng.ranges():
            ns = np.where( t1 <= t0[s] )[0]
            if len(ns) > 0 :
                ns = ns[-1]
            else:
                ns = 0

            if e == rng.shape[0]:
                ne = new_rng.shape[0]
            else:
                ne = np.where( t1 >= t0[e] )[0]
                if len(ne) > 0:
                    ne = ne[0]
                else:
                    ne = new_rng.shape[0]
                    
            if ns == 0 and ne == 0:
                continue
            if ns == new_rng.shape[0] and ne == new_rng.shape[0]:
                continue

            new_rng.add_interval( ns,ne )
    

    if isinstance(flag, Ranges):
        new_flag = flag.__class__( len(t1) )
        transfer( flag, new_flag)

    elif isinstance(flag, RangesMatrix):
        new_flag = flag.__class__.full( (flag.shape[0], len(t1)), False)

        for r, rng in enumerate(flag):
            transfer(rng, new_flag[r])
    return new_flag

def down_sample_aman(aman, ds_factor=40, axis='samps'):
    t0 = np.arange(aman[axis].count)
    t1 = t0[::ds_factor]
    
    new_axes = []
    for k, v in aman._axes.items():
        if k == axis:
            new_axes.append(core.OffsetAxis(axis, len(t1)))
        else:
            new_axes.append(v)
    
    if isinstance(aman, core.FlagManager):
        dest = core.FlagManager(*new_axes)
    else:
        dest = core.AxisManager(*new_axes)

    for k, assign in aman._assignments.items():
        if axis in assign:
            if isinstance(aman[k], core.AxisManager):
                dest.wrap(k, down_sample_aman(aman[k], ds_factor=ds_factor, axis=axis,))
            elif (isinstance(aman[k], RangesMatrix) or 
                    isinstance(aman[k], Ranges)):
                
                dest.wrap(k, resample_cuts(aman[k], t0, t1))
                dest._assignments[k] = aman._assignments[k]
                
            elif isinstance(aman[k], np.ndarray):
                shape = list(aman[k].shape)
                for i, a in enumerate(assign):
                    if a is not None:
                        shape[i] = a
                dest.wrap_new(k, shape=shape, dtype=aman[k].dtype)
                if len(shape) == 1:
                    dest[k][:] = aman[k][::ds_factor]

                elif len(shape) == 2:
                    if (shape[-1] != axis):
                        logger.warning(f'dropping {k}')
                        continue
                    for i, y in enumerate(aman[k]):
                        dest[k][i, :] = aman[k][i, ::ds_factor]

            else:
                raise ValueError('Data type in axis manager not supported '+
                                 'in interpolation')
        else:
            dest.wrap(k, aman[k])
            dest._assignments[k] = aman._assignments[k]
    return dest

def demod_tod_and_wrap_to_ds(aman, bman, ds_factor,
                             signal=None, demod_mode=4, bpf_cfg=None, 
                             lpf_cfg_dsT=None,lpf_cfg_demod=None,):
    if signal is None:
        #signal_name variable to be deleted when tod_ops.fourier_filter is updated
        signal_name = 'signal'
        signal = aman[signal_name]
    elif isinstance(signal, str):
        signal_name = signal
        signal = aman[signal_name]
    elif isinstance(signal, np.ndarray):
        raise TypeError("Currently ndarray not supported, need update to tod_ops.fourier_filter module to remove signal_name argument.")
    else:
        raise TypeError("Signal must be None, str, or ndarray")
    
    # HWP speed in Hz
    speed = (np.sum(np.abs(np.diff(np.unwrap(aman.hwp_angle)))) /
            (aman.timestamps[-1] - aman.timestamps[0])) / (2 * np.pi)
    
    if bpf_cfg is None:
        bpf_center = demod_mode * speed
        #bpf_width = speed * 2. * 0.95
        bpf_width = 2. * 1.9
        bpf_cfg = {'type': 'sine2',
                   'center': bpf_center,
                   'width': bpf_width,
                   'trans_width': 0.1}
    bpf = filters.get_bpf(bpf_cfg)
    
    if lpf_cfg_dsT is None:
        #lpf_cutoff = speed * 0.95
        lpf_cutoff = 1.9
        lpf_cfg_dsT = {'type': 'sine2',
                       'cutoff': lpf_cutoff,
                       'trans_width': 0.1}
    lpf_dsT = filters.get_lpf(lpf_cfg_dsT)
    
    if lpf_cfg_demod is None:
        #lpf_cutoff = speed * 0.95
        lpf_cutoff = 1.9
        lpf_cfg_demod = {'type': 'sine2',
                       'cutoff': lpf_cutoff,
                       'trans_width': 0.1}
    lpf_demod = filters.get_lpf(lpf_cfg_demod)
        
    # dsT
    bman.wrap_new('dsT', dtype='float32', shape=('dets', 'samps'))
    bman['dsT'] = tod_ops.fourier_filter(
        aman, lpf_dsT, signal_name=signal_name, detrend=None)[:, ::ds_factor]
    
    # demod
    phasor = np.exp(demod_mode * 1.j * aman.hwp_angle)
    demod = tod_ops.fourier_filter(aman, bpf, detrend=None,
                                   signal_name=signal_name) * phasor
    
    # demodQ
    bman.wrap_new('demodQ', dtype='float32', shape=('dets', 'samps'))
    aman[signal_name] = demod.real
    bman['demodQ'] = tod_ops.fourier_filter(
        aman, lpf_demod, signal_name=signal_name, detrend=None)[:, ::ds_factor] * 2.
    
    # demodU
    bman.wrap_new('demodU', dtype='float32', shape=('dets', 'samps'))
    aman[signal_name] = demod.imag
    bman['demodU'] = tod_ops.fourier_filter(
        aman, lpf_demod, signal_name=signal_name, detrend=None)[:, ::ds_factor] * 2.
    
    aman.move(signal_name, None)
    return

def subtract_sss_lr(aman, signal_name, bins=50, bad_flags=None, 
                    apodize_edges_sec=100, apodize_flags_sec=1,
                    method='fit', max_mode=10,
                   ):
    apodize_edges_samps = int(apodize_edges_sec * np.round(1/np.median(np.diff(aman.timestamps))))
    apodize_flags_samps = int(apodize_flags_sec * np.round(1/np.median(np.diff(aman.timestamps))))
    
    if bad_flags is not None:
        mask_for_valid_left_scans = aman.flags.right_scan + bad_flags
        mask_for_valid_right_scans = aman.flags.left_scan + bad_flags
    else:
        mask_for_valid_left_scans = aman.flags.right_scan
        mask_for_valid_right_scans = aman.flags.left_scan

    # left scan
    azss_stats_left, azss_model_left = azss.get_azss(aman, signal=signal_name,
                                                     merge_stats=False, merge_model=False,
                                                     flags=mask_for_valid_left_scans, 
                                                     bins=bins, apodize_flags=True, 
                                                     method=method, max_mode=max_mode,
                                                     apodize_flags_samps=apodize_flags_samps,
                                                     apodize_edges_samps=apodize_edges_samps)
    left_flags = aman.flags.left_scan[0].mask()
    aman[signal_name][:, left_flags] -= azss_model_left[:, left_flags]
    
    del azss_model_left
    
    # right scan
    azss_stats_right, azss_model_right = azss.get_azss(aman, signal=signal_name, 
                                                       merge_stats=False, merge_model=False,
                                                       flags=mask_for_valid_right_scans, 
                                                       bins=bins, apodize_flags=True,
                                                       method=method, max_mode=max_mode,
                                                       apodize_flags_samps=apodize_flags_samps,
                                                       apodize_edges_samps=apodize_edges_samps)    
    right_flags = aman.flags.right_scan[0].mask()
    aman[signal_name][:, right_flags] -= azss_model_right[:, right_flags]
    
    del azss_model_right
    
    return azss_stats_left, azss_stats_right

def subtract_sss_lr_all(aman, bins=50, bad_flags=None, 
                        apodize_edges_sec=100, apodize_flags_sec=1,
                        method_dsT='fit', max_mode_dsT=11, 
                        method_demod='fit', max_mode_demod=5, 
                        merge=True):
    azss_stats_left_dsT, azss_stats_right_dsT = subtract_sss_lr(aman=aman, signal_name='dsT', bins=bins, bad_flags=bad_flags, 
                                                               apodize_edges_sec=apodize_edges_sec, apodize_flags_sec=apodize_flags_sec,
                                                               method=method_dsT, max_mode=max_mode_dsT)
    azss_stats_left_demodQ, azss_stats_right_demodQ = subtract_sss_lr(aman=aman, signal_name='demodQ', bins=bins, bad_flags=bad_flags, 
                                                           apodize_edges_sec=apodize_edges_sec, apodize_flags_sec=apodize_flags_sec,
                                                           method=method_demod, max_mode=max_mode_demod)
    azss_stats_left_demodU, azss_stats_right_demodU = subtract_sss_lr(aman=aman, signal_name='demodU', bins=bins, bad_flags=bad_flags, 
                                                               apodize_edges_sec=apodize_edges_sec, apodize_flags_sec=apodize_flags_sec,
                                                               method=method_demod, max_mode=max_mode_demod)
    
    if merge:
        _azss_stats_dict = {
        'left_dsT': azss_stats_left_dsT,
        'right_dsT': azss_stats_right_dsT,
        'left_demodQ': azss_stats_left_demodQ,
        'right_demodQ': azss_stats_right_demodQ,
        'left_demodU': azss_stats_left_demodU,
        'right_demodU': azss_stats_right_demodU,
                       }

        azss_stats = core.AxisManager(azss_stats_left_dsT.dets, azss_stats_left_dsT.bin_az_samps)
        azss_stats.wrap('binned_az', azss_stats_left_dsT['binned_az'], [(0, 'bin_az_samps')])
        for key, val in _azss_stats_dict.items():
            azss_stats.wrap('binned_signal_'+key, val['binned_signal'], [(0, 'dets'), (1, 'bin_az_samps')])
            azss_stats.wrap('binned_signal_sigma_'+key, val['binned_signal_sigma'], [(0, 'dets'), (1, 'bin_az_samps')])
            azss_stats.wrap('uniform_binned_signal_sigma_'+key, val['uniform_binned_signal_sigma'], [(0, 'dets')])
            
            if key[-3:] == 'dsT' and method_dsT=='fit':
                azss_stats.wrap('binned_model_'+key, val['binned_model'], [(0, 'dets'), (1, 'bin_az_samps')])
                azss_stats.wrap('redchi2s_'+key, val['redchi2s'], [(0, 'dets')])
            if key[-6:] in ['demodQ', 'demodU'] and method_demod=='fit':
                azss_stats.wrap('binned_model_'+key, val['binned_model'], [(0, 'dets'), (1, 'bin_az_samps')])
                azss_stats.wrap('redchi2s_'+key, val['redchi2s'], [(0, 'dets')])

        aman.wrap('azss_stats', azss_stats)
    return
    
def model_func(x, sigma, fk, alpha):
    return sigma**2 * (1 + (x/fk)**alpha)

def log_fit_func(x, sigma, fk, alpha):
    return np.log(model_func(x, sigma, fk, alpha))

def get_noise_model(aman, freqs, Pxx, pre_computed_wnl, 
                    merge=False, merge_suffix=None, 
                    f_low_wnl=None,
                    freq_lims = (1e-4, 1.9),
                    fk_alpha_init=(0.01, -2.),
                    fk_bounds = (1e-4, 1.),
                    alpha_bounds = (-5, -0.5),
                   ):
    sigma = np.zeros(aman.dets.count, dtype='float32')
    fk = np.zeros(aman.dets.count, dtype='float32')
    alpha = np.zeros(aman.dets.count, dtype='float32')
    
    mask_valid_freqs = ( freq_lims[0]<freqs ) & (freqs < freq_lims[1])
    x = freqs[mask_valid_freqs]
    mask_for_valid_dets_fit = np.full(aman.dets.count, True)
    for di, det in enumerate(aman.dets.vals):
        y = Pxx[di, mask_valid_freqs]
        
        if pre_computed_wnl is not None:
            _sigma = pre_computed_wnl[di]
        else:
            _sigma = np.sqrt(np.median(y[x>f_low_wnl]))
        _fit_func = lambda x, _fk, _alpha: log_fit_func(x, _sigma, _fk, _alpha)
        try:
            popt, pcov = curve_fit(_fit_func, x, np.log(y), 
                                   p0 = fk_alpha_init,
                                   bounds = ((fk_bounds[0], alpha_bounds[0]), 
                                             (fk_bounds[1], alpha_bounds[1])),
                                   maxfev = 10000)
        except:
            mask_for_valid_dets_fit[di] = False
            popt [None,None]
        sigma[di] = _sigma
        fk[di] = popt[0]
        alpha[di] = popt[1]
    
    if merge:    
        aman.wrap(f'sigma_{merge_suffix}', sigma, [(0, 'dets')])
        aman.wrap(f'fk_{merge_suffix}', fk, [(0, 'dets')])
        aman.wrap(f'alpha_{merge_suffix}', alpha, [(0, 'dets')])
    aman.restrict('dets', aman.dets.vals[mask_for_valid_dets_fit])
    return sigma, fk, alpha

def get_bad_subscan_flags(aman, nstd_threshold=3., kurt_threshold=1., skew_threshold=0.5, 
                         Tptp_subscan_pW_threshold=0.5):
    subscan_Tptps = np.zeros([aman.dets.count, aman.subscan_samps.count])
    subscan_Tstds = np.zeros([aman.dets.count, aman.subscan_samps.count])
    subscan_Qstds = np.zeros([aman.dets.count, aman.subscan_samps.count])
    subscan_Ustds = np.zeros([aman.dets.count, aman.subscan_samps.count])
    
    subscan_Qkurt = np.zeros([aman.dets.count, aman.subscan_samps.count])
    subscan_Ukurt = np.zeros([aman.dets.count, aman.subscan_samps.count])
    subscan_Qskew = np.zeros([aman.dets.count, aman.subscan_samps.count])
    subscan_Uskew = np.zeros([aman.dets.count, aman.subscan_samps.count])

    for subscan_i, (subscan_s, subscan_e) in enumerate(zip(aman.subscan_indices_start, aman.subscan_indices_end)):
        _Tsig= aman.dsT[:,subscan_s:subscan_e]
        _Qsig= aman.demodQ[:,subscan_s:subscan_e]
        _Usig= aman.demodU[:,subscan_s:subscan_e]

        _Tptp = np.ptp(_Tsig, axis=1)
        _Tstd = np.std(_Tsig, axis=1)
        _Qstd = np.std(_Qsig, axis=1)
        _Ustd = np.std(_Usig, axis=1)

        _Qkurt = kurtosis(_Qsig, axis=1)
        _Ukurt = kurtosis(_Usig, axis=1)

        _Qskew = skew(_Qsig, axis=1)
        _Uskew = skew(_Usig, axis=1)

        subscan_Tptps[:, subscan_i] = _Tptp
        subscan_Tstds[:, subscan_i] = _Tstd
        subscan_Qstds[:, subscan_i] = _Qstd
        subscan_Ustds[:, subscan_i] = _Ustd
        subscan_Qkurt[:, subscan_i] = _Qkurt
        subscan_Ukurt[:, subscan_i] = _Ukurt
        subscan_Qskew[:, subscan_i] = _Qskew
        subscan_Uskew[:, subscan_i] = _Uskew

    badsubscan_indicator =  (subscan_Tptps > Tptp_subscan_pW_threshold)|\
                            (subscan_Qstds > nstd_threshold*np.median(subscan_Qstds, axis=1)[:, np.newaxis])|\
                            (subscan_Ustds > nstd_threshold*np.median(subscan_Ustds, axis=1)[:, np.newaxis])|\
                            (np.abs(subscan_Qkurt) > kurt_threshold) | (np.abs(subscan_Ukurt) > kurt_threshold) |\
                            (np.abs(subscan_Qskew) > skew_threshold) | (np.abs(subscan_Uskew) > skew_threshold)

    badsubscan_flags = np.zeros([aman.dets.count, aman.samps.count], dtype='bool')

    for subscan_i, (subscan_s, subscan_e) in enumerate(zip(aman.subscan_indices_start, aman.subscan_indices_end)):
        badsubscan_flags[:, subscan_s:subscan_e] = badsubscan_indicator[:, subscan_i, np.newaxis]
    
    totally_bad_subscan_flags = np.mean(badsubscan_flags, axis=0) > 0.5
    totally_bad_detector_flags = np.mean(badsubscan_flags, axis=1) > 0.5

    badsubscan_flags = RangesMatrix.from_mask(badsubscan_flags)
    totally_bad_subscan_flags = Ranges.from_mask(totally_bad_subscan_flags)

    aman.flags.wrap('_bad_subscan', badsubscan_flags, [(0, 'dets'), (1, 'samps')])
    aman.flags.wrap('totally_bad_subscan', totally_bad_subscan_flags, [(0, 'samps')])
    aman.flags.reduce(flags=['_bad_subscan', 'totally_bad_subscan'], method='union', wrap=True,
                     new_flag='bad_subscan', remove_reduced=True)

    aman.wrap('subscan_Tptps', subscan_Tptps, [(0, 'dets'), (1, 'subscan_samps')])
    aman.wrap('subscan_Tstds', subscan_Tstds, [(0, 'dets'), (1, 'subscan_samps')])
    aman.wrap('subscan_Qstds', subscan_Qstds, [(0, 'dets'), (1, 'subscan_samps')])
    aman.wrap('subscan_Ustds', subscan_Ustds, [(0, 'dets'), (1, 'subscan_samps')])
    aman.wrap('subscan_Qkurt', subscan_Qkurt, [(0, 'dets'), (1, 'subscan_samps')])
    aman.wrap('subscan_Ukurt', subscan_Ukurt, [(0, 'dets'), (1, 'subscan_samps')])
    aman.wrap('subscan_Qskew', subscan_Qskew, [(0, 'dets'), (1, 'subscan_samps')])
    aman.wrap('subscan_Uskew', subscan_Uskew, [(0, 'dets'), (1, 'subscan_samps')])
    aman.wrap('badsubscan_indicator', badsubscan_indicator, [(0, 'dets'), (1, 'subscan_samps')])
    
    print(f'dets: {aman.dets.count}')
    print('removing detectors which are not good > 50% of duration')
    aman.restrict('dets', aman.dets.vals[~totally_bad_detector_flags])
    # filt_buffer is included in bad_subscan
    print(f'dets: {aman.dets.count}')
    badsubscan_flags = aman.flags.reduce(flags=['bad_subscan', 'filt_buffer'], method='union', wrap=False)
    aman.flags.bad_subscan = badsubscan_flags
    
    return

def get_glitches(aman, buffer=5, t_glitch=0.1, n_sig_dsT=10, n_sig_demod=7, edge_guard=40):
    glitches_T = tod_ops.flags.get_glitch_flags(aman, signal_name='dsT', merge=True, name='glitches_T', 
                                               buffer=buffer, t_glitch=t_glitch, n_sig=n_sig_dsT, edge_guard=edge_guard)
    
    glitches_Q = tod_ops.flags.get_glitch_flags(aman, signal_name='demodQ', merge=True, name='glitches_Q', 
                                               buffer=buffer, t_glitch=t_glitch, n_sig=n_sig_demod, edge_guard=edge_guard)
    
    glitches_U = tod_ops.flags.get_glitch_flags(aman, signal_name='demodU', merge=True, name='glitches_U', 
                                               buffer=buffer, t_glitch=t_glitch, n_sig=n_sig_demod, edge_guard=edge_guard)
    aman.flags.reduce(flags=['glitches_T', 'glitches_Q', 'glitches_U'], method='union',
                      wrap=True, new_flag='glitches', remove_reduced=True)
    return

def get_demod_wnl(aman):
    freqs_demod, Pxx_demodQ = calc_wrap_psd(aman, signal_name='demodQ', merge=False, 
                                               merge_wn=False, nperseg=int(200*100/ds_factor))
    freqs_demod, Pxx_demodU = calc_wrap_psd(aman, signal_name='demodU', merge=False, 
                                               merge_wn=False, nperseg=int(200*100/ds_factor))

    _fmask = (0.5<freqs_demod)&(freqs_demod<1.75)
    wn_demodQ = np.mean(np.sqrt(Pxx_demodQ[:, _fmask]), axis=1)
    wn_demodU = np.mean(np.sqrt(Pxx_demodU[:, _fmask]), axis=1)

    good_demod_wnl_flags = (wn_demodQ/ aman.wn_signal > 0.66*np.sqrt(2))&(wn_demodU/aman.wn_signal > 0.66*np.sqrt(2))
    good_demod_wnl_flags = np.logical_and(good_demod_wnl_flags, (wn_demodQ/ aman.wn_signal < 1.33*np.sqrt(2))&(wn_demodU/aman.wn_signal < 1.33*np.sqrt(2)))
    
    _ = aman.restrict('dets', aman.dets.vals[good_demod_wnl_flags])
    return

def get_t2p_coeffs_in_freq(aman, T_sig_name='dsT', Q_sig_name='demodQ', U_sig_name='demodU',
                           fs=None, fit_freq_range=(0.01, 0.1), wn_freq_range=(0.2, 1.9),
                           subtract_sig=False, merge_stats=True, t2p_stats_name='t2p_stats',
                          ):
    """
    Compute the leakage coefficients from temperature (T) to polarization (Q and U) in Fourier
    domain. Return an axismanager of the coefficients with their statistical uncertainties and 
    reduced chi-squared values for the fit.

    Parameters
    ----------
    aman : AxisManager
        AxisManager object containing the TOD data.
    T_sig_name : str
        Name of the temperature signal in `aman`. Default is 'dsT'.
    Q_sig_name : str
        Name of the Q polarization signal in `aman`. Default is 'demodQ'.
    U_sig_name : str
        Name of the U polarization signal in `aman`. Default is 'demodU'.
    fs: float
        The sampling frequency. If it is None, it will be calculated. Default is None.
    fit_range_freq: tuple
        The start/end frequencies of the t2p fit. Default is (0.01, 0.1).
    wn_freq_range: tuple
        The start/end frequencies to calculate the white noise level of demod signal. 
        Default is (0.2, 1.9).
    subtract_sig : bool
        Whether to subtract the calculated leakage from the polarization signals. Default is False.
    merge_stats : bool
        Whether to merge the calculated statistics back into `aman`. Default is True.
    t2p_stats_name : str
        Name under which to wrap the output AxisManager containing statistics. Default is 't2p_stats'.

    Returns
    -------
    out_aman : AxisManager
                An AxisManager containing leakage coefficients, their errors, and reduced
                chi-squared statistics.
    """
    if fs is None:
        fs = np.median(1/np.diff(aman.timestamps))
    N = aman.samps.count
    freqs = rfftfreq(N, d=1/fs)
    I_fs = rfft(aman[T_sig_name], axis=1)
    Q_fs = rfft(aman[Q_sig_name], axis=1)
    U_fs = rfft(aman[U_sig_name], axis=1)
    
    coeffsQ = np.zeros(aman.dets.count)
    errorsQ = np.zeros(aman.dets.count)
    redchi2sQ = np.zeros(aman.dets.count)
    coeffsU = np.zeros(aman.dets.count)
    errorsU = np.zeros(aman.dets.count)
    redchi2sU = np.zeros(aman.dets.count)

    fit_mask = (fit_freq_range[0] < freqs) & (freqs < fit_freq_range[1])
    wn_mask = (wn_freq_range[0] < freqs) & (freqs < wn_freq_range[1])

    def leakage_model(B, x):
        return B[0] * x

    model = Model(leakage_model)

    for i in range(aman.dets.count):
        # fit Q
        Q_wnl = np.nanmean(np.abs(Q_fs[i][wn_mask]))
        x = np.real(I_fs[i])[fit_mask]
        y = np.real(Q_fs[i])[fit_mask]
        sx = Q_wnl / np.sqrt(2) * np.ones_like(x)
        sy = Q_wnl * np.ones_like(y)
        try:
            data = RealData(x=x, 
                            y=y, 
                            sx=sx, 
                            sy=sy)
            odr = ODR(data, model, beta0=[1e-3])
            output = odr.run()
            coeffsQ[i] = output.beta[0]
            errorsQ[i] = output.sd_beta[0]
            redchi2sQ[i] = output.sum_square / (len(x) - 2)
        except:
            coeffsQ[i] = np.nan
            errorsQ[i] = np.nan
            redchi2sQ[i] = np.nan

        #fit U
        U_wnl = np.nanmean(np.abs(U_fs[i][wn_mask]))
        x = np.real(I_fs[i])[fit_mask]
        y = np.real(U_fs[i])[fit_mask]
        sx = U_wnl / np.sqrt(2) * np.ones_like(x)
        sy = U_wnl * np.ones_like(y)
        try:
            data = RealData(x=x, 
                            y=y, 
                            sx=sx, 
                            sy=sy)
            odr = ODR(data, model, beta0=[1e-3])
            output = odr.run()
            coeffsU[i] = output.beta[0]
            errorsU[i] = output.sd_beta[0]
            redchi2sU[i] = output.sum_square / (len(x) - 2)
        except:
            coeffsU[i] = np.nan
            errorsU[i] = np.nan
            redchi2sU[i] = np.nan

    out_aman = core.AxisManager(aman.dets, aman.samps)
    out_aman.wrap('coeffsQ', coeffsQ, [(0, 'dets')])
    out_aman.wrap('errorsQ', errorsQ, [(0, 'dets')])
    out_aman.wrap('redchi2sQ', redchi2sQ, [(0, 'dets')])
    out_aman.wrap('coeffsU', coeffsU, [(0, 'dets')])
    out_aman.wrap('errorsU', errorsU, [(0, 'dets')])
    out_aman.wrap('redchi2sU', redchi2sU, [(0, 'dets')])

    if subtract_sig:
        subtract_t2p(aman, out_aman)
    if merge_stats:
        aman.wrap(t2p_stats_name, out_aman)

    return out_aman


def subtract_t2p(aman):
    t2p_aman = get_t2p_coeffs_in_freq(aman)
    t2p.subtract_t2p(aman, t2p_aman)
    return

def get_and_subtract_i2p(aman, filt_order=1, cutoff_lim=(0.00001, 0.005), wrap=True, wrap_name='t2p_stats'):
    def split_I(I_obs, cutoff, fs=4):
        b_low, a_low = butter(filt_order, cutoff / (0.5 * fs), btype='low')
        b_high, a_high = butter(filt_order, cutoff / (0.5 * fs), btype='high')
    
        I_atm = filtfilt(b_low, a_low, I_obs)  # slow drift
        I_g = filtfilt(b_high, a_high, I_obs)  # fast drift
        return I_g, I_atm
    
    def loss(params, I_obs, Q_obs, cutoff_lim):
        cutoff, eps1, eps2 = params
        if not (cutoff_lim[0]<cutoff < cutoff_lim[1]):  # cutoffの現実的制約
            return np.inf
        
        I_g, I_atm = split_I(I_obs, cutoff)
        Q_est = Q_obs - eps1 * I_g - eps2 * I_atm
        return np.sum(Q_est**2)  

    x0 = [1E-3, 0.005, 0.005]
    coeffsQ_slow = np.zeros(aman.dets.count)
    coeffsQ_fast = np.zeros(aman.dets.count)
    cut_freqQ = np.zeros(aman.dets.count)
    redchi2sQ = np.zeros(aman.dets.count)
    
    coeffsU_slow = np.zeros(aman.dets.count)
    coeffsU_fast = np.zeros(aman.dets.count)
    cut_freqU = np.zeros(aman.dets.count)
    redchi2sU = np.zeros(aman.dets.count)
    
    for i in tqdm(range(aman.dets.count)):
        res = minimize(loss, x0, args=(aman.dsT[i], aman.demodQ[i], cutoff_lim), method='Nelder-Mead')
        if res.success is False:
            redchi2sQ[i] = np.inf
            continue
        else: 
            redchi2sQ[i] = res.fun/np.std(aman.demodQ[i])
        I_g, I_atm = split_I(aman.dsT[i], cutoff=res.x[0])
        aman.demodQ[i] = aman.demodQ[i] - res.x[1] * I_g - res.x[2] * I_atm
        coeffsQ_slow[i] = res.x[2]
        coeffsQ_fast[i] = res.x[1]
        cut_freqQ[i] = res.x[0]
    
    for i in tqdm(range(aman.dets.count)):
        res = minimize(loss, x0, args=(aman.dsT[i], aman.demodU[i], cutoff_lim), method='Nelder-Mead')
        if res.success is False:
            redchi2sU[i] = np.inf
            continue
        else: 
            redchi2sU[i] = res.fun/np.std(aman.demodU[i])
        I_g, I_atm = split_I(aman.dsT[i], cutoff=res.x[0])
        aman.demodU[i] = aman.demodU[i] - res.x[1] * I_g - res.x[2] * I_atm
        coeffsU_slow[i] = res.x[2]
        coeffsU_fast[i] = res.x[1]
        cut_freqU[i] = res.x[0]
        
    out_aman = core.AxisManager(aman.dets, aman.samps)
    out_aman.wrap('coeffsQ_slow', coeffsQ_slow, [(0, 'dets')])
    out_aman.wrap('coeffsQ_fast', coeffsQ_fast, [(0, 'dets')])
    out_aman.wrap('redchi2sQ', redchi2sQ, [(0, 'dets')])
    out_aman.wrap('cut_freqQ', cut_freqQ, [(0, 'dets')])
    
    out_aman.wrap('coeffsU_slow', coeffsU_slow, [(0, 'dets')])
    out_aman.wrap('coeffsU_fast', coeffsU_fast, [(0, 'dets')])
    out_aman.wrap('redchi2sU', redchi2sU, [(0, 'dets')])
    out_aman.wrap('cut_freqU', cut_freqU, [(0, 'dets')])
    if wrap:
        aman.wrap(wrap_name, out_aman)
    return out_aman

def restrict_dets_from_t2p_fit(aman, t2p_aman=None, redchi2s=True, error=True, lam=False, 
                               redchi2s_lims=(0.1, 3), error_lims=(0, 0.03), lam_lims=(0, 0.01)):
    """
    Restrict detectors based on the t2p fit stats or t2p coefficient.

    Parameters
    ----------
    aman : AxisManager
        The tod.
    t2p_aman : AxisManager
        Axis manager with Q and U leakage coeffients.
        If joint fitting was used in get_t2p_coeffs, Q coeffs are in 
        fields ``lamQ`` and ``AQ`` and U coeffs are in ``lamU`` and 
        ``AU``. Otherwise Q coeff is in field ``coeffsQ`` and U coeff 
        in ``coeffsU``.
    redchi2s : bool
        If True, restrict detectors based on redchi2s values.
    error : bool
        If True, restrict detectors based on fit errors of lamQ and lamU.
    lam : bool
        If True, restrict detectors based on the amplitude of lamQ and lamU.
    redchi2s_lims : tuple
        The lower and upper limit of acceptable redchi2s.
    error_lims : tuple
        The lower and upper limit of acceptable errors.
    lam_lims : tuple
        The lower and upper limit of acceptable leakage coefficient.
    """
    if t2p_aman is None:
        t2p_aman = aman.t2p_stats
    mask = np.ones(aman.dets.count, dtype=bool)
    if redchi2s:
        redchi2s_t2p = t2p_aman.redchi2s
        mask_redchi2s = (redchi2s_lims[0] < redchi2s_t2p) & (redchi2s_t2p < redchi2s_lims[1])
        mask = np.logical_and(mask, mask_redchi2s)
    if error:
        error_t2p = np.sqrt(t2p_aman.lamQ_error**2+t2p_aman.lamU_error**2)
        mask_error = (error_lims[0] < error_t2p) & (error_t2p < error_lims[1])
        mask = np.logical_and(mask, mask_error)
    if lam:
        lam_t2p = np.sqrt(t2p_aman.lamQ**2 + t2p_aman.lamU**2)
        mask_lam = (lam_lims[0] < lam_t2p) & (lam_t2p < lam_lims[1])
        mask = np.logical_and(mask, mask_lam)
    aman.restrict('dets', aman.dets.vals[mask]) 

def get_common_mode_Pxx_model(aman, signal_name, 
                    f_low_wnl=0.3,
                    freq_lims = (1e-4, 1.9),
                    fk_alpha_init=(0.01, -2.),
                    fk_bounds = (1e-4, 1.),
                    alpha_bounds = (-5, -0.5),
                   ):
    model = pca.get_pca_model(aman, signal=aman[signal_name], n_modes=1)
    freqs, Pxx = calc_psd(aman, signal=np.array([model.modes[0]]), nperseg=200*1000, merge=False)
    mask_valid_freqs = ( freq_lims[0]<freqs ) & (freqs < freq_lims[1])
    x = freqs[mask_valid_freqs]
    y = Pxx[0][mask_valid_freqs]
    _sigma = np.sqrt(np.median(y[x>f_low_wnl]))
    _fit_func = lambda x, _fk, _alpha: log_fit_func(x, _sigma, _fk, _alpha)
    popt, pcov = curve_fit(_fit_func, x, np.log(y), 
                           p0 = fk_alpha_init,
                           bounds = ((fk_bounds[0], alpha_bounds[0]), 
                                     (fk_bounds[1], alpha_bounds[1])),
                           maxfev = 10000)
    sigma = _sigma
    fk = popt[0]
    alpha = popt[1]

    return model, sigma, fk, alpha

def common_c1f(aman, signal_name, fs=4, overwrite=False):
    """
    apply common mode counter 1/f filter

    Parameters
    ----------
    aman : axismanager
    signal_name : str
    fs : float, optional
        Sampling frequency [Hz]（Default is 4 Hz）
    overwrite : bool, optional
        If True, overwrite the input signal

    Returns
    -------
    processed_signal : array
    """
    def F_inv(freqs, fk, n, c):
        return c*(fk/freqs)**n/(1+(fk/freqs)**n)

    model, sigma, fk, alpha= get_common_mode_Pxx_model(aman, signal_name)
    N = aman.samps.count
    freqs = rfftfreq(N, d=1/fs)
    dc_f = rfft(model.modes[0])
    processed_signal = []
    for i, d_t in enumerate(tqdm(aman[signal_name])):
        d_f = rfft(d_t)
        F_inv_i = F_inv(freqs, fk, -alpha, model.weights.T[0][i])
        F_inv_i[0] = F_inv_i[1]
        d_f_prime = d_f - F_inv_i * dc_f
        d_t_prime = irfft(d_f_prime, n=aman.samps.count)
        processed_signal.append(d_t_prime)
    processed_signal = np.array(processed_signal)

    if overwrite:
        aman[signal_name] = processed_signal
    
    return processed_signal

def apply_common_c1f_per_wafer(aman, signal_name, fs=4):
    wafer_slots = np.unique(aman.det_info.wafer_slot)
    for ws in wafer_slots[:]:
        mask_ws = aman.det_info.wafer_slot == ws
        aman_ws = aman.restrict('dets', aman.dets.vals[mask_ws], in_place=False)
        common_c1f(aman_ws, signal_name, fs=4, overwrite=True)
        aman[signal_name][mask_ws, :] = aman_ws[signal_name]
    
    return

def subtract_common_mode_ws_pca(aman, signal_names, n_modes=1):
    wafer_slots = np.unique(aman.det_info.wafer_slot)
    for ws in wafer_slots:
        mask_ws = aman.det_info.wafer_slot == ws
        aman_ws = aman.restrict('dets', aman.dets.vals[mask_ws], in_place=False)
        
        for signal_name in signal_names:
            try:
                model_ws = pca.get_pca_model(aman_ws, signal=aman_ws[signal_name], n_modes=n_modes)
                aman[signal_name][mask_ws] = pca.add_model(aman_ws, model_ws, signal=aman_ws[signal_name], scale=-1.)
            except:
                print('numdets are too small skipping pca')
    return

def apply_common_mode_sub(aman, signal_name='demodQ', n_modes=2):
    aman_common = get_common_mode_noise_params(aman)
    lpf_cfg = {'type': 'sine2', 'cutoff': aman_common[f'fk_{signal_name}'][0], 'trans_width': aman_common[f'fk_{signal_name}'][0]}
    hpf_cfg = {'type': 'sine2', 'cutoff': 0.005, 'trans_width': 0.005}
    filt = filters.get_lpf(lpf_cfg) * filters.get_hpf(hpf_cfg)
    lpf_signal = tod_ops.fourier_filter(aman, filt, signal_name=signal_name, detrend='linear')
    model = pca.get_pca_model(aman, signal=lpf_signal, n_modes=n_modes)
    demod_subtracted = pca.add_model(aman, model, signal=aman[signal_name], scale=-1.)
    return

def subtract_subscan_pca(aman, signal_name, n_modes,
                        hpf_cfg = {'type': 'sine2', 'cutoff': 1.35, 'trans_width': 0.025},
                        lpf_cfg = {'type': 'sine2', 'cutoff': 1.45, 'trans_width': 0.025},
                        ds_factor=10):
    # mainly to subtract the pulse tube signal at 1.4
    
    filt = filters.get_lpf(lpf_cfg) * filters.get_hpf(hpf_cfg)
    for i, (_s,_e)  in enumerate(zip(aman.subscan_indices_start[1:-1], aman.subscan_indices_end[1:-1])):
        aman_subscan = aman.restrict('samps', (aman.samps.offset+_s-int(200*10/ds_factor), aman.samps.offset+_e+int(200*10/ds_factor)), in_place=False)
        
        aman_subscan.wrap(f'{signal_name}_temp', aman_subscan[signal_name].copy(), [(0, 'dets'), (1, 'samps')])
        tod_ops.apodize_cosine(aman_subscan, signal_name=f'{signal_name}_temp', apodize_samps=int(200*5/ds_factor))
        aman_subscan[f'{signal_name}_temp'] = tod_ops.fourier_filter(aman_subscan, filt, signal_name=f'{signal_name}_temp', detrend='linear')
        
        pca_matrix = pca.get_pca(aman_subscan, signal=aman_subscan[f'{signal_name}_temp'])
        pca_model = pca.get_pca_model(aman_subscan, signal=aman_subscan[f'{signal_name}_temp'], pca=pca_matrix, n_modes=n_modes)
        aman_subscan[signal_name] = pca.add_model(aman_subscan, pca_model, signal=aman_subscan[signal_name], scale=-1.)
        aman[signal_name][:, _s:_e] = aman_subscan[signal_name][:,int(200*10/ds_factor):-int(200*10/ds_factor)]
    return

def wrap_subscan_idx_info(aman):
    subscan_indices_l = sub_polyf._get_subscan_range_index(aman.flags["left_scan"][0].mask())
    subscan_indices_l_start = subscan_indices_l[:, 0]
    subscan_indices_l_end = subscan_indices_l[:, 1]
    subscan_indices_r = sub_polyf._get_subscan_range_index(aman.flags["right_scan"][0].mask())
    subscan_indices_r_start = subscan_indices_r[:, 0]
    subscan_indices_r_end = subscan_indices_r[:, 1]

    subscan_indices_start = np.hstack([subscan_indices_l_start, subscan_indices_r_start])
    subscan_indices_start= subscan_indices_start[np.argsort(subscan_indices_start)]
    subscan_indices_end = np.hstack([subscan_indices_l_end, subscan_indices_r_end])
    subscan_indices_end= subscan_indices_end[np.argsort(subscan_indices_end)]
    
    subscan_lr = np.where(np.in1d(subscan_indices_start, subscan_indices_l_start), 'l', 'r')
    subscan_samps = core.OffsetAxis('subscan_samps', subscan_indices_start.shape[0])

    aman.wrap('subscan_indices_start', subscan_indices_start, [(0, subscan_samps)])
    aman.wrap('subscan_indices_end', subscan_indices_end, [(0, subscan_samps)])
    aman.wrap('subscan_lr', subscan_lr, [(0, subscan_samps)])

    flag_buffer_scan = np.zeros((aman.dets.count, aman.samps.count), dtype=bool)
    flag_buffer_scan[:, :subscan_indices_start[1]] = True
    flag_buffer_scan[:, subscan_indices_end[-2]:] = True
    flag_buffer_scan = RangesMatrix.from_mask(flag_buffer_scan)
    aman.flags.wrap('buffer_scan', flag_buffer_scan, [(0, 'dets'), (1, 'samps')])
    return


def calc_subscan_psds(aman, signal_name, ds_factor, nperseg, wrap=True):
    for i, (_s,_e)  in enumerate(zip(aman.subscan_indices_start[1:-1], aman.subscan_indices_end[1:-1])):
        freqs_subscan, Pxx_subscan = welch(aman[signal_name][:, _s:_e+1], fs=int(200/ds_factor), nperseg=nperseg,
                                          detrend='linear')
        if i == 0:
            Pxx_subscans = np.full((aman.dets.count, aman.subscan_samps.count, len(freqs_subscan)), np.nan)
            nusamps_subscan = core.OffsetAxis('nusamps_subscan', len(freqs_subscan))
        Pxx_subscans[:, i+1, :] = Pxx_subscan
        
    if wrap:
        aman.wrap(f'freqs_subscan', freqs_subscan, [(0, nusamps_subscan)])
        aman.wrap(f'Pxx_subscans_{signal_name}', Pxx_subscans, [(0, 'dets'), (1, 'subscan_samps'), (2, nusamps_subscan)])
        
    return freqs_subscan, Pxx_subscans

def subscan_detrend(aman, signal_name='dsT', nsamps=5):
    for i, (_s, _e)  in enumerate(zip(aman.subscan_indices_start[:], aman.subscan_indices_end[:])):
        _dx = _e - _s
        _x = np.arange(0, _dx)
        
        _y1 = np.median(aman[signal_name][:, _s : _s+nsamps], axis=1)
        _y2 = np.median(aman[signal_name][:, _e-nsamps : _e], axis=1)
        
        _trend = ((_y2 - _y1)/_dx * _x[:, np.newaxis] + _y1).T
        
        aman[signal_name][:, _s:_e] -= _trend
    return

def wrap_and_fit_lr_Pxx(aman, freq_lims=(0.1, 7.), fk_alpha_init=(0.8, -3), fk_bounds=(0.1, 5.), alpha_bounds = (-5.0, -2.0), remove_at_limit=True):
    mask = aman.badsubscan_indicator.copy()
    mask[:, 0] = True
    mask[:, -1] = True
    mask_r = mask.copy()
    mask_l = mask.copy()
    mask_r[:, aman.subscan_lr=='l'] = True
    mask_l[:, aman.subscan_lr=='r'] = True
    Pxx_subscans_dsT_r_ma = np.ma.masked_array(aman.Pxx_subscans_dsT, mask=np.tile(mask_r[:, :, np.newaxis], (1, 1, aman.nusamps_subscan.count)))
    Pxx_subscans_dsT_r = np.ma.mean(Pxx_subscans_dsT_r_ma, axis=1).data
    Pxx_subscans_dsT_l_ma = np.ma.masked_array(aman.Pxx_subscans_dsT, mask=np.tile(mask_l[:, :, np.newaxis], (1, 1, aman.nusamps_subscan.count)))
    Pxx_subscans_dsT_l = np.ma.mean(Pxx_subscans_dsT_l_ma, axis=1).data
    aman.wrap('Pxx_subscans_dsT_r', Pxx_subscans_dsT_r, [(0, 'dets'), (1, 'nusamps_subscan')])
    aman.wrap('Pxx_subscans_dsT_l', Pxx_subscans_dsT_l, [(0, 'dets'), (1, 'nusamps_subscan')])

    hwpss_mask = get_mask_for_hwpss(aman.freqs_subscan, hwp_freq=np.abs(aman.fhwp))
    ptc_mask = get_mask_for_single_peak(aman.freqs_subscan, peak_freq=1.4, peak_width=(-0.1, +0.1))
    ptc_mask = np.logical_or(ptc_mask, get_mask_for_single_peak(aman.freqs_subscan, peak_freq=2.8, peak_width=(-0.15, +0.15)))
    peaks_mask = np.logical_or(hwpss_mask, ptc_mask)
    
    _sigma_r, _fk_r, _alpha_r = get_noise_model(aman, aman.freqs_subscan[~peaks_mask], aman.Pxx_subscans_dsT_r[:, ~peaks_mask], pre_computed_wnl=aman.wn_signal, 
                                                         freq_lims=freq_lims, fk_alpha_init=fk_alpha_init, 
                                                         fk_bounds=fk_bounds, alpha_bounds=alpha_bounds,
                                                         merge=False)
    _sigma_l, _fk_l, _alpha_l = get_noise_model(aman, aman.freqs_subscan[~peaks_mask], aman.Pxx_subscans_dsT_l[:, ~peaks_mask], pre_computed_wnl=aman.wn_signal, 
                                                         freq_lims=freq_lims, fk_alpha_init=fk_alpha_init, 
                                                         fk_bounds=fk_bounds, alpha_bounds=alpha_bounds,
                                                         merge=False)

    aman.wrap(f'sigma_dsT_r', _sigma_r, [(0, 'dets'),])
    aman.wrap(f'fk_dsT_r', _fk_r, [(0, 'dets'),])
    aman.wrap(f'alpha_dsT_r', _alpha_r, [(0, 'dets'),])

    aman.wrap(f'sigma_dsT_l', _sigma_l, [(0, 'dets'),])
    aman.wrap(f'fk_dsT_l', _fk_l, [(0, 'dets'),])
    aman.wrap(f'alpha_dsT_l', _alpha_l, [(0, 'dets'),])
    
    if remove_at_limit:
        print('remove detectors with alpha is at limit')
        aman.restrict('dets', aman.dets.vals[~np.logical_or(np.in1d(aman.alpha_dsT_l, alpha_bounds), np.in1d(aman.alpha_dsT_r, alpha_bounds))])
    return

def apply_subscan_c1f_fitler(aman, signal_name, ds_factor, ):
    aman.wrap(f'{signal_name}_new', np.zeros_like(aman[signal_name]), [(0, 'dets'), (1, 'samps')])
    for i, (_s,_e)  in enumerate(zip(aman.subscan_indices_start, aman.subscan_indices_end)):
        if i not in [0, aman.subscan_samps.count-1]:
            restrict_s_idx = _s
            valid_s_idx = _s
            restrict_e_idx = _e
            valid_e_idx = _e

            aman_subscan = aman.restrict('samps', (aman.samps.offset + restrict_s_idx, aman.samps.offset + restrict_e_idx), in_place=False)
            if aman.subscan_lr[i] == 'l':
                subscan_c1f_filt = filters.counter_1_over_f(aman.fk_dsT_l, -aman.alpha_dsT_l)
            elif aman.subscan_lr[i] == 'r':
                subscan_c1f_filt = filters.counter_1_over_f(aman.fk_dsT_r, -aman.alpha_dsT_r)
            aman_subscan[signal_name] = tod_ops.fourier_filter(aman_subscan, subscan_c1f_filt, signal_name=signal_name, )
            aman[f'{signal_name}_new'][:, valid_s_idx:valid_e_idx] = \
                                aman_subscan[signal_name][:, :]
    aman.move(signal_name, None)
    aman.move(f'{signal_name}_new', signal_name)
    return

def get_demod_Pxxs(aman, nperseg=int(200*1000/ds_factor), base=1.05):
    #freqs_demod, Pxx_dsT = calc_wrap_psd(aman, signal_name='dsT', merge=False, merge_wn=False, nperseg=nperseg)
    freqs_demod, Pxx_demodQ = calc_wrap_psd(aman, signal_name='demodQ', merge=False, merge_wn=False, nperseg=nperseg)
    freqs_demod, Pxx_demodU = calc_wrap_psd(aman, signal_name='demodU', merge=False, merge_wn=False, nperseg=nperseg)

    freqs_demod_binned = log_binning(freqs_demod, base=base)
    #Pxx_dsT_binned = log_binning(Pxx_dsT, base=base)
    Pxx_demodQ_binned = log_binning(Pxx_demodQ, base=base)
    Pxx_demodU_binned = log_binning(Pxx_demodU, base=base)
    nusamps_demod_binned = core.OffsetAxis('nusamps_demod_binned', len(freqs_demod_binned))

    _fmask = (0.5<freqs_demod_binned)&(freqs_demod_binned<1.75)
    wn_demodQ = np.mean(np.sqrt(Pxx_demodQ_binned[:, _fmask]), axis=1)
    wn_demodU = np.mean(np.sqrt(Pxx_demodU_binned[:, _fmask]), axis=1)
    
    if 'freqs_demod_binned' in aman._fields:
        aman.move('freqs_demod_binned', None)
    if 'Pxx_demodQ_binned' in aman._fields:
        aman.move('Pxx_demodQ_binned', None)
    if 'Pxx_demodU_binned' in aman._fields:
        aman.move('Pxx_demodU_binned', None)
    if 'wn_demodQ' in aman._fields:
        aman.move('wn_demodQ', None)     
    if 'wn_demodU' in aman._fields:
        aman.move('wn_demodU', None)

    _ = aman.wrap('freqs_demod_binned', freqs_demod_binned, [(0, nusamps_demod_binned, )])
    #_ = aman.wrap('Pxx_dsT_binned', Pxx_dsT_binned, [(0, 'dets'), (1, 'nusamps_demod_binned')])
    _ = aman.wrap('Pxx_demodQ_binned', Pxx_demodQ_binned, [(0, 'dets'), (1, 'nusamps_demod_binned')])
    _ = aman.wrap('Pxx_demodU_binned', Pxx_demodU_binned, [(0, 'dets'), (1, 'nusamps_demod_binned')])
    _ = aman.wrap('wn_demodQ', wn_demodQ, [(0, 'dets')])
    _ = aman.wrap('wn_demodU', wn_demodU, [(0, 'dets')])
    return

def do_signal_noise_fit(aman, freq_lims=(0.3, 50), fk_alpha_init = (1.5, -3), fk_bounds=(0.3, 20), alpha_bounds = (-6., -1.)):
    # noise fit for signal
    _ = get_noise_model(aman, aman.freqs_raw_binned, aman.Pxx_signal_binned, pre_computed_wnl=aman.wn_signal, 
                           freq_lims = freq_lims, fk_alpha_init=fk_alpha_init, fk_bounds=fk_bounds, alpha_bounds=alpha_bounds,
                           merge=True, merge_suffix='signal')
    return

def do_demod_noise_fit(aman,
                      freq_lims = (1e-4, 1.9), fk_alpha_init=(0.001, -2.), fk_bounds = (1e-4, 1.), alpha_bounds = (-3., -0.1),
                      merge_suffixQ="demodQ", merge_suffixU="demodU"):
    for suffix in (merge_suffixQ, merge_suffixU):
        for params in ('sigma', 'fk', 'alpha'):
            if f'{params}_{suffix}' in aman:
                aman.move(f'{params}_{suffix}', None)
    # noise fit for demod
    _ = get_noise_model(aman, aman.freqs_demod_binned, aman.Pxx_demodQ_binned, pre_computed_wnl=aman.wn_demodQ, 
                        merge=True, merge_suffix=merge_suffixQ, 
                        freq_lims = freq_lims, fk_alpha_init=fk_alpha_init, fk_bounds=fk_bounds, alpha_bounds=alpha_bounds
                       )
    _ = get_noise_model(aman, aman.freqs_demod_binned, aman.Pxx_demodU_binned, pre_computed_wnl=aman.wn_demodU,
                        merge=True, merge_suffix=merge_suffixU,
                        freq_lims = freq_lims, fk_alpha_init=fk_alpha_init, fk_bounds=fk_bounds, alpha_bounds=alpha_bounds)
    return

def remove_signal_fk_outliers(aman, fk_signal_max=6.):
    fk_signal = aman.fk_signal
    good_abs_fk_signal_flags = fk_signal < fk_signal_max

    good_fraction = np.count_nonzero(good_abs_fk_signal_flags) / aman.dets.count
    print(f'tot: {np.count_nonzero(good_abs_fk_signal_flags)}/{aman.dets.count} ({100*good_fraction:.1f}%)')
    aman.restrict('dets', aman.dets.vals[good_abs_fk_signal_flags])
    return

def remove_demod_fk_outliers(aman, fk_demod_max=0.1):
    fk_demod = (aman.fk_demodQ + aman.fk_demodU)/2.
    good_abs_fk_demod_flags = fk_demod < fk_demod_max

    good_fraction = np.count_nonzero(good_abs_fk_demod_flags) / aman.dets.count
    print(f'tot: {np.count_nonzero(good_abs_fk_demod_flags)}/{aman.dets.count} ({100*good_fraction:.1f}%)')
    aman.restrict('dets', aman.dets.vals[good_abs_fk_demod_flags])
    return

def apply_c1f_filters_demod(aman):
    # apodize.apodize_cosine(aman, signal_name='demodQ', apodize_samps=int(200*(5+2+5+10)/ds_factor))
    # apodize.apodize_cosine(aman, signal_name='demodU', apodize_samps=int(200*(5+2+5+10)/ds_factor))   
    # apodize.apodize_cosine(aman, signal_name='demodQ', apodize_samps=int(200*5/ds_factor))
    # apodize.apodize_cosine(aman, signal_name='demodU', apodize_samps=int(200*5/ds_factor))
    filt = filters.counter_1_over_f(aman.fk_demodQ, -aman.alpha_demodQ)
    aman.demodQ = tod_ops.fourier_filter(aman, filt, signal_name='demodQ')
    filt = filters.counter_1_over_f(aman.fk_demodU, -aman.alpha_demodU)
    aman.demodU = tod_ops.fourier_filter(aman, filt, signal_name='demodU')
    return

def get_common_mode_noise_params(aman, freq_lims=(1e-2, 1.9)):
    aman_common = core.AxisManager(core.LabelAxis('dets', ['common']), aman.samps)
    demodQ_c = np.nanmedian(aman.demodQ, axis=0)
    demodU_c = np.nanmedian(aman.demodU, axis=0)
    aman_common.wrap('demodQ', np.array([demodQ_c]), [(0, 'dets'), (1, 'samps')])
    aman_common.wrap('demodU', np.array([demodU_c]), [(0, 'dets'), (1, 'samps')])
    aman_common.wrap('timestamps', aman.timestamps, [(0, 'samps')])
    get_demod_Pxxs(aman_common)
    do_demod_noise_fit(aman_common, freq_lims=freq_lims)
    return aman_common

def subtract_overall_mean(aman):
    dsT_overall_mean = np.ma.mean(np.ma.masked_array(aman.dsT, mask=aman.flags.exclude.mask()), axis=1).data
    demodQ_overall_mean = np.ma.mean(np.ma.masked_array(aman.demodQ, mask=aman.flags.exclude.mask()), axis=1).data
    demodU_overall_mean = np.ma.mean(np.ma.masked_array(aman.demodU, mask=aman.flags.exclude.mask()), axis=1).data

    aman.dsT = aman.dsT - dsT_overall_mean[:, np.newaxis].astype(np.float32)
    aman.demodQ = aman.demodQ - demodQ_overall_mean[:, np.newaxis].astype(np.float32)
    aman.demodU = aman.demodU - demodU_overall_mean[:, np.newaxis].astype(np.float32)
    return

def get_apodizer_turnarounds(aman, apodize_sec=1.7):
    fsamp = np.round(np.median(1/np.diff(aman.timestamps))).astype(int)
    apodize_samps = (apodize_sec*fsamp).astype(int)
    apodizer = apodize.get_apodize_window_from_flags(aman, flags=aman.flags.turnarounds_narrow[0], apodize_samps=apodize_samps)
    return apodizer

def remove_dropout_wafers(aman):
    wafer_slots = np.unique(aman.det_info.wafer_slot)
    for ws in wafer_slots[:]:
        mask_ws = aman.det_info.wafer_slot == ws
        if np.count_nonzero(mask_ws) < 10:
            print(f'removing {ws} because only {np.count_nonzero(mask_ws)} detectors are aliving')
            aman.restrict('dets', aman.dets.vals[~mask_ws])
    return

def get_inv_var(aman):
    # the factor 1.9 [Hz] stands for bandpass frequency
    var_demod = 1.9 * ((aman.wn_demodQ+aman.wn_demodU)/2.)**2
    inv_var_demod = 1/var_demod
    aman.wrap('inv_var', inv_var_demod, [(0, 'dets')])
    return

def scan_freq_cut(aman):
    scan_freq = 1/(aman.obs_info.az_throw*4/aman.scanspeed)
    hpf_cfg = {'type': 'sine2', 'cutoff': scan_freq, 'trans_width': scan_freq/10}
    filt = filters.get_hpf(hpf_cfg)
    
    aman.dsT = tod_ops.fourier_filter(aman, filt, signal_name='dsT', detrend='linear')
    aman.demodQ = tod_ops.fourier_filter(aman, filt, signal_name='demodQ', detrend='linear')
    aman.demodU = tod_ops.fourier_filter(aman, filt, signal_name='demodU', detrend='linear')
    return

def log_binning(psd, unbinned_mode=3, base=1.05, mask=None,
                return_bin_size=False, drop_nan=False):
    """
    Function to bin PSD or frequency. First several Fourier modes are left un-binned.
    Fourier modes higher than that are averaged into logspace bins.
    Parameters
    ----------
    psd : numpy.ndarray
        PSD (or frequency) to be binned. Can be a 1D or 2D array.
    unbinned_mode : int, optional
        First Fourier modes up to this number are left un-binned. Defaults to 3.
    base : float, optional
        Base of the logspace bins. Must be greater than 1. Defaults to 1.05.
    mask : numpy.ndarray, optional
        Mask for psd. If all values in a bin are masked, the value becomes np.nan.
        Should be a 1D array.
    return_bin_size : bool, optional
        If True, the number of data points in the bins are returned. Defaults to False.
    drop_nan : bool, optional
        If True, drop the indices where psd is NaN. Defaults to False.
    Returns
    -------
    binned_psd : numpy.ndarray
        The binned PSD. If the input is 2D, the output will also be 2D with the same number of rows.
    bin_size : numpy.ndarray, optional
        The number of data points in each bin, only returned if return_bin_size is True.
    """
    if base <= 1:
        raise ValueError("base must be greater than 1")

    is_1d = psd.ndim == 1

    # Ensure psd is at least 2D for consistent processing
    psd = np.atleast_2d(psd)
    num_signals, num_samples = psd.shape

    if mask is not None:
        # Ensure mask is at least 2D and has the same shape as psd
        mask = np.atleast_2d(mask)
        if mask.shape[1] != num_samples:
            raise ValueError("Mask must have the same number of columns as psd")
        psd = np.ma.masked_array(psd, mask=np.tile(mask, (num_signals, 1)))

    # Initialize the binned PSD and optionally the bin sizes
    binned_psd = np.zeros((num_signals, unbinned_mode + 1))
    binned_psd[:, :unbinned_mode + 1] = psd[:, :unbinned_mode + 1]
    bin_size = np.ones((num_signals, unbinned_mode + 1)) if return_bin_size else None

    # Determine the number of bins and their indices
    N = int(np.ceil(np.emath.logn(base, num_samples - unbinned_mode)))
    binning_idx = np.unique(np.logspace(base, N, N, base=base, dtype=int) + unbinned_mode - 1)

    # Bin the PSD values for each signal
    new_binned_psd = []
    new_bin_size = []
    for start, end in zip(binning_idx[:-1], binning_idx[1:]):
        bin_mean = np.nanmean(psd[:, start:end], axis=1)
        new_binned_psd.append(bin_mean)
        if return_bin_size:
            new_bin_size.append(end - start)

    # Convert lists to numpy arrays and concatenate with initial values
    new_binned_psd = np.array(new_binned_psd).T  # Transpose to match dimensions
    binned_psd = np.hstack([binned_psd, new_binned_psd])
    if return_bin_size:
        new_bin_size = np.array(new_bin_size)
        bin_size = np.hstack([bin_size, np.tile(new_bin_size, (num_signals, 1))])

    if drop_nan:
        valid_indices = ~np.isnan(binned_psd).any(axis=0)
        binned_psd = binned_psd[:, valid_indices]
        if return_bin_size:
            bin_size = bin_size[:, valid_indices]

    if is_1d:
        binned_psd = binned_psd.flatten()
        if return_bin_size:
            bin_size = bin_size.flatten()

    if return_bin_size:
        return binned_psd, bin_size
    return binned_psd

def get_mask_for_single_peak(f, peak_freq, peak_width=(-0.002, +0.002)):
    mask = (f > peak_freq + peak_width[0])&(f < peak_freq + peak_width[1])
    return mask

def get_mask_for_hwpss(freq, hwp_freq, max_mode=3, width=(-0.15, 0.15)):
    if isinstance(width, (float, int)):
        width_minus = -width/2
        width_plus = width/2
        mask_arrays = []
        for n in range(max_mode):
            mask_arrays.append(get_mask_for_single_peak(freq, hwp_freq*(n+1), peak_width=(width_minus, width_plus)))
    elif isinstance(width, (np.ndarray, list, tuple)):
        width = np.array(width)
        if len(width.shape) == 1:
            # mask for 1f
            width_minus = -width[0]/2
            width_plus = width[0]/2
            mask_arrays = [get_mask_for_single_peak(freq, hwp_freq, peak_width=(width_minus, width_plus))]
            # masks for Nf
            width_minus = -width[1]/2
            width_plus = width[1]/2
            for n in range(max_mode-1):
                mask_arrays.append(get_mask_for_single_peak(freq, hwp_freq*(n+2), peak_width=(width_minus, width_plus)))
        elif len(width.shape) == 2:
            # mask for 1f
            width_minus = width[0][0]
            width_plus = width[0][1]
            mask_arrays = [get_mask_for_single_peak(freq, hwp_freq, peak_width=(width_minus, width_plus))]
            # masks for Nf
            width_minus = width[1][0]
            width_plus = width[1][1]
            for n in range(max_mode-1):
                mask_arrays.append(get_mask_for_single_peak(freq, hwp_freq*(n+2), peak_width=(width_minus, width_plus)))
    mask = np.any(np.array(mask_arrays), axis=0)
    return mask

def remove_heavy_fields(aman):
    to_be_removed = ['signal', 'iir_params', 'pwv_class', 'hwp_angle', 'dsT', 'demodQ', 'demodU']
    
    # [timestamps, boresight, flags]
    for _key in to_be_removed:
        if _key in aman._assignments.keys():
            aman.move(_key, None)
        else:
            pass
    return

def map_make(aman, res=0.3*coords.DEG, full_sky=True, cuts=None, det_weights_dsT=None, det_weights_demod=None):
    if cuts is None:
        cuts = aman.flags.exclude
        
    if det_weights_dsT is None:
        det_weights_dsT = 2*aman.inv_var
        
    if det_weights_demod is None:
        det_weights_demod = aman.inv_var
    
    _t0 = time()
    if full_sky:
        shape, wcs = enmap.fullsky_geometry(res=res, proj='car')
        P = coords.P.for_tod(aman, geom=(shape, wcs), comps='QU', cuts=cuts, hwp=True)
    else: 
        wcs = coords.get_wcs_kernel('car', 0, 0, res)
        P = coords.P.for_tod(aman, wcs_kernel=wcs, comps='QU', cuts=cuts, hwp=True)    
        
    result = demod_mm.make_map(aman, P=P, 
                               det_weights=det_weights_dsT, det_weights_demod=det_weights_demod)    
    
    if full_sky:
        P = coords.P.for_tod(aman, geom=(shape, wcs), comps='T', cuts=cuts)
    else:
        P = coords.P.for_tod(aman, wcs_kernel=wcs, comps='T', cuts=cuts)
        
    hitmap = P.to_weights(aman, signal='dsT', comps='T')[0][0]
    result['hitmap'] = hitmap
    
    print(f'{time()-_t0:.1f} [sec] for map making')
    return result

def map_make_det_split(aman, res=0.3*coords.DEG, full_sky=True, 
                       cuts=None, det_mask_sp=None,
                       det_weights_dsT=None, det_weights_demod=None,
                       save=True, 
                       sp_map_root_dir=None,
                       bandpass=None,
                       obs_id=None,
                       split_definition_name=None,                       
                       sp=None,
                       address=None,
                       do_plot=True
                      ):
    print(f'making maps for {split_definition_name}, {sp}')
    assert det_mask_sp is not None
    _aman = aman.restrict('dets', aman.dets.vals[det_mask_sp], in_place=False)
        
    if cuts is None:        
        cuts = _aman.flags.exclude        
    if det_weights_dsT is None:
        det_weights_dsT = 2*_aman.inv_var
    if det_weights_demod is None:
        det_weights_demod = _aman.inv_var
        
    _t0 = time()

    if full_sky:
        shape, wcs = enmap.fullsky_geometry(res=res, proj='car')
        P = coords.P.for_tod(_aman, geom=(shape, wcs), comps='QU', cuts=cuts, hwp=True, threads=False)
    else:
        wcs = coords.get_wcs_kernel('car', 0, 0, res)
        P = coords.P.for_tod(_aman, wcs_kernel=wcs, comps='QU', cuts=cuts, hwp=True, threads=False)
    
    result = demod_mm.make_map(_aman, P=P, 
                               det_weights=det_weights_dsT, 
                               det_weights_demod=det_weights_demod)
    
    if full_sky:
        P = coords.P.for_tod(_aman, geom=(shape, wcs), comps='T', cuts=cuts)
    else:
        P = coords.P.for_tod(_aman, wcs_kernel=wcs, comps='T', cuts=cuts)
        
    hitmap = P.to_weights(_aman, signal='dsT', comps='T')[0][0]
    result['hitmap'] = hitmap
    print(f'{time()-_t0:.1f} [sec] for map making')
    if save:        
        assert sp_map_root_dir is not None
        assert split_definition_name is not None
        assert sp is not None
        save_dir_sp = os.path.join(sp_map_root_dir, split_definition_name, sp)
        os.makedirs(save_dir_sp, exist_ok=True)
        save_result(result, save_dir_sp, bandpass=bandpass, obs_id=obs_id, address=address, do_plot=do_plot)
        print(f'{time()-_t0:.1f} [sec] for saving')
        del result
        del _aman
        return
    else:
        return result

def map_make_scan_split(aman, res=0.3*coords.DEG, full_sky=True, 
                       cuts=None, scan_direction=None,
                       det_weights_dsT=None, det_weights_demod=None,
                       save=True, 
                       sp_map_root_dir=None,
                       bandpass=None,
                        obs_id=None,
                       split_definition_name=None,
                       sp=None,
                       address=None,
                       do_plot=True
                      ):
    _aman = aman.copy()
    if cuts is None:        
        assert scan_direction in ['right', 'left']
        if scan_direction == 'right':
            cuts = _aman.flags.reduce(flags=['exclude', 'left_scan'], wrap=False, remove_reduced=False, )            
        elif scan_direction == 'left':
            cuts = _aman.flags.reduce(flags=['exclude', 'right_scan'], wrap=False, remove_reduced=False, )
            
    if det_weights_dsT is None:
        det_weights_dsT = 2*_aman.inv_var        
    if det_weights_demod is None:
        det_weights_demod = _aman.inv_var                 
    
    if full_sky:
        shape, wcs = enmap.fullsky_geometry(res=res, proj='car')
        P = coords.P.for_tod(_aman, geom=(shape, wcs), comps='QU', cuts=cuts, hwp=True, threads=False)
    else:
        wcs = coords.get_wcs_kernel('car', 0, 0, res)
        P = coords.P.for_tod(_aman, wcs_kernel=wcs, comps='QU', cuts=cuts, hwp=True, threads=False)
        
    result = demod_mm.make_map(_aman, P=P, 
                               det_weights=det_weights_dsT, det_weights_demod=det_weights_demod)
    
    if full_sky:
        P = coords.P.for_tod(_aman, geom=(shape, wcs), comps='T', cuts=cuts)
    else:
        P = coords.P.for_tod(_aman, wcs_kernel=wcs, comps='T', cuts=cuts)
        
    hitmap = P.to_weights(_aman, signal='dsT', comps='T')[0][0]
    result['hitmap'] = hitmap
    if save:
        assert sp_map_root_dir is not None
        assert split_definition_name is not None
        assert sp is not None
        save_dir_sp = os.path.join(sp_map_root_dir, split_definition_name, sp)
        os.makedirs(save_dir_sp, exist_ok=True)
        save_result(result, save_dir_sp, bandpass=bandpass, obs_id=obs_id, address=address, do_plot=do_plot)
        del result
        del _aman
        return
    else:
        return result

def save_aman(aman, map_root_dir, bandpass, obs_id):
    aman_dir = os.path.join(map_root_dir, bandpass, 'aman')
    os.makedirs(aman_dir, exist_ok=True)
    aman.save(os.path.join(aman_dir, f'{obs_id}.hdf'), overwrite=True)
    return

def save_result(result, map_root_dir, bandpass, obs_id, address=None, do_plot=True, save_all=True):
    map = result['map']
    weighted_map = result['weighted_map']
    weight = result['weight']
    weight_diag = enmap.enmap([weight[0][0], weight[1][1], weight[2][2]])
    hitmap = result['hitmap']

    map_dir = os.path.join(map_root_dir, bandpass, 'map')
    weighted_map_dir = os.path.join(map_root_dir, bandpass, 'weighted_map')
    weight_dir = os.path.join(map_root_dir, bandpass, 'weight')
    hitmap_dir = os.path.join(map_root_dir, bandpass, 'hitmap')
    plots_dir = os.path.join(map_root_dir, bandpass, 'plots')

    for _dir in [map_dir, weighted_map_dir, weight_dir, hitmap_dir, plots_dir]:
        os.makedirs(_dir, exist_ok=True)

    # hdf
    hdf_name = f'{obs_id}.hdf'
    enmap.write_hdf(os.path.join(weighted_map_dir, hdf_name), weighted_map, address=address)

    if save_all:
        enmap.write_hdf(os.path.join(map_dir, hdf_name), map, address=address)
        enmap.write_hdf(os.path.join(weight_dir, hdf_name), weight_diag, address=address)
        enmap.write_hdf(os.path.join(hitmap_dir, hdf_name), hitmap, address=address)
    #plots
    if do_plot:
        map[map==0] = np.nan
        plotT = enplot.plot(map[0]*1.1*1e6, ticks=30, colorbar=True)
        plot_nameT = f'plot_{obs_id}_T'
        enplot.write(os.path.join(plots_dir, plot_nameT), plotT)

        plotQ = enplot.plot(map[1]*1.1*1e6, ticks=30, colorbar=True)
        plot_nameQ = f'plot_{obs_id}_Q'
        enplot.write(os.path.join(plots_dir, plot_nameQ), plotQ)

        plotU = enplot.plot(map[2]*1.1*1e6, ticks=30, colorbar=True)
        plot_nameU = f'plot_{obs_id}_U'
        enplot.write(os.path.join(plots_dir, plot_nameU), plotU)
    return

def initiate_db(output_dir, db_name):
    os.makedirs(output_dir, exist_ok=True)
    db_filename = os.path.join(output_dir, db_name)

    if os.path.exists(db_filename):
        db = core.metadata.ManifestDb(db_filename)
    else:
        scheme = core.metadata.ManifestScheme()
        scheme.add_exact_match('obs_id_bandpass')

        scheme.add_data_field('obs_id')
        scheme.add_data_field('bandpass')
        scheme.add_data_field('ctime')
        
        scheme.add_data_field('aman_fp')
        scheme.add_data_field('map_fp')
        scheme.add_data_field('weight_fp')
        scheme.add_data_field('weighted_map_fp')
        scheme.add_data_field('hitmap_fp')
        
        scheme.add_data_field('scanspeed')
        scheme.add_data_field('az_min')
        scheme.add_data_field('az_max')
        scheme.add_data_field('fhwp')
        scheme.add_data_field('pwv')
        scheme.add_data_field('dpwv')
        scheme.add_data_field('az_center')
        scheme.add_data_field('el_center')
        scheme.add_data_field('roll_center')
        
        scheme.add_data_field('fk_demod')
        scheme.add_data_field('alpha_demod')
        scheme.add_data_field('fk_signal')
        scheme.add_data_field('alpha_signal')
        
        scheme.add_data_field('ra_center')
        scheme.add_data_field('dec_center')
        scheme.add_data_field('ra_width')
        scheme.add_data_field('dec_width')
        
        scheme.add_data_field('duration')
        scheme.add_data_field('effective_duration')
        scheme.add_data_field('NEP_det_median_aW')
        scheme.add_data_field('NEP_array_aW')
        scheme.add_data_field('num_dets_perfect')
        scheme.add_data_field('num_dets_streamed')
        scheme.add_data_field('num_dets_valid_biasstep')
        scheme.add_data_field('num_dets_wg_available')
        scheme.add_data_field('num_dets_valid_fp')
        scheme.add_data_field('num_dets_valid_wnl')
        scheme.add_data_field('num_dets_valid_wafers')
        scheme.add_data_field('num_dets_second')
        scheme.add_data_field('num_dets_good_leakage_estimation')
        scheme.add_data_field('num_dets_final')

        db = core.metadata.ManifestDb(db_filename, scheme=scheme)
    return db

def get_entry_dict(aman, result, obs_id, bandpass, num_dets_dict, map_root_dir):
    entry_dict = {}
    entry_dict['obs_id_bandpass'] = f'{obs_id}_{bandpass}'
    entry_dict['obs_id'] = obs_id
    entry_dict['bandpass'] = bandpass
    entry_dict['ctime'] = int(aman.ctime)
    

    hdf_name = f'{obs_id}.hdf'
    entry_dict['aman_fp'] = os.path.join(map_root_dir, bandpass, 'aman', hdf_name)
    entry_dict['map_fp'] = os.path.join(map_root_dir, bandpass, 'map', hdf_name)
    entry_dict['weight_fp'] = os.path.join(map_root_dir, bandpass, 'weight', hdf_name)
    entry_dict['weighted_map_fp'] = os.path.join(map_root_dir, bandpass, 'weighted_map', hdf_name)
    entry_dict['hitmap_fp'] = os.path.join(map_root_dir, bandpass, 'hitmap', hdf_name)

    entry_dict['scanspeed'] = float(aman.scanspeed)
    entry_dict['az_min'] = float(aman.az_min)
    entry_dict['az_max'] = float(aman.az_max)
    
    entry_dict['fhwp'] = float(aman.fhwp)
    entry_dict['pwv'] = float(aman.pwv)
    entry_dict['dpwv'] = float(aman.dpwv)
    
    entry_dict['az_center'] = float(aman.obs_info['az_center'])
    entry_dict['el_center'] = float(aman.obs_info['el_center'])
    entry_dict['roll_center'] = float(aman.obs_info['roll_center'])
    
    entry_dict['fk_signal'] = float(np.median(aman.fk_signal, axis=0))
    entry_dict['alpha_signal'] = float(np.median(aman.alpha_signal, axis=0))
    entry_dict['fk_demod'] = float(np.median((aman.fk_demodQ + aman.fk_demodU)/2, axis=0))
    entry_dict['alpha_demod'] = float(np.median((aman.alpha_demodQ + aman.alpha_demodU)/2, axis=0))
    
    hitmap = result['hitmap']
    hitbit = hitmap != 0.
    decs, ras = hitmap.posmap()
    dec_center = circular_mean(decs[hitbit])
    ra_center = circular_mean(ras[hitbit])
    dec_width = circular_std(decs[hitbit])
    ra_width = circular_std(ras[hitbit])

    entry_dict['ra_center'] = ra_center
    entry_dict['dec_center'] = dec_center
    entry_dict['ra_width'] = ra_width
    entry_dict['dec_width'] = dec_width
    
    entry_dict['duration'] = aman.obs_info.duration
    entry_dict['effective_duration'] = aman.obs_info.duration * np.mean(~aman.flags.exclude.mask())
    
    entry_dict['NEP_det_median_aW'] = np.median((aman.wn_demodQ+aman.wn_demodU)/2.)*1e6
    entry_dict['NEP_array_aW'] = np.sqrt( 1 / np.sum(1/(((aman.wn_demodQ+aman.wn_demodU)/2.*1e6)**2)))
    
    for key, val in num_dets_dict.items():
        entry_dict[key] = val
    return entry_dict


# for map making with splitting focal plane
# split_definition_names_f090 = ['split_left_right',                                    
#                            'split_lower_higher',                                    
#                            'split_polA_polB',]
split_definition_names_f090 = ['split_inner_outer',
                           'split_left_right',
                           'split_lower_higher',
                           'split_horizontal_vertical',
                           'split_polA_polB',]

split_definition_names_f150 = ['split_inner_outer',
                           'split_left_right',
                           'split_lower_higher',
                           'split_horizontal_vertical',
                           'split_polA_polB',]

split_definition_names_f090 +=   ['split_random00A_random00B',
                                 'split_random01A_random01B',
                                 'split_random02A_random02B',
                                 'split_random03A_random03B',
                                 'split_random04A_random04B',
                                 'split_random05A_random05B',
                                 # 'split_random06A_random06B',
                                 # 'split_random07A_random07B',
                                 # 'split_random08A_random08B',
                                 # 'split_random09A_random09B',
                                 # 'split_random10A_random10B',
                                 # 'split_random11A_random11B',
                                 # 'split_random12A_random12B',
                                 # 'split_random13A_random13B',
                                 # 'split_random14A_random14B',
                                 # 'split_random15A_random15B',
                                 ]

split_definition_names_f150 +=   ['split_random00A_random00B',
                                 'split_random01A_random01B',
                                 'split_random02A_random02B',
                                 'split_random03A_random03B',
                                 'split_random04A_random04B',
                                 'split_random05A_random05B',
                                 # 'split_random06A_random06B',
                                 # 'split_random07A_random07B',
                                 # 'split_random08A_random08B',
                                 # 'split_random09A_random09B',
                                 # 'split_random10A_random10B',
                                 # 'split_random11A_random11B',
                                 # 'split_random12A_random12B',
                                 # 'split_random13A_random13B',
                                 # 'split_random14A_random14B',
                                 # 'split_random15A_random15B',
                                 ]


def wrap_split_definitions(aman, bandpass):
    satp = aman.obs_info.obs_id.split('_')[-2]
    split_definition_aman = core.AxisManager.load(f'/scratch/gpfs/SIMONSOBS/users/js7893/common_scripts/detector_database/{satp}/split_definition_aman_{bandpass}.hdf')
    if bandpass == 'f090':
        split_definition_names = split_definition_names_f090
    elif bandpass == 'f150':
        split_definition_names = split_definition_names_f150
    
    for split_definition_name in split_definition_names:
        for sp in ['sp0', 'sp1']:
            aman.wrap_new(f'{split_definition_name}_{sp}', ('dets', ), dtype=bool)
    
    for di, det in enumerate(aman.dets.vals):
        di_split_definition_aman = np.where(split_definition_aman.dets.vals == det)[0]
        if len(di_split_definition_aman)==1:
            di_split_definition_aman = di_split_definition_aman[0]        
            for split_definition_name in split_definition_names:
                for sp in ['sp0', 'sp1']:
                    aman[f'{split_definition_name}_{sp}'][di] = split_definition_aman[f'{split_definition_name}_{sp}'][di_split_definition_aman]
    return

def process_all_wafer_before_mapmake(ctx_file, obs_id, bandpass, debug=False, debug_style='one_ws', ws=None,):
    t0 = time()
    aman, num_dets_dict = load_data(ctx_file=ctx_file, obs_id=obs_id, bandpass=bandpass, debug=debug, debug_style=debug_style,
                                  do_calibration=True, 
                                  do_relcal=True,
                                  do_get_wn_signal=True,
                                  do_first_data_cut=True,
                                  remove_useless=True,
                                  ws = ws,
                                 )
    # flag buffer for filter
    flag_filt_buffer(aman)

    print('apply timeconstant')
    tod_ops.apodize_cosine(aman, apodize_samps=200*(5+2)) #2sec
    aman.signal = tod_ops.fourier_filter(aman, filters.timeconst_filter(invert=True, timeconst=aman.det_cal.tau_eff), detrend=None)

    print('get and subtract hwpss')
    hwp.get_hwpss(aman)
    hwp.subtract_hwpss(aman, in_place=True)

    print(f'calculate psd for signal for {aman.dets.count} detectors')
    freqs_raw, Pxx_signal = calc_wrap_psd(aman, signal_name='signal', merge=False, merge_wn=False, nperseg=200*50)
    freqs_raw_binned = log_binning(freqs_raw, base=1.05)
    Pxx_signal_binned = log_binning(Pxx_signal, base=1.05)
    nusamps_raw_binned = core.OffsetAxis('nusamps_raw_binned', len(freqs_raw_binned))
    _ = aman.wrap('freqs_raw_binned', freqs_raw_binned, [(0, nusamps_raw_binned, )])
    _ = aman.wrap(f'Pxx_signal_binned', Pxx_signal_binned, [(0, 'dets'), (1, 'nusamps_raw_binned')])
    
    print('fit signal noise parameters')
    do_signal_noise_fit(aman)
    if bandpass=='f090':
        fk_signal_max = 6.0
    elif bandpass=='f150':
        fk_signal_max = 7.0
    remove_signal_fk_outliers(aman, fk_signal_max)
    
    print('remove wafers with too small number of surviving detectors')
    remove_dropout_wafers(aman)
    num_dets_dict['num_dets_second'] = aman.dets.count
    
    print('demodulation and down sampling')
    tod_ops.detrend_tod(aman, method='median')
    tod_ops.apodize_cosine(aman, apodize_samps=200*(5+2+5)) #5sec
    bman = down_sample_aman(aman, ds_factor=ds_factor)
    demod_tod_and_wrap_to_ds(aman=aman, bman=bman, ds_factor=ds_factor)
    aman = bman
    aman.move('signal', None)
    print('cut edge and detrend')
    aman.restrict('samps', (aman.samps.offset + 20, aman.samps.count - 20))

    # ADD PLOT
    mws = (aman.det_info.wafer_slot == plot_ws)
    f, pxx = welch(aman.demodQ[mws][::20], fs=4., nperseg = 2**18//50, noverlap=0)
    avgtod = np.mean(aman.demodQ[mws], axis=0)
    favg, pxxavg = welch(avgtod, fs=4., nperseg = 2**18//50, noverlap=0)
    plt.figure()
    m = f < 1.9
    plt.loglog(f[m], pxx[:,m].T, 'C0', alpha=0.1)
    plt.loglog(f[m], np.mean(pxx, axis=0)[m], 'C1')
    plt.loglog(favg[m], pxxavg[m], 'k')
    plt.title(f'{aman.obs_info.obs_id}\nPost demod, no filter')
    plt.savefig(os.path.join(plot_path, 'filter_impact_plots',f'{aman.obs_info.obs_id}_{plot_ws}_post_demod_no_filt.png'))

    tod_ops.detrend_tod(aman, method='median', signal_name='dsT')
    tod_ops.detrend_tod(aman, method='median', signal_name='demodQ')
    tod_ops.detrend_tod(aman, method='median', signal_name='demodU')

    print('apply ptp cuts')
    apply_ptp_data_cuts(aman, ptp_signal_name='dsT')
    
    print('get turnaround flags')
    turnarounds_flags, left_scan_flags, right_scan_flags = get_turnaround_flags(aman, t_buffer=4., kernel_size=400//ds_factor,
                                                                            merge=False, merge_lr=False, merge_subscans=False)
    turnarounds_narrow_flags, _, _ = get_turnaround_flags(aman, t_buffer=0.5, kernel_size=400//ds_factor,
                                                            merge=False, merge_lr=False, merge_subscans=False)
    aman.flags.wrap('turnarounds', turnarounds_flags, [(0, 'dets'), (1, 'samps')])
    aman.flags.wrap('turnarounds_narrow', turnarounds_narrow_flags, [(0, 'dets'), (1, 'samps')])
    aman.flags.wrap('left_scan', left_scan_flags, [(0, 'dets'), (1, 'samps')])
    aman.flags.wrap('right_scan', right_scan_flags, [(0, 'dets'), (1, 'samps')])
    
    print('wrap_subscan_idx_info')
    wrap_subscan_idx_info(aman)
    
    print('apply wnl based data cut')
    apply_wnl_data_cuts(aman, wn_name='wn_signal')

    print('get bad subscan')
    if bandpass=='f090':
        Tptp_subscan_pW_threshold = 0.25
    elif bandpass=='f150':
        Tptp_subscan_pW_threshold = 0.5
    get_bad_subscan_flags(aman, Tptp_subscan_pW_threshold=Tptp_subscan_pW_threshold)
    
    print('get glitches')
    get_glitches(aman)

    print('take union for exclude masks')
    _ = aman.flags.reduce(flags=['turnarounds', 'bad_subscan', 'glitches', 'buffer_scan'], method='union', 
                              wrap=True, new_flag='exclude', remove_reduced=False)
    
    print('t2p subtraction')
    tod_ops.detrend_tod(aman, method='linear', signal_name='dsT')
    tod_ops.detrend_tod(aman, method='linear', signal_name='demodQ')
    tod_ops.detrend_tod(aman, method='linear', signal_name='demodU')
    tod_ops.apodize_cosine(aman, signal_name='demodQ', apodize_samps=int(200*(5)/ds_factor))
    tod_ops.apodize_cosine(aman, signal_name='demodU', apodize_samps=int(200*(5)/ds_factor))
    tod_ops.apodize_cosine(aman, signal_name='dsT', apodize_samps=int(200*(5)/ds_factor))
    subtract_t2p(aman)

    print('remove weird leakage estimation from fit statistics')
    redchi2s_t2p = aman.t2p_stats.redchi2sQ + aman.t2p_stats.redchi2sU
    is_good_leakage_estimation = (0.1 < redchi2s_t2p) & (redchi2s_t2p < 3)
    aman.restrict('dets', aman.dets.vals[is_good_leakage_estimation])    
    num_dets_dict['num_dets_good_leakage_estimation'] = aman.dets.count
    print(f'dets count: {aman.dets.count}')

    # ADD PLOT
    mws = (aman.det_info.wafer_slot == plot_ws)
    f, pxx = welch(aman.demodQ[mws][::20], fs=4., nperseg = 2**18//50, noverlap=0)
    avgtod = np.mean(aman.demodQ[mws], axis=0)
    favg, pxxavg = welch(avgtod, fs=4., nperseg = 2**18//50, noverlap=0)
    plt.figure()
    m = f < 1.9
    plt.loglog(f[m], pxx[:,m].T, 'C0', alpha=0.1)
    plt.loglog(f[m], np.mean(pxx, axis=0)[m], 'C1')
    plt.loglog(favg[m], pxxavg[m], 'k')
    plt.title(f'{aman.obs_info.obs_id}\nPost T2P Subtraction')
    plt.savefig(os.path.join(plot_path, 'filter_impact_plots',f'{aman.obs_info.obs_id}_{plot_ws}_T2P.png'))

    print('subtract scan-ss for all')
    subtract_sss_lr_all(aman, bins=50, bad_flags=aman.flags.bad_subscan, apodize_edges_sec=100, apodize_flags_sec=1,
                        method_dsT='interpolate', max_mode_dsT=None, # method_dsT='interpolate' until 2025-11-20
                        method_demod='fit', max_mode_demod=5)

    # ADD PLOT
    mws = (aman.det_info.wafer_slot == plot_ws)
    f, pxx = welch(aman.demodQ[mws][::20], fs=4., nperseg = 2**18//50, noverlap=0)
    avgtod = np.mean(aman.demodQ[mws], axis=0)
    favg, pxxavg = welch(avgtod, fs=4., nperseg = 2**18//50, noverlap=0)
    plt.figure()
    m = f < 1.9
    plt.loglog(f[m], pxx[:,m].T, 'C0', alpha=0.1)
    plt.loglog(f[m], np.mean(pxx, axis=0)[m], 'C1')
    plt.loglog(favg[m], pxxavg[m], 'k')
    plt.title(f'{aman.obs_info.obs_id}\nPost AzSS Subtraction')
    plt.savefig(os.path.join(plot_path, 'filter_impact_plots',f'{aman.obs_info.obs_id}_{plot_ws}_after_azss.png'))
    
    print('remove wafers with too small number of surviving detectors')
    remove_dropout_wafers(aman)
    if aman.dets.count < 50:
        raise ValueError(f'Only {aman.dets.count} in total are surviving. Killing this process for this obs_id')

    print('mild counter 1 over f filter for dsT')
    apodizer = get_apodizer_turnarounds(aman)
    aman.dsT *= apodizer
    fk = 1.0
    n = 1.5
    hpf = filters.get_hpf({'type': 'sine2', 'cutoff': 0.05, 'trans_width': 0.08})
    c1f_filt = filters.counter_1_over_f(fk, n)
    aman['dsT'] = tod_ops.fourier_filter(aman, c1f_filt*hpf, signal_name='dsT')
    
    print('subtract common mode for dsT')
    subtract_common_mode_ws_pca(aman, signal_names=['dsT'], n_modes=1)

    return aman, num_dets_dict

def process_all_wafer_to_mapmake(aman, num_dets_dict, obs_id, bandpass,  
                      res=0.3*coords.DEG, full_sky=True, save_map=True, save_db=True, do_map_make_for_split=True,
                      map_root_dir=None, sp_map_root_dir=None):
    
    print('subscan baseline subtraction for demod')
    subscan_polyfilter(aman, degree=1, signal_name='demodQ', mask=aman.flags.exclude)
    subscan_polyfilter(aman, degree=1, signal_name='demodU', mask=aman.flags.exclude)
    
    print('apodize turnarounds for demod')
    apodizer = get_apodizer_turnarounds(aman)
    aman.demodQ *= apodizer
    aman.demodU *= apodizer

    # ADD PLOT
    mws = (aman.det_info.wafer_slot == plot_ws)
    f, pxx = welch(aman.demodQ[mws][::20], fs=4., nperseg = 2**18//50, noverlap=0)
    avgtod = np.mean(aman.demodQ[mws], axis=0)
    favg, pxxavg = welch(avgtod, fs=4., nperseg = 2**18//50, noverlap=0)
    plt.figure()
    m = f < 1.9
    plt.loglog(f[m], pxx[:,m].T, 'C0', alpha=0.1)
    plt.loglog(f[m], np.mean(pxx, axis=0)[m], 'C1')
    plt.loglog(favg[m], pxxavg[m], 'k')
    plt.title(f'{aman.obs_info.obs_id}\nPost subscan poly')
    plt.savefig(os.path.join(plot_path, 'filter_impact_plots',f'{aman.obs_info.obs_id}_{plot_ws}_post_subscan_poly.png'))

    raise BreakError
    
    print('subtract overall mean')
    subtract_overall_mean(aman)
    
    print('get inverse variance')
    get_inv_var(aman)
    
    num_dets_dict['num_dets_final'] = aman.dets.count
    for key, val in num_dets_dict.items():
        print(f'{key}: {val}')
    
    print('map making')
    result = map_make(aman, res=res, full_sky=full_sky)
    if save_map:
        print('saving map')
        save_result(result, map_root_dir, bandpass=bandpass, obs_id=obs_id)
    
    
    if do_map_make_for_split:
        print('making maps for focal plane split')
        wrap_split_definitions(aman, bandpass)
        if bandpass == 'f090':
            split_definition_names = split_definition_names_f090
        elif bandpass == 'f150':
            split_definition_names = split_definition_names_f150
            
        futures = []
        with ThreadPoolExecutor(max_workers=11) as executor:
            for split_definition_name in split_definition_names:                        
                for sp in ['sp0', 'sp1']:
                    futures.append(executor.submit(
                                                   map_make_det_split,
                                                   aman=aman, 
                                                   res=res, 
                                                   full_sky=full_sky,
                                                   det_mask_sp = aman[f'{split_definition_name}_{sp}'],
                                                   save=True,
                                                   sp_map_root_dir=sp_map_root_dir,
                                                   bandpass=bandpass,
                                                   obs_id=obs_id,
                                                   split_definition_name=split_definition_name,
                                                   sp=sp,)
                                  )
                                        
            print('making maps for scan direction split')
            split_definition_name = 'split_leftscan_rightscan'
            for sp, scan_direction in zip(['sp0', 'sp1'], ['left', 'right']):                
                futures.append(executor.submit(
                                                map_make_scan_split,
                                                aman=aman,
                                                res=res, 
                                                full_sky=full_sky, 
                                                scan_direction=scan_direction,
                                                save=True,
                                                sp_map_root_dir=sp_map_root_dir,
                                                bandpass=bandpass,
                                                obs_id=obs_id,
                                                split_definition_name=split_definition_name,
                                                sp=sp,)
                              )
        for future in futures:
            _ = future.result()
                            
    if save_db:
        print('saving db')
        db_file = os.path.join(map_root_dir, bandpass, 'map_summary.sqlite')
        db = initiate_db(output_dir = os.path.join(map_root_dir, bandpass),
                         db_name='map_summary.sqlite')
        entry_dict = get_entry_dict(aman=aman, result=result, 
                                    obs_id=obs_id, bandpass=bandpass, 
                                    num_dets_dict=num_dets_dict,
                                    map_root_dir=map_root_dir)
        
        print('reduce aman')
        remove_heavy_fields(aman)
        
        print('save aman')
        save_aman(aman=aman, map_root_dir=map_root_dir, bandpass=bandpass, obs_id=obs_id)
        
        # overwrite if exist
        obs_id_bandpass = f'{obs_id}_{bandpass}'
        with sqlite3.connect(db_file) as conn:
            cur = conn.cursor()
            cur = cur.execute(f'select * from map where obs_id_bandpass="{obs_id_bandpass}"')
            resultset = ResultSet.from_cursor(cur)
        if len(resultset)==1:
            with sqlite3.connect(db_file) as conn:
                print('removing existing entry')
                cur = conn.cursor()
                cur.execute('delete from map where obs_id_bandpass = ?', (obs_id_bandpass,))
                conn.commit()
                
        db.add_entry(entry_dict)
    else:
        pass
    print('-'*30)
    return aman, num_dets_dict, result

    
def ws_to_bit_value(ws_list):
    max_ws = 6
    bit_value = ['0'] * (max_ws + 1)
    for ws in ws_list:
        index = int(ws[2])
        bit_value[index] = '1'
    return ''.join(bit_value)

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
        
    parser.add_argument(
        'bandpass',
        help='bandpass to be processed')
    
    parser.add_argument(
        'obs_id',
        help='obs_id to be processed')
    
    parser.add_argument(
        '--ws',
        help='ws to be processed')
    
    return parser

def main(args):
    bandpass = args.bandpass
    obs_id = args.obs_id
    ws = args.ws
    satp = obs_id.split('_')[-2]
    
    res = 0.3 * coords.DEG
    
    debug = False
    debug_style='one_ws'
    if ws == 'debug':
        ws = ['ws0', 'ws1']
    try_except = False
    full_sky = True
    ctx_file = f'/scratch/gpfs/SIMONSOBS/so/tracked/metadata/{satp}/contexts/use_this_local.yaml'
    root_dir = os.path.dirname(os.path.abspath(__file__))
    map_root_dir = os.path.join(root_dir, 'each_obs_result')
    sp_map_root_dir = os.path.join(root_dir, 'each_obs_result_fp_splits')
    os.makedirs(os.path.join(map_root_dir, bandpass), exist_ok=True)    
    os.makedirs(os.path.join(sp_map_root_dir, bandpass), exist_ok=True)
    os.makedirs(os.path.join(map_root_dir, bandpass, 'plots'), exist_ok=True)
    do_map_make_for_split = False
    
    print('-'*50)
    print(f'processing: {obs_id}, {bandpass}')

    aman, num_dets_dict = process_all_wafer_before_mapmake(ctx_file=ctx_file, 
                                                        obs_id=obs_id, bandpass=bandpass, ws=ws,
                                                        debug=debug, debug_style=debug_style,)

    print('calc PSDs for demodQ, demodU')
    get_demod_Pxxs(aman)
    print('fit demod noise parameters')
    do_demod_noise_fit(aman)
    print('apply c1f for demod')
    apply_c1f_filters_demod(aman)

    # ADD PLOT
    mws = (aman.det_info.wafer_slot == plot_ws)
    f, pxx = welch(aman.demodQ[mws][::20], fs=4., nperseg = 2**18//50, noverlap=0)
    avgtod = np.mean(aman.demodQ[mws], axis=0)
    favg, pxxavg = welch(avgtod, fs=4., nperseg = 2**18//50, noverlap=0)
    plt.figure()
    m = f < 1.9
    plt.loglog(f[m], pxx[:,m].T, 'C0', alpha=0.1)
    plt.loglog(f[m], np.mean(pxx, axis=0)[m], 'C1')
    plt.loglog(favg[m], pxxavg[m], 'k')
    plt.title(f'{aman.obs_info.obs_id}\nPost counter 1/f')
    plt.savefig(os.path.join(plot_path, 'filter_impact_plots',f'{aman.obs_info.obs_id}_{plot_ws}_post_c1f.png'))

    print('apply scan frequency cut')
    scan_freq_cut(aman)

    # ADD PLOT
    mws = (aman.det_info.wafer_slot == plot_ws)
    f, pxx = welch(aman.demodQ[mws][::20], fs=4., nperseg = 2**18//50, noverlap=0)
    avgtod = np.mean(aman.demodQ[mws], axis=0)
    favg, pxxavg = welch(avgtod, fs=4., nperseg = 2**18//50, noverlap=0)
    plt.figure()
    m = f < 1.9
    plt.loglog(f[m], pxx[:,m].T, 'C0', alpha=0.1)
    plt.loglog(f[m], np.mean(pxx, axis=0)[m], 'C1')
    plt.loglog(favg[m], pxxavg[m], 'k')
    plt.title(f'{aman.obs_info.obs_id}\nPost scan freq cut')
    plt.savefig(os.path.join(plot_path, 'filter_impact_plots',f'{aman.obs_info.obs_id}_{plot_ws}_post_scan_freq_cut.png'))

    print('mapmake_for_c1f and common sub')
    process_all_wafer_to_mapmake(aman, num_dets_dict, obs_id, bandpass,
                                      res=res, full_sky=full_sky, save_map=True, save_db=True,
                                                        do_map_make_for_split=do_map_make_for_split,
                                                        map_root_dir=map_root_dir,
                                                        sp_map_root_dir=sp_map_root_dir,
                                                       )
    make_gallery(os.path.join(map_root_dir, bandpass, 'plots'), '_gallery.html')
    print(f'done processing: {obs_id}, {bandpass}')
    
    return


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)

# Example:        
# python /global/cfs/cdirs/sobs/users/ttera/cmb_ttera_2411_4/do_process.py f150 obs_1714176018_satp1_1111111
