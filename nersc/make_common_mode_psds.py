import numpy as np
import h5py
import time
import yaml
import os
import matplotlib.pyplot as plt

from sotodlib import core, preprocess
from sotodlib.coords import demod
from sotodlib.preprocess.pcore import update_full_aman
from sotodlib.site_pipeline import preprocess_tod as pt
from sotodlib.core.flagman import count_cuts, has_all_cut, has_any_cuts
from sotodlib import tod_ops
from pixell import utils, enmap, enplot
from scipy.signal import welch

def rotate_demodQU(aman, offset = 0):
    demodC = ((aman.demodQ + 1j*aman.demodU).T * np.exp(-2j*aman.focal_plane.gamma + 1j*np.deg2rad(offset))).T
    aman.demodQ = demodC.real
    aman.demodU = demodC.imag
    del demodC

def main():
    configs = '/global/homes/m/msilvafe/so_home/preprocess/configs/20240529_satp1_cmb_preproc_cfg.yaml'
    configs = yaml.safe_load(open(configs, "r"))
    ctx = core.Context(configs['context_file'])
    obslist = ctx.obsdb.query(f'type=="obs" and subtype=="cmb" and start_time>1711337000'
                              + f' and start_time<1722180119')
    savedir = '/global/cfs/cdirs/sobs/www/users/msilvafe/20240807_tod_common_mode'
    for ob in obslist:
        obsid = ob['obs_id']
        print(obsid)
        dets={'wafer_slot':'ws0', 'wafer.bandpass':'f150'}
        configs = '/global/homes/m/msilvafe/so_home/preprocess/configs/20240529_satp1_cmb_preproc_cfg.yaml'
        configs = yaml.safe_load(open(configs, "r"))
        ctx = core.Context(configs['context_file'])
        try:
            aman = ctx.get_obs(obsid, dets=dets)
        except:
            print('Failed to load.')
            continue

        pipe = preprocess.Pipeline(configs["process_pipe"])
        proc_aman = core.AxisManager( aman.dets, aman.samps)
        full = core.AxisManager( aman.dets, aman.samps)
        print('Running starting processing')
        fail = False
        for i, process in enumerate(pipe[:14]):
            try:
                process.process(aman, proc_aman)
                process.calc_and_save(aman, proc_aman)
                update_full_aman(proc_aman, full, True)
                process.select(aman, proc_aman)
                proc_aman.restrict('dets', aman.dets.vals)
                if aman.dets.count < 60:
                    print(f'Stopping because too few detectors on step {process.name}')
                    continue
            except:
                fail = True
                continue
        if fail:
            continue
        if aman.dets.count < 60:
            print(f'Stopping because too few detectors on step {process.name}')
            continue
        print('Demodulating')
        for i, process in enumerate(pipe[22:24]):
            process.process(aman, proc_aman)
            process.calc_and_save(aman, proc_aman)
            update_full_aman(proc_aman, full, True)
            process.select(aman, proc_aman)
            proc_aman.restrict('dets', aman.dets.vals)
        print('Rotating and making plots')
        rotate_demodQU(aman)
        
        medT = np.median(aman.dsT, axis=0)
        medQ = np.median(aman.demodQ, axis=0)
        medU = np.median(aman.demodU, axis=0)
        f, pxxT = welch(aman.dsT, fs=200, nperseg = 131072)
        _, mpxxT = welch(medT, fs=200, nperseg = 131072)
        _, pxxQ = welch(aman.demodQ, fs=200, nperseg = 131072)
        _, mpxxQ = welch(medQ, fs=200, nperseg = 131072)
        _, pxxU = welch(aman.demodQ, fs=200, nperseg = 131072)
        _, mpxxU = welch(medU, fs=200, nperseg = 131072)
        
        eix = np.argmin(np.abs(f-1.9))
        fig, axes = plt.subplots(1,2,figsize=(8,4))
        ax = axes.flatten()
        ax[0].loglog(f[:eix], pxxT[::20, :eix].T, 'C0', alpha=0.05)
        ax[0].loglog(f[:eix], mpxxT[:eix], 'C0')
        ax[1].loglog(f[:eix], pxxQ[::20, :eix].T, 'C1', alpha=0.05)
        ax[1].loglog(f[:eix], mpxxQ[:eix], 'C1')
        ax[1].loglog(f[:eix], pxxU[::20, :eix].T, 'C2', alpha=0.05)
        ax[1].loglog(f[:eix], mpxxU[:eix], 'C2')

        scan_speed = np.median(np.abs(np.diff(np.rad2deg(aman.boresight.az))/np.diff(aman.timestamps)))
        scan_f = 1/(2*40/scan_speed)
        print(scan_f)

        ax[0].axvline(scan_f, c='k', ls=':', alpha=0.5)
        ax[1].axvline(scan_f, c='k', ls=':', alpha=0.5)
        ax[0].set_xlabel('Frequency [Hz]')
        ax[0].set_ylabel('PSD raw units [rad^2/Hz]')
        ax[1].set_xlabel('Frequency [Hz]')
        plt.suptitle(f'{obsid} ws0 f150')
        plt.tight_layout()
        print('Saving and closing plots')
        plt.savefig(os.path.join(savedir,f'{obsid}_ws0_f150'))
        plt.close()

if __name__ == '__main__':
    main()
