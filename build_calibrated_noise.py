#!/usr/bin/env python3
"""Write K_CMB copies of the noise aggregation files.

Why
---
The 260702 preprocessing configs dropped the final
``calibrate: {cal_array: abscal.abscal_cmb}`` step that every earlier
generation applied before ``demodulate``. So ``white_noise`` in
``aggregation/noise_aggregation_<band>.h5`` is in relatively-calibrated pW,
not K_CMB, and the usual ``(x/2)*1e6`` conversion under-reports by roughly a
factor of 10.

``calibrate`` is a pure multiplicative per-detector rescaling and white noise
is a linear amplitude, so the correction is exact after the fact:

    white_noise_K_CMB = white_noise_relcal_pW * abscal_cmb(t, wafer, band)

abscal is one scalar per (wafer_slot, bandpass) per time epoch, read straight
from the product's own ManifestDb by ``abscal_lookup``. Rather than teach
every downstream script to apply it, this writes a calibrated copy once and
lets everything build off that.

Output
------
``aggregation/noise_aggregation_<band>_cmb.h5``, same layout as the input plus

    abscal        one value per index entry, the factor applied
    valid         per detector: finite and strictly positive after scaling

and attributes recording the source file, context and abscal database, so a
calibrated file can always be traced back.

``valid`` matters because the aggregation carries genuine zeros -- 2.0M of
31.7M satp3 f090 detectors -- which are not measurements. They are left in
place rather than dropped, so the file stays row-aligned with its source, but
everything downstream should select on ``valid``.

Usage
-----
    python3 build_calibrated_noise.py satp2_260702 \\
        --context contexts/satp2/use_this_local_260824_abscalr2.yaml

Run from the iso-sat v4 root so the context path resolves.
"""

import argparse
import os
import sqlite3
import sys
import time

import h5py
import numpy as np

from abscal_lookup import AbscalTable, abscal_db_from_context

CUTS_STATS = '/scratch/gpfs/SIMONSOBS/sat-iso/v4/cuts_stats'


def obs_timestamps(archive_dir, telescope):
    """obs_id -> timestamp from the cuts/stats database, if it is there.

    Preferred over parsing the ctime out of obs_id: the two differ by up to a
    second, which only matters for an observation sitting on an abscal epoch
    boundary, but there is no reason to take the risk when the exact value is
    on hand.
    """
    import glob
    hits = glob.glob(os.path.join(archive_dir, 'databases',
                                  f'iso_v4_{telescope}_*.sqlite'))
    hits = [h for h in hits if '_copy' not in os.path.basename(h)]
    if not hits:
        return {}
    con = sqlite3.connect(f'file:{sorted(hits)[0]}?mode=ro', uri=True)
    try:
        return {r[0]: float(r[1]) for r in
                con.execute('SELECT DISTINCT obs_id, timestamp FROM obs '
                            'WHERE timestamp IS NOT NULL')}
    except sqlite3.DatabaseError:
        return {}
    finally:
        con.close()


def ctime_of(obs_id):
    try:
        return float(obs_id.split('_')[1])
    except (IndexError, ValueError):
        return None


def _as_str(arr):
    return np.array([a.decode() if isinstance(a, bytes) else str(a) for a in arr])


def calibrate_band(src_path, dst_path, band, table, ts_lookup,
                   boundary_warn_s=600.0, overwrite=False):
    if os.path.exists(dst_path) and not overwrite:
        print(f'  {os.path.basename(dst_path)} exists, skipping (use --overwrite)')
        return None

    with h5py.File(src_path, 'r') as f:
        entry_unit = f.attrs.get('entry_unit')
        obs_ids = _as_str(f['obs_ids'][:])
        starts = f['obs_starts'][:]
        ends = f['obs_ends'][:]
        wn = f['white_noise'][:]
        wafer_slots = f['wafer_slots'][:]
        det_ids = f['det_ids'][:]
        has_obs_wafers = 'obs_wafers' in f
        obs_wafers = _as_str(f['obs_wafers'][:]) if has_obs_wafers else None
        src_attrs = dict(f.attrs)

    n = len(wn)
    covered = int((ends - starts).sum())
    if entry_unit != 'obs_wafer':
        print(f'  WARNING: {os.path.basename(src_path)} has entry_unit='
              f'{entry_unit!r}, not "obs_wafer". Older files index only the '
              f'last wafer per observation; this one reaches '
              f'{100*covered/n:.1f}% of its detectors. Calibrating only what '
              f'is indexed.')

    abscal_per_entry = np.full(len(starts), np.nan)
    n_missing = n_boundary = 0
    scaled = wn.copy()

    for i, (obs_id, s, e) in enumerate(zip(obs_ids, starts, ends)):
        if e <= s:
            continue
        if has_obs_wafers:
            wafer = obs_wafers[i]
        else:
            # Each index entry covers one wafer, so the slot is constant.
            uniq = np.unique(wafer_slots[s:e])
            wafer = f'ws{int(uniq[0])}' if len(uniq) == 1 else None
            if wafer is None:
                n_missing += 1
                continue

        ts = ts_lookup.get(obs_id)
        if ts is None:
            ts = ctime_of(obs_id)
        if ts is None:
            n_missing += 1
            continue

        ep = table.epoch(ts)
        if ep is not None:
            lo, hi = ep[0], ep[1]
            if min(abs(ts - lo), abs(ts - hi)) < boundary_warn_s:
                n_boundary += 1

        val = table.get(ts, wafer, band)
        if val is None:
            n_missing += 1
            continue
        abscal_per_entry[i] = val
        scaled[s:e] = wn[s:e] * val

    valid = np.isfinite(scaled) & (scaled > 0)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp = dst_path + '.tmp'
    with h5py.File(tmp, 'w') as f:
        f.create_dataset('white_noise', data=scaled, compression='gzip')
        f.create_dataset('valid', data=valid, compression='gzip')
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('det_ids', data=np.asarray(det_ids, dtype=object),
                         dtype=dt, compression='gzip')
        f.create_dataset('wafer_slots', data=wafer_slots, compression='gzip')
        f.create_dataset('obs_ids', data=np.asarray(obs_ids, dtype=object), dtype=dt)
        if has_obs_wafers:
            f.create_dataset('obs_wafers', data=np.asarray(obs_wafers, dtype=object),
                             dtype=dt)
        f.create_dataset('obs_starts', data=starts)
        f.create_dataset('obs_ends', data=ends)
        f.create_dataset('abscal', data=abscal_per_entry)
        for k, v in src_attrs.items():
            f.attrs[k] = v
        f.attrs['units'] = 'K_CMB per rtHz (apply /2*1e6 for uK_CMB*rt(s))'
        f.attrs['calibrated'] = True
        f.attrs['abscal_db'] = table.db_path
        f.attrs['source_file'] = src_path
        f.attrs['built'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        f.attrs['n_entries_uncalibrated'] = int(np.isnan(abscal_per_entry).sum())
    os.replace(tmp, dst_path)

    ok = np.isfinite(abscal_per_entry)
    med_raw = np.median(wn[valid]) if valid.any() else np.nan
    med_cal = np.median(scaled[valid]) if valid.any() else np.nan
    print(f'  {band}: {len(starts):,} entries, {int(ok.sum()):,} calibrated, '
          f'{n_missing:,} without abscal'
          + (f', {n_boundary:,} near an epoch boundary' if n_boundary else ''))
    print(f'       abscal {np.nanmin(abscal_per_entry):.3f}-'
          f'{np.nanmax(abscal_per_entry):.3f}; '
          f'median white noise {med_raw/2*1e6:.1f} -> {med_cal/2*1e6:.1f} uK*rt(s); '
          f'{int((~valid).sum()):,} of {n:,} detectors not valid')
    return dst_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('archive', help='archive directory name, e.g. satp2_260702')
    ap.add_argument('--context', required=True,
                    help='context whose abscal entry supplies the factors')
    ap.add_argument('--base-dir', default=CUTS_STATS)
    ap.add_argument('--bands', nargs='+', default=None,
                    help='default: every noise_aggregation_*.h5 present')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    archive_dir = os.path.join(args.base_dir, args.archive)
    agg = os.path.join(archive_dir, 'aggregation')
    telescope = args.archive.split('_')[0]

    if not os.path.isdir(agg):
        sys.exit(f'no aggregation directory: {agg}')

    bands = args.bands
    if bands is None:
        import glob
        bands = sorted(
            os.path.basename(p)[len('noise_aggregation_'):-len('.h5')]
            for p in glob.glob(os.path.join(agg, 'noise_aggregation_*.h5'))
            if not p.endswith('_cmb.h5'))
    if not bands:
        sys.exit(f'no noise_aggregation_*.h5 under {agg}')

    db_path = abscal_db_from_context(args.context)
    table = AbscalTable(db_path)
    print(f'{args.archive}: abscal from {db_path}')
    print(f'bands: {bands}')

    ts_lookup = obs_timestamps(archive_dir, telescope)
    print(f'{len(ts_lookup):,} observation timestamps from the stats database')

    for band in bands:
        src = os.path.join(agg, f'noise_aggregation_{band}.h5')
        dst = os.path.join(agg, f'noise_aggregation_{band}_cmb.h5')
        if not os.path.exists(src):
            print(f'  {band}: no {src}, skipping')
            continue
        calibrate_band(src, dst, band, table, ts_lookup, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
