from __future__ import annotations

import math
import re
import sqlite3
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

matplotlib.use('Agg')

SAMPLE_RATE_HZ = 200.0
DETECTORS_PER_WAFER_PER_BAND = 860.0
MAX_DETECTORS_PER_BAND = 6020.0
MAX_DETECTORS_TOTAL = 12040.0

NOTEBOOK_ROOT = Path('/home/ms3067/so_data_exploration/tiger/202601_v4')
ANALYSIS_ROOT = NOTEBOOK_ROOT / 'plots' / 'analysis'

SQLITE_PATHS = {
    'satp1': Path('/scratch/gpfs/SIMONSOBS/sat-iso/v4/cuts_stats/satp1_20260417/databases/iso_v4_satp1_20260417.sqlite'),
    'satp3': Path('/scratch/gpfs/SIMONSOBS/sat-iso/v4/cuts_stats/satp3_20260417/databases/iso_v4_satp3_20260417.sqlite'),
}

SAMPLE_FLAG_STEPS = [
    'smurfgaps',
    'turnaround_flags',
    'jumps_slow',
    'jumps_2pi',
    'glitches_pre_hwpss',
    'source_flags',
    'glitches_post_hwpss',
    'noisy_subscan_flags',
]

NOISE_Q50_METRICS = [
    ('white_noise_q50', 'White Noise q50'),
    ('fknee_q50', 'Knee Frequency q50'),
    ('alpha_q50', 'Alpha q50'),
]

NOISE_POLS = ['T', 'Q', 'U']

AGG_NET_CONVERSION_FACTOR = 1e6 / 2.0

# Keep PDF size reasonable while retaining full on-disk figure outputs.
AGG_NOISE_SCENARIO1_WAFERS_FOR_REPORT = ['ws00', 'ws01', 'ws02', 'ws03']

# Split very tall figures into report-friendly chunks instead of shrinking to illegibility.
MAX_REPORT_PANEL_ASPECT = 1.15  # height / width threshold
REPORT_IMAGE_MAX_HEIGHT_FRACTION = 0.82


def latex_escape(text: str) -> str:
    repl = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
    }
    out = text
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def fmt_num(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 'nan'
    av = abs(float(v))
    if av >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if av >= 1_000:
        return f"{v / 1_000:.2f}k"
    if av >= 100:
        return f"{v:.1f}"
    if av >= 10:
        return f"{v:.2f}"
    return f"{v:.4f}"


def fmt_pct(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 'nan'
    return f"{100.0 * float(v):.2f}\\%"


def extract_obs_timestamp(obs_id: str) -> int | None:
    m = re.search(r'obs_(\d+)_', obs_id)
    return int(m.group(1)) if m else None


@dataclass
class TelescopeData:
    name: str
    step_cols: list[str]
    step_names: list[str]
    success_rows: list[dict]
    wafer_names: list[str]
    bands: list[str]
    obs_band_agg: dict[tuple[int, str], dict]
    obs_total_agg: dict[int, dict]
    sample_step_columns_available: bool


def load_telescope_data(name: str, sqlite_path: Path) -> TelescopeData:
    with sqlite3.connect(sqlite_path) as conn:
        conn.row_factory = sqlite3.Row
        cols = [r['name'] for r in conn.execute('PRAGMA table_info(obs)').fetchall()]
        step_cols = sorted(
            [c for c in cols if re.match(r'^det_\d+_.+_ndets$', c)],
            key=lambda c: int(re.match(r'^det_(\d+)_', c).group(1)),
        )
        step_names = [re.sub(r'^det_\d+_|_ndets$', '', c) for c in step_cols]
        rows = [dict(r) for r in conn.execute('SELECT * FROM obs').fetchall()]

    success_rows = [r for r in rows if r.get('success') == 1]
    wafer_names = sorted({r.get('wafer') for r in success_rows if r.get('wafer') is not None})
    bands = sorted({r.get('band') for r in success_rows if r.get('band') is not None})

    sample_step_columns_available = any(re.match(r'^samp_\d+_.+_nsamps$', c) for c in cols)

    # For sample denominators, estimate per (obs, band) samples_per_detector baseline.
    samp_per_det_ref: dict[tuple[int, str], float] = defaultdict(float)
    for r in success_rows:
        ts = extract_obs_timestamp(str(r.get('obs_id', '')))
        band = r.get('band')
        det0 = float(r.get(step_cols[0]) or 0.0)
        total_samples = float(r.get('total_samples') or 0.0)
        if ts is None or band is None or det0 <= 0:
            continue
        spd = total_samples / det0
        key = (ts, str(band))
        if spd > samp_per_det_ref[key]:
            samp_per_det_ref[key] = spd

    obs_band_agg: dict[tuple[int, str], dict] = {}
    for r in success_rows:
        ts = extract_obs_timestamp(str(r.get('obs_id', '')))
        band = r.get('band')
        if ts is None or band is None:
            continue
        band = str(band)
        key = (ts, band)
        if key not in obs_band_agg:
            obs_band_agg[key] = {
                'step_sums': np.zeros(len(step_cols), dtype=float),
                'duration_h': 0.0,
                'final_samples': 0.0,
                'possible_samples': 0.0,
            }

        step_vals = np.array([float(r.get(c) or 0.0) for c in step_cols], dtype=float)
        obs_band_agg[key]['step_sums'] += step_vals

        det0 = float(r.get(step_cols[0]) or 0.0)
        total_samples = float(r.get('total_samples') or 0.0)
        if det0 > 0:
            spd = total_samples / det0
            obs_band_agg[key]['duration_h'] = max(obs_band_agg[key]['duration_h'], spd / SAMPLE_RATE_HZ / 3600.0)

        obs_band_agg[key]['final_samples'] += total_samples
        obs_band_agg[key]['possible_samples'] += det0 * samp_per_det_ref.get((ts, band), 0.0)

    obs_total_agg: dict[int, dict] = {}
    for (ts, _band), info in obs_band_agg.items():
        if ts not in obs_total_agg:
            obs_total_agg[ts] = {
                'step_sums': np.zeros(len(step_cols), dtype=float),
                'duration_h_sum': 0.0,
                'final_samples': 0.0,
                'possible_samples': 0.0,
            }
        obs_total_agg[ts]['step_sums'] += info['step_sums']
        obs_total_agg[ts]['duration_h_sum'] += float(info['duration_h'])
        obs_total_agg[ts]['final_samples'] += float(info['final_samples'])
        obs_total_agg[ts]['possible_samples'] += float(info['possible_samples'])

    return TelescopeData(
        name=name,
        step_cols=step_cols,
        step_names=step_names,
        success_rows=success_rows,
        wafer_names=wafer_names,
        bands=bands,
        obs_band_agg=obs_band_agg,
        obs_total_agg=obs_total_agg,
        sample_step_columns_available=sample_step_columns_available,
    )


def step_cut_fracs(step_values: np.ndarray, denom: float) -> np.ndarray:
    cuts = np.zeros_like(step_values, dtype=float)
    if len(step_values) == 0 or denom <= 0:
        return cuts
    cuts[0] = max(0.0, denom - step_values[0])
    if len(step_values) > 1:
        prev = step_values[:-1]
        cur = step_values[1:]
        cuts[1:] = np.maximum(0.0, prev - cur)
    return cuts / denom


def compute_hours_stats(td: TelescopeData) -> dict[str, float]:
    det_hours_by_band = {'f090': 0.0, 'f150': 0.0}
    obs_hours_by_band = {'f090': 0.0, 'f150': 0.0}

    for (_ts, band), info in td.obs_band_agg.items():
        final_ndets = float(info['step_sums'][-1])
        duration_h = float(info['duration_h'])
        if band in det_hours_by_band:
            det_hours_by_band[band] += final_ndets * duration_h
            if final_ndets > 0:
                obs_hours_by_band[band] += duration_h

    tss = sorted(td.obs_total_agg.keys())
    cal_hours = 0.0
    if len(tss) >= 2:
        cal_hours = (tss[-1] - tss[0]) / 3600.0

    out = {
        'det_hours_f090': det_hours_by_band.get('f090', 0.0),
        'det_hours_f150': det_hours_by_band.get('f150', 0.0),
        'det_hours_combined': det_hours_by_band.get('f090', 0.0) + det_hours_by_band.get('f150', 0.0),
        'possible_det_hours_f090': cal_hours * MAX_DETECTORS_PER_BAND,
        'possible_det_hours_f150': cal_hours * MAX_DETECTORS_PER_BAND,
        'possible_det_hours_combined': cal_hours * MAX_DETECTORS_TOTAL,
        'calendar_hours': cal_hours,
        'obs_hours_f090': obs_hours_by_band.get('f090', 0.0),
        'obs_hours_f150': obs_hours_by_band.get('f150', 0.0),
        'obs_hours_combined': obs_hours_by_band.get('f090', 0.0) + obs_hours_by_band.get('f150', 0.0),
    }
    out['det_frac_f090'] = out['det_hours_f090'] / out['possible_det_hours_f090'] if out['possible_det_hours_f090'] > 0 else np.nan
    out['det_frac_f150'] = out['det_hours_f150'] / out['possible_det_hours_f150'] if out['possible_det_hours_f150'] > 0 else np.nan
    out['det_frac_combined'] = out['det_hours_combined'] / out['possible_det_hours_combined'] if out['possible_det_hours_combined'] > 0 else np.nan
    out['obs_frac_f090'] = out['obs_hours_f090'] / out['calendar_hours'] if out['calendar_hours'] > 0 else np.nan
    out['obs_frac_f150'] = out['obs_hours_f150'] / out['calendar_hours'] if out['calendar_hours'] > 0 else np.nan
    out['obs_frac_combined'] = out['obs_hours_combined'] / out['calendar_hours'] if out['calendar_hours'] > 0 else np.nan
    return out


def _stats(vals: list[float]) -> tuple[float, float, float]:
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    return float(np.mean(arr)), float(np.median(arr)), float(np.max(arr))


def compute_detector_yield_rows(td: TelescopeData) -> list[tuple[str, str, tuple[float, float, float]]]:
    rows: list[tuple[str, str, tuple[float, float, float]]] = []

    # Total across both bands, summed by timestamp.
    total_vals = [float(v['step_sums'][-1]) / MAX_DETECTORS_TOTAL for _, v in sorted(td.obs_total_agg.items())]
    rows.append(('All wafers', 'f090+f150', _stats(total_vals)))

    # All wafers per band, summed by (timestamp, band).
    for band in ['f090', 'f150']:
        vals = [
            float(v['step_sums'][-1]) / MAX_DETECTORS_PER_BAND
            for (ts, b), v in sorted(td.obs_band_agg.items())
            if b == band
        ]
        rows.append(('All wafers', band, _stats(vals)))

    # Each wafer, split by band.
    for wafer in td.wafer_names:
        for band in ['f090', 'f150']:
            vals = []
            for r in td.success_rows:
                if r.get('wafer') == wafer and r.get('band') == band:
                    vals.append(float(r.get(td.step_cols[-1]) or 0.0) / DETECTORS_PER_WAFER_PER_BAND)
            rows.append((str(wafer), band, _stats(vals)))

    return rows


def compute_cut_step_stats(td: TelescopeData) -> tuple[list[str], list[tuple[str, list[tuple[float, float]]]]]:
    # Use steps that have non-zero mean cut in total row.
    total_step_fracs = []
    for _ts, info in sorted(td.obs_total_agg.items()):
        total_step_fracs.append(step_cut_fracs(info['step_sums'], MAX_DETECTORS_TOTAL))
    if not total_step_fracs:
        return td.step_names, []
    total_arr = np.vstack(total_step_fracs)
    keep_indices = [i for i in range(total_arr.shape[1]) if float(np.nanmean(total_arr[:, i])) > 0.0]
    step_names = [td.step_names[i] for i in keep_indices]

    rows: list[tuple[str, list[tuple[float, float]]]] = []

    def row_stats(frac_rows: list[np.ndarray], label: str) -> None:
        if not frac_rows:
            rows.append((label, [(np.nan, np.nan) for _ in keep_indices]))
            return
        arr = np.vstack(frac_rows)
        stats = []
        for i in keep_indices:
            col = arr[:, i]
            stats.append((float(np.nanmean(col)), float(np.nanmedian(col))))
        rows.append((label, stats))

    row_stats([step_cut_fracs(v['step_sums'], MAX_DETECTORS_TOTAL) for _, v in sorted(td.obs_total_agg.items())], 'Total f090+f150')
    for band in ['f090', 'f150']:
        row_stats([step_cut_fracs(v['step_sums'], MAX_DETECTORS_PER_BAND) for (ts, b), v in sorted(td.obs_band_agg.items()) if b == band], f'All wafers {band}')
    for wafer in td.wafer_names:
        for band in ['f090', 'f150']:
            frac_rows = []
            for r in td.success_rows:
                if r.get('wafer') == wafer and r.get('band') == band:
                    step_vals = np.array([float(r.get(c) or 0.0) for c in td.step_cols], dtype=float)
                    frac_rows.append(step_cut_fracs(step_vals, DETECTORS_PER_WAFER_PER_BAND))
            row_stats(frac_rows, f'{wafer} {band}')

    return step_names, rows


def compute_sample_summary_rows(td: TelescopeData) -> list[tuple[str, float, float, float, float]]:
    rows: list[tuple[str, float, float, float, float]] = []

    def summarize(possible: list[float], final: list[float], label: str) -> None:
        p = float(np.nansum(possible))
        f = float(np.nansum(final))
        keep = f / p if p > 0 else np.nan
        flagged = 1.0 - keep if p > 0 else np.nan
        rows.append((label, p, f, keep, flagged))

    summarize([v['possible_samples'] for _, v in sorted(td.obs_total_agg.items())], [v['final_samples'] for _, v in sorted(td.obs_total_agg.items())], 'Total f090+f150')

    for band in ['f090', 'f150']:
        summarize(
            [v['possible_samples'] for (ts, b), v in sorted(td.obs_band_agg.items()) if b == band],
            [v['final_samples'] for (ts, b), v in sorted(td.obs_band_agg.items()) if b == band],
            f'All wafers {band}',
        )

    for wafer in td.wafer_names:
        for band in ['f090', 'f150']:
            possible = []
            final = []
            for r in td.success_rows:
                if r.get('wafer') != wafer or r.get('band') != band:
                    continue
                ts = extract_obs_timestamp(str(r.get('obs_id', '')))
                if ts is None:
                    continue
                key = (ts, band)
                # Same reference as load_telescope_data:
                det0 = float(r.get(td.step_cols[0]) or 0.0)
                total_samples = float(r.get('total_samples') or 0.0)
                spd_ref = 0.0
                if key in td.obs_band_agg:
                    band_possible = float(td.obs_band_agg[key]['possible_samples'])
                    band_det0 = float(td.obs_band_agg[key]['step_sums'][0])
                    if band_det0 > 0:
                        spd_ref = band_possible / band_det0
                possible.append(det0 * spd_ref)
                final.append(total_samples)
            summarize(possible, final, f'{wafer} {band}')

    return rows


def compute_sample_step_proxy(td: TelescopeData) -> tuple[list[str], list[tuple[str, list[tuple[float, float]]]], bool]:
    # No true sample-step columns are present in current obsdb schema.
    # Build proxy by apportioning total sample flagged fraction by detector step cut fractions.
    step_names, det_rows = compute_cut_step_stats(td)

    # Determine indices in full step list.
    full_indices = [td.step_names.index(s) for s in step_names]

    def proxy_rows_for_level(level: str) -> list[np.ndarray]:
        fracs = []
        if level == 'total':
            for _ts, info in sorted(td.obs_total_agg.items()):
                det_cuts = step_cut_fracs(info['step_sums'], MAX_DETECTORS_TOTAL)
                dsum = float(np.sum(det_cuts))
                possible = float(info['possible_samples'])
                final = float(info['final_samples'])
                sample_flag = (1.0 - final / possible) if possible > 0 else np.nan
                if dsum > 0 and np.isfinite(sample_flag):
                    fracs.append((sample_flag * det_cuts / dsum))
                else:
                    fracs.append(np.zeros_like(det_cuts))
        elif level in ('f090', 'f150'):
            for (ts, band), info in sorted(td.obs_band_agg.items()):
                if band != level:
                    continue
                det_cuts = step_cut_fracs(info['step_sums'], MAX_DETECTORS_PER_BAND)
                dsum = float(np.sum(det_cuts))
                possible = float(info['possible_samples'])
                final = float(info['final_samples'])
                sample_flag = (1.0 - final / possible) if possible > 0 else np.nan
                if dsum > 0 and np.isfinite(sample_flag):
                    fracs.append((sample_flag * det_cuts / dsum))
                else:
                    fracs.append(np.zeros_like(det_cuts))
        else:
            wafer, band = level.split('|')
            for r in td.success_rows:
                if str(r.get('wafer')) != wafer or str(r.get('band')) != band:
                    continue
                step_vals = np.array([float(r.get(c) or 0.0) for c in td.step_cols], dtype=float)
                det_cuts = step_cut_fracs(step_vals, DETECTORS_PER_WAFER_PER_BAND)
                dsum = float(np.sum(det_cuts))
                ts = extract_obs_timestamp(str(r.get('obs_id', '')))
                key = (ts, band)
                det0 = float(r.get(td.step_cols[0]) or 0.0)
                total_samples = float(r.get('total_samples') or 0.0)
                spd_ref = 0.0
                if key in td.obs_band_agg:
                    band_possible = float(td.obs_band_agg[key]['possible_samples'])
                    band_det0 = float(td.obs_band_agg[key]['step_sums'][0])
                    if band_det0 > 0:
                        spd_ref = band_possible / band_det0
                possible = det0 * spd_ref
                final = total_samples
                sample_flag = (1.0 - final / possible) if possible > 0 else np.nan
                if dsum > 0 and np.isfinite(sample_flag):
                    fracs.append((sample_flag * det_cuts / dsum))
                else:
                    fracs.append(np.zeros_like(det_cuts))
        return fracs

    out_rows: list[tuple[str, list[tuple[float, float]]]] = []

    def append_proxy(label: str, level: str) -> None:
        fracs = proxy_rows_for_level(level)
        if not fracs:
            out_rows.append((label, [(np.nan, np.nan) for _ in step_names]))
            return
        arr = np.vstack(fracs)
        stats = []
        for i in full_indices:
            col = arr[:, i]
            stats.append((float(np.nanmean(col)), float(np.nanmedian(col))))
        out_rows.append((label, stats))

    append_proxy('Total f090+f150', 'total')
    append_proxy('All wafers f090', 'f090')
    append_proxy('All wafers f150', 'f150')
    for wafer in td.wafer_names:
        for band in ['f090', 'f150']:
            append_proxy(f'{wafer} {band}', f'{wafer}|{band}')

    return step_names, out_rows, td.sample_step_columns_available


def _weighted_percent_summary(sub_df: pd.DataFrame, total_col: str, value_cols: list[str]) -> dict[str, float]:
    total = float(sub_df[total_col].sum())
    out: dict[str, float] = {}
    if total <= 0:
        for col in value_cols:
            out[col] = np.nan
        return out
    for col in value_cols:
        out[col] = 100.0 * float(sub_df[col].sum()) / total
    return out


def _plot_hist_grid_percent(
    df: pd.DataFrame,
    value_cols: list[str],
    title: str,
    save_path: Path,
    x_label: str,
    x_range: tuple[float, float],
    bins: int = 50,
) -> None:
    n = len(value_cols)
    n_cols = min(4, n)
    n_rows = int(math.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.9 * n_rows), squeeze=False)
    for i, col in enumerate(value_cols):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        values = pd.to_numeric(df[col], errors='coerce').to_numpy()
        values = values[np.isfinite(values)]
        if values.size > 0:
            ax.hist(values, bins=bins, range=x_range, alpha=0.8, edgecolor='none')
            ax.axvline(np.mean(values), color='k', linestyle='--', linewidth=1)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlim(*x_range)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2, axis='y')
    for j in range(n, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r][c].set_visible(False)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def compute_sample_flag_products(telescope: str, sqlite_path: Path) -> dict[str, object]:
    output_dir = ANALYSIS_ROOT / telescope / 'sample_flag_percentages'
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cols = ['obs_id', 'wafer', 'band', 'total_samples'] + SAMPLE_FLAG_STEPS
    with sqlite3.connect(sqlite_path) as conn:
        schema = pd.read_sql_query('PRAGMA table_info(obs)', conn)
        obs_columns = set(schema['name'].tolist())
        missing = [c for c in base_cols if c not in obs_columns]
        if missing:
            raise ValueError(f'{telescope}: missing sample-flag columns in obs table: {missing}')
        df = pd.read_sql_query('SELECT ' + ', '.join(base_cols) + ' FROM obs', conn)

    df['wafer'] = df['wafer'].astype(str)
    df['band'] = df['band'].astype(str)
    for col in ['total_samples'] + SAMPLE_FLAG_STEPS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    df = df[df['total_samples'] > 0].copy()
    if df.empty:
        raise ValueError(f'{telescope}: no rows with total_samples > 0')

    df['flagged_samples_sum'] = df[SAMPLE_FLAG_STEPS].sum(axis=1)
    df['pct_all_flagged'] = 100.0 * np.clip(df['flagged_samples_sum'] / df['total_samples'], 0.0, 1.0)
    pct_cols = []
    for step in SAMPLE_FLAG_STEPS:
        pct_col = f'pct_{step}'
        df[pct_col] = 100.0 * np.clip(df[step] / df['total_samples'], 0.0, 1.0)
        pct_cols.append(pct_col)

    summary_cols = ['pct_all_flagged'] + pct_cols

    wafer_band_records: list[dict[str, object]] = []
    for (wafer, band_name), g in df.groupby(['wafer', 'band'], dropna=False, sort=False):
        row = {
            'wafer': str(wafer),
            'band': str(band_name),
            **_weighted_percent_summary(g, 'total_samples', ['flagged_samples_sum'] + SAMPLE_FLAG_STEPS),
        }
        wafer_band_records.append(row)
    wafer_band = (
        pd.DataFrame(wafer_band_records)
        .sort_values(['band', 'wafer'])
        .reset_index(drop=True)
        .rename(columns={'flagged_samples_sum': 'pct_all_flagged', **{s: f'pct_{s}' for s in SAMPLE_FLAG_STEPS}})
    )

    band_records: list[dict[str, object]] = []
    for band_name, g in df.groupby(['band'], dropna=False, sort=False):
        row = {
            'band': str(band_name),
            **_weighted_percent_summary(g, 'total_samples', ['flagged_samples_sum'] + SAMPLE_FLAG_STEPS),
        }
        band_records.append(row)
    band = (
        pd.DataFrame(band_records)
        .sort_values(['band'])
        .reset_index(drop=True)
        .rename(columns={'flagged_samples_sum': 'pct_all_flagged', **{s: f'pct_{s}' for s in SAMPLE_FLAG_STEPS}})
    )
    global_df = pd.DataFrame([
        {
            'group': 'all_wafers_all_bands',
            **_weighted_percent_summary(df, 'total_samples', ['flagged_samples_sum'] + SAMPLE_FLAG_STEPS),
        }
    ]).rename(columns={'flagged_samples_sum': 'pct_all_flagged', **{s: f'pct_{s}' for s in SAMPLE_FLAG_STEPS}})

    wafer_band.to_csv(output_dir / 'sample_flag_percent_summary_by_wafer_band.csv', index=False)
    band.to_csv(output_dir / 'sample_flag_percent_summary_by_band.csv', index=False)
    global_df.to_csv(output_dir / 'sample_flag_percent_summary_global.csv', index=False)

    for b in sorted(df['band'].dropna().unique()):
        sub = df[df['band'] == b]
        _plot_hist_grid_percent(
            sub,
            pct_cols,
            f'{telescope} sample-flag percentage histograms by step: band={b}',
            output_dir / f'sample_flag_hists_percent_band_{b}.png',
            'Flagged samples [% of total_samples]',
            (0.0, 100.0),
        )
    _plot_hist_grid_percent(
        df,
        pct_cols,
        f'{telescope} sample-flag percentage histograms by step: all bands and wafers',
        output_dir / 'sample_flag_hists_percent_all_wafers_all_bands.png',
        'Flagged samples [% of total_samples]',
        (0.0, 100.0),
    )

    return {
        'wafer_band': wafer_band,
        'band': band,
        'global': global_df,
        'summary_cols': summary_cols,
        'plot_paths': [
            str((output_dir / 'sample_flag_hists_percent_all_wafers_all_bands.png').relative_to(ANALYSIS_ROOT)),
            *[
                str((output_dir / f'sample_flag_hists_percent_band_{b}.png').relative_to(ANALYSIS_ROOT))
                for b in sorted(df['band'].dropna().unique())
            ],
        ],
    }


def compute_noise_q50_products(telescope: str, sqlite_path: Path) -> dict[str, object]:
    output_dir = ANALYSIS_ROOT / telescope / 'sample_flag_percentages'
    output_dir.mkdir(parents=True, exist_ok=True)

    noise_cols = [f'noise_{pol}_{metric}' for pol in NOISE_POLS for metric, _ in NOISE_Q50_METRICS]
    select_cols = ['wafer', 'band', 'total_samples'] + noise_cols

    with sqlite3.connect(sqlite_path) as conn:
        schema = pd.read_sql_query('PRAGMA table_info(obs)', conn)
        obs_columns = set(schema['name'].tolist())
        missing = [c for c in select_cols if c not in obs_columns]
        if missing:
            raise ValueError(f'{telescope}: missing noise-q50 columns in obs table: {missing}')
        df = pd.read_sql_query('SELECT ' + ', '.join(select_cols) + ' FROM obs', conn)

    df['wafer'] = df['wafer'].astype(str)
    df['band'] = df['band'].astype(str)
    for col in ['total_samples'] + noise_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[(df['total_samples'] > 0) & df[noise_cols].notna().any(axis=1)].copy()
    if df.empty:
        raise ValueError(f'{telescope}: no usable rows for noise-q50 summaries')

    def weighted_noise_summary(sub_df: pd.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {}
        vals_w = sub_df['total_samples'].to_numpy()
        for col in noise_cols:
            vals = sub_df[col].to_numpy()
            mask = np.isfinite(vals) & np.isfinite(vals_w) & (vals_w > 0)
            out[col] = float(np.average(vals[mask], weights=vals_w[mask])) if np.any(mask) else np.nan
        return out

    wafer_band_records: list[dict[str, object]] = []
    for (wafer, band_name), g in df.groupby(['wafer', 'band'], dropna=False, sort=False):
        wafer_band_records.append(
            {
                'wafer': str(wafer),
                'band': str(band_name),
                **weighted_noise_summary(g),
            }
        )
    wafer_band = pd.DataFrame(wafer_band_records).sort_values(['band', 'wafer']).reset_index(drop=True)

    band_records: list[dict[str, object]] = []
    for band_name, g in df.groupby(['band'], dropna=False, sort=False):
        band_records.append({'band': str(band_name), **weighted_noise_summary(g)})
    band = pd.DataFrame(band_records).sort_values(['band']).reset_index(drop=True)
    global_df = pd.DataFrame([{'group': 'all_wafers_all_bands', **weighted_noise_summary(df)}])

    wafer_band.to_csv(output_dir / 'noise_stats_weighted_by_wafer_band.csv', index=False)
    band.to_csv(output_dir / 'noise_stats_weighted_by_band.csv', index=False)
    global_df.to_csv(output_dir / 'noise_stats_weighted_global.csv', index=False)

    for b in sorted(df['band'].dropna().unique()):
        sub = df[df['band'] == b]
        _plot_hist_grid_percent(
            sub,
            noise_cols,
            f'{telescope} noise-q50 histograms by step: band={b}',
            output_dir / f'noise_stats_hists_band_{b}.png',
            'Noise q50 value',
            (
                float(np.nanpercentile(sub[noise_cols].to_numpy(), 1)),
                float(np.nanpercentile(sub[noise_cols].to_numpy(), 99)),
            ),
            bins=50,
        )
    _plot_hist_grid_percent(
        df,
        noise_cols,
        f'{telescope} noise-q50 histograms by step: all bands and wafers',
        output_dir / 'noise_stats_hists_all_wafers_all_bands.png',
        'Noise q50 value',
        (
            float(np.nanpercentile(df[noise_cols].to_numpy(), 1)),
            float(np.nanpercentile(df[noise_cols].to_numpy(), 99)),
        ),
        bins=50,
    )

    return {
        'wafer_band': wafer_band,
        'band': band,
        'global': global_df,
        'noise_cols': noise_cols,
        'plot_paths': [
            str((output_dir / 'noise_stats_hists_all_wafers_all_bands.png').relative_to(ANALYSIS_ROOT)),
            *[
                str((output_dir / f'noise_stats_hists_band_{b}.png').relative_to(ANALYSIS_ROOT))
                for b in sorted(df['band'].dropna().unique())
            ],
        ],
    }


def _decode_h5_string_array(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == object:
        return np.array([x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in arr])
    if np.issubdtype(arr.dtype, np.bytes_):
        return arr.astype(str)
    return arr


def _compute_net_array(net_values: np.ndarray) -> tuple[float, int]:
    valid = net_values[np.isfinite(net_values) & (net_values > 0)]
    if valid.size == 0:
        return np.nan, 0
    return float(1.0 / np.sqrt(np.sum(1.0 / np.square(valid)))), int(valid.size)


def _band_from_h5(path: Path) -> str:
    m = re.search(r'f(\d+)', path.name)
    return f'f{m.group(1)}' if m else path.stem


def compute_agg_noise_products(telescope: str) -> dict[str, object]:
    output_dir = ANALYSIS_ROOT / telescope / 'agg_noise'
    output_dir.mkdir(parents=True, exist_ok=True)

    agg_dir = SQLITE_PATHS[telescope].parent.parent / 'aggregation'
    h5_files = sorted(agg_dir.glob('noise_aggregation_*.h5'))
    if not h5_files:
        raise FileNotFoundError(f'{telescope}: no aggregation files found in {agg_dir}')

    scenario1_records: list[dict[str, object]] = []
    scenario2_records: list[dict[str, object]] = []
    scenario3_records: list[dict[str, object]] = []

    for h5_path in h5_files:
        band = _band_from_h5(h5_path)
        with h5py.File(h5_path, 'r') as f:
            obs_ids = _decode_h5_string_array(f['obs_ids'][()])
            starts = f['obs_starts'][()]
            ends = f['obs_ends'][()]
            wafer_slots = f['wafer_slots'][()]
            white_noise = f['white_noise'][()] * AGG_NET_CONVERSION_FACTOR

        net_array, n_det = _compute_net_array(white_noise)
        scenario3_records.append(
            {
                'scenario': '3_per_band_total',
                'band': band,
                'n_det': n_det,
                'net_array': net_array,
                'typ_det_net': net_array * np.sqrt(n_det) if n_det > 0 else np.nan,
            }
        )

        for obs_id, s, e in zip(obs_ids, starts, ends):
            s = int(s)
            e = int(e)
            if e <= s:
                continue
            wn = white_noise[s:e]
            ws = wafer_slots[s:e]

            net2, nd2 = _compute_net_array(wn)
            if nd2 > 0:
                scenario2_records.append(
                    {
                        'scenario': '2_per_band_obsid',
                        'band': band,
                        'obs_id': str(obs_id),
                        'n_det': nd2,
                        'net_array': net2,
                        'typ_det_net': net2 * np.sqrt(nd2),
                    }
                )

            for wafer in np.unique(ws):
                mask = ws == wafer
                net1, nd1 = _compute_net_array(wn[mask])
                if nd1 == 0:
                    continue
                scenario1_records.append(
                    {
                        'scenario': '1_per_wafer_band_obsid',
                        'wafer': f'ws{int(wafer):02d}',
                        'band': band,
                        'obs_id': str(obs_id),
                        'n_det': nd1,
                        'net_array': net1,
                        'typ_det_net': net1 * np.sqrt(nd1),
                    }
                )

    scenario1_df = pd.DataFrame(scenario1_records)
    scenario2_df = pd.DataFrame(scenario2_records)
    scenario3_df = pd.DataFrame(scenario3_records)

    def summarize_distribution(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        return (
            df.groupby(group_cols, dropna=False)
            .agg(
                median_net_array=('net_array', 'median'),
                mean_net_array=('net_array', 'mean'),
                mean_typ_det_net=('typ_det_net', 'mean'),
                median_typ_det_net=('typ_det_net', 'median'),
                n_obs=('obs_id', 'nunique'),
                mean_n_det=('n_det', 'mean'),
            )
            .reset_index()
        )

    summary1 = summarize_distribution(scenario1_df, ['wafer', 'band']).sort_values(['wafer', 'band']).reset_index(drop=True)
    summary1.insert(0, 'scenario', '1_per_wafer_band')
    summary1['row_label'] = summary1['wafer'] + '-' + summary1['band']
    summary2 = summarize_distribution(scenario2_df, ['band']).sort_values(['band']).reset_index(drop=True)
    summary2.insert(0, 'scenario', '2_per_band')
    summary2['row_label'] = summary2['band']
    summary3 = pd.DataFrame(
        [
            {
                'scenario': '3_per_band_total',
                'row_label': 'all_bands',
                'median_net_array': scenario3_df['net_array'].median(),
                'mean_net_array': scenario3_df['net_array'].mean(),
                'mean_typ_det_net': scenario3_df['typ_det_net'].mean(),
                'median_typ_det_net': scenario3_df['typ_det_net'].median(),
                'n_obs': np.nan,
                'mean_n_det': scenario3_df['n_det'].mean(),
            }
        ]
    )
    summary_table = pd.concat(
        [
            summary1[['scenario', 'row_label', 'median_net_array', 'mean_net_array', 'mean_typ_det_net', 'median_typ_det_net', 'n_obs', 'mean_n_det']],
            summary2[['scenario', 'row_label', 'median_net_array', 'mean_net_array', 'mean_typ_det_net', 'median_typ_det_net', 'n_obs', 'mean_n_det']],
            summary3[['scenario', 'row_label', 'median_net_array', 'mean_net_array', 'mean_typ_det_net', 'median_typ_det_net', 'n_obs', 'mean_n_det']],
        ],
        ignore_index=True,
    ).rename(
        columns={
            'median_net_array': 'median_net_array_uK_rt_s',
            'mean_net_array': 'mean_net_array_uK_rt_s',
            'mean_typ_det_net': 'mean_typ_det_net_uK_rt_s',
            'median_typ_det_net': 'median_typ_det_net_uK_rt_s',
        }
    )

    summary1.to_csv(output_dir / 'agg_noise_summary_scenario1_wafer_band.csv', index=False)
    summary2.to_csv(output_dir / 'agg_noise_summary_scenario2_band.csv', index=False)
    summary3.to_csv(output_dir / 'agg_noise_summary_scenario3_global.csv', index=False)
    summary_table.to_csv(output_dir / 'agg_noise_summary_table.csv', index=False)

    plot_paths: list[str] = []

    def plot_distribution_pair(df: pd.DataFrame, title: str, save_path: Path) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        net_array_bins = np.linspace(0, 100, 41)
        typ_det_bins = np.linspace(150, 550, 41)
        axes[0].hist(df['net_array'].dropna(), bins=net_array_bins, range=(0, 100), alpha=0.85, edgecolor='black')
        axes[0].set_xlim(0, 100)
        axes[0].set_title(f'{title} - NET_array')
        axes[0].set_xlabel('NET_array [uK*sqrt(s)]')
        axes[0].set_ylabel('Count')
        axes[1].hist(df['typ_det_net'].dropna(), bins=typ_det_bins, range=(150, 550), alpha=0.85, edgecolor='black')
        axes[1].set_xlim(150, 550)
        axes[1].set_title(f'{title} - Typical Detector NET')
        axes[1].set_xlabel('Typical detector NET [uK*sqrt(s)]')
        axes[1].set_ylabel('Count')
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    for _, row in summary1.iterrows():
        wafer = row['wafer']
        band = row['band']
        subset = scenario1_df[(scenario1_df['wafer'] == wafer) & (scenario1_df['band'] == band)]
        save_path = output_dir / f'agg_noise_hist_scenario1_{wafer}_{band}.png'
        plot_distribution_pair(subset, f'Scenario 1: {wafer}-{band}', save_path)
        plot_paths.append(str(save_path.relative_to(ANALYSIS_ROOT)))

    for _, row in summary2.iterrows():
        band = row['band']
        subset = scenario2_df[scenario2_df['band'] == band]
        save_path = output_dir / f'agg_noise_hist_scenario2_{band}.png'
        plot_distribution_pair(subset, f'Scenario 2: {band}', save_path)
        plot_paths.append(str(save_path.relative_to(ANALYSIS_ROOT)))

    return {
        'summary1': summary1,
        'summary2': summary2,
        'summary3': summary3,
        'summary_table': summary_table,
        'plot_paths': plot_paths,
    }


def latex_hours_table(stats1: dict[str, float], stats3: dict[str, float]) -> str:
    rows = [
        ('Det-hours f090', 'det_hours_f090'),
        ('Det-hours f150', 'det_hours_f150'),
        ('Det-hours total', 'det_hours_combined'),
        ('Possible det-hours f090', 'possible_det_hours_f090'),
        ('Possible det-hours f150', 'possible_det_hours_f150'),
        ('Possible det-hours total', 'possible_det_hours_combined'),
        ('Det-hour fraction f090', 'det_frac_f090'),
        ('Det-hour fraction f150', 'det_frac_f150'),
        ('Det-hour fraction total', 'det_frac_combined'),
        ('Calendar hours', 'calendar_hours'),
        ('Obs hours f090', 'obs_hours_f090'),
        ('Obs hours f150', 'obs_hours_f150'),
        ('Obs hours total', 'obs_hours_combined'),
        ('Obs/calendar f090', 'obs_frac_f090'),
        ('Obs/calendar f150', 'obs_frac_f150'),
        ('Obs/calendar total', 'obs_frac_combined'),
    ]
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Detector-hours and calendar-hours summary.}',
        r'\begin{tabular}{lcc}',
        r'\toprule',
        r'Metric & SATP1 & SATP3 \\',
        r'\midrule',
    ]
    for label, key in rows:
        if 'frac' in key:
            v1 = fmt_pct(stats1[key])
            v3 = fmt_pct(stats3[key])
        else:
            v1 = fmt_num(stats1[key])
            v3 = fmt_num(stats3[key])
        lines.append(f"{latex_escape(label)} & {v1} & {v3} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def latex_yield_table(rows1: list[tuple[str, str, tuple[float, float, float]]], rows3: list[tuple[str, str, tuple[float, float, float]]], wafer_names: list[str]) -> str:
    m1 = {(a, b): c for a, b, c in rows1}
    m3 = {(a, b): c for a, b, c in rows3}

    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Detector yield summary (fraction of possible detectors retained at final step).}',
        r'\small',
        r'\begin{tabular}{llcccccc}',
        r'\toprule',
        r'& & \multicolumn{3}{c}{SATP1} & \multicolumn{3}{c}{SATP3} \\',
        r'Category & Band & Avg & Median & Max & Avg & Median & Max \\',
        r'\midrule',
    ]

    def add_row(cat: str, band: str):
        a1, m1v, x1 = m1.get((cat, band), (np.nan, np.nan, np.nan))
        a3, m3v, x3 = m3.get((cat, band), (np.nan, np.nan, np.nan))
        lines.append(
            f"{latex_escape(cat)} & {latex_escape(band)} & {fmt_pct(a1)} & {fmt_pct(m1v)} & {fmt_pct(x1)} & {fmt_pct(a3)} & {fmt_pct(m3v)} & {fmt_pct(x3)} \\\\")

    # Opposite ordering requested: all wafers first, each wafer last.
    add_row('All wafers', 'f090+f150')
    lines.append(r'\midrule')

    lines.append('All wafers & f090 & ' +
                 f"{fmt_pct(m1.get(('All wafers', 'f090'), (np.nan, np.nan, np.nan))[0])} & " +
                 f"{fmt_pct(m1.get(('All wafers', 'f090'), (np.nan, np.nan, np.nan))[1])} & " +
                 f"{fmt_pct(m1.get(('All wafers', 'f090'), (np.nan, np.nan, np.nan))[2])} & " +
                 f"{fmt_pct(m3.get(('All wafers', 'f090'), (np.nan, np.nan, np.nan))[0])} & " +
                 f"{fmt_pct(m3.get(('All wafers', 'f090'), (np.nan, np.nan, np.nan))[1])} & " +
                 f"{fmt_pct(m3.get(('All wafers', 'f090'), (np.nan, np.nan, np.nan))[2])} \\\\")
    lines.append('All wafers & f150 & ' +
                 f"{fmt_pct(m1.get(('All wafers', 'f150'), (np.nan, np.nan, np.nan))[0])} & " +
                 f"{fmt_pct(m1.get(('All wafers', 'f150'), (np.nan, np.nan, np.nan))[1])} & " +
                 f"{fmt_pct(m1.get(('All wafers', 'f150'), (np.nan, np.nan, np.nan))[2])} & " +
                 f"{fmt_pct(m3.get(('All wafers', 'f150'), (np.nan, np.nan, np.nan))[0])} & " +
                 f"{fmt_pct(m3.get(('All wafers', 'f150'), (np.nan, np.nan, np.nan))[1])} & " +
                 f"{fmt_pct(m3.get(('All wafers', 'f150'), (np.nan, np.nan, np.nan))[2])} \\\\")

    lines.append(r'\midrule')
    for wafer in wafer_names:
        key = str(wafer)
        lines.append(rf'{latex_escape(key)} & f090 & ' +
                     f"{fmt_pct(m1.get((key, 'f090'), (np.nan, np.nan, np.nan))[0])} & " +
                     f"{fmt_pct(m1.get((key, 'f090'), (np.nan, np.nan, np.nan))[1])} & " +
                     f"{fmt_pct(m1.get((key, 'f090'), (np.nan, np.nan, np.nan))[2])} & " +
                     f"{fmt_pct(m3.get((key, 'f090'), (np.nan, np.nan, np.nan))[0])} & " +
                     f"{fmt_pct(m3.get((key, 'f090'), (np.nan, np.nan, np.nan))[1])} & " +
                     f"{fmt_pct(m3.get((key, 'f090'), (np.nan, np.nan, np.nan))[2])} \\\\")
        lines.append(rf'{latex_escape(key)} & f150 & ' +
                     f"{fmt_pct(m1.get((key, 'f150'), (np.nan, np.nan, np.nan))[0])} & " +
                     f"{fmt_pct(m1.get((key, 'f150'), (np.nan, np.nan, np.nan))[1])} & " +
                     f"{fmt_pct(m1.get((key, 'f150'), (np.nan, np.nan, np.nan))[2])} & " +
                     f"{fmt_pct(m3.get((key, 'f150'), (np.nan, np.nan, np.nan))[0])} & " +
                     f"{fmt_pct(m3.get((key, 'f150'), (np.nan, np.nan, np.nan))[1])} & " +
                     f"{fmt_pct(m3.get((key, 'f150'), (np.nan, np.nan, np.nan))[2])} \\\\")

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def render_step_table_as_png(
    title: str,
    step_names: list[str],
    rows: list[tuple[str, list[tuple[float, float]]]],
    save_path: Path,
    pct: bool = True,
    chunk_size: int = 6,
) -> None:
    """Render a step-impact table as a horizontally-chunked heatmap PNG.

    Wide tables (many steps) are tiled as stacked subplot grids so no text
    is squashed and the image always fits on a landscape page in the PDF.
    """
    n_rows = len(rows)
    n_steps = len(step_names)
    row_labels = [label for label, _ in rows]

    # Build full mean/median arrays (percent if requested).
    scale = 100.0 if pct else 1.0
    mean_arr = np.full((n_rows, n_steps), np.nan)
    med_arr = np.full((n_rows, n_steps), np.nan)
    for i, (_, vals) in enumerate(rows):
        for j, (m, med) in enumerate(vals):
            if np.isfinite(float(m)):
                mean_arr[i, j] = float(m) * scale
            if np.isfinite(float(med)):
                med_arr[i, j] = float(med) * scale

    chunks = [
        (step_names[k: k + chunk_size], mean_arr[:, k: k + chunk_size], med_arr[:, k: k + chunk_size])
        for k in range(0, n_steps, chunk_size)
    ]
    n_chunks = len(chunks)

    cell_w = 1.6
    label_w = max(len(r) for r in row_labels) * 0.09
    fig_w = label_w + cell_w * chunk_size + 1.2
    cell_h = 0.38
    pad_h = 1.4
    fig_h = (cell_h * n_rows + pad_h) * n_chunks

    fig, axes = plt.subplots(n_chunks, 1, figsize=(fig_w, max(fig_h, 3 * n_chunks)),
                              squeeze=False)
    fig.suptitle(title, fontsize=9, fontweight='bold', y=1.0 - 0.2 / max(fig_h, 1))

    unit_str = '%' if pct else ''

    for ci, (s_names_chunk, m_chunk, med_chunk) in enumerate(chunks):
        ax = axes[ci][0]
        n_c = len(s_names_chunk)
        vmax = float(np.nanmax(m_chunk)) if np.any(np.isfinite(m_chunk)) else 1.0
        vmax = max(vmax, 0.01)
        im = ax.imshow(m_chunk, aspect='auto', cmap='YlOrRd', vmin=0.0, vmax=vmax,
                        interpolation='nearest')
        ax.set_xticks(range(n_c))
        ax.set_xticklabels(s_names_chunk, rotation=35, ha='right', fontsize=7)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.tick_params(length=2, pad=2)

        for ri in range(n_rows):
            for ci2 in range(n_c):
                mv = m_chunk[ri, ci2]
                mdv = med_chunk[ri, ci2]
                if np.isfinite(mv):
                    txt = f'{mv:.1f}{unit_str}\n({mdv:.1f}{unit_str})' if np.isfinite(mdv) else f'{mv:.2f}{unit_str}'
                    contrast = 'white' if mv > 0.55 * vmax else 'black'
                    ax.text(ci2, ri, txt, ha='center', va='center', fontsize=5.5,
                            color=contrast, linespacing=1.1)

        plt.colorbar(im, ax=ax, pad=0.01, fraction=0.015,
                     label=f'Mean {unit_str}')
        if n_chunks > 1:
            ax.set_title(f'Steps {ci * chunk_size + 1}–{ci * chunk_size + n_c}', fontsize=8)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def latex_sample_summary_table(rows1: list[tuple[str, float, float, float, float]], rows3: list[tuple[str, float, float, float, float]]) -> str:
    m1 = {k: (p, f, keep, flag) for k, p, f, keep, flag in rows1}
    m3 = {k: (p, f, keep, flag) for k, p, f, keep, flag in rows3}
    order = [r[0] for r in rows1]

    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Sample retention/flag summary from available \texttt{total\_samples}.}',
        r'\small',
        r'\begin{tabular}{lcccccc}',
        r'\toprule',
        r'& \multicolumn{3}{c}{SATP1} & \multicolumn{3}{c}{SATP3} \\',
        r'Row & Possible samples & Final samples & Flagged \% & Possible samples & Final samples & Flagged \% \\',
        r'\midrule',
    ]
    for k in order:
        p1, f1, _keep1, flag1 = m1.get(k, (np.nan, np.nan, np.nan, np.nan))
        p3, f3, _keep3, flag3 = m3.get(k, (np.nan, np.nan, np.nan, np.nan))
        lines.append(f"{latex_escape(k)} & {fmt_num(p1)} & {fmt_num(f1)} & {fmt_pct(flag1)} & {fmt_num(p3)} & {fmt_num(f3)} & {fmt_pct(flag3)} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def latex_sample_flag_global_table(sf1: dict[str, object], sf3: dict[str, object]) -> str:
    # Table 4: average number of flagged samples for each wafer-band, for SATP1 and SATP3, for the 3 largest cuts (turnarounds, noisy subscans, smurfgaps)
    wb1 = sf1['wafer_band']
    wb3 = sf3['wafer_band']
    g1 = sf1['global'].iloc[0].to_dict()
    step_cols = [c for c in g1.keys() if c.startswith('pct_') and c != 'pct_all_flagged']
    top3 = sorted(step_cols, key=lambda c: -g1[c])[:3]
    step_map = {'pct_turnaround_flags': 'Turnarounds', 'pct_noisy_subscan_flags': 'Noisy subscans', 'pct_smurfgaps': 'Smurfgaps'}
    col_names = [step_map.get(c, c) for c in top3]
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Average flagged sample percentage for each wafer-band and cut type.}',
        r'\small',
        r'\begin{tabular}{llccc|ccc}',
        r'\toprule',
        r'Wafer & Band & \multicolumn{3}{c}{SATP1} & \multicolumn{3}{c}{SATP3} \\',
        r'& & ' + ' & '.join(col_names) + ' & ' + ' & '.join(col_names) + r' \\',
        r'\midrule',
    ]
    for idx, row in wb1.iterrows():
        wafer = row['wafer']
        band = row['band']
        row3 = wb3[(wb3['wafer'] == wafer) & (wb3['band'] == band)]
        if row3.empty:
            continue
        row3 = row3.iloc[0]
        vals1 = [fmt_pct(row[c]/100.0) for c in top3]
        vals3 = [fmt_pct(row3[c]/100.0) for c in top3]
        lines.append(f"{latex_escape(wafer)} & {latex_escape(band)} & " + ' & '.join(vals1) + ' & ' + ' & '.join(vals3) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def latex_sample_flag_band_table(sf1: dict[str, object], sf3: dict[str, object]) -> str:
    b1 = sf1['band'].set_index('band').to_dict(orient='index')
    b3 = sf3['band'].set_index('band').to_dict(orient='index')
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Sample-flag weighted percentage by band (all wafers).}',
        r'\small',
        r'\begin{tabular}{lcc}',
        r'\toprule',
        r'Band & SATP1 flagged \% & SATP3 flagged \% \\',
        r'\midrule',
    ]
    for band in ['f090', 'f150']:
        v1 = float(b1.get(band, {}).get('pct_all_flagged', np.nan))
        v3 = float(b3.get(band, {}).get('pct_all_flagged', np.nan))
        lines.append(f"{latex_escape(band)} & {fmt_pct(v1 / 100.0)} & {fmt_pct(v3 / 100.0)} " + r"\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def latex_noise_q50_global_table(n1: dict[str, object], n3: dict[str, object]) -> str:
    g1 = n1['global'].iloc[0].to_dict()
    g3 = n3['global'].iloc[0].to_dict()
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Noise-q50 weighted means from ObsDb noise columns (global all wafers and bands).}',
        r'\small',
        r'\begin{tabular}{lcc}',
        r'\toprule',
        r'Metric & SATP1 & SATP3 \\',
        r'\midrule',
    ]
    for pol in NOISE_POLS:
        for metric, _label in NOISE_Q50_METRICS:
            col = f'noise_{pol}_{metric}'
            lines.append(f"{latex_escape(col)} & {fmt_num(float(g1.get(col, np.nan)))} & {fmt_num(float(g3.get(col, np.nan)))} \\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def latex_agg_noise_compare_table(a1: dict[str, object], a3: dict[str, object]) -> str:
    s2_1 = a1['summary2'].set_index('band').to_dict(orient='index')
    s2_3 = a3['summary2'].set_index('band').to_dict(orient='index')
    s3_1 = a1['summary3'].iloc[0].to_dict()
    s3_3 = a3['summary3'].iloc[0].to_dict()

    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Aggregate-noise NET summary (units: $\mu$K$\sqrt{s}$).}',
        r'\small',
        r'\begin{tabular}{llcc}',
        r'\toprule',
        r'Scenario & Row & SATP1 mean NET\_array & SATP3 mean NET\_array \\',
        r'\midrule',
    ]
    for band in ['f090', 'f150']:
        v1 = float(s2_1.get(band, {}).get('mean_net_array', np.nan))
        v3 = float(s2_3.get(band, {}).get('mean_net_array', np.nan))
        lines.append(f"2\\_per\\_band & {latex_escape(band)} & {fmt_num(v1)} & {fmt_num(v3)} \\")
    lines.append(
        f"3\\_per\\_band\\_total & all\\_bands & {fmt_num(float(s3_1.get('mean_net_array', np.nan)))} & {fmt_num(float(s3_3.get('mean_net_array', np.nan)))} \\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def gather_plot_paths() -> dict[str, list[str]]:
    out = {'hist': [], 'non_time': [], 'time': [], 'sample_flag': [], 'agg_noise': [], 'step_tables': []}

    # Pre-generated step-table heatmap PNGs (created during build_report before this call).
    tables_dir = ANALYSIS_ROOT / 'step_tables'
    for p in sorted(tables_dir.glob('*.png')):
        out['step_tables'].append(str(p.relative_to(ANALYSIS_ROOT)))

    # Histograms first.
    for tel in ['satp1', 'satp3']:
        for fname in [
            f'{tel}/{tel}_det_counts_per_wafer_band.png',
            f'{tel}/{tel}_det_fractions_per_wafer_band.png',
            f'{tel}/{tel}_det_histograms_by_band.png',
            f'{tel}/{tel}_det_histograms_combined.png',
        ]:
            if (ANALYSIS_ROOT / fname).exists():
                out['hist'].append(fname)
        for p in sorted((ANALYSIS_ROOT / tel).glob(f'{tel}_ws*_f*_cuts_histograms_by_step.png')):
            out['hist'].append(str(p.relative_to(ANALYSIS_ROOT)))

    # Non-time x-axis plots.
    for tel in ['satp1', 'satp3']:
        for band in ['f090', 'f150']:
            for fname in [
                f'{tel}/{tel}_{band}_retention_median_by_wafer.png',
                f'{tel}/{tel}_{band}_cuts_median_by_wafer.png',
            ]:
                if (ANALYSIS_ROOT / fname).exists():
                    out['non_time'].append(fname)

    # Time-axis plots.
    for tel in ['satp1', 'satp3']:
        for fname in [
            f'{tel}/{tel}_det_hours_daily.png',
            f'{tel}/{tel}_det_hours_weekly.png',
            f'{tel}/{tel}_cumulative_hours.png',
            f'{tel}/{tel}_efficiency.png',
            f'{tel}/{tel}_f090_cuts_timeseries_daily_by_wafer.png',
            f'{tel}/{tel}_f150_cuts_timeseries_daily_by_wafer.png',
            f'{tel}/{tel}_f090_cuts_timeseries_daily_grid.png',
            f'{tel}/{tel}_f150_cuts_timeseries_daily_grid.png',
        ]:
            if (ANALYSIS_ROOT / fname).exists():
                out['time'].append(fname)

    # Sample-flag and noise-q50 plots from ISO_V4_SAMPLE_FLAG_ANALYSIS coverage.
    for tel in ['satp1', 'satp3']:
        for fname in [
            f'{tel}/sample_flag_percentages/sample_flag_hists_percent_all_wafers_all_bands.png',
            f'{tel}/sample_flag_percentages/sample_flag_hists_percent_band_f090.png',
            f'{tel}/sample_flag_percentages/sample_flag_hists_percent_band_f150.png',
            f'{tel}/sample_flag_percentages/noise_stats_hists_all_wafers_all_bands.png',
            f'{tel}/sample_flag_percentages/noise_stats_hists_band_f090.png',
            f'{tel}/sample_flag_percentages/noise_stats_hists_band_f150.png',
        ]:
            if (ANALYSIS_ROOT / fname).exists():
                out['sample_flag'].append(fname)

    # Aggregate-noise NET plots from ISO_V4_AGG_NOISE_ANALYSIS coverage.
    for tel in ['satp1', 'satp3']:
        agg_dir = ANALYSIS_ROOT / tel / 'agg_noise'
        # Always include scenario2 (band-level) histograms.
        for p in sorted(agg_dir.glob('agg_noise_hist_scenario2_*.png')):
            out['agg_noise'].append(str(p.relative_to(ANALYSIS_ROOT)))
        # Include a curated subset of scenario1 wafer-band histograms for readability.
        for wafer in AGG_NOISE_SCENARIO1_WAFERS_FOR_REPORT:
            for band in ['f090', 'f150']:
                p = agg_dir / f'agg_noise_hist_scenario1_{wafer}_{band}.png'
                if p.exists():
                    out['agg_noise'].append(str(p.relative_to(ANALYSIS_ROOT)))

    return out


def fig_block(path: str, caption: str) -> str:
    cap = latex_escape(caption)
    lbl = 'fig:' + re.sub(r'[^a-zA-Z0-9]+', '_', path)
    return '\n'.join([
        r'\begin{figure}[H]',
        r'\centering',
        rf'\includegraphics[width=0.96\textwidth,height={REPORT_IMAGE_MAX_HEIGHT_FRACTION}\textheight,keepaspectratio]{{{path}}}',
        rf'\caption{{{cap}}}',
        rf'\label{{{lbl}}}',
        r'\end{figure}',
    ])


def split_tall_plot_for_report(rel_path: str) -> list[str]:
    src = ANALYSIS_ROOT / rel_path
    if not src.exists():
        return [rel_path]

    try:
        img = plt.imread(src)
    except Exception:
        return [rel_path]

    if img.ndim < 2:
        return [rel_path]

    h, w = img.shape[0], img.shape[1]
    if h <= 0 or w <= 0:
        return [rel_path]

    aspect = float(h) / float(w)
    if aspect <= MAX_REPORT_PANEL_ASPECT:
        return [rel_path]

    n_parts = int(math.ceil(aspect / MAX_REPORT_PANEL_ASPECT))
    n_parts = max(2, n_parts)
    part_h = int(math.ceil(h / n_parts))

    stem = src.stem
    suffix = src.suffix
    split_dir = src.parent / 'report_split'
    split_dir.mkdir(parents=True, exist_ok=True)

    out_paths: list[str] = []
    for i in range(n_parts):
        y0 = i * part_h
        y1 = min((i + 1) * part_h, h)
        if y0 >= y1:
            continue
        chunk = img[y0:y1, :, ...]
        out_file = split_dir / f'{stem}_part{i + 1:02d}{suffix}'
        plt.imsave(out_file, chunk)
        out_paths.append(str(out_file.relative_to(ANALYSIS_ROOT)))

    return out_paths or [rel_path]


def expand_plots_for_report_pages(plot_paths: list[str]) -> list[str]:
    expanded: list[str] = []
    for p in plot_paths:
        expanded.extend(split_tall_plot_for_report(p))
    return expanded


def build_report() -> None:
    satp1 = load_telescope_data('satp1', SQLITE_PATHS['satp1'])
    satp3 = load_telescope_data('satp3', SQLITE_PATHS['satp3'])

    sample_flag_satp1 = compute_sample_flag_products('satp1', SQLITE_PATHS['satp1'])
    sample_flag_satp3 = compute_sample_flag_products('satp3', SQLITE_PATHS['satp3'])
    noise_q50_satp1 = compute_noise_q50_products('satp1', SQLITE_PATHS['satp1'])
    noise_q50_satp3 = compute_noise_q50_products('satp3', SQLITE_PATHS['satp3'])
    agg_noise_satp1 = compute_agg_noise_products('satp1')
    agg_noise_satp3 = compute_agg_noise_products('satp3')

    hours1 = compute_hours_stats(satp1)
    hours3 = compute_hours_stats(satp3)

    y1 = compute_detector_yield_rows(satp1)
    y3 = compute_detector_yield_rows(satp3)

    det_steps1, det_cut_rows1 = compute_cut_step_stats(satp1)
    det_steps3, det_cut_rows3 = compute_cut_step_stats(satp3)

    samp_rows1 = compute_sample_summary_rows(satp1)
    samp_rows3 = compute_sample_summary_rows(satp3)

    samp_steps1, samp_proxy_rows1, sample_step_cols_present1 = compute_sample_step_proxy(satp1)
    samp_steps3, samp_proxy_rows3, sample_step_cols_present3 = compute_sample_step_proxy(satp3)

    # Render step-impact tables as heatmap PNGs so they fit on the page
    # and are available as standalone files.
    step_tables_dir = ANALYSIS_ROOT / 'step_tables'
    step_tables_dir.mkdir(parents=True, exist_ok=True)
    _step_png_paths: dict[str, Path] = {}
    for _slug, _title, _steps, _rows in [
        ('satp1_det_cuts',  'SATP1 detector cut fractions by step', det_steps1, det_cut_rows1),
        ('satp3_det_cuts',  'SATP3 detector cut fractions by step', det_steps3, det_cut_rows3),
        ('satp1_samp_proxy', 'SATP1 sample-flag proxy contribution by step', samp_steps1, samp_proxy_rows1),
        ('satp3_samp_proxy', 'SATP3 sample-flag proxy contribution by step', samp_steps3, samp_proxy_rows3),
    ]:
        _p = step_tables_dir / f'{_slug}.png'
        render_step_table_as_png(_title, _steps, _rows, _p, pct=True)
        _step_png_paths[_slug] = _p

    plots = gather_plot_paths()
    plots = {
        k: expand_plots_for_report_pages(v)
        for k, v in plots.items()
    }

    tex_lines: list[str] = [
        r'\documentclass[11pt,a4paper]{article}',
        r'\usepackage[margin=1.8cm]{geometry}',
        r'\usepackage{graphicx}',
        r'\usepackage{booktabs}',
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}',
        r'\usepackage{float}',
        r'\usepackage{pdflscape}',
        r'\usepackage[colorlinks=true,linkcolor=blue]{hyperref}',
        r'\graphicspath{{./}}',
        '',
        r'\title{ISO V4 Detector Retention, Cuts, and Sample-Flag Summary}',
        rf'\date{{{date.today().isoformat()}}}',
        r'\begin{document}',
        r'\maketitle',
        r'\tableofcontents',
        r'\clearpage',
        r'\section{Computation Template and Definitions}',
        r'This report is generated from ObsDb detector-step columns and \texttt{total\_samples}. Definitions:',
        r'\begin{itemize}',
        r'\item Per wafer-band denominators: $N_{\max}^{\mathrm{wb}} = 860$ detectors.',
        r'\item Per band denominators: $N_{\max}^{\mathrm{band}} = 7 \times 860 = 6020$.',
        r'\item All-band denominators: $N_{\max}^{\mathrm{all}} = 2 \times 6020 = 12040$.',
        r'\item Detector retention fraction at final step: $f_{\mathrm{ret}} = N_{\mathrm{final}} / N_{\max}$.',
        r'\item Detector cut increment at step $k$: $c_k = (N_{k-1} - N_k)/N_{\max}$ with $c_0 = (N_{\max}-N_0)/N_{\max}$.',
        r'\item Count distributions are histograms of per-observation detector counts or fractions.',
        r'\item Retention plots show retained detectors by pipeline step; cut plots show incremental losses by step.',
        r'\item Observation hours: $h_{\mathrm{obs}} = \sum \Delta t$ over successful observations.',
        r'\item Detector-hours: $h_{\mathrm{det}} = \sum (N_{\mathrm{final}} \cdot \Delta t)$.',
        r'\item Calendar hours: $h_{\mathrm{cal}} = (t_{\max}-t_{\min})/3600$.',
        r'\item Detector-hour fraction: $h_{\mathrm{det}} / (N_{\max} h_{\mathrm{cal}})$.',
        r'\item Obs-hour fraction: $h_{\mathrm{obs}} / h_{\mathrm{cal}}$.',
        r'\end{itemize}',
        r'For sample accounting, ObsDb currently provides \texttt{total\_samples} but no explicit sample-step columns. We estimate possible samples from a per-observation-band baseline samples-per-detector, then compare final to possible samples. Step-by-step sample impact is reported as a detector-cut-weighted proxy to avoid double counting.',
        '',
        r'\section{Summary Tables}',
        r'\subsection{Detector-Hours and Calendar-Hours}',
        latex_hours_table(hours1, hours3),
        r'\subsection{Detector Yield (separate table)}',
        latex_yield_table(y1, y3, satp1.wafer_names),
        r'\subsection{Detector-Cut Step Impact (Mean/Median by step)}',
        fig_block(str(_step_png_paths['satp1_det_cuts'].relative_to(ANALYSIS_ROOT)),
                  'SATP1 detector cut fractions by step (rows: total, all wafers by band, then each wafer-band). Color = mean %; parentheses = median %.'),
        fig_block(str(_step_png_paths['satp3_det_cuts'].relative_to(ANALYSIS_ROOT)),
                  'SATP3 detector cut fractions by step (rows: total, all wafers by band, then each wafer-band). Color = mean %; parentheses = median %.'),
        r'\subsection{Sample Flags Summary}',
        latex_sample_summary_table(samp_rows1, samp_rows3),
        r'\subsection{Sample-Flag Weighted Percentages (from sample-step columns)}',
        latex_sample_flag_band_table(sample_flag_satp1, sample_flag_satp3),
        latex_sample_flag_global_table(sample_flag_satp1, sample_flag_satp3),
        r'\subsection{Noise q50 Weighted Means (from ObsDb noise columns)}',
        latex_noise_q50_global_table(noise_q50_satp1, noise_q50_satp3),
        r'\subsection{Sample-Flag Step Impact (Proxy)}',
        rf'Sample-step columns present in SATP1 ObsDb: {"yes" if sample_step_cols_present1 else "no"}; SATP3: {"yes" if sample_step_cols_present3 else "no"}.',
        r'Figures below show proxy allocations of total flagged sample fraction across detector-cut steps, weighted by each observation''s incremental detector cuts (non-overlapping by construction).',
        fig_block(str(_step_png_paths['satp1_samp_proxy'].relative_to(ANALYSIS_ROOT)),
                  'SATP1 sample-flag proxy contribution by step (mean/median fractions). Color = mean %; parentheses = median %.'),
        fig_block(str(_step_png_paths['satp3_samp_proxy'].relative_to(ANALYSIS_ROOT)),
                  'SATP3 sample-flag proxy contribution by step (mean/median fractions). Color = mean %; parentheses = median %.'),
        r'\subsection{Aggregate Noise NET Summary}',
        latex_agg_noise_compare_table(agg_noise_satp1, agg_noise_satp3),
        r'\clearpage',
        r'\section{Plots (all plots moved after tables)}',
        r'\subsection{Histograms first}',
    ]

    for p in plots['hist']:
        tex_lines.append(fig_block(p, f'Histogram: {p}'))

    tex_lines += [r'\subsection{Other non-time plots}']
    for p in plots['non_time']:
        tex_lines.append(fig_block(p, f'Non-time plot: {p}'))

    tex_lines += [r'\subsection{Time-axis plots}']
    for p in plots['time']:
        tex_lines.append(fig_block(p, f'Time-axis plot: {p}'))

    tex_lines += [r'\subsection{Sample-Flag and Noise-q50 plots}']
    for p in plots['sample_flag']:
        tex_lines.append(fig_block(p, f'Sample-flag or noise-q50 plot: {p}'))

    tex_lines += [r'\subsection{Aggregate-noise NET plots}']
    for p in plots['agg_noise']:
        tex_lines.append(fig_block(p, f'Aggregate-noise NET distribution plot: {p}'))

    tex_lines += [r'\end{document}']

    tex_path = ANALYSIS_ROOT / 'iso_v4_analysis_report.tex'
    pdf_path = ANALYSIS_ROOT / 'iso_v4_analysis_report.pdf'
    tex_path.write_text('\n'.join(tex_lines), encoding='utf-8')

    for _ in range(2):
        subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-output-directory', str(ANALYSIS_ROOT), str(tex_path)],
            cwd=str(ANALYSIS_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )

    if not pdf_path.exists():
        raise RuntimeError('PDF was not produced.')

    print(f'Wrote: {tex_path}')
    print(f'Wrote: {pdf_path} ({pdf_path.stat().st_size / 1024:.0f} kB)')


if __name__ == '__main__':
    build_report()
