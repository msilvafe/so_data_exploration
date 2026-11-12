#!/usr/bin/env python3
"""
Generate per-platform-band summary tables of:
  1) average detector cuts (total, and the top 5 individual cut steps)
  2) average sample-flagged fraction, restricted to uncut/surviving detectors
     (total, and the top 5 individual flagging steps)

Detector cut columns (det_<i>_<step>_ndets) must be ordered by their numeric
step index <i>, NOT by naive lexicographic string sort -- "det_10_..." sorts
before "det_2_..." as plain strings, which silently scrambles the pipeline
order and was found to be a real bug in analyze_cuts_and_stats.py (its
sorted(det_cols)[-1] resolves to step 9 instead of the true final step ~73).

Sample-flag columns (smurfgaps, turnaround_flags, ...) are not numbered; the
"possible samples" denominator per row is reconstructed as
total_samples + sum(flag columns), matching exactly how
track_cuts_and_stats.py's extract_sample_cuts/count_ranges_matrix_samples
built these numbers in the first place (total_samples is already restricted
to uncut/surviving detectors' unflagged sample-time, and each flag column is
the exclusive, non-overlapping sample count removed by that step).

Fractions (of NUM_DETECTORS_PER_BAND) require summing det_step columns
across wafers *within an observation* before averaging/dividing -- the obs
table has one row per (observation, wafer, band) (7 wafers/band), so
averaging a single wafer-row's count and dividing by the all-7-wafers total
would understate every fraction by ~7x. avg_final_frac in the detector cuts
table is computed the same way generate_yield_and_efficiency_plots.py
computes detector yield (sum final_ndets across wafers with final_ndets>10
per observation, then average across observations), so it should match that
plot's reported average yield. avg_total_cut_frac will *not* generally equal
(1 - yield) though: it only counts loss from the tracked pipeline steps
(starting -> final), while (1 - yield) also includes whatever gap already
exists between NUM_DETECTORS_PER_BAND and the "starting" count itself (e.g.
detectors absent before step tracking begins, wafers missing from a given
observation entirely). See avg_starting_frac to see that gap directly.
"""

import sqlite3
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DB_PATHS = {
    'satp1': '/scratch/gpfs/SIMONSOBS/sat-iso/v4/cuts_stats/satp1_20260520/databases/iso_v4_satp1_20260520.sqlite',
    'satp3': '/scratch/gpfs/SIMONSOBS/sat-iso/v4/cuts_stats/satp3_20260520/databases/iso_v4_satp3_20260520.sqlite',
}
VERSION_LABEL = '20260520'

# Versions with no same-version peer to pair into the DB_PATHS table above get
# their own standalone table/PNG pair: same platform_band-row format, their own
# version label in the output filenames. Each entry is (version_label, db_paths).
STANDALONE_PASSES = [
    ('satp1_260702', {
        'satp1': '/scratch/gpfs/SIMONSOBS/sat-iso/v4/cuts_stats/satp1_260702/databases/iso_v4_satp1_260702.sqlite',
    }),
    ('satp2_260702', {
        'satp2': '/scratch/gpfs/SIMONSOBS/sat-iso/v4/cuts_stats/satp2_260702/databases/iso_v4_satp2_260702.sqlite',
    }),
    ('satp3_260702', {
        'satp3': '/scratch/gpfs/SIMONSOBS/sat-iso/v4/cuts_stats/satp3_260702/databases/iso_v4_satp3_260702.sqlite',
    }),
]

OUTPUT_DIR = '/scratch/gpfs/SIMONSOBS/sat-iso/v4/cuts_stats/summary_plots'
TOP_N = 5

# Matches NUM_DETECTORS_PER_BAND in generate_yield_and_efficiency_plots.py --
# keep these in sync if that value ever changes.
NUM_DETECTORS_PER_BAND = 6020

DET_COL_RE = re.compile(r'^det_(\d+)_(.+)_ndets$')


def load_columns(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('PRAGMA table_info(obs)')
    cols = [c[1] for c in cur.fetchall()]
    conn.close()
    return cols


def get_ordered_det_cols(cols):
    """Return (step_index, process_name, col_name) sorted by numeric step index."""
    parsed = []
    for c in cols:
        m = DET_COL_RE.match(c)
        if m:
            parsed.append((int(m.group(1)), m.group(2), c))
    parsed.sort(key=lambda t: t[0])
    return parsed


def get_flag_cols(cols):
    known_prefixes = ('det_', 'noise_', 't2p_')
    known_exact = {'timestamp', 'obs_id', 'wafer', 'band', 'total_samples',
                    'success', 'failure_step', 'error_message'}
    return [c for c in cols if c not in known_exact and not c.startswith(known_prefixes)]


def build_detector_cuts_table(db_paths=DB_PATHS):
    """
    obs rows are per (observation, wafer, band) -- e.g. 7 wafer rows per
    observation+band. To make fractions comparable to the detector-yield plot
    (which sums final_ndets across a wafer > 10 detector filter per
    observation, then divides by NUM_DETECTORS_PER_BAND), we must sum each
    det_step column across wafers *within an observation* first, and only
    then average across observations and divide by NUM_DETECTORS_PER_BAND.
    Dividing a single wafer-row's average directly by NUM_DETECTORS_PER_BAND
    (the all-7-wafers total) would understate every fraction by ~7x.
    """
    rows = []
    for sat, db_path in db_paths.items():
        cols = load_columns(db_path)
        det_steps = get_ordered_det_cols(cols)
        if not det_steps:
            continue
        conn = sqlite3.connect(db_path)
        col_list = ', '.join(f'`{c}`' for _, _, c in det_steps)
        final_col = det_steps[-1][2]
        bands = [r[0] for r in conn.execute(
            "SELECT DISTINCT band FROM obs WHERE total_samples > 0 AND band IS NOT NULL"
        ).fetchall()]
        for band in sorted(bands):
            df = pd.read_sql_query(
                f"SELECT obs_id, wafer, band, {col_list} FROM obs "
                f"WHERE total_samples > 0 AND band = ?",
                conn, params=(band,)
            )
            if df.empty:
                continue

            # Match the yield plot's wafer-quality filter before summing.
            df = df[df[final_col] > 10]
            if df.empty:
                continue

            step_cols = [c for _, _, c in det_steps]
            per_obs_sums = df.groupby('obs_id')[step_cols].sum()
            n_obs = len(per_obs_sums)

            starting_col = det_steps[0][2]
            avg_starting = per_obs_sums[starting_col].mean()
            avg_final = per_obs_sums[final_col].mean()
            avg_total_cut = avg_starting - avg_final

            step_losses = []
            for i in range(len(det_steps) - 1):
                _, _, col_before = det_steps[i]
                step_idx, process_name, col_after = det_steps[i + 1]
                avg_loss = (per_obs_sums[col_before] - per_obs_sums[col_after]).mean()
                step_losses.append((process_name, avg_loss))

            step_losses.sort(key=lambda x: -x[1])
            top_steps = step_losses[:TOP_N]

            row = {
                'platform_band': f'{sat}-{band}',
                'avg_starting_dets': avg_starting,
                'avg_starting_frac': avg_starting / NUM_DETECTORS_PER_BAND,
                'avg_final_dets': avg_final,
                'avg_final_frac': avg_final / NUM_DETECTORS_PER_BAND,
                'avg_total_cut': avg_total_cut,
                'avg_total_cut_frac': avg_total_cut / NUM_DETECTORS_PER_BAND,
                'n_observations': n_obs,
            }
            for rank, (name, val) in enumerate(top_steps, start=1):
                row[f'rank{rank}_step'] = name
                row[f'rank{rank}_avg_cut'] = val
                row[f'rank{rank}_avg_cut_frac'] = val / NUM_DETECTORS_PER_BAND
            rows.append(row)
        conn.close()
    return pd.DataFrame(rows)


def build_sample_flags_table(db_paths=DB_PATHS):
    rows = []
    for sat, db_path in db_paths.items():
        cols = load_columns(db_path)
        flag_cols = get_flag_cols(cols)
        if not flag_cols:
            continue
        conn = sqlite3.connect(db_path)
        col_list = ', '.join(f'`{c}`' for c in flag_cols)
        bands = [r[0] for r in conn.execute(
            "SELECT DISTINCT band FROM obs WHERE total_samples > 0 AND band IS NOT NULL"
        ).fetchall()]
        for band in sorted(bands):
            df = pd.read_sql_query(
                f"SELECT band, total_samples, {col_list} FROM obs "
                f"WHERE total_samples > 0 AND band = ?",
                conn, params=(band,)
            )
            if df.empty:
                continue

            # possible = final unflagged samples (already restricted to uncut
            # detectors) + everything that got flagged along the way
            possible = df['total_samples'] + df[flag_cols].sum(axis=1)
            total_flagged = possible - df['total_samples']

            avg_possible = possible.sum() / len(df)
            avg_total_flagged_frac = total_flagged.sum() / possible.sum()

            flag_fracs = []
            for c in flag_cols:
                frac = df[c].sum() / possible.sum()
                flag_fracs.append((c, frac))
            flag_fracs.sort(key=lambda x: -x[1])
            top_flags = flag_fracs[:TOP_N]

            row = {
                'platform_band': f'{sat}-{band}',
                'avg_possible_samples': avg_possible,
                'avg_final_unflagged_samples': df['total_samples'].sum() / len(df),
                'avg_total_flagged_frac': avg_total_flagged_frac,
                'n_obs_wafer_rows': len(df),
            }
            for rank, (name, frac) in enumerate(top_flags, start=1):
                row[f'rank{rank}_step'] = name
                row[f'rank{rank}_avg_flagged_frac'] = frac
            rows.append(row)
        conn.close()
    return pd.DataFrame(rows)


def detector_cuts_display_table(df):
    """Build the (col_labels, cell_text) shown in the PNG -- shared with the
    display CSV so the two can never drift apart."""
    col_labels = ['Platform-Band', 'Avg Starting Frac', 'Avg Total Cut (starting-final)']
    for r in range(1, TOP_N + 1):
        col_labels.append(f'#{r} Step (avg cut)')

    cell_text = []
    for _, row in df.iterrows():
        line = [row['platform_band'], f"{row['avg_starting_frac']*100:.2f}%", f"{row['avg_total_cut_frac']*100:.2f}%"]
        for r in range(1, TOP_N + 1):
            step = row.get(f'rank{r}_step', '')
            val = row.get(f'rank{r}_avg_cut_frac', np.nan)
            line.append(f"{step} ({val*100:.2f}%)" if step else '')
        cell_text.append(line)
    return col_labels, cell_text


def sample_flags_display_table(df):
    """Build the (col_labels, cell_text) shown in the PNG -- shared with the
    display CSV so the two can never drift apart."""
    col_labels = ['Platform-Band', 'Avg Total Flagged %']
    for r in range(1, TOP_N + 1):
        col_labels.append(f'#{r} Step (avg %)')

    cell_text = []
    for _, row in df.iterrows():
        line = [row['platform_band'], f"{row['avg_total_flagged_frac']*100:.2f}%"]
        for r in range(1, TOP_N + 1):
            step = row.get(f'rank{r}_step', '')
            val = row.get(f'rank{r}_avg_flagged_frac', np.nan)
            line.append(f"{step} ({val*100:.2f}%)" if step else '')
        cell_text.append(line)
    return col_labels, cell_text


def write_display_csv(col_labels, cell_text, out_path):
    pd.DataFrame(cell_text, columns=col_labels).to_csv(out_path, index=False)
    print(f"✓ Created: {out_path}")


def render_detector_cuts_png(df, out_path, version_label=VERSION_LABEL):
    fig, ax = plt.subplots(figsize=(16, 1.2 + 0.9 * len(df)))
    ax.axis('off')

    col_labels, cell_text = detector_cuts_display_table(df)
    # Table cells use newlines between step name and (%) for readability;
    # the CSV keeps them on one line -- re-split here for the PNG only.
    cell_text = [[c.replace(' (', '\n(') for c in line] for line in cell_text]

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#f2f2f2' if r % 2 == 0 else 'white')

    ax.set_title(f'Average Detector Cuts by Platform-Band ({version_label})\n'
                 f'Fraction of {NUM_DETECTORS_PER_BAND} possible detectors/band; '
                 f'top {TOP_N} steps ranked by avg incremental loss',
                 fontsize=12, fontweight='bold', pad=20)
    # Spell out both columns' definitions: a reader should not have to infer
    # that these are fractions of a fixed 6020, nor that "total cut" is a
    # difference of two fractions rather than a fraction of the starting set.
    fig.text(0.5, 0.005,
             r'Starting Frac $=\;\dfrac{N_{\mathrm{dets\ loaded}}}{%d}$        '
             r'Total Cut $=\;\dfrac{N_{\mathrm{loaded}}-N_{\mathrm{uncut\ final}}}{%d}$        '
             r'Step cut $=\;\dfrac{N_{\mathrm{before\ step}}-N_{\mathrm{after\ step}}}{%d}$'
             % (NUM_DETECTORS_PER_BAND, NUM_DETECTORS_PER_BAND, NUM_DETECTORS_PER_BAND),
             ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Created: {out_path}")


def render_sample_flags_png(df, out_path, version_label=VERSION_LABEL):
    fig, ax = plt.subplots(figsize=(16, 1.2 + 0.9 * len(df)))
    ax.axis('off')

    col_labels, cell_text = sample_flags_display_table(df)
    # Table cells use newlines between step name and (%) for readability;
    # the CSV keeps them on one line -- re-split here for the PNG only.
    cell_text = [[c.replace(' (', '\n(') for c in line] for line in cell_text]

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#f2f2f2' if r % 2 == 0 else 'white')

    ax.set_title(f'Average Sample-Flagged Fraction by Platform-Band ({version_label})\n'
                 f'Restricted to uncut/surviving detectors; top {TOP_N} steps ranked by avg flagged fraction '
                 f'(weighted: Σflagged / Σpossible)',
                 fontsize=12, fontweight='bold', pad=20)
    fig.text(0.5, 0.005,
             r'Flagged fraction $=\;\dfrac{\sum_{\mathrm{obs}} N_{\mathrm{flagged\ samples}}}'
             r'{\sum_{\mathrm{obs}} N_{\mathrm{dets\ surviving}} \times N_{\mathrm{samples}}}$'
             '        (summed over observations, not an average of per-observation ratios)',
             ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Created: {out_path}")


def generate_summary_tables(db_paths, version_label):
    det_df = build_detector_cuts_table(db_paths)
    det_csv = os.path.join(OUTPUT_DIR, f'detector_cuts_summary_{version_label}.csv')
    det_df.to_csv(det_csv, index=False)
    print(f"✓ Created: {det_csv}")
    render_detector_cuts_png(det_df, os.path.join(OUTPUT_DIR, f'detector_cuts_summary_{version_label}.png'),
                              version_label=version_label)
    det_col_labels, det_cell_text = detector_cuts_display_table(det_df)
    write_display_csv(det_col_labels, det_cell_text,
                       os.path.join(OUTPUT_DIR, f'detector_cuts_summary_{version_label}_display.csv'))

    print()
    flag_df = build_sample_flags_table(db_paths)
    flag_csv = os.path.join(OUTPUT_DIR, f'sample_flags_summary_{version_label}.csv')
    flag_df.to_csv(flag_csv, index=False)
    print(f"✓ Created: {flag_csv}")
    render_sample_flags_png(flag_df, os.path.join(OUTPUT_DIR, f'sample_flags_summary_{version_label}.png'),
                             version_label=version_label)
    flag_col_labels, flag_cell_text = sample_flags_display_table(flag_df)
    write_display_csv(flag_col_labels, flag_cell_text,
                       os.path.join(OUTPUT_DIR, f'sample_flags_summary_{version_label}_display.csv'))

    print()
    print(det_df.to_string(index=False))
    print()
    print(flag_df.to_string(index=False))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generate_summary_tables(DB_PATHS, VERSION_LABEL)

    # Standalone passes write their own output files and don't touch the
    # paired satp1/satp3 table above.
    for version_label, db_paths in STANDALONE_PASSES:
        missing = [p for p in db_paths.values() if not os.path.exists(p)]
        if missing:
            for p in missing:
                print(f"\n[WARN] database not found, skipping {version_label}: {p}")
            continue
        print()
        print(f"{'='*80}")
        print(f"standalone summary ({version_label})")
        print(f"{'='*80}")
        generate_summary_tables(db_paths, version_label)


if __name__ == '__main__':
    main()
