#!/usr/bin/env python3
"""Render preprocessing-attrition summary tables as PNGs.

Secondary step to track_preprocessing_attrition.py: that script measures the
attrition per platform-datecode and writes
<cuts_stats>/<platform>_<datecode>/analysis/preprocessing_attrition_summary_*.csv;
this one collects those CSVs and renders readable tables.

For each datecode it produces one platform-comparison PNG in
<cuts_stats>/preprocessing_attrition/, holding two stacked tables:

  stage    -- expected / reached init / reached proc / lost before init /
              lost init->proc, each as a count with its share of expected
  causes   -- the top reasons groups never reached the init archive, one
              column per platform, as count with share of that platform's loss

With --per-platform it also writes a single-platform PNG next to each summary
CSV it read.

Usage
-----
    python3 render_preprocessing_attrition_tables.py                 # every datecode found
    python3 render_preprocessing_attrition_tables.py --datecode 260702
    python3 render_preprocessing_attrition_tables.py --per-platform
"""

import argparse
import collections
import csv
import glob
import os
import re
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CUTS_STATS_ROOT = '/scratch/gpfs/SIMONSOBS/sat-iso/v4/cuts_stats'
COMBINED_DIR = f'{CUTS_STATS_ROOT}/preprocessing_attrition'

# Matches the other summary-table PNGs in this repo.
HEADER_BG = '#2c3e50'
ALT_BG = '#f2f2f2'
# Reason labels the tracker emits in parentheses are bookkeeping outcomes, not
# pipeline steps: the run never got to the group, or no log survives to say.
# They are shown but tinted, so a big "(no log coverage...)" number is never
# mistaken for a pipeline failure mode.
NON_FAILURE_BG = '#fdf0d5'

TOP_N_REASONS = 8

STAGE_ROWS = [
    ('expected (obs, wafer, band)', 'expected_groups', False),
    ('reached init archive', 'reached_init', True),
    ('reached proc archive', 'reached_proc', True),
    ('LOST before init archive', 'lost_before_init', True),
    ('LOST init -> proc', 'lost_init_to_proc', True),
]


def read_summary(path):
    """Read one preprocessing_attrition_summary_<label>.csv."""
    metrics, reasons = {}, collections.Counter()
    with open(path) as f:
        for row in csv.DictReader(f):
            metric, value = row['metric'], row['value']
            if metric.startswith('reason:'):
                reasons[metric[len('reason:'):]] = int(value)
            else:
                metrics[metric] = value
    return metrics, reasons


def discover(datecode=None):
    """Find summary CSVs, grouped by datecode then platform."""
    pattern = os.path.join(
        CUTS_STATS_ROOT,
        f'satp?_{datecode or "*"}',
        'analysis',
        'preprocessing_attrition_summary_*.csv',
    )
    found = collections.defaultdict(dict)
    for path in sorted(glob.glob(pattern)):
        # .../satp2_260702/analysis/preprocessing_attrition_summary_satp2_260702.csv
        m = re.search(r'/(satp\d)_([0-9]+)/analysis/', path)
        if not m:
            continue
        found[m.group(2)][m.group(1)] = path
    return found


def fmt(n, denom=None):
    """'12,443\n(8.3%)' -- count over its share, as the prose tables read."""
    if denom:
        return f'{n:,}\n({100 * n / denom:.1f}%)'
    return f'{n:,}'


def is_non_failure(reason):
    return reason.startswith('(')


def build_stage_table(platforms, data):
    col_labels = [''] + [p.upper() for p in platforms]
    cell_text, highlight = [], []
    for label, key, as_frac in STAGE_ROWS:
        row = [label]
        for p in platforms:
            metrics = data[p][0]
            n = int(metrics[key])
            denom = int(metrics['expected_groups']) if as_frac else None
            row.append(fmt(n, denom))
        cell_text.append(row)
        highlight.append(key.startswith('lost'))

    # Whole observations that lost every wafer/band, and log coverage.
    row = ['observations losing every wafer/band']
    for p in platforms:
        metrics = data[p][0]
        row.append(fmt(int(metrics['obs_fully_lost']),
                       int(metrics['obs_expected']) if 'obs_expected' in metrics else None))
    cell_text.append(row)
    highlight.append(True)

    row = ['of the archive the logs explain']
    for p in platforms:
        cov = data[p][0].get('log_coverage_pct', '')
        row.append(f'{float(cov):.1f}%' if cov else 'n/a')
    cell_text.append(row)
    highlight.append(False)

    return col_labels, cell_text, highlight


def build_cause_table(platforms, data, top_n=TOP_N_REASONS):
    """Top reasons across platforms, ranked by their largest single share."""
    lost = {p: int(data[p][0]['lost_before_init']) for p in platforms}
    reasons = {p: data[p][1] for p in platforms}

    all_reasons = set()
    for p in platforms:
        all_reasons |= set(reasons[p])

    def rank(reason):
        return max((reasons[p].get(reason, 0) / lost[p]) if lost[p] else 0
                   for p in platforms)

    ordered = sorted(all_reasons, key=rank, reverse=True)[:top_n]

    col_labels = [''] + [p.upper() for p in platforms]
    cell_text, non_failure = [], []
    for reason in ordered:
        # Wrap rather than truncate: the first column is narrow in the
        # single-platform layout, and the value cells are already two lines
        # tall, so a second line costs nothing.
        lines = textwrap.wrap(reason, 42)[:2]
        if len(reason) > len(' '.join(lines)):
            lines[-1] = lines[-1][:39] + '...'
        row = ['\n'.join(lines)]
        for p in platforms:
            n = reasons[p].get(reason, 0)
            row.append(fmt(n, lost[p]) if n else '--')
        cell_text.append(row)
        non_failure.append(is_non_failure(reason))

    # Everything outside the top N, so the column still adds up.
    row = [f'all other reasons']
    any_rest = False
    for p in platforms:
        rest = sum(n for r, n in reasons[p].items() if r not in ordered)
        row.append(fmt(rest, lost[p]) if rest else '--')
        any_rest = any_rest or bool(rest)
    if any_rest:
        cell_text.append(row)
        non_failure.append(False)

    return col_labels, cell_text, non_failure


def style(table, cell_text, first_col_width=None, highlight=None,
          tint=None, fontsize=9):
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 2.4)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color='white', fontweight='bold')
            continue
        i = r - 1
        if tint and i < len(tint) and tint[i]:
            cell.set_facecolor(NON_FAILURE_BG)
        else:
            cell.set_facecolor(ALT_BG if r % 2 == 0 else 'white')
        if c == 0:
            cell.set_text_props(ha='left', fontweight='bold'
                                if highlight and i < len(highlight) and highlight[i]
                                else 'normal')
        elif highlight and i < len(highlight) and highlight[i]:
            cell.set_text_props(fontweight='bold')


def render(datecode, data, out_path, per_platform_label=None):
    platforms = sorted(data)
    s_cols, s_text, s_hi = build_stage_table(platforms, data)
    c_cols, c_text, c_tint = build_cause_table(platforms, data)

    width = 5.4 + 2.6 * len(platforms)
    h_stage = 0.62 * (len(s_text) + 1)
    h_cause = 0.62 * (len(c_text) + 1)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(width, 1.7 + h_stage + h_cause),
        gridspec_kw={'height_ratios': [h_stage, h_cause]})

    who = per_platform_label.upper() if per_platform_label else 'SATP1 / SATP2 / SATP3'
    fig.suptitle(
        f'Preprocessing attrition --- {who} --- {datecode}\n'
        'Groups lost before reaching the preprocessing archive, and therefore '
        'absent from every cuts_and_stats number',
        fontsize=12.5, fontweight='bold')

    for ax in (ax1, ax2):
        ax.axis('off')

    ax1.set_title('Where (obs, wafer, band) groups are lost  '
                  '[count / share of expected]', fontsize=10.5,
                  fontweight='bold', pad=6)
    t1 = ax1.table(cellText=s_text, colLabels=s_cols, loc='center',
                   cellLoc='center', colWidths=[0.40] +
                   [0.60 / len(platforms)] * len(platforms))
    style(t1, s_text, highlight=s_hi)

    ax2.set_title('Why they never reached the init archive  '
                  '[count / share of that platform\'s loss]', fontsize=10.5,
                  fontweight='bold', pad=6)
    t2 = ax2.table(cellText=c_text, colLabels=c_cols, loc='center',
                   cellLoc='center', colWidths=[0.40] +
                   [0.60 / len(platforms)] * len(platforms))
    style(t2, c_text, tint=c_tint)

    note = ('All percentages are of expected (obs, wafer, band) groups, where expected = '
            'each obs in the list x its obsdb wafer_slots_list x the bands of its tube_flavor.\n'
            '  reached init / proc = groups present in that archive / expected.   '
            'LOST before init = (expected - reached init) / expected.   '
            'LOST init->proc = (reached init - reached proc) / expected.\n'
            'Reason percentages are of that platform\'s LOST-before-init total, not of expected.\n'
            'Shaded rows are bookkeeping outcomes, not pipeline failures: the run never '
            'reached the group, or no surviving log says why.\n'
            'Where "of the archive the logs explain" is well below 100%, the causes are '
            'incomplete rather than wrong -- preprocessing run scripts\n'
            'redirect with ">" to a fixed log path, so each rerun truncates the previous '
            "run's log.")
    fig.text(0.01, 0.005, note, fontsize=7.6, va='bottom', color='#444444')

    fig.tight_layout(rect=[0, 0.05, 1, 0.955])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✓ Created: {out_path}')


def write_display_csv(datecode, data, out_path):
    """Same numbers as the PNG, for pasting into slides or a doc."""
    platforms = sorted(data)
    s_cols, s_text, _ = build_stage_table(platforms, data)
    c_cols, c_text, _ = build_cause_table(platforms, data)
    with open(out_path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow([f'preprocessing attrition {datecode}'])
        wr.writerow(s_cols)
        for row in s_text:
            wr.writerow([c.replace('\n', ' ') for c in row])
        wr.writerow([])
        wr.writerow(['why never reached init archive'] + c_cols[1:])
        for row in c_text:
            wr.writerow([c.replace('\n', ' ') for c in row])
    print(f'✓ Created: {out_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datecode', help='e.g. 260702; default is every datecode found')
    ap.add_argument('--per-platform', action='store_true',
                    help='also write a single-platform PNG beside each summary CSV')
    ap.add_argument('--output-dir', default=COMBINED_DIR)
    args = ap.parse_args()

    found = discover(args.datecode)
    if not found:
        print('No preprocessing_attrition_summary_*.csv found under '
              f'{CUTS_STATS_ROOT}/satp?_{args.datecode or "*"}/analysis/.')
        print('Run track_preprocessing_attrition.py first.')
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    for datecode, by_platform in sorted(found.items()):
        data = {p: read_summary(path) for p, path in by_platform.items()}
        print(f'\n=== {datecode}: {", ".join(sorted(data))} '
              f'({len(data)} platform(s))')

        out = os.path.join(args.output_dir,
                           f'preprocessing_attrition_comparison_{datecode}.png')
        render(datecode, data, out)
        write_display_csv(
            datecode, data,
            os.path.join(args.output_dir,
                         f'preprocessing_attrition_comparison_{datecode}_display.csv'))

        if args.per_platform:
            for platform, path in sorted(by_platform.items()):
                render(datecode, {platform: data[platform]},
                       os.path.join(os.path.dirname(path),
                                    f'preprocessing_attrition_{platform}_{datecode}.png'),
                       per_platform_label=platform)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
