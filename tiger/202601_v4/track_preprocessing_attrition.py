#!/usr/bin/env python3
"""Track observations lost in preprocessing before they reach the archives.

The cuts_and_stats database can only describe (obs, wafer, band) groups that
preprocessing actually wrote to its archives. Groups that died inside the
preprocessing pipeline never appear there at all, so their loss is invisible in
every downstream yield/efficiency number -- those numbers are conditioned on
"made it into the archive". This script measures that missing stage.

For one processing version it compares three populations:

  expected   -- every (obs, wafer, band) implied by the submitted obs list,
                using each observation's real wafer_slots_list from the obsdb
                and the bands implied by its tube_flavor
  in init    -- groups present in the init archive's process_archive.sqlite
  in proc    -- groups present in the proc archive's process_archive.sqlite

and attributes each missing group to a failing pipeline step by parsing the
preprocessing run logs, which report failures in three forms:

  "Init Pipeline Step Error for <obs>: [ws, band]" + "Failed at step <step>"
  "Proc Pipeline Step Error for <obs>: [ws, band]" + "Failed at step <step>"
  "Pipeline Run Error for <obs>: [ws, band]"       + "<class 'Exc'>: message"

Archive membership is authoritative for the counts; the logs only supply the
reason. A group can appear in the logs and still be in the archive (a later
retry succeeded), and a group can be missing with no log line at all (the run
never reached it, e.g. it was killed by a walltime limit), so both are reported
separately rather than being assumed to agree.

Usage
-----
    # built-in versions
    python3 track_preprocessing_attrition.py --version satp2_260702
    python3 track_preprocessing_attrition.py --version all

    # explicit paths
    python3 track_preprocessing_attrition.py \
        --telescope satp2 --label satp2_260702 \
        --obs-list  .../satp2_260702_obslist_20261103.txt \
        --context   .../contexts/satp2/use_this_local_260824.yaml \
        --init-archive .../satp2_20260702_init/process_archive.sqlite \
        --proc-archive .../satp2_20260702_proc/process_archive.sqlite \
        --log .../log_preproc_satp2_20260702 \
        --log .../log_preproc_satp2_20260702_update
"""

import argparse
import collections
import csv
import datetime
import os
import re
import sqlite3
import sys

PREPROC_ROOT = '/scratch/gpfs/SIMONSOBS/sat-iso/v4/preprocessing'
CUTS_STATS_ROOT = '/scratch/gpfs/SIMONSOBS/sat-iso/v4/cuts_stats'
REPO_ROOT = os.path.expanduser('~/repos/iso-sat/v4')

# Bands carried by each tube flavor. Mirrors track_cuts_and_stats_parallel.py;
# note obsdb spells the UHF flavor 'uhf', not 'uf'.
TUBE_FLAVOR_TO_BANDS = {
    'lf': ['f030', 'f040'],
    'mf': ['f090', 'f150'],
    'uhf': ['f220', 'f280'],
}

# Known processing versions. Several logs per version is normal: the original
# full-survey run plus each incremental update that extended the same archive.
VERSIONS = {
    'satp1_260702': dict(
        telescope='satp1',
        obs_list=f'{CUTS_STATS_ROOT}/satp1_260702/satp1_260702_obslist_20261103.txt',
        context=f'{REPO_ROOT}/contexts/satp1/use_this_local_261102.yaml',
        init_archive=f'{PREPROC_ROOT}/satp1_20260702_init/process_archive.sqlite',
        proc_archive=f'{PREPROC_ROOT}/satp1_20260702_proc/process_archive.sqlite',
        # The first two logs are this archive's own runs, but between them they
        # only speak about 2025-11..2026-11 while the obs list spans
        # 2024-05..2026-11, leaving 7,690 pre-init losses unattributable
        # ("(no log coverage for this epoch)"). The run scripts redirect with
        # '>' to a fixed path, so the earlier 260702 runs' logs were truncated.
        #
        # The remaining logs are from EARLIER ARCHIVE GENERATIONS and do cover
        # 2024-05..2025-10. Adding them takes log coverage of this archive from
        # 23.1% to 96.4% and drops the unattributed count to zero. Their
        # configs differ from 260702's, so for this archive the reasons they
        # supply are INDICATIVE, NOT AUTHORITATIVE -- they corroborate well
        # (the same three steps dominate where 260702's own logs survive), but
        # do not quote them as this archive's measured failure modes without
        # that caveat. See ANOMALIES_260702.md section 7.
        logs=[
            f'{PREPROC_ROOT}/log_preproc_satp1_20260702',
            f'{PREPROC_ROOT}/log_preproc_satp1_20260702_update',
            f'{PREPROC_ROOT}/log_preproc_satp1_20251216',
            f'{PREPROC_ROOT}/log_preproc_satp1_20260330',
            f'{PREPROC_ROOT}/log_preproc_satp1_20260417',
            f'{PREPROC_ROOT}/log_preproc_satp1_20260425',
            f'{PREPROC_ROOT}/log_preproc_satp1_20260520',
        ],
    ),
    'satp2_260702': dict(
        telescope='satp2',
        obs_list=f'{CUTS_STATS_ROOT}/satp2_260702/satp2_260702_obslist_20261103.txt',
        context=f'{REPO_ROOT}/contexts/satp2/use_this_local_260824.yaml',
        init_archive=f'{PREPROC_ROOT}/satp2_20260702_init/process_archive.sqlite',
        proc_archive=f'{PREPROC_ROOT}/satp2_20260702_proc/process_archive.sqlite',
        logs=[
            f'{PREPROC_ROOT}/log_preproc_satp2_20260702',
            f'{PREPROC_ROOT}/log_preproc_satp2_20260702_update',
        ],
    ),
    'satp3_260702': dict(
        telescope='satp3',
        obs_list=f'{CUTS_STATS_ROOT}/satp3_260702/satp3_260702_obslist_20261103.txt',
        context=f'{REPO_ROOT}/contexts/satp3/use_this_local_261102.yaml',
        # The _temp dirs are the live ones the _update configs write to; the
        # non-_temp pair is an older partial run.
        init_archive=f'{PREPROC_ROOT}/satp3_20260702_init_temp/process_archive.sqlite',
        proc_archive=f'{PREPROC_ROOT}/satp3_20260702_proc_temp/process_archive.sqlite',
        # NOTE: preprocessing run scripts redirect with '>' to a fixed log
        # path, so each rerun truncates the previous run's log. The archive's
        # h5 files were written 2026-07-19 but log_..._temp only speaks about
        # obs from 2025-11 on, so most of this archive has no surviving log and
        # its losses cannot be attributed to a step. Use '>>' or a %j-stamped
        # log name in the run script to keep this from recurring.
        # As for satp1: the first three are this archive's own runs and leave a
        # large unattributed residual; the rest are earlier generations added
        # to cover the gap, and their reasons are indicative, not
        # authoritative. See ANOMALIES_260702.md section 7.
        logs=[
            f'{PREPROC_ROOT}/log_preproc_satp3_20260702',
            f'{PREPROC_ROOT}/log_preproc_satp3_20260702_temp',
            f'{PREPROC_ROOT}/log_preproc_satp3_20260702_temp_update',
            f'{PREPROC_ROOT}/log_preproc_satp3_20251216',
            f'{PREPROC_ROOT}/log_preproc_satp3_20260330',
            f'{PREPROC_ROOT}/log_preproc_satp3_20260330_wgcal',
            f'{PREPROC_ROOT}/log_preproc_satp3_20260425_noazss',
            f'{PREPROC_ROOT}/log_preproc_satp3_20260501',
            f'{PREPROC_ROOT}/log_preproc_satp3_20260520',
        ],
    ),
}

# Results belong with the rest of that version's analysis products, so each
# version writes into its own cuts_stats analysis directory by default. A
# combined summary across versions also goes in the shared directory below.
def default_output_dir(label):
    return f'{CUTS_STATS_ROOT}/{label}/analysis'


COMBINED_DIR = f'{CUTS_STATS_ROOT}/preprocessing_attrition'


# ---------------------------------------------------------------- populations

def read_obs_list(path):
    """Read an obs list, one obs_id per line (extra columns ignored)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                out.append(line.split()[0])
    return out


def build_expected(obs_ids, context_path):
    """Expand obs ids into the (obs, wafer, band) groups preprocessing should
    have produced, using each observation's own wafer list and tube flavor.

    Returns (expected set, per-obs info dict, list of ids absent from obsdb).
    """
    from sotodlib.core import Context

    cwd = os.getcwd()
    try:
        # Contexts use relative tag paths, so resolve them from the repo root.
        os.chdir(REPO_ROOT)
        ctx = Context(context_path)
        rows = ctx.obsdb.query()
    finally:
        os.chdir(cwd)

    info = {}
    for r in rows:
        wafers = [w for w in (r['wafer_slots_list'] or '').split(',') if w]
        info[r['obs_id']] = dict(
            wafers=wafers,
            bands=TUBE_FLAVOR_TO_BANDS.get(r['tube_flavor'], []),
            timestamp=r['timestamp'],
            subtype=r['subtype'],
        )

    expected = set()
    not_in_obsdb = []
    for oid in obs_ids:
        meta = info.get(oid)
        if meta is None:
            not_in_obsdb.append(oid)
            continue
        for w in meta['wafers']:
            for b in meta['bands']:
                expected.add((oid, w, b))
    return expected, info, not_in_obsdb


def read_archive(path):
    """Read the (obs, wafer, band) groups recorded in a process_archive.sqlite."""
    if not os.path.exists(path):
        return None
    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        cols = [r[1] for r in con.execute('PRAGMA table_info(map)')]
        if 'dets:wafer_slot' not in cols:
            raise RuntimeError(f'{path}: map table has unexpected columns {cols}')
        return {
            (o, w, b) for o, w, b in con.execute(
                'SELECT "obs:obs_id", "dets:wafer_slot", "dets:wafer.bandpass" FROM map'
            )
        }
    finally:
        con.close()


# -------------------------------------------------------------- log attribution

# Any log line naming a single (obs, wafer, band). Covers both the failures
#   "  19.310: Init Pipeline Step Error for obs_1788968110_satp2_1111011: [np.str_('ws3'), np.str_('f220')]"
# and the successes
#   "   3.301: Adding future result to db for obs_1723355324_satp3_1111111: [np.str_('ws2'), np.str_('f090')]"
# so the set of groups the run actually attempted can be recovered.
GROUP_LINE = re.compile(
    r'(?P<kind>[A-Za-z][A-Za-z ]*?) for (?P<obs>\S+?):\s*\['
    r"(?:np\.str_\()?'(?P<wafer>[^']+)'\)?,\s*"
    r"(?:np\.str_\()?'(?P<band>[^']+)'\)?\]"
)
FAILED_AT_STEP = re.compile(r'Failed at step (?P<step>[A-Za-z0-9_]+)')
EXC_CLASS = re.compile(r"<class '(?P<exc>[^']+)'>:\s*(?P<msg>.*)")


def parse_logs(paths):
    """Map (obs, wafer, band) -> (kind, reason) from preprocessing run logs.

    The reason sits on the line after the header. Later entries win, so for a
    group that was retried the final outcome is the one reported.
    """
    attributed = {}
    attempted = set()
    counts = collections.Counter()
    missing_logs = []
    log_ranges = []   # (basename, n_obs, min_ctime, max_ctime)

    # Log files are listed authoritative-first: the archive's OWN run logs,
    # then older-generation logs added only to raise coverage. A reason from an
    # older generation must NEVER overwrite one from the archive's own run --
    # those older runs used superseded metadata (e.g. det_cal 251020m, which
    # ends 2026-01-08), so in them every post-boundary group died with a
    # LoaderError that has nothing to do with why it failed in THIS archive.
    # Letting that overwrite mislabelled 1,883 satp1 + 1,797 satp3 groups as
    # metadata losses when their own log records an ordinary step failure.
    # Within one file later entries still win, so a rerun supersedes its
    # earlier attempt as before.
    for path in paths:
        if not os.path.exists(path):
            missing_logs.append(path)
            continue
        pending = None  # header awaiting its reason line
        seen_here = set()
        claimed_here = set()   # keys this file has attributed
        with open(path, errors='replace') as f:
            for line in f:
                if pending is not None:
                    key, kind = pending
                    pending = None
                    m = FAILED_AT_STEP.search(line)
                    if m:
                        reason = m.group('step')
                    else:
                        m = EXC_CLASS.search(line)
                        if m:
                            # Keep the exception type plus a trimmed message:
                            # the full LoaderError text embeds the whole obsdb
                            # row and is useless for grouping.
                            exc = m.group('exc').rsplit('.', 1)[-1]
                            # Messages often embed the obs_id/ctime, which
                            # would make every occurrence its own category;
                            # collapse long digit runs so they group.
                            msg = re.sub(r'\d{4,}', '<N>', m.group('msg').strip())
                            reason = f"{exc}: {msg[:60]}"
                        else:
                            reason = 'unknown'
                    if key in attributed and key not in claimed_here:
                        # Already attributed by a higher-priority (earlier) log.
                        continue
                    attributed[key] = (kind, reason)
                    claimed_here.add(key)
                    counts[os.path.basename(path)] += 1
                    continue

                m = GROUP_LINE.search(line)
                if not m:
                    continue
                key = (m.group('obs'), m.group('wafer'), m.group('band'))
                # Every line naming a group means the run reached it, whether
                # it then succeeded ("Adding future result to db...") or failed.
                # That distinguishes "tried and failed" from "never attempted".
                attempted.add(key)
                seen_here.add(key[0])
                kind = m.group('kind').strip()
                if kind.endswith('Error'):
                    pending = (key, kind)

        if seen_here:
            times = [obs_ctime(o) for o in seen_here]
            times = [t for t in times if t is not None]
            if times:
                log_ranges.append((os.path.basename(path), len(seen_here),
                                   min(times), max(times)))

    return attributed, attempted, counts, missing_logs, log_ranges


# ------------------------------------------------------------------- reporting

def obs_ctime(obs_id):
    """ctime embedded in <obs|oper>_<ctime>_<telescope>_<wafers>."""
    m = re.match(r'[A-Za-z]+_(\d+)_', obs_id)
    return int(m.group(1)) if m else None


def month_of(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.UTC).strftime('%Y-%m')


def analyze(label, cfg, output_dir):
    print('=' * 78)
    print(f'preprocessing attrition: {label}')
    print('=' * 78)

    obs_ids = read_obs_list(cfg['obs_list'])
    expected, info, not_in_obsdb = build_expected(obs_ids, cfg['context'])
    in_init = read_archive(cfg['init_archive'])
    in_proc = read_archive(cfg['proc_archive'])

    if in_init is None:
        print(f"[ERROR] init archive not found: {cfg['init_archive']}")
        return None
    if in_proc is None:
        print(f"[WARN] proc archive not found: {cfg['proc_archive']}")
        in_proc = set()

    # Restrict archive sets to the obs we asked about, so an archive holding
    # unrelated observations doesn't inflate the comparison.
    wanted = set(obs_ids)
    in_init_w = {g for g in in_init if g[0] in wanted}
    in_proc_w = {g for g in in_proc if g[0] in wanted}

    never_init = expected - in_init_w
    init_not_proc = in_init_w - in_proc_w
    # Groups the archive has but we never expected: a wafer/band outside the
    # obsdb wafer_slots_list. Reported because it means one of the two sources
    # disagrees, not because it is attrition.
    unexpected = in_init_w - expected

    (attributed, attempted, per_log, missing_logs,
     log_ranges) = parse_logs(cfg['logs'])
    for p in missing_logs:
        print(f'[WARN] log not found, reasons from it unavailable: {p}')

    n_exp = len(expected)
    print(f'\nobs ids in list                 : {len(obs_ids)}')
    if not_in_obsdb:
        print(f'  not present in obsdb          : {len(not_in_obsdb)}')
    print(f'expected (obs, wafer, band)     : {n_exp}')
    print(f'reached init archive            : {len(in_init_w)}'
          f'  ({100 * len(in_init_w) / n_exp:.1f}% of expected)')
    print(f'reached proc archive            : {len(in_proc_w)}'
          f'  ({100 * len(in_proc_w) / n_exp:.1f}% of expected)')
    print(f'\nLOST before the init archive    : {len(never_init)}'
          f'  ({100 * len(never_init) / n_exp:.1f}% of expected)')
    print(f'LOST between init and proc      : {len(init_not_proc)}'
          f'  ({100 * len(init_not_proc) / n_exp:.1f}% of expected)')
    if unexpected:
        print(f'[note] in init archive but not expected: {len(unexpected)}')

    # Attribution for the groups that never reached init. Absence of an error
    # line is not the same as absence of an attempt, so split those apart:
    # a group the run never touched is a scheduling/coverage gap, not a
    # pipeline failure, and the two call for different follow-up.
    attempted_obs = {g[0] for g in attempted}

    def in_logged_period(obs_id):
        """Does any available log cover this observation's epoch?

        An archive is usually built by several runs over time, and the logs for
        the earlier ones may be absent or have been overwritten. Inside a
        covered window, a group the logs never mention really was not attempted
        by that run; outside one, nothing can be concluded.
        """
        ts = obs_ctime(obs_id)
        if ts is None:
            return False
        return any(lo <= ts <= hi for _, _, lo, hi in log_ranges)

    def classify(g):
        entry = attributed.get(g)
        if entry:
            return entry[1]
        if g in attempted:
            return '(attempted, no failure logged)'
        if g[0] in attempted_obs:
            return '(this wafer/band not attempted)'
        if in_logged_period(g[0]):
            return '(not attempted in the logged run)'
        return '(no log coverage for this epoch)'

    print('\nlog coverage (obs epochs each available log speaks about):')
    if not log_ranges:
        print('  none -- no reasons can be attributed')
    for name, n, lo, hi in sorted(log_ranges, key=lambda r: r[2]):
        print(f'  {name:46} {n:6} obs  '
              f'{month_of(lo)} .. {month_of(hi)}')
    archived_seen = len(in_init_w & attempted)
    if in_init_w:
        frac = 100 * archived_seen / len(in_init_w)
        print(f'  groups in the init archive that the logs also mention: '
              f'{archived_seen}/{len(in_init_w)} ({frac:.1f}%)')
        if frac < 99:
            print('  [WARN] the logs do not account for the whole archive, so '
                  'absence of a log line\n         cannot be read as absence '
                  'of an attempt')

    by_reason = collections.Counter(classify(g) for g in never_init)

    print(f'\nwhy the {len(never_init)} never reached the init archive:')
    print(f"  {'reason':52} {'groups':>7}  {'share':>6}")
    for reason, n in by_reason.most_common():
        print(f'  {reason[:52]:52} {n:7}  {100 * n / max(len(never_init), 1):5.1f}%')

    # Per-month view: shows whether loss is uniform or concentrated in a period.
    per_month_exp = collections.Counter()
    per_month_lost = collections.Counter()
    for oid, w, b in expected:
        per_month_exp[month_of(info[oid]['timestamp'])] += 1
    for oid, w, b in never_init:
        per_month_lost[month_of(info[oid]['timestamp'])] += 1

    print('\nloss before init archive, by month:')
    print(f"  {'month':9} {'expected':>9} {'lost':>7}  {'lost%':>6}")
    for m in sorted(per_month_exp):
        e, l = per_month_exp[m], per_month_lost[m]
        print(f'  {m:9} {e:9} {l:7}  {100 * l / e:5.1f}%')

    # Whole observations lost outright, i.e. not a single wafer/band survived.
    lost_obs = collections.Counter(g[0] for g in never_init)
    exp_obs = collections.Counter(g[0] for g in expected)
    fully_lost = sorted(o for o, n in lost_obs.items() if n == exp_obs[o])
    print(f'\nobservations with no surviving group at all: {len(fully_lost)}'
          f' of {len(exp_obs)}')

    os.makedirs(output_dir, exist_ok=True)

    detail = os.path.join(output_dir, f'preprocessing_attrition_{label}.csv')
    with open(detail, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['obs_id', 'wafer', 'band', 'timestamp', 'month',
                     'lost_stage', 'error_kind', 'reason'])
        for stage, groups in (('before_init', never_init),
                              ('init_to_proc', init_not_proc)):
            for oid, w, b in sorted(groups):
                kind = attributed.get((oid, w, b), ('', ''))[0]
                reason = classify((oid, w, b))
                ts = info[oid]['timestamp']
                wr.writerow([oid, w, b, f'{ts:.0f}', month_of(ts),
                             stage, kind, reason])
    print(f'\n-> {detail}')

    summary = os.path.join(output_dir, f'preprocessing_attrition_summary_{label}.csv')
    with open(summary, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['label', 'metric', 'value'])
        for k, v in [
            ('obs_in_list', len(obs_ids)),
            ('expected_groups', n_exp),
            ('reached_init', len(in_init_w)),
            ('reached_proc', len(in_proc_w)),
            ('lost_before_init', len(never_init)),
            ('lost_init_to_proc', len(init_not_proc)),
            ('obs_fully_lost', len(fully_lost)),
            ('obs_expected', len(exp_obs)),
            # How much of the archive the available logs speak about. Where
            # this is low, the reasons below are incomplete rather than wrong.
            ('log_coverage_pct',
             f'{100 * archived_seen / len(in_init_w):.2f}' if in_init_w else ''),
        ]:
            wr.writerow([label, k, v])
        for reason, n in by_reason.most_common():
            wr.writerow([label, f'reason:{reason}', n])
    print(f'-> {summary}')

    return dict(label=label, expected=n_exp, reached_init=len(in_init_w),
                reached_proc=len(in_proc_w), lost_before_init=len(never_init),
                lost_init_to_proc=len(init_not_proc), by_reason=by_reason)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', help=f"built-in version: {', '.join(VERSIONS)}, or 'all'")
    ap.add_argument('--telescope')
    ap.add_argument('--label')
    ap.add_argument('--obs-list')
    ap.add_argument('--context')
    ap.add_argument('--init-archive')
    ap.add_argument('--proc-archive')
    ap.add_argument('--log', action='append', default=[])
    ap.add_argument('--output-dir', default=None,
                    help='override; default is <cuts_stats>/<label>/analysis')
    args = ap.parse_args()

    if args.version == 'all':
        targets = list(VERSIONS.items())
    elif args.version:
        if args.version not in VERSIONS:
            ap.error(f'unknown --version {args.version!r}; known: {", ".join(VERSIONS)}')
        targets = [(args.version, VERSIONS[args.version])]
    else:
        required = ['label', 'obs_list', 'context', 'init_archive']
        missing = [r for r in required if not getattr(args, r)]
        if missing:
            ap.error('without --version these are required: '
                     + ', '.join('--' + m.replace("_", "-") for m in missing))
        targets = [(args.label, dict(
            telescope=args.telescope, obs_list=args.obs_list, context=args.context,
            init_archive=args.init_archive, proc_archive=args.proc_archive or '',
            logs=args.log))]

    results = [r for r in (analyze(l, c, args.output_dir or default_output_dir(l))
                           for l, c in targets) if r]

    if len(results) > 1:
        print('\n' + '=' * 78)
        print('combined')
        print('=' * 78)
        print(f"  {'version':16} {'expected':>9} {'init':>9} {'proc':>9} "
              f"{'lost pre-init':>14}")
        for r in results:
            print(f"  {r['label']:16} {r['expected']:9} {r['reached_init']:9} "
                  f"{r['reached_proc']:9} {r['lost_before_init']:9} "
                  f"({100 * r['lost_before_init'] / r['expected']:.1f}%)")

        os.makedirs(COMBINED_DIR, exist_ok=True)
        combined = os.path.join(COMBINED_DIR, 'preprocessing_attrition_all_versions.csv')
        with open(combined, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['version', 'expected_groups', 'reached_init', 'reached_proc',
                         'lost_before_init', 'lost_before_init_pct',
                         'lost_init_to_proc', 'lost_init_to_proc_pct',
                         'top_reason', 'top_reason_groups'])
            for r in results:
                top = r['by_reason'].most_common(1)
                wr.writerow([
                    r['label'], r['expected'], r['reached_init'], r['reached_proc'],
                    r['lost_before_init'],
                    f"{100 * r['lost_before_init'] / r['expected']:.2f}",
                    r['lost_init_to_proc'],
                    f"{100 * r['lost_init_to_proc'] / r['expected']:.2f}",
                    top[0][0] if top else '', top[0][1] if top else 0,
                ])
        print(f'\n-> {combined}')

    return 0 if results else 1


if __name__ == '__main__':
    sys.exit(main())
