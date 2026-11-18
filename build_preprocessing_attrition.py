#!/usr/bin/env python3
"""Build a database of where every (obs, wafer, band) group went in a preprocess run.

The cuts/stats database (track_cuts_and_stats_parallel.py) replays the pipeline on what
the preprocess archive saved, so it only describes groups that made it into the archive.
Groups whose detectors were all cut, or that raised, never reach the archive. This
script accounts for every group from the run's own outputs, and from the context,
obsdb and metadata it ran with. It uses nothing derived afterwards.

Inputs, all found from one preprocess config:

  config            process_pipe (the step list and which steps select), subobs,
                    archive index, context_file, and jobdb if the run used one
  jobdb             every queued group: state, error tag, visit count
  archive index     process_archive.sqlite: groups that were saved
  errlog.txt        next to the archive index: error class and detail per failure
  run logs          --log: preprocess_tod output. With a per-line worker pid (the
                    '%(asctime)s pid=%(process)d: %(message)s' format) each group is
                    followed through every step ("Running <step>" / "N detectors
                    remaining"), so failed groups get a full per-step funnel. Stock
                    logs (relative times, no pid) interleave workers, so from those
                    only the per-group error and success lines are used.
  context / obsdb   obs metadata, the expected groups for --obs-list, and with
                    --load-meta the detectors loaded per group and det_cal r_frac

Tables (keyed by `run`, a label per config, so several runs can share one database):

  runs        one row per run: inputs, versions, build time
  steps       the run's pipeline steps: index, name, a key that is the same across
              platforms ('name[tag]', repeats numbered), and whether the step selects
  groups      one row per group seen anywhere (jobdb, archive, errlog, logs, obs list)
              with what each source says and a combined `outcome`
  step_dets   per group and step: detectors remaining after the step, start time,
              duration (pid-tagged logs only; the group's last attempt)
  obs_info    the obsdb row of every obs in the run (not `obs`, which is the
              cuts/stats database's table, so the two can share one file)
  group_meta  (--load-meta) per group and det type: loaded dets, finite r_frac, on
              transition / below / above, median r_frac and p_bias on transition
  obs_meta    (--load-meta) per obs: get_meta status, dets, seconds, error

`outcome`: 'done' (in the archive), 'cut' (all dets cut; `outcome_step` is the step),
'error' (raised; `outcome_step` is the step if known, the error class in
`outcome_detail`), 'not_run' (queued but never attempted), 'not_queued' (expected from
--obs-list but not in the jobdb), or 'unknown'.

Usage
-----
    python build_preprocessing_attrition.py \\
        --config .../satp1/preprocess_satp1_init_20261029.yaml \\
        --log '/lcrc/.../satp1_init/logs/preprocess_*.log' \\
        --obs-list .../production/obs_lists/satp1.txt \\
        --db attrition_20261029.sqlite [--load-meta --nproc 8] [--overwrite]

Run once per config, into the same --db. --log takes globs and can be repeated. Light
except for --load-meta (one get_meta per obs, about 2-3 s each): run that on a compute node.
"""

import argparse
import concurrent.futures as cf
import datetime
import glob
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict

import yaml

TUBE_FLAVOR_TO_BANDS = {
    'lf': ['f030', 'f040'],
    'mf': ['f090', 'f150'],
    'uhf': ['f220', 'f280'],
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run TEXT PRIMARY KEY, platform TEXT, config TEXT, config_sha256 TEXT,
    context_file TEXT, archive_index TEXT, jobdb TEXT, errlog TEXT, logs TEXT,
    obs_list TEXT, n_steps INTEGER, rfrac_range TEXT, sotodlib_version TEXT,
    script_rev TEXT, built TEXT);
CREATE TABLE IF NOT EXISTS steps (
    run TEXT, step_idx INTEGER, name TEXT, key TEXT, selects INTEGER,
    PRIMARY KEY (run, step_idx));
CREATE TABLE IF NOT EXISTS groups (
    run TEXT, obs_id TEXT, wafer TEXT, band TEXT,
    expected INTEGER, in_jobdb INTEGER, jstate TEXT, job_error TEXT,
    visit_count INTEGER, visit_time REAL,
    in_archive INTEGER, archive_file TEXT,
    errlog_error TEXT, errlog_detail TEXT, errlog_time REAL, errlog_count INTEGER,
    log_status TEXT, log_fail_step TEXT, log_fail_step_idx INTEGER, log_error TEXT,
    log_file TEXT, log_attempts INTEGER, log_t_start REAL, log_t_end REAL,
    log_n_steps INTEGER, log_n_after_first INTEGER, log_n_final INTEGER,
    n_loaded INTEGER,
    outcome TEXT, outcome_step TEXT, outcome_detail TEXT,
    PRIMARY KEY (run, obs_id, wafer, band));
CREATE TABLE IF NOT EXISTS step_dets (
    run TEXT, obs_id TEXT, wafer TEXT, band TEXT, step_idx INTEGER,
    n_dets INTEGER, t_start REAL, dt REAL,
    PRIMARY KEY (run, obs_id, wafer, band, step_idx));
CREATE TABLE IF NOT EXISTS obs_info (run TEXT, obs_id TEXT, PRIMARY KEY (run, obs_id));
CREATE TABLE IF NOT EXISTS group_meta (
    run TEXT, obs_id TEXT, wafer TEXT, band TEXT, det_type TEXT,
    n_dets INTEGER, n_rfrac INTEGER, n_trans INTEGER, n_below INTEGER, n_above INTEGER,
    med_rfrac_trans REAL, med_pbias_trans REAL,
    PRIMARY KEY (run, obs_id, wafer, band, det_type));
CREATE TABLE IF NOT EXISTS obs_meta (
    run TEXT, obs_id TEXT, status TEXT, n_dets INTEGER, seconds REAL, error TEXT,
    PRIMARY KEY (run, obs_id));
"""
RUN_TABLES = ['runs', 'steps', 'groups', 'step_dets', 'obs_info', 'group_meta', 'obs_meta']


# ----------------------------------------------------------------------------- config

def step_table(process_pipe):
    """(idx, name, key, selects) per step. The key is 'name[tag]', with repeats
    numbered, so the same step matches across platforms whose configs differ in
    length (satp2 has no `move`, so the indices shift)."""
    rows, seen = [], Counter()
    for i, s in enumerate(process_pipe, 1):
        tag = (s.get('glitch_name') or s.get('wrap') or s.get('hwpss_stats')
               or s.get('psd') or s.get('flags') or '')
        for k in ('calc', 'process', 'save'):
            v = s.get(k)
            if isinstance(v, dict) and not tag:
                tag = (v.get('jumps_name') or v.get('jumps_aman') or v.get('total_flags_label')
                       or v.get('hwpss_stats_name') or v.get('wrap_name')
                       or v.get('glitch_flags') or v.get('subtract_name') or '')
        key = s['name'] + (f'[{tag}]' if tag else '')
        seen[key] += 1
        if seen[key] > 1:
            key = f'{key} ({seen[key]})'
        sel = s.get('select')
        rows.append((i, s['name'], key, int(bool(sel)) if sel is not None else 0))
    return rows


def rfrac_range(process_pipe):
    """r_frac transition range from the det_bias_flags step, or (0.2, 0.8)."""
    for s in process_pipe:
        if s['name'] == 'det_bias_flags':
            for k in ('calc', 'select'):
                v = s.get(k)
                if isinstance(v, dict) and 'rfrac_range' in v:
                    return tuple(float(x) for x in v['rfrac_range'])
    return (0.2, 0.8)


def group_columns(configs):
    """Positions of wafer_slot and wafer.bandpass in the subobs grouping."""
    use = list(configs.get('subobs', {}).get('use', ['wafer_slot', 'wafer.bandpass']))
    iw = use.index('wafer_slot') if 'wafer_slot' in use else 0
    ib = use.index('wafer.bandpass') if 'wafer.bandpass' in use else 1
    return iw, ib


# ----------------------------------------------------------------------------- sources

def read_jobdb(path, iw_key='dets:wafer_slot', ib_key='dets:wafer.bandpass'):
    out = {}
    if not path or not os.path.exists(path):
        return out
    c = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    tags = defaultdict(dict)
    for jid, k, v in c.execute('SELECT job_id, key, value FROM tags'):
        tags[jid][k] = v
    for jid, jstate, vc, vt in c.execute('SELECT id, jstate, visit_count, visit_time FROM jobs'):
        t = tags.get(jid, {})
        key = (t.get('obs:obs_id'), t.get(iw_key), t.get(ib_key))
        if None in key:
            continue
        out[key] = dict(jstate=jstate, job_error=t.get('error') or None,
                        visit_count=vc, visit_time=vt)
    c.close()
    return out


def read_archive(path):
    out = {}
    if not path or not os.path.exists(path):
        return out
    c = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    cols = [r[1] for r in c.execute('PRAGMA table_info(map)')]
    need = ['obs:obs_id', 'dets:wafer_slot', 'dets:wafer.bandpass']
    if not all(n in cols for n in need):
        raise RuntimeError(f'{path}: map table lacks {need}; has {cols}')
    q = ('SELECT m.`obs:obs_id`, m.`dets:wafer_slot`, m.`dets:wafer.bandpass`, f.name '
         'FROM map m LEFT JOIN files f ON m.file_id = f.id')
    for o, w, b, fn in c.execute(q):
        out[(o, w, b)] = fn
    c.close()
    return out


ERRLOG_HEAD = re.compile(r'^(\d+(?:\.\d+)?), (\S+), (.*), (\w+)\s*$')
QUOTED = re.compile(r"'([^']+)'")


def read_errlog(path, iw, ib):
    """{group: (error, detail, time, count)}; the last entry for a group wins.
    Each entry is a header line '<time>, <obs_id>, <group>, <error>' followed by
    '\\t<message><traceback>'. For pipeline_step_error the message is the step."""
    out, count = {}, Counter()
    if not path or not os.path.exists(path):
        return out
    cur = None

    def flush():
        if cur is None:
            return
        key, err, t, lines = cur
        detail = next((ln.strip() for ln in lines if ln.strip()), '')
        count[key] += 1
        out[key] = [err, detail[:300], t, count[key]]

    for line in open(path, errors='replace'):
        m = ERRLOG_HEAD.match(line.rstrip('\n'))
        if m:
            flush()
            vals = QUOTED.findall(m.group(3))
            if len(vals) > max(iw, ib):
                key = (m.group(2), vals[iw], vals[ib])
            else:   # cleanup_obs writes 'n/a' for the group
                key = (m.group(2), 'n/a', 'n/a')
            cur = (key, m.group(4), float(m.group(1)), [])
        elif cur is not None:
            cur[3].append(line)
    flush()
    for k in out:
        out[k][3] = count[k]
    return out


# One regex per log format. Both put the level in ' (LEVEL)' at the end.
LOG_PID = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) pid=(\d+): (.*)$')
LOG_STOCK = re.compile(r'^\s*(\d+\.\d+): (.*)$')
LEVEL = re.compile(r' \((DEBUG|INFO|WARNING|ERROR|CRITICAL)\)$')
GROUP_IN = re.compile(r' for (obs_\S+?):\s*(\[.*?\])')
ERROR_HEADS = ('Init Pipeline Step Error', 'Proc Pipeline Step Error', 'Pipeline Run Error',
               'Get configs/context failed', 'Initial layer Pipeline Load Error',
               'Multilayer Pipeline Load Error', 'Executor Future Result Error')
EXC_CLASS = re.compile(r"<class '([^']+)'>:\s*(.*)")


def _abs_time(s):
    return datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S,%f').timestamp()


def parse_logs(paths, iw, ib):
    """Follow groups through preprocess_tod logs.

    Returns {group: attempt}, the last attempt per group over all logs (logs in time
    order), and per-log info. An attempt has status ok / allcut / error / incomplete,
    the failing step and error text, its log, start/end times, and steps as
    [name, t_start, n_after]. Steps are only filled for pid-tagged logs."""
    attempts = {}
    n_attempts = Counter()
    info = []
    for path in paths:
        fmt, n_lines, groups_here = None, 0, set()
        current = {}          # pid -> attempt (pid logs)
        pending = None        # attempt awaiting its reason line
        with open(path, errors='replace') as f:
            for raw in f:
                line = raw.rstrip('\n')
                m = LOG_PID.match(line)
                if m:
                    fmt = fmt or 'pid'
                    t, pid, msg = _abs_time(m.group(1)), int(m.group(2)), m.group(3)
                else:
                    # Stock lines are '<reltime>: <msg>'. Error headers carry no
                    # ' (LEVEL)' suffix (their message continues on the next line),
                    # so the level cannot be required here.
                    m = LOG_STOCK.match(line)
                    if m:
                        fmt = fmt or 'stock'
                        t, pid, msg = None, None, m.group(2)   # relative time: unusable
                    else:
                        # continuation line: reason for the last error header
                        if pending is not None and line.strip():
                            s = LEVEL.sub('', line.strip())
                            ms = re.match(r'Failed at step (\S+)', s)
                            if ms:
                                pending['fail_step'] = ms.group(1)
                            elif not pending['error']:
                                me = EXC_CLASS.search(s)
                                if me:
                                    msg_ = re.sub(r'\d{4,}', '<N>', me.group(2))
                                    pending['error'] = f"{me.group(1).rsplit('.', 1)[-1]}: {msg_[:120]}"
                            if pending['fail_step'] or pending['error']:
                                pending = None
                        continue
                n_lines += 1
                msg = LEVEL.sub('', msg)
                g = current.get(pid) if pid is not None else None

                if msg.startswith('Generating new init db entry') or msg.startswith('Generating new proc db entry'):
                    gm = GROUP_IN.search(msg)
                    if gm and pid is not None:
                        vals = QUOTED.findall(gm.group(2))
                        key = (gm.group(1), vals[iw], vals[ib])
                        if g is not None and g['key'] == key:
                            continue          # proc layer of the same group
                        g = dict(key=key, status='incomplete', fail_step=None, error=None,
                                 log=os.path.basename(path), t_start=t, t_end=None, steps=[])
                        current[pid] = g
                    continue
                if pid is not None and g is not None:
                    if msg.startswith('Running '):
                        g['steps'].append([msg[8:].strip(), t, None])
                        continue
                    if msg.endswith('detectors remaining') and g['steps']:
                        try:
                            g['steps'][-1][2] = int(msg.split()[0])
                        except ValueError:
                            pass
                        continue
                    if msg.startswith('Saving preprocessing axis manager'):
                        g['t_end'] = t
                        continue
                    if msg.startswith('preproc_or_load_group finished successfully'):
                        g['status'] = 'ok'
                        g['t_end'] = g['t_end'] or t
                        attempts[g['key']] = g
                        n_attempts[g['key']] += 1
                        groups_here.add(g['key'])
                        current.pop(pid)
                        continue

                head = next((h for h in ERROR_HEADS if msg.startswith(h)), None)
                gm = GROUP_IN.search(msg) if (head or msg.startswith('Adding future result')) else None
                if not gm:
                    continue
                vals = QUOTED.findall(gm.group(2))
                if len(vals) <= max(iw, ib):
                    continue
                key = (gm.group(1), vals[iw], vals[ib])
                groups_here.add(key)
                if head is None:
                    # 'Adding future result to db' (main process): success in stock logs
                    if fmt == 'stock':
                        a = attempts.get(key)
                        if a is None or a['log'] != os.path.basename(path):
                            a = dict(key=key, status='ok', fail_step=None, error=None,
                                     log=os.path.basename(path), t_start=None, t_end=None, steps=[])
                            n_attempts[key] += 1
                        a['status'] = 'ok'
                        attempts[key] = a
                    continue
                # an error header. In pid logs it closes that worker's attempt.
                if pid is not None and g is not None and g['key'] == key:
                    a = g
                    current.pop(pid)
                else:
                    a = dict(key=key, status='incomplete', fail_step=None, error=None,
                             log=os.path.basename(path), t_start=t, t_end=None, steps=[])
                a['t_end'] = t
                a['status'] = 'allcut' if 'Step Error' in head else 'error'
                if a['status'] == 'error' and a['steps']:
                    a['fail_step'] = a['steps'][-1][0]
                attempts[key] = a
                n_attempts[key] += 1
                pending = a
        # groups still open at the end of a pid log were cut off (walltime, crash)
        for g in current.values():
            if g['key'] not in attempts or attempts[g['key']]['log'] != g['log']:
                attempts[g['key']] = g
                n_attempts[g['key']] += 1
        info.append(dict(log=path, format=fmt, lines=n_lines, groups=len(groups_here)))
    for k, a in attempts.items():
        a['attempts'] = n_attempts[k]
    return attempts, info


def read_obs_list(path):
    out = []
    for line in open(path):
        s = line.strip()
        if s and not s.startswith('#'):
            out.append(s.split()[0])
    return out


# ----------------------------------------------------------------------------- get_meta

_CTX = None


def _meta_worker(context_file, obs_id, lo, hi):
    """get_meta for one obs; per (wafer_slot, bandpass, det type) counts."""
    import numpy as np
    from sotodlib.core import Context
    global _CTX
    if _CTX is None:
        _CTX = Context(context_file)
    t0 = time.time()
    try:
        meta = _CTX.get_meta(obs_id)
    except Exception as e:
        return obs_id, 'fail', None, time.time() - t0, f'{type(e).__name__}: {e}'[:500], []
    di = meta.det_info
    ws = np.asarray(di.wafer_slot).astype(str)
    bp = np.asarray(di.wafer.bandpass).astype(str)
    ty = np.asarray(di.wafer.type).astype(str) if 'type' in di.wafer._fields else np.full(len(ws), 'NA')
    has_cal = 'det_cal' in meta._fields
    r = np.asarray(meta.det_cal.r_frac, dtype=float) if has_cal and 'r_frac' in meta.det_cal._fields else np.full(len(ws), np.nan)
    pb = np.asarray(meta.det_cal.p_bias, dtype=float) if has_cal and 'p_bias' in meta.det_cal._fields else np.full(len(ws), np.nan)
    rows = []
    for key in sorted(set(zip(ws, bp, ty))):
        m = (ws == key[0]) & (bp == key[1]) & (ty == key[2])
        rr, pp = r[m], pb[m]
        fin = np.isfinite(rr)
        tr = fin & (rr > lo) & (rr < hi)
        rows.append((key[0], key[1], key[2], int(m.sum()), int(fin.sum()), int(tr.sum()),
                     int((fin & (rr <= lo)).sum()), int((fin & (rr >= hi)).sum()),
                     float(np.median(rr[tr])) if tr.any() else None,
                     float(np.median(pp[tr])) if tr.any() else None))
    return obs_id, 'ok', int(meta.dets.count), time.time() - t0, None, rows


# ----------------------------------------------------------------------------- outcome

def classify(row):
    if row['in_archive']:
        return 'done', None, None
    step = row['log_fail_step']
    if row['errlog_error'] == 'pipeline_step_error' or row['log_status'] == 'allcut' \
            or row['job_error'] == 'pipeline_step_error':
        if not step and row['errlog_error'] == 'pipeline_step_error':
            step = row['errlog_detail'] or None
        return 'cut', step, None
    err = row['errlog_error'] or row['job_error'] or (row['log_error'] if row['log_status'] == 'error' else None)
    if err or row['log_status'] == 'error':
        return 'error', step, (row['log_error'] or row['errlog_detail'] or err)
    if row['in_jobdb'] and row['jstate'] == 'open' and not row['log_status']:
        return 'not_run', None, None
    if row['expected'] and not row['in_jobdb'] and not row['log_status']:
        return 'not_queued', None, None
    if row['log_status'] == 'incomplete':
        return 'unknown', step, 'started in the log but never finished'
    return 'unknown', None, None


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='preprocess config the run used')
    ap.add_argument('--db', required=True, help='output sqlite (created or appended to)')
    ap.add_argument('--log', action='append', default=[],
                    help='preprocess_tod log file or glob; repeat for several')
    ap.add_argument('--obs-list', help='obs ids the run was meant to cover; adds expected groups '
                    'from the obsdb (wafer_slots_list x tube_flavor bands)')
    ap.add_argument('--run', help='label for this run (default: the config file stem)')
    ap.add_argument('--jobdb', help='override the config jobdb path')
    ap.add_argument('--errlog', help='override errlog path (default: next to the archive index)')
    ap.add_argument('--load-meta', action='store_true',
                    help='get_meta every obs: loaded dets per group and det_cal r_frac counts')
    ap.add_argument('--nproc', type=int, default=1, help='processes for --load-meta')
    ap.add_argument('--meta-max', type=int, help='--load-meta on only the first N obs (testing)')
    ap.add_argument('--overwrite', action='store_true', help='replace this run in --db')
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    configs = yaml.safe_load(open(cfg_path))
    run = args.run or os.path.splitext(os.path.basename(cfg_path))[0]
    iw, ib = group_columns(configs)
    steps = step_table(configs['process_pipe'])
    lo, hi = rfrac_range(configs['process_pipe'])
    archive_index = configs['archive']['index']
    jobdb_path = args.jobdb or configs.get('jobdb')
    errlog_path = args.errlog or os.path.join(os.path.dirname(archive_index), 'errlog.txt')
    logs = []
    for pat in args.log:
        hits = sorted(glob.glob(pat))
        if not hits:
            print(f'[WARN] no log matches {pat}')
        logs.extend(hits)
    logs = sorted(set(logs), key=lambda p: (os.path.getmtime(p), p))

    con = sqlite3.connect(args.db)
    con.executescript(SCHEMA)
    if con.execute('SELECT 1 FROM runs WHERE run = ?', (run,)).fetchone():
        if not args.overwrite:
            sys.exit(f'run {run!r} already in {args.db}; pass --overwrite to replace it')
        for t in RUN_TABLES:
            con.execute(f'DELETE FROM {t} WHERE run = ?', (run,))

    print(f'run {run}: {len(steps)} steps, groups by {configs.get("subobs", {}).get("use")}')
    jobs = read_jobdb(jobdb_path)
    print(f'jobdb {jobdb_path}: {len(jobs)} jobs {dict(Counter(j["jstate"] for j in jobs.values()))}')
    arch = read_archive(archive_index)
    print(f'archive {archive_index}: {len(arch)} groups')
    errs = read_errlog(errlog_path, iw, ib)
    print(f'errlog {errlog_path}: {len(errs)} groups {dict(Counter(e[0] for e in errs.values()))}')
    attempts, loginfo = parse_logs(logs, iw, ib)
    for li in loginfo:
        print(f'log {li["log"]}: format {li["format"]}, {li["groups"]} groups')

    # context and obsdb: obs rows for every obs touched, expected groups from the obs list
    from sotodlib.core import Context
    import sotodlib
    ctx = Context(configs['context_file'])
    platform = None
    obs_ids = set(k[0] for src in (jobs, arch, attempts) for k in src) | \
        set(k[0] for k in errs if k[1] != 'n/a')
    wanted = read_obs_list(args.obs_list) if args.obs_list else []
    obs_ids |= set(wanted)
    obs_rows = {}
    if ctx.obsdb is not None:
        for r in ctx.obsdb.query():
            if r['obs_id'] in obs_ids:
                obs_rows[r['obs_id']] = dict(r)
    missing_obsdb = sorted(obs_ids - set(obs_rows))
    if missing_obsdb:
        print(f'[WARN] {len(missing_obsdb)} obs not in the obsdb, e.g. {missing_obsdb[:3]}')
    platform = next((r.get('telescope') for r in obs_rows.values() if r.get('telescope')), None)
    if platform is None and obs_ids:   # obs_<ctime>_<telescope>_<wafers>
        parts = sorted(obs_ids)[0].split('_')
        platform = parts[2] if len(parts) > 2 else None

    expected = set()
    if wanted:
        for o in wanted:
            r = obs_rows.get(o)
            if r is None:
                continue
            wafers = [w for w in (r.get('wafer_slots_list') or '').split(',') if w]
            bands = TUBE_FLAVOR_TO_BANDS.get(r.get('tube_flavor'), [])
            if not wafers or not bands:
                print(f'[WARN] obsdb has no wafer_slots_list/tube_flavor for {o}; expected groups skipped')
                continue
            expected |= {(o, w, b) for w in wafers for b in bands}
        print(f'obs list {args.obs_list}: {len(wanted)} obs, {len(expected)} expected groups')

    # optional get_meta pass
    meta_rows, obs_meta, n_loaded = [], [], Counter()
    if args.load_meta:
        todo = sorted(o for o in obs_ids if o in obs_rows)[:args.meta_max]
        print(f'get_meta on {len(todo)} obs with {args.nproc} processes ...')
        t0 = time.time()
        with cf.ProcessPoolExecutor(max_workers=args.nproc) as ex:
            futs = [ex.submit(_meta_worker, configs['context_file'], o, lo, hi) for o in todo]
            for i, fut in enumerate(cf.as_completed(futs), 1):
                o, status, ndet, sec, err, rows = fut.result()
                obs_meta.append((run, o, status, ndet, sec, err))
                for w, b, ty, *vals in rows:
                    meta_rows.append((run, o, w, b, ty, *vals))
                    n_loaded[(o, w, b)] += vals[0]
                if i % 100 == 0:
                    print(f'  {i}/{len(todo)} obs, {time.time() - t0:.0f} s')
        print(f'get_meta done in {time.time() - t0:.0f} s, '
              f'{sum(1 for m in obs_meta if m[2] != "ok")} failed')

    # assemble groups
    keys = set(jobs) | set(arch) | set(attempts) | expected | {k for k in errs if k[1] != 'n/a'}
    step_idx = {}
    names = [s[1] for s in steps]
    out_groups, out_steps, mismatched = [], [], 0
    for key in sorted(keys):
        j, a, e = jobs.get(key, {}), attempts.get(key), errs.get(key)
        row = dict(run=run, obs_id=key[0], wafer=key[1], band=key[2],
                   expected=(int(key in expected) if wanted else None),
                   in_jobdb=int(key in jobs), jstate=j.get('jstate'), job_error=j.get('job_error'),
                   visit_count=j.get('visit_count'), visit_time=j.get('visit_time'),
                   in_archive=int(key in arch), archive_file=arch.get(key),
                   errlog_error=e[0] if e else None, errlog_detail=e[1] if e else None,
                   errlog_time=e[2] if e else None, errlog_count=e[3] if e else None,
                   log_status=None, log_fail_step=None, log_fail_step_idx=None, log_error=None,
                   log_file=None, log_attempts=None, log_t_start=None, log_t_end=None,
                   log_n_steps=None, log_n_after_first=None, log_n_final=None,
                   n_loaded=(n_loaded.get(key) if args.load_meta else None))
        if a is not None:
            st = a['steps']
            if st and [s[0] for s in st] != names[:len(st)]:
                mismatched += 1
            if a['status'] == 'allcut' and st and not a['fail_step']:
                a['fail_step'] = st[-1][0]
            fail_idx = None
            if a['status'] in ('allcut', 'error') and st:
                fail_idx = len(st)
            row.update(log_status=a['status'], log_fail_step=a['fail_step'],
                       log_fail_step_idx=fail_idx, log_error=a['error'], log_file=a['log'],
                       log_attempts=a['attempts'], log_t_start=a['t_start'], log_t_end=a['t_end'],
                       log_n_steps=len(st) or None,
                       log_n_after_first=st[0][2] if st else None,
                       log_n_final=st[-1][2] if st else None)
            ends = [s[1] for s in st[1:]] + [a['t_end']]
            for k, ((nm, t_s, n), t_e) in enumerate(zip(st, ends), 1):
                dt = (t_e - t_s) if (t_e is not None and t_s is not None) else None
                out_steps.append((run, *key, k, n, t_s, dt))
        row['outcome'], row['outcome_step'], row['outcome_detail'] = classify(row)
        out_groups.append(row)
    if mismatched:
        print(f'[WARN] {mismatched} groups ran steps in a different order from the config '
              '(sim skips, or the log is from another config)')

    # write
    cols = list(out_groups[0].keys()) if out_groups else []
    con.executemany(f'INSERT INTO groups ({",".join(cols)}) VALUES ({",".join("?" * len(cols))})',
                    [tuple(g[c] for c in cols) for g in out_groups])
    con.executemany('INSERT INTO step_dets VALUES (?,?,?,?,?,?,?,?)', out_steps)
    con.executemany('INSERT INTO steps VALUES (?,?,?,?,?)', [(run, *s) for s in steps])
    con.executemany('INSERT INTO group_meta VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', meta_rows)
    con.executemany('INSERT INTO obs_meta VALUES (?,?,?,?,?,?)', obs_meta)
    if obs_rows:
        have = {r[1] for r in con.execute('PRAGMA table_info(obs_info)')}
        allcols = sorted({c for r in obs_rows.values() for c in r} - {'obs_id'})
        for c in allcols:
            if c not in have:
                con.execute(f'ALTER TABLE obs_info ADD COLUMN "{c}"')
        q = f'INSERT INTO obs_info (run, obs_id, {",".join(chr(34) + c + chr(34) for c in allcols)}) ' \
            f'VALUES ({",".join("?" * (len(allcols) + 2))})'
        con.executemany(q, [(run, o, *[r.get(c) for c in allcols]) for o, r in obs_rows.items()])

    try:
        rev = subprocess.run(['git', '-C', os.path.dirname(os.path.abspath(__file__)), 'rev-parse',
                              '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', os.path.dirname(os.path.abspath(__file__)), 'status',
                                '--porcelain', '--', os.path.basename(__file__)],
                               capture_output=True, text=True).stdout.strip()
        rev = rev + (' (dirty)' if dirty else '')
    except OSError:
        rev = None
    con.execute('INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (
        run, platform, cfg_path, hashlib.sha256(open(cfg_path, 'rb').read()).hexdigest(),
        configs['context_file'], archive_index, jobdb_path, errlog_path, json.dumps(loginfo),
        os.path.abspath(args.obs_list) if args.obs_list else None, len(steps), json.dumps([lo, hi]),
        sotodlib.__version__, rev, datetime.datetime.now().isoformat(timespec='seconds')))
    con.commit()

    # summary and consistency checks
    oc = Counter(g['outcome'] for g in out_groups)
    print(f'\n{len(out_groups)} groups: {dict(oc)}')
    cut = Counter(g['outcome_step'] for g in out_groups if g['outcome'] == 'cut')
    print('cut at step:', dict(cut.most_common()))
    er = Counter(g['outcome_detail'] for g in out_groups if g['outcome'] == 'error')
    if er:
        print('errors:', dict(er.most_common(10)))
    checks = [
        ('jobdb done but not in archive', [g for g in out_groups if g['jstate'] == 'done' and not g['in_archive']]),
        ('in archive but jobdb not done', [g for g in out_groups if g['in_archive'] and g['in_jobdb'] and g['jstate'] != 'done']),
        ('jobdb failed, no errlog entry', [g for g in out_groups if g['jstate'] == 'failed' and not g['errlog_error']]),
        ('errlog step != log step', [g for g in out_groups if g['errlog_error'] == 'pipeline_step_error'
                                     and g['log_fail_step'] and g['errlog_detail'] != g['log_fail_step']]),
        ('cut, but last log step count != 0', [g for g in out_groups if g['outcome'] == 'cut'
                                              and g['log_n_final'] not in (None, 0)]),
        ('in archive, never seen in a log', [g for g in out_groups if g['in_archive'] and not g['log_status']]),
        ('expected, not in jobdb', [g for g in out_groups if g['expected'] == 1 and not g['in_jobdb']]),
        ('in jobdb, not expected', [g for g in out_groups if g['expected'] == 0 and g['in_jobdb']]),
    ]
    for name, rows in checks:
        eg = f' e.g. {rows[0]["obs_id"]} {rows[0]["wafer"]} {rows[0]["band"]}' if rows else ''
        print(f'check: {name}: {len(rows)}{eg}')
    print(f'\nwrote run {run!r} to {args.db}')


if __name__ == '__main__':
    main()
