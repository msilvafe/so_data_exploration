#!/usr/bin/env python3
import argparse
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml
from tqdm import tqdm

from sotodlib.core import metadata
from sotodlib.preprocess import preprocess_util as pp_util
from sotodlib.preprocess import Pipeline

import multiprocessing as mp
mp.set_start_method("spawn", force=True)


CUT_NAMES = [
    '0_starting', '1_dark_dets', '2_fp_flags',
    '3_detcal_nan_cuts', '4_det_bias_flags', '5_trends',
    '6_jumps', '7_jumps', '8_glitches', '9_source_flags',
    '10_glitches', '11_ptp_flags', '12_noise', '13_cut_bad_dist',
    '14_calibrate', '15_noisy_subscan_flags', '16_estimate_t2p',
    '17_noise', '18_noise', '19_inv_var_flags'
]


def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_obsid_to_wafers(db_path: str, bandpass: str):
    db = metadata.ManifestDb(db_path)
    obsid_to_wafers = defaultdict(list)
    for entry in db.inspect({'dets:wafer.bandpass': bandpass}):
        obsid_to_wafers[entry['obs:obs_id']].append(entry['dets:wafer_slot'])
    obsids = sorted(obsid_to_wafers.keys())
    return obsids, obsid_to_wafers


def process_one_obsid(obsid, wafer_slots, cfg_init, cfg_proc, db_path, bandpass, min_survive, cut_names):
    n_cuts = len(cut_names)
    n_tries = 1

    try:
        # Silence the specific sotodlib logger that spams tqdm
        logging.getLogger("sotodlib.core.metadata.loader").setLevel(logging.ERROR)

        # Build contexts/pipelines *inside* the worker to avoid pickling issues
        configs_init, context_init = pp_util.get_preprocess_context(cfg_init)
        pipe_init = Pipeline(configs_init["process_pipe"])

        configs_proc, context_proc = pp_util.get_preprocess_context(cfg_proc)
        pipe_proc = Pipeline(configs_proc["process_pipe"])

        cuts_all_wafer = np.zeros(n_cuts, dtype=float)

        for wafer_slot in wafer_slots:
            dets = {'wafer_slot': wafer_slot, 'wafer.bandpass': bandpass}
            break_group = False
            cuts_group = np.zeros(n_cuts)
            nc = 0

            # ---- init pipe ----
            try:
                meta = context_init.get_meta(obsid, dets=dets)
            except Exception as exc:
                print(f'{wafer_slot} for {obsid} failed to load')
                cuts_all_wafer += cuts_group
                continue
            cuts_group[nc] = meta.dets.count
            nc += 1

            keep_all = np.ones(meta.dets.count, dtype=bool)
            for proc in pipe_init:
                keep = proc.select(meta, in_place=False)
                if isinstance(keep, np.ndarray):
                    keep_all &= keep
                    if keep_all.sum() < min_survive:
                        print(f'{wafer_slot} for {obsid} has {keep_all.sum()} dets left moving on.')
                        break_group = True
                        break
                    cuts_group[nc] = keep_all.sum()
                    nc += 1
            if break_group:
                cuts_all_wafer += cuts_group
                continue


            # ---- proc pipe ----
            try:
                meta = context_proc.get_meta(obsid, dets=dets)
            except Exception as exc:
                cuts_all_wafer += cuts_group
                continue
            keep_all = np.ones(meta.dets.count, dtype=bool)
            for proc in pipe_proc:
                keep = proc.select(meta, in_place=False)
                if isinstance(keep, np.ndarray):
                    keep_all &= keep
                    if keep_all.sum() < min_survive:
                        break_group = True
                        break
                    cuts_group[nc] = keep_all.sum()
                    nc += 1
            if break_group:
                cuts_all_wafer += cuts_group
                continue

            # Guard: if you accidentally produced the wrong length, flag it
            if len(cuts_group) != n_cuts:
                return np.full(n_cuts, np.nan), f"{obsid}: cuts_group length {len(cuts_group)} != {n_cuts}"

            cuts_all_wafer += cuts_group

        # Your requested “catch error”: all 20 cut values equal (exactly)
        if np.allclose(cuts_all_wafer, cuts_all_wafer[0], rtol=0, atol=0):
            return cuts_all_wafer, f"{obsid}: suspicious — all cut totals identical ({cuts_all_wafer[0]})"

        return cuts_all_wafer, None

    except Exception as exc:
        return np.zeros(n_cuts), f"{obsid}: exception {type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to process_archive.sqlite")
    ap.add_argument("--bandpass", default="f150", choices=["f090", "f150"])
    ap.add_argument("--cfg-init", required=True, help="YAML config for init preprocess")
    ap.add_argument("--cfg-proc", required=True, help="YAML config for proc preprocess")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-survive", type=int, default=10)
    ap.add_argument("--out", default="cuts_all.npy")
    ap.add_argument("--out-obsids", default="obsids.npy")
    ap.add_argument("--out-errors", default="errors.txt")
    args = ap.parse_args()

    # global-ish logging defaults
    logging.basicConfig(level=logging.WARNING)

    cfg_init = load_cfg(args.cfg_init)
    cfg_proc = load_cfg(args.cfg_proc)

    obsids, obsid_to_wafers = build_obsid_to_wafers(args.db, args.bandpass)
    cut_names = CUT_NAMES
    n_cuts = len(cut_names)

    cuts_all = np.zeros((len(obsids), n_cuts), dtype=float)
    errors = []

    # Avoid over-subscription in many scientific libs
    # (You can also set these in the sbatch script)
    # os.environ["OMP_NUM_THREADS"] = "1"

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fut_to_i = {
            ex.submit(
                process_one_obsid,
                obsid,
                obsid_to_wafers[obsid],
                cfg_init,
                cfg_proc,
                args.db,
                args.bandpass,
                args.min_survive,
                cut_names,
            ): i
            for i, obsid in enumerate(obsids)
        }

        for fut in tqdm(as_completed(fut_to_i), total=len(fut_to_i), desc="obsids", dynamic_ncols=True):
            i = fut_to_i[fut]
            vec, err = fut.result()
            cuts_all[i, :] = vec
            if err is not None:
                errors.append(err)

    np.save(args.out, cuts_all)
    np.save(args.out_obsids, np.array(obsids, dtype=object))

    with open(args.out_errors, "w") as f:
        for e in errors:
            f.write(e + "\n")

    print(f"Wrote: {args.out}  shape={cuts_all.shape}")
    print(f"Wrote: {args.out_obsids}  n={len(obsids)}")
    print(f"Wrote: {args.out_errors}  n={len(errors)}")


if __name__ == "__main__":
    main()

