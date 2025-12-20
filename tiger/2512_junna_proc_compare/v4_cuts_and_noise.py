#!/usr/bin/env python3
"""
Parallel extraction of:
  1) per-detector survival after each *select* step (from init+proc configs)
  2) per-detector noise-fit parameters for any wrap that looks like a noise fit

Output:
  AxisManager with LabelAxes: dets, obsids, cuts, wafer_bands
  - keep[dets, obsids, cuts] : bool (True if det survives through that cut step)
  - det_present[dets, obsids] : bool (det exists in that obsid/bandpass)
  - noise_params[dets, obsids, noise_params] : float (NaN if missing)
  - cut_survival_frac_by_wb[wafer_bands, obsids, cuts]
  - noise_param_mean_by_wb[wafer_bands, obsids, noise_params]

Parallel model:
  - Use sotodlib.utils.procs_pool.get_exec_env(nproc) like your older examples
    (iso_v1_stats_20250718.py / check_dark_dets.py) 
  - Build context/pipeline *inside* workers (like parallel_cuts.py) :contentReference[oaicite:6]{index=6}
"""

import argparse
import os
import traceback
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import yaml

from sotodlib.core import AxisManager, LabelAxis, metadata as core_metadata
from sotodlib.preprocess import preprocess_util as pp_util
from sotodlib.preprocess import Pipeline
from sotodlib.utils.procs_pool import get_exec_env


# ----------------------------
# YAML helpers
# ----------------------------

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def derive_cut_labels_from_cfg(cfg_init: dict, cfg_proc: dict) -> List[str]:
    """
    Build ordered cut labels:
      0_starting,
      then every process_pipe item that has a 'select' key (init then proc),
      numbered in the order they appear.
    """
    cuts = ["0_starting"]
    k = 1

    for step in cfg_init.get("process_pipe", []):
        if "select" in step:
            cuts.append(f"{k}_{step['name']}")
            k += 1

    for step in cfg_proc.get("process_pipe", []):
        if "select" in step:
            cuts.append(f"{k}_{step['name']}")
            k += 1

    return cuts

def build_obsid_to_wafers(db_path: str, bandpass: str) -> Dict[str, List[str]]:
    """
    Use archive manifest DB to map obsid -> wafer_slots for this bandpass.
    Pattern mirrors your parallel_cuts helper :contentReference[oaicite:7]{index=7}.
    """
    db = core_metadata.ManifestDb(db_path)
    obsid_to_wafers = defaultdict(list)
    for entry in db.inspect({"dets:wafer.bandpass": bandpass}):
        obsid_to_wafers[entry["obs:obs_id"]].append(entry["dets:wafer_slot"])
    # de-dupe and sort
    return {o: sorted(set(ws)) for o, ws in obsid_to_wafers.items()}


# ----------------------------
# Noise-fit extraction (robust-ish)
# ----------------------------

def _looks_like_noise_fit(obj: Any) -> bool:
    """
    We don't assume a specific wrap name (noiseT/noiseQ/noiseU/etc).
    We just look for AxisManagers (or similar) that contain white-noise-like fields.
    """
    # AxisManager has ._fields; support both AxisManager and dict-like wrappers
    fields = getattr(obj, "_fields", None)
    if fields is None:
        return False
    keys = set(fields.keys())
    # common: white_noise, fknee, alpha (or wn, f_knee, etc)
    return ("white_noise" in keys) or (("fknee" in keys or "f_knee" in keys) and ("alpha" in keys))

def extract_noise_fit_params(preproc: AxisManager) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns:
      {wrap_name: {"white_noise": arr, "fknee": arr, "alpha": arr}}
    arrays are per-detector (aligned to preproc.dets axis).
    Missing fields are omitted for that wrap.
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name in getattr(preproc, "_fields", {}).keys():
        obj = preproc[name]
        if not _looks_like_noise_fit(obj):
            continue

        d: Dict[str, np.ndarray] = {}
        for key, aliases in [
            ("white_noise", ["white_noise", "wn"]),
            ("fknee", ["fknee", "f_knee"]),
            ("alpha", ["alpha"]),
        ]:
            for a in aliases:
                if hasattr(obj, a):
                    d[key] = np.asarray(getattr(obj, a))
                    break
        if len(d):
            out[name] = d
    return out


# ----------------------------
# Worker: process one (obsid, wafer_slot)
# ----------------------------

def process_one_subobs(
    obsid: str,
    wafer_slot: str,
    bandpass: str,
    cfg_init_path: str,
    cfg_proc_path: str,
    cut_labels: List[str],
) -> Tuple[str, str, str, Optional[dict], Optional[str]]:
    """
    Returns:
      (obsid, wafer_slot, bandpass, payload, err)

    payload = {
      "dets": np.ndarray[str] (ndet,),
      "keep": np.ndarray[bool] (ndet, ncuts),
      "noise": {wrap: {param: np.ndarray[float](ndet,)}},
    }
    """
    try:
        cfg_init = load_yaml(cfg_init_path)
        cfg_proc = load_yaml(cfg_proc_path)

        # Build contexts/pipelines INSIDE worker (matches your newer pattern) :contentReference[oaicite:8]{index=8}
        configs_init, context_init = pp_util.get_preprocess_context(cfg_init)
        pipe_init = Pipeline(configs_init["process_pipe"])

        configs_proc, context_proc = pp_util.get_preprocess_context(cfg_proc)
        pipe_proc = Pipeline(configs_proc["process_pipe"])

        dets_query = {"wafer_slot": wafer_slot, "wafer.bandpass": bandpass}

        # Load meta for init/proc
        meta_init = context_init.get_meta(obsid, dets=dets_query)
        meta_proc = context_proc.get_meta(obsid, dets=dets_query)

        dets = np.array(meta_proc.dets.vals, dtype=str)
        ndet = dets.size
        ncuts = len(cut_labels)

        keep = np.zeros((ndet, ncuts), dtype=bool)
        keep[:, 0] = True  # 0_starting

        # We will fill cuts in the exact order defined by cut_labels,
        # by stepping through init then proc and only advancing when proc.name matches.
        # This avoids ambiguity with repeated names.
        next_cut_i = 1

        def _maybe_record(proc_name: str, mask: np.ndarray, running_keep: np.ndarray):
            nonlocal next_cut_i
            if next_cut_i >= ncuts:
                return
            expected = cut_labels[next_cut_i].split("_", 1)[1]
            if proc_name == expected:
                keep[:, next_cut_i] = running_keep & mask
                next_cut_i += 1

        # ---- init pipe: apply select masks in-order (like parallel_cuts) :contentReference[oaicite:9]{index=9}
        running = np.ones(ndet, dtype=bool)
        for proc in pipe_init:
            sel = proc.select(meta_init, in_place=False)
            if isinstance(sel, np.ndarray):
                running &= sel
                _maybe_record(proc.name, sel, running)  # record after applying

        # ---- proc pipe
        running = np.ones(ndet, dtype=bool)
        for proc in pipe_proc:
            sel = proc.select(meta_proc, in_place=False)
            if isinstance(sel, np.ndarray):
                running &= sel
                _maybe_record(proc.name, sel, running)

        # Noise fits: scan preprocess products for anything that looks like a noise-fit wrap.
        # (Your init has a noise step with fit: True) :contentReference[oaicite:10]{index=10}
        noise = {}
        if hasattr(meta_init, "preprocess"):
            noise.update(extract_noise_fit_params(meta_init.preprocess))
        if hasattr(meta_proc, "preprocess"):
            noise.update(extract_noise_fit_params(meta_proc.preprocess))

        payload = {"dets": dets, "keep": keep, "noise": noise}
        return obsid, wafer_slot, bandpass, payload, None

    except Exception as e:
        errmsg = f"{obsid} {wafer_slot} {bandpass}: {type(e).__name__}: {e}"
        tb = "".join(traceback.format_tb(e.__traceback__))
        return obsid, wafer_slot, bandpass, None, f"{errmsg}\n{tb}"


# ----------------------------
# Rank-0 assembly
# ----------------------------

def wafer_bands_axis() -> np.ndarray:
    ws_list = [f"ws{i}" for i in range(7)]
    wb = [f"f090_{w}" for w in ws_list] + [f"f150_{w}" for w in ws_list] + ["f090_all", "f150_all"]
    return np.array(wb, dtype=str)

def wb_for(wafer_slot: str, bandpass: str) -> str:
    # wafer_slot like "ws0", bandpass "f090"/"f150"
    return f"{bandpass}_{wafer_slot}"

def make_output_axisman(
    obsids: List[str],
    cut_labels: List[str],
    subobs_results: Dict[Tuple[str, str], dict],
) -> AxisManager:
    """
    subobs_results[(obsid, wafer_slot)] = payload from worker
    """
    # Union dets across all subobs
    det_union = set()
    for payload in subobs_results.values():
        det_union.update(payload["dets"])
    dets_all = np.array(sorted(det_union), dtype=str)

    ax_d = LabelAxis("dets", dets_all)
    ax_o = LabelAxis("obsids", np.array(sorted(set(obsids)), dtype=str))
    ax_c = LabelAxis("cuts", np.array(cut_labels, dtype=str))
    ax_wb = LabelAxis("wafer_bands", wafer_bands_axis())

    aman = AxisManager(ax_d, ax_o, ax_c, ax_wb)

    det_index = {d: i for i, d in enumerate(dets_all)}
    obs_index = {o: i for i, o in enumerate(ax_o.vals)}
    wb_index = {w: i for i, w in enumerate(ax_wb.vals)}

    keep = np.zeros((ax_d.count, ax_o.count, ax_c.count), dtype=bool)
    det_present = np.zeros((ax_d.count, ax_o.count), dtype=bool)

    # Build a global "noise_params" axis by union of all wrap/param pairs found.
    noise_keys = set()
    for payload in subobs_results.values():
        for wrap, params in payload["noise"].items():
            for p in params.keys():
                noise_keys.add(f"{wrap}:{p}")
    noise_params = np.array(sorted(noise_keys), dtype=str)
    ax_np = LabelAxis("noise_params", noise_params)
    aman.add_axis(ax_np)

    noise_arr = np.full((ax_d.count, ax_o.count, ax_np.count), np.nan, dtype=float)
    np_index = {n: i for i, n in enumerate(noise_params)}

    # Fill det-level arrays
    for (obsid, wafer_slot), payload in subobs_results.items():
        oi = obs_index[obsid]
        dets = payload["dets"]
        k_sub = payload["keep"]  # (ndet, ncuts)

        gidx = np.array([det_index[d] for d in dets], dtype=int)
        det_present[gidx, oi] = True
        keep[gidx, oi, :] = k_sub

        # noise
        for wrap, params in payload["noise"].items():
            for p, arr in params.items():
                key = f"{wrap}:{p}"
                if key in np_index and arr.shape[0] == dets.shape[0]:
                    noise_arr[gidx, oi, np_index[key]] = arr.astype(float)

    # Wafer-band aggregates
    cut_surv_frac_by_wb = np.full((ax_wb.count, ax_o.count, ax_c.count), np.nan, dtype=float)
    noise_mean_by_wb = np.full((ax_wb.count, ax_o.count, ax_np.count), np.nan, dtype=float)

    # For each obsid, for each wafer_band bucket, aggregate over dets present in that bucket
    for obsid in ax_o.vals:
        oi = obs_index[obsid]
        # group det indices by wafer_slot using subobs_results
        wb_to_gidx = defaultdict(list)

        for (o, ws), payload in subobs_results.items():
            if o != obsid:
                continue
            wb = wb_for(ws, payload.get("bandpass", None) or "")  # might not exist
        # easier: derive from key and obsid, using bandpass via payload
        for (o, ws), payload in subobs_results.items():
            if o != obsid:
                continue
            # bandpass not stored above; reconstruct from wb label:
            # we can infer by taking first det’s bandpass isn't accessible; so store it in payload:
            # (we stored it as function arg in worker return tuple; in main we’ll add it.)
            wb = payload["wafer_band"]
            wi = wb_index.get(wb)
            if wi is None:
                continue
            dets = payload["dets"]
            gidx = [det_index[d] for d in dets]
            wb_to_gidx[wi].extend(gidx)

        # now compute per-wb, and also _all buckets
        for wi, gidx_list in wb_to_gidx.items():
            idx = np.array(sorted(set(gidx_list)), dtype=int)
            if idx.size == 0:
                continue
            cut_surv_frac_by_wb[wi, oi, :] = keep[idx, oi, :].mean(axis=0)
            noise_mean_by_wb[wi, oi, :] = np.nanmean(noise_arr[idx, oi, :], axis=0)

        # all-wafers bucket for each bandpass (f090_all / f150_all)
        for bp in ["f090", "f150"]:
            wi_all = wb_index[f"{bp}_all"]
            # include all wafer slots for this bp
            idxs = []
            for ws in [f"ws{i}" for i in range(7)]:
                wi = wb_index.get(f"{bp}_{ws}")
                if wi is not None and wi in wb_to_gidx:
                    idxs.extend(wb_to_gidx[wi])
            idx = np.array(sorted(set(idxs)), dtype=int)
            if idx.size == 0:
                continue
            cut_surv_frac_by_wb[wi_all, oi, :] = keep[idx, oi, :].mean(axis=0)
            noise_mean_by_wb[wi_all, oi, :] = np.nanmean(noise_arr[idx, oi, :], axis=0)

    aman.wrap("keep", keep, [(0, "dets"), (1, "obsids"), (2, "cuts")])
    aman.wrap("det_present", det_present, [(0, "dets"), (1, "obsids")])
    aman.wrap("noise_params_values", noise_arr, [(0, "dets"), (1, "obsids"), (2, "noise_params")])
    aman.wrap("cut_survival_frac_by_wb", cut_surv_frac_by_wb, [(0, "wafer_bands"), (1, "obsids"), (2, "cuts")])
    aman.wrap("noise_param_mean_by_wb", noise_mean_by_wb, [(0, "wafer_bands"), (1, "obsids"), (2, "noise_params")])

    return aman


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-init", required=True, help="init preprocess YAML")
    ap.add_argument("--cfg-proc", required=True, help="proc preprocess YAML")
    ap.add_argument("--bandpass", required=True, choices=["f090", "f150"])
    ap.add_argument("--nproc", type=int, default=16, help="parallel workers / mpi ranks")
    ap.add_argument("--out", default="preproc_cuts_noise_summary.h5")
    ap.add_argument("--errlog", default="preproc_cuts_noise_errors.txt")
    args = ap.parse_args()

    # Avoid oversubscription (matches your usual Tiger setup)
    for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(v, "1")

    cfg_init = load_yaml(args.cfg_init)
    cfg_proc = load_yaml(args.cfg_proc)

    # Where to read obs list from: proc archive index is explicit in cfg :contentReference[oaicite:11]{index=11}
    db_path = cfg_proc["archive"]["index"]

    cut_labels = derive_cut_labels_from_cfg(cfg_init, cfg_proc)

    obsid_to_wafers = build_obsid_to_wafers(db_path, args.bandpass)
    obsids = sorted(obsid_to_wafers.keys())

    # Make work list = (obsid, wafer_slot)
    tasks = [(o, ws) for o in obsids for ws in obsid_to_wafers[o]]

    rank, executor, as_completed_callable = get_exec_env(args.nproc)

    # Submit tasks
    futures = [
        executor.submit(
            process_one_subobs,
            obsid=o,
            wafer_slot=ws,
            bandpass=args.bandpass,
            cfg_init_path=args.cfg_init,
            cfg_proc_path=args.cfg_proc,
            cut_labels=cut_labels,
        )
        for (o, ws) in tasks
    ]

    # Rank-0 gathers results and writes output
    if rank == 0:
        subobs_results: Dict[Tuple[str, str], dict] = {}
        errors: List[str] = []

        n = 0
        ntot = len(futures)
        for fut in as_completed_callable(futures):
            n += 1
            try:
                obsid, wafer_slot, bandpass, payload, err = fut.result()
            except Exception as e:
                errors.append(f"future.result failed: {type(e).__name__}: {e}\n" +
                              "".join(traceback.format_tb(e.__traceback__)))
                continue

            if err is not None:
                errors.append(err)
                continue

            # stash band/wafer_band for aggregation
            payload["bandpass"] = bandpass
            payload["wafer_band"] = f"{bandpass}_{wafer_slot}"

            subobs_results[(obsid, wafer_slot)] = payload

            if n % 200 == 0 or n == ntot:
                print(f"[rank0] collected {n}/{ntot}")

        # Write errors
        if errors:
            with open(args.errlog, "w") as f:
                f.write("\n\n".join(errors))
            print(f"[rank0] wrote error log: {args.errlog}  (n={len(errors)})")

        # Assemble + save AxisManager
        aman = make_output_axisman(obsids=obsids, cut_labels=cut_labels, subobs_results=subobs_results)
        aman.save(args.out)
        print(f"[rank0] wrote: {args.out}")

if __name__ == "__main__":
    main()

