#!/usr/bin/env python3
"""
Generate detector yield and observing efficiency plots.

1) Detector yield: # uncut detectors / 6020 vs time
2) Observing efficiency: accounts for sample flags using actual observation duration from obsdb
"""

import argparse
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
from pathlib import Path
from collections import defaultdict
from sotodlib.core.metadata.obsdb import ObsDb
import glob
import yaml
import h5py

# Number of possible detectors per band, used as the denominator for detector
# yield, mapping efficiency, and the map-depth baseline/goal benchmarks.
NUM_DETECTORS_PER_BAND = 6020

# White-noise NET baseline/goal targets (uK*sqrt(s)), used for the white-noise
# and map-depth benchmark lines.
#
# Two sets, because the MF numbers this script has always used and the MF
# numbers in the V3r6 bolocalc model are not the same, and there is no scaling
# that reconciles them:
#
# 'deck'  f090/f150 are the 312/395 and 236/263 that appear in
#         tiger/202601_v4/f2f_data_retention.pdf. They are NOT V3r6
#         sensitivity.txt values. They are the so_noise_models v3.1 SAT array
#         NETs per optics tube (MF baseline 3.4/4.3, goal 2.4/2.7 uK*sqrt(s))
#         converted to a per-detector NET by multiplying by
#         sqrt(N_det * yield), with N_det = 12096 per MF band (two MF SATs)
#         and yield 0.7 baseline / 0.8 goal:
#             3.4*sqrt(12096*0.7) = 312.9    4.3*sqrt(12096*0.7) = 395.7
#             2.4*sqrt(12096*0.8) = 236.1    2.7*sqrt(12096*0.8) = 265.6
#         reproducing 312/395/236 to 0.3% and 263 to 1%.
#
# 'v3r6'  every band straight from the Detector NET_CMB column of
#         bolocalc-so-model/V3r6/V3r6_{Baseline,Goal}/SAT/{MF,UHF}/sensitivity.txt
#         (el = 50 deg, PWV = 1.0 mm). V3r6's MF values sit well below the
#         deck's, so switching sets visibly moves the f090/f150 lines.
#
# f220/f280 exist only in V3r6 -- the deck never had UHF benchmarks -- so both
# sets carry the same UHF numbers. The default set is 'deck' so satp1-vs-satp3
# output stays identical to what has already been circulated.
NET_BENCHMARK_SETS = {
    'deck': {
        'f090': {'baseline': 312.0, 'goal': 236.0,
                 'source': 'so_noise_models v3.1 SAT MF tube NET x sqrt(12096*yield)'},
        'f150': {'baseline': 395.0, 'goal': 263.0,
                 'source': 'so_noise_models v3.1 SAT MF tube NET x sqrt(12096*yield)'},
        'f220': {'baseline': 540.4, 'goal': 467.3,
                 'source': 'V3r6 SAT UHF sensitivity.txt (UHF_1 Detector NET_CMB)'},
        'f280': {'baseline': 1397.6, 'goal': 1218.1,
                 'source': 'V3r6 SAT UHF sensitivity.txt (UHF_2 Detector NET_CMB)'},
    },
    'v3r6': {
        'f090': {'baseline': 245.1, 'goal': 179.8,
                 'source': 'V3r6 SAT MF sensitivity.txt (MF_1 Detector NET_CMB)'},
        'f150': {'baseline': 250.3, 'goal': 204.4,
                 'source': 'V3r6 SAT MF sensitivity.txt (MF_2 Detector NET_CMB)'},
        'f220': {'baseline': 540.4, 'goal': 467.3,
                 'source': 'V3r6 SAT UHF sensitivity.txt (UHF_1 Detector NET_CMB)'},
        'f280': {'baseline': 1397.6, 'goal': 1218.1,
                 'source': 'V3r6 SAT UHF sensitivity.txt (UHF_2 Detector NET_CMB)'},
    },
}

# Active set; --net-source rebinds this before any plotting happens.
NET_BENCHMARKS = NET_BENCHMARK_SETS['deck']

# Color styling used to distinguish platforms on combined plots. satp1/satp3
# keep their original blue/orange; satp2 gets a third, clearly distinct hue.
TELESCOPE_STYLE = {
    'satp1': {'scatter': '#1f77b4', 'line': 'darkblue'},
    'satp2': {'scatter': '#9467bd', 'line': '#54278f'},
    'satp3': {'scatter': '#ff7f0e', 'line': '#cc5500'},
}

# Panel order for combined plots, so a mixed MF+UHF figure reads low-to-high
# frequency. Bands outside this list are appended alphabetically.
BAND_ORDER = ['f090', 'f150', 'f220', 'f280']

# Marker/linestyle used to distinguish the 3 observing-efficiency curves on combined plots
EFFICIENCY_CURVE_STYLE = {
    'curve1': {'marker': 'o', 'linestyle': '-', 'label': 'All CMB Observations'},
    'curve2': {'marker': 's', 'linestyle': '--', 'label': '+ PWV < 2mm & $N_{\\mathrm{dets}}$/wafer > 10'},
    'curve3': {'marker': '^', 'linestyle': ':', 'label': '+ excluding flagged samples'},
}

def load_noise_aggregation(agg_dir, band, prefer_calibrated=True):
    """Per-observation white-noise summaries from the aggregation HDF5.

    Returns ``(per_obs, path, calibrated)``. ``per_obs`` maps obs_id to a dict
    in the file's own raw units -- callers apply their own ``(x/2)*1e6``:

        median         median over every valid detector of the observation
        inv_var_mean   1/sqrt(mean(1/wn^2)), the per-detector-equivalent noise
        array_net      1/sqrt(sum(1/wn^2)), the array noise of the observation
        n_dets         valid detectors contributing

    Two things this gets right that reading the file inline did not:

    * The index carries one entry per (obs_id, wafer), so an observation spans
      several entries. Keying a dict on obs_id while looping entries silently
      kept only the last wafer. Entries are accumulated here instead.
    * Zero white-noise values are not measurements -- 2.0M of 31.7M satp3 f090
      detectors -- and a zero makes 1/wn^2 infinite, collapsing array_net to
      exactly 0. Only finite, strictly positive values are used, via the
      file's own ``valid`` mask when it has one.

    The ``_cmb`` file written by build_calibrated_noise.py is preferred when
    present, since the 260702 archives are otherwise in relcal'd pW and not
    comparable to any NET benchmark.
    """
    import h5py

    cal_path = os.path.join(agg_dir, f'noise_aggregation_{band}_cmb.h5')
    raw_path = os.path.join(agg_dir, f'noise_aggregation_{band}.h5')
    path, calibrated = (cal_path, True) if (prefer_calibrated and
                                            os.path.exists(cal_path)) else (raw_path, False)
    if not os.path.exists(path):
        return {}, None, False

    with h5py.File(path, 'r') as f:
        obs_ids = f['obs_ids'][:]
        starts = f['obs_starts'][:]
        ends = f['obs_ends'][:]
        wn = f['white_noise'][:]
        valid = f['valid'][:] if 'valid' in f else None

    if valid is None:
        valid = np.isfinite(wn) & (wn > 0)

    slices = defaultdict(list)
    for obs_id, s, e in zip(obs_ids, starts, ends):
        if e <= s:
            continue
        oid = obs_id.decode() if isinstance(obs_id, bytes) else str(obs_id)
        slices[oid].append((int(s), int(e)))

    per_obs = {}
    for oid, spans in slices.items():
        vals = np.concatenate([wn[s:e][valid[s:e]] for s, e in spans]) \
            if spans else np.array([])
        if vals.size == 0:
            continue
        inv_var = 1.0 / vals ** 2
        per_obs[oid] = {
            'median': float(np.median(vals)),
            'inv_var_mean': float(1.0 / np.sqrt(inv_var.mean())),
            'array_net': float(1.0 / np.sqrt(inv_var.sum())),
            'n_dets': int(vals.size),
        }
    return per_obs, path, calibrated


# Every ratio/percentage plot carries its own definition, so a reader never
# has to guess what the denominator was. Kept as raw strings with mathtext
# fractions; NUM_DETECTORS_PER_BAND is substituted so the figure shows the
# actual number used, not a symbol.
RATIO_FORMULAS = {
    'yield': (r'Detector Yield $=\;\dfrac{N_{\mathrm{dets\ uncut\ after\ preprocessing}}}'
              r'{%d\ \mathrm{possible\ per\ band}}$   '
              r'(wafers with $\leq$10 uncut dets excluded)'),
    'efficiency': (r'Observing Efficiency $=\;\dfrac{\sum \mathrm{CMB\ observation\ hours\ in\ week}}'
                   r'{\mathrm{calendar\ hours\ in\ week}}\times100\%$'),
    'mapping_efficiency': (r'Mapping Efficiency $=\;\dfrac{\sum \mathrm{detector\ seconds\ into\ map\ in\ week}}'
                           r'{%d \times \mathrm{calendar\ seconds\ in\ week}}\times100\%%$'),
}


def formula_text(key):
    """The definition string for a ratio plot, with the denominator filled in."""
    s = RATIO_FORMULAS[key]
    return s % NUM_DETECTORS_PER_BAND if '%d' in s else s


def add_formula(fig, key, y=0.01):
    """Write the ratio's definition along the bottom of a figure."""
    fig.text(0.5, y, formula_text(key), ha='center', va='bottom', fontsize=11)


def map_depth_from_net(net, yield_frac, obs_efficiency=0.2, f_sky=0.1):
    """Weekly map depth (uK*arcmin) implied by a per-detector NET target.

    Detector-seconds available in a calendar week at the target yield and
    observing efficiency become an inverse-variance weight, which becomes a
    map depth over f_sky. Same chain as the per-week measurement, so the
    benchmark line and the plotted points are directly comparable.
    """
    seconds_per_week = 7 * 24 * 3600
    det_seconds = NUM_DETECTORS_PER_BAND * yield_frac * obs_efficiency * seconds_per_week
    weight = det_seconds / (net ** 2)
    return np.sqrt(8.0 * np.pi * f_sky / weight) * (10800.0 / np.pi)


def extract_timestamp_from_obs_id(obs_id):
    """Extract Unix timestamp from observation ID."""
    parts = obs_id.split('_')
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None

def get_final_detector_column(cursor):
    """Query the schema to find the final detector ndets column."""
    cursor.execute("PRAGMA table_info(obs);")
    columns = cursor.fetchall()
    
    final_det_col = None
    for col in columns:
        col_name = col[1]
        if 'ndets' in col_name and col_name.startswith('det_'):
            final_det_col = col_name
    
    return final_det_col

def find_preprocessing_config(version_dir, telescope):
    """Find the preprocessing config files for a given version directory.
    
    Returns tuple of (init_config_path, proc_config_path)
    """
    # Extract version number from directory name (e.g., 'satp1_20260417' -> '20260417')
    version_match = version_dir.split('_')[-1]
    
    # Search in repos directory
    config_dir = os.path.expanduser(f'~/repos/iso-sat/v4/preprocessing/{telescope}')
    
    init_config = os.path.join(config_dir, f'preprocessing_config_{version_match}_init.yaml')
    proc_config = os.path.join(config_dir, f'preprocessing_config_{version_match}_proc.yaml')
    
    if os.path.exists(init_config):
        return init_config, proc_config if os.path.exists(proc_config) else None
    
    return None, None

def generate_detector_yield_and_efficiency_plots(db_path, telescope, version_dir, output_dir=None):
    """
    Generate detector yield and observing efficiency plots.
    
    Parameters
    ----------
    db_path : str
        Path to the sqlite database
    telescope : str
        Telescope name (e.g., 'satp1', 'satp3')
    version_dir : str
        Base directory name (e.g., 'satp1_20260417')
    output_dir : str, optional
        Output directory. If None, uses version_dir/analysis
    """
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return False
    
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(db_path)),
            'analysis',
            telescope
        )
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get the final detector column name
        final_det_col = get_final_detector_column(cursor)
        if not final_det_col:
            print(f"Could not find final detector column in {db_path}")
            return False
        
        # Get all observations with detector counts and samples
        query = f"""
            SELECT obs_id, timestamp, det_0_starting_ndets, {final_det_col}, total_samples 
            FROM obs 
            WHERE total_samples > 0
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            print(f"No observations with samples found in {db_path}")
            return False

        # Discover this telescope's actual bands from the data (e.g. satp1/satp3
        # are f090/f150, satp2 is UHF -> f220/f280) instead of assuming f090/f150.
        cursor.execute("SELECT DISTINCT band FROM obs WHERE total_samples > 0 AND band IS NOT NULL")
        bands = sorted(r[0] for r in cursor.fetchall())

        # Collected series data, keyed by plot type then band, used later to build
        # combined satp1-vs-satp3 plots. Does not affect the per-platform plots below.
        plot_data = {'yield': {}, 'efficiency': {}, 'white_noise': {}, 'array_net': {},
                     'map_depth': {}, 'mapping_efficiency': {}}

        # ===== PLOT 1: DETECTOR YIELD (Per Band) =====

        # Load ObsDb for start_time timestamps
        from sotodlib.preprocess import preprocess_util as pp_util
        
        init_config_path, proc_config_path = find_preprocessing_config(version_dir, telescope)
        
        obsdb_yield = None
        if init_config_path and os.path.exists(init_config_path):
            try:
                with open(init_config_path, 'r') as f:
                    cfg_init = yaml.safe_load(f)
                
                repo_dir = os.path.expanduser('~/repos/iso-sat/v4')
                orig_cwd = os.getcwd()
                os.chdir(repo_dir)
                
                try:
                    cfg_init_full, ctx_init = pp_util.get_preprocess_context(cfg_init)
                    if hasattr(ctx_init, 'obsdb') and ctx_init.obsdb is not None:
                        obsdb_yield = ctx_init.obsdb
                finally:
                    os.chdir(orig_cwd)
            except Exception as e:
                pass
        
        # Re-query to get band information
        cursor.execute(f"""
            SELECT obs_id, timestamp, {final_det_col}, total_samples, wafer, band
            FROM obs 
            WHERE total_samples > 0
        """)
        rows_with_band = cursor.fetchall()
        
        # Reorganize data by (observation, band) - aggregate across wafers per band
        # Filter wafers at obs-wafer-band level: exclude wafers with ndets <= 10
        obs_band_yield_data = defaultdict(lambda: defaultdict(lambda: {'ndets': 0}))
        
        for obs_id, timestamp, final_ndets, total_samples, wafer, band in rows_with_band:
            # final_ndets is NULL for groups whose init stage succeeded but whose
            # proc stage never ran (no proc-archive entry), so there is no final
            # count to judge -- treat those like a wafer that fails the cut.
            if final_ndets is None:
                continue
            # Only include this wafer if it has > 10 uncut detectors
            if final_ndets > 10:
                obs_band_yield_data[obs_id][band]['ndets'] += final_ndets
        
        # Generate detector yield plots per band
        for band in bands:
            obs_times = []
            detector_yields = []
            
            # Weekly aggregates
            weekly_yield = defaultdict(list)
            week_dates = {}
            
            for obs_id in sorted(obs_band_yield_data.keys()):
                if band not in obs_band_yield_data[obs_id]:
                    continue
                
                total_ndets = obs_band_yield_data[obs_id][band]['ndets']
                
                # Skip observations where summed ndets from quality wafers is 0
                if total_ndets == 0:
                    continue
                
                # Try to get start_time from obsdb first
                timestamp_extracted = None
                if obsdb_yield is not None:
                    try:
                        obs_query_data = obsdb_yield.query(f"obs_id='{obs_id}'")
                        if len(obs_query_data) > 0 and 'start_time' in obs_query_data[0]:
                            timestamp_extracted = obs_query_data[0]['start_time']
                    except:
                        pass
                
                # Fall back to extracting from obs_id if needed
                if timestamp_extracted is None:
                    timestamp_extracted = extract_timestamp_from_obs_id(obs_id)
                
                if timestamp_extracted is None:
                    continue
                
                try:
                    dt = datetime.fromtimestamp(timestamp_extracted)
                except (ValueError, OSError):
                    continue
                
                # Detector yield for this observation and band
                # 6020 total possible detectors per band
                yield_frac = total_ndets / NUM_DETECTORS_PER_BAND
                
                obs_times.append(dt)
                detector_yields.append(yield_frac)
                
                # Weekly aggregate
                week_start = dt - timedelta(days=dt.weekday())
                week_key = week_start.strftime('%Y-%m-%d')
                weekly_yield[week_key].append(yield_frac)
                week_dates[week_key] = week_start
            
            if not obs_times:
                continue
            
            # Sort observations by time
            sorted_indices = np.argsort(obs_times)
            obs_times_sorted = [obs_times[i] for i in sorted_indices]
            detector_yields_sorted = [detector_yields[i] for i in sorted_indices]
            
            # Compute weekly averages
            sorted_weeks = sorted(weekly_yield.keys())
            week_dates_list = [week_dates[w] for w in sorted_weeks]
            weekly_yields_mean = [np.mean(weekly_yield[w]) for w in sorted_weeks]
            
            # Create detector yield plot per band
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot per-observation points (light)
            ax.scatter(obs_times_sorted, detector_yields_sorted, s=20, alpha=0.3, 
                       color='#1f77b4', label='Per-observation')
            
            # Plot weekly averages
            ax.plot(week_dates_list, weekly_yields_mean, 'o-', linewidth=2, markersize=8,
                    color='darkblue', label='Weekly average')
            
            # Calculate and plot average over whole time range
            avg_yield = np.mean(detector_yields_sorted)
            ax.axhline(y=avg_yield, color='gray', linestyle='--', linewidth=2,
                       label=f'Average ({avg_yield:.2f})')

            plot_data['yield'][band] = {
                'obs_times': obs_times_sorted,
                'obs_yield': detector_yields_sorted,
                'week_dates': week_dates_list,
                'week_yield': weekly_yields_mean,
                'avg': avg_yield,
            }
            
            # Reference lines
            ax.axhline(y=0.7, color='red', linestyle='--', linewidth=2, 
                       label='Baseline (0.7)')
            ax.axhline(y=0.8, color='orange', linestyle='--', linewidth=2,
                       label='Goal (0.8)')
            
            # Formatting
            ax.set_xlabel('Observation Time', fontsize=12)
            ax.set_ylabel('Detector Yield (uncut detectors / 6020)', fontsize=12)
            ax.set_title(f'Detector Yield: {telescope.upper()} - {band.lower()} - {version_dir}', 
                        fontsize=14, fontweight='bold')
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Set y-axis limits
            ax.set_ylim([0, 1.0])
        
            # Grid and legend
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.set_axisbelow(True)
            ax.legend(loc='lower right', fontsize=10)
            
            plt.tight_layout()
            
            # Save per-band
            add_formula(fig, 'yield')
            output_path = os.path.join(output_dir, f'{telescope}_detector_yield_{band}.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Created: {output_path}")
            plt.close(fig)
            
            # Print summary
            yield_values = weekly_yields_mean if weekly_yields_mean else detector_yields_sorted
            if yield_values:
                print(f"\n=== Detector Yield Summary ({telescope.upper()} - {band.upper()}) ===")
                print(f"Yield (min/max/mean): {min(yield_values):.3f} / {max(yield_values):.3f} / {np.mean(yield_values):.3f}")
        
        # ===== PLOT 2: OBSERVING EFFICIENCY =====
         
        # Reuse obsdb_yield if already loaded, otherwise load fresh
        obsdb = obsdb_yield
        if obsdb is None:
            # Load preprocessing config to access ObsDb
            from sotodlib.preprocess import preprocess_util as pp_util
             
            init_config_path, proc_config_path = find_preprocessing_config(version_dir, telescope)
             
            if init_config_path and os.path.exists(init_config_path):
                try:
                    # Load the init config
                    with open(init_config_path, 'r') as f:
                        cfg_init = yaml.safe_load(f)
                     
                    # Change to repo directory so relative paths in config work
                    repo_dir = os.path.expanduser('~/repos/iso-sat/v4')
                    orig_cwd = os.getcwd()
                    os.chdir(repo_dir)
                     
                    try:
                        # Get the preprocess context which includes ObsDb
                        cfg_init_full, ctx_init = pp_util.get_preprocess_context(cfg_init)
                         
                        # Access the ObsDb from the context
                        if hasattr(ctx_init, 'obsdb') and ctx_init.obsdb is not None:
                            obsdb = ctx_init.obsdb
                            print(f"Loaded ObsDb from context for {version_dir}")
                    finally:
                        os.chdir(orig_cwd)
                except Exception as e:
                    pass
         
        # Load metadata obsdb for PWV information and duration fallback
        metadata_obsdb = None
        metadata_obsdb_path = f'/scratch/gpfs/SIMONSOBS/sat-iso/metadata/{telescope}_obsdb_251203r1.sqlite'
        if os.path.exists(metadata_obsdb_path):
            try:
               metadata_obsdb = ObsDb(metadata_obsdb_path)
               print(f"Loaded metadata ObsDb from {metadata_obsdb_path}")
            except Exception as e:
               pass
         
        # If obsdb is still None, try using metadata_obsdb
        if obsdb is None and metadata_obsdb is not None:
            obsdb = metadata_obsdb
            print(f"Using metadata ObsDb for duration information in {version_dir}")
         
        # Re-query for efficiency calculation and white noise
        # Query all rows (one per obs_id, wafer, band combination)
        cursor.execute(f"""
            SELECT obs_id, timestamp, {final_det_col}, total_samples, wafer, band, noise_Q_array_noise, noise_Q_n_array_noise
            FROM obs 
            WHERE total_samples > 0
        """)
        rows = cursor.fetchall()
         
        # Organize by observation and band
        obs_band_data = defaultdict(lambda: defaultdict(lambda: {'ndets': 0, 'samples': 0, 'array_noise': []}))
        obs_timestamps = {}
        
        for obs_id, timestamp, final_ndets, total_samples, wafer, band, array_noise, n_array_noise in rows:
            # See above: NULL final count means the proc stage produced nothing
            # for this wafer, so skip it rather than adding its samples against
            # an unknown detector count.
            if final_ndets is None:
                continue
            obs_band_data[obs_id][band]['ndets'] += final_ndets
            obs_band_data[obs_id][band]['samples'] += total_samples
            # Store array noise estimate: array_noise * sqrt(n_array_noise)
            if array_noise is not None and n_array_noise is not None:
                try:
                    if not np.isnan(array_noise) and n_array_noise > 0:
                        per_det_noise = array_noise * np.sqrt(n_array_noise)
                        obs_band_data[obs_id][band]['array_noise'].append(per_det_noise)
                except (TypeError, ValueError):
                    pass
            if obs_id not in obs_timestamps:
                obs_timestamps[obs_id] = timestamp
         
        # ===== PLOT 2: OBSERVING EFFICIENCY (only if obsdb available) =====
        if obsdb is not None:
            band_times_eff = defaultdict(list)
            band_weekly_cmb_time = defaultdict(lambda: defaultdict(float))
            band_weekly_first_obs_time = defaultdict(dict)
            band_weekly_last_obs_time = defaultdict(dict)
            band_obs_times = defaultdict(list)
             
            # Filtered versions - curve 2: obs_duration with cuts
            band_times_eff_filtered = defaultdict(list)
            band_weekly_cmb_time_filtered = defaultdict(lambda: defaultdict(float))
            band_weekly_first_obs_time_filtered = defaultdict(dict)
            band_weekly_last_obs_time_filtered = defaultdict(dict)
            band_obs_times_filtered = defaultdict(list)
             
            # Curve 3: etio with cuts
            band_weekly_cmb_time_etio = defaultdict(lambda: defaultdict(float))
            band_weekly_first_obs_time_etio = defaultdict(dict)
            band_weekly_last_obs_time_etio = defaultdict(dict)
        
            for obs_id in sorted(obs_band_data.keys()):
                timestamp_extracted = extract_timestamp_from_obs_id(obs_id)
            
                if timestamp_extracted is None:
                    continue
            
                try:
                    dt = datetime.fromtimestamp(timestamp_extracted)
                except (ValueError, OSError):
                    continue
            
                # Get actual duration from obsdb
                try:
                    obs_query_data = obsdb.query(f"obs_id='{obs_id}'")
                    if len(obs_query_data) == 0:
                        continue
                    obs_duration_seconds = obs_query_data[0]['duration']
                except Exception as e:
                    continue
            
                # Get PWV from metadata obsdb
                pwv = None
                if metadata_obsdb is not None:
                    try:
                        metadata_query_data = metadata_obsdb.query(f"obs_id='{obs_id}'")
                        if len(metadata_query_data) > 0:
                            pwv = metadata_query_data[0].get('pwv_mean', None)
                    except Exception as e:
                        pass
            
                obs_duration_hours = obs_duration_seconds / 3600.0
            
                # For each band in this observation:
                # etio = total_samples / total_uncut_dets / 200 (in seconds)
                for band, data in obs_band_data[obs_id].items():
                    total_samples_band = data['samples']
                    total_uncut_dets_band = data['ndets']
                
                    if total_uncut_dets_band > 0:
                        avg_samples_per_det = total_samples_band / total_uncut_dets_band
                        etio_seconds = avg_samples_per_det / 200.0
                        etio_hours = etio_seconds / 3600.0
                    
                        # For curve 1: obs_duration_hours (no cuts at all)
                        week_start = dt - timedelta(days=dt.weekday())
                        week_key = week_start.strftime('%Y-%m-%d')
                    
                        # Curve 1: All obs, no cuts - just raw observation duration
                        if band not in band_weekly_cmb_time:
                            band_weekly_cmb_time[band] = defaultdict(float)
                        band_weekly_cmb_time[band][week_key] += obs_duration_hours
                    
                        # Track times for curve 1
                        if band not in band_weekly_first_obs_time:
                            band_weekly_first_obs_time[band] = {}
                        if band not in band_weekly_last_obs_time:
                            band_weekly_last_obs_time[band] = {}
                        if week_key not in band_weekly_first_obs_time[band]:
                            band_weekly_first_obs_time[band][week_key] = dt
                        band_weekly_last_obs_time[band][week_key] = dt
                    
                     
                        # Curve 2 & 3: With PWV and wafer cuts
                        if (pwv is None or pwv < 2.0) and total_uncut_dets_band > 10:
                            # Curve 2: obs_duration_hours (with cuts applied)
                            if band not in band_weekly_cmb_time_filtered:
                                band_weekly_cmb_time_filtered[band] = defaultdict(float)
                            band_weekly_cmb_time_filtered[band][week_key] += obs_duration_hours
                        
                            # Track times for curve 2
                            if band not in band_weekly_first_obs_time_filtered:
                                band_weekly_first_obs_time_filtered[band] = {}
                            if band not in band_weekly_last_obs_time_filtered:
                                band_weekly_last_obs_time_filtered[band] = {}
                            if week_key not in band_weekly_first_obs_time_filtered[band]:
                                band_weekly_first_obs_time_filtered[band][week_key] = dt
                            band_weekly_last_obs_time_filtered[band][week_key] = dt
                        
                            # Curve 3: etio_hours (with cuts applied)
                            if band not in band_weekly_cmb_time_etio:
                                band_weekly_cmb_time_etio[band] = defaultdict(float)
                            band_weekly_cmb_time_etio[band][week_key] += etio_hours
                        
                            # Track times for curve 3 (reuse same as curve 2)
                            if band not in band_weekly_first_obs_time_etio:
                                band_weekly_first_obs_time_etio[band] = {}
                            if band not in band_weekly_last_obs_time_etio:
                                band_weekly_last_obs_time_etio[band] = {}
                            if week_key not in band_weekly_first_obs_time_etio[band]:
                                band_weekly_first_obs_time_etio[band][week_key] = dt
                            band_weekly_last_obs_time_etio[band][week_key] = dt
        
            # Generate efficiency plots per band
            for band in bands:
                if band not in band_weekly_cmb_time or not band_weekly_cmb_time[band]:
                    print(f"No valid observations found for band {band}")
                    continue
            
                # Weekly efficiency - Curve 1: all obs, obs_duration (no cuts)
                weekly_eff = {}
                for week_key in sorted(band_weekly_cmb_time[band].keys()):
                    first_time = band_weekly_first_obs_time[band][week_key]
                
                    # Calendar week is 168 hours
                    calendar_hours = 168.0
                    eff = (band_weekly_cmb_time[band][week_key] / calendar_hours) * 100.0
                    weekly_eff[week_key] = (first_time, eff)
            
                # Weekly efficiency - Curve 2: obs_duration with cuts (PWV < 2.0mm AND ndets > 10)
                weekly_eff_filtered = {}
                for week_key in sorted(band_weekly_cmb_time_filtered[band].keys()):
                    first_time = band_weekly_first_obs_time_filtered[band][week_key]
                
                    # Calendar week is 168 hours
                    calendar_hours = 168.0
                    eff = (band_weekly_cmb_time_filtered[band][week_key] / calendar_hours) * 100.0
                    weekly_eff_filtered[week_key] = (first_time, eff)
            
                # Weekly efficiency - Curve 3: etio with cuts (PWV < 2.0mm AND ndets > 10)
                weekly_eff_etio = {}
                for week_key in sorted(band_weekly_cmb_time_etio[band].keys()):
                    first_time = band_weekly_first_obs_time_etio[band][week_key]
                
                    # Calendar week is 168 hours
                    calendar_hours = 168.0
                    eff = (band_weekly_cmb_time_etio[band][week_key] / calendar_hours) * 100.0
                    weekly_eff_etio[week_key] = (first_time, eff)
            
                # Debug output for weekly
                if weekly_eff:
                    first_week = sorted(weekly_eff.keys())[0]
                    first_time, eff_val = weekly_eff[first_week]
             
                # Plot efficiency per band
                fig, ax = plt.subplots(figsize=(14, 6))
            
                # Weekly average - Curve 1: all obs, obs_duration (no cuts)
                week_dates = []
                week_eff = []
                for week_key in sorted(weekly_eff.keys()):
                    dt, eff_val = weekly_eff[week_key]
                    week_dates.append(dt)
                    week_eff.append(eff_val)
            
                avg_eff_1 = None
                if week_dates:
                    ax.plot(week_dates, week_eff, 'b-o', linewidth=2, markersize=8,
                            label='All CMB Observations', alpha=0.8)
                    # Average for curve 1
                    avg_eff_1 = np.mean(week_eff)
                    ax.axhline(y=avg_eff_1, color='blue', linestyle='--', linewidth=1.5, alpha=0.5,
                               label=f'Average ({avg_eff_1:.2f}%)')
            
                # Weekly average - Curve 2: obs_duration with cuts
                week_dates_filtered = []
                week_eff_filtered = []
                for week_key in sorted(weekly_eff_filtered.keys()):
                    dt, eff_val = weekly_eff_filtered[week_key]
                    week_dates_filtered.append(dt)
                    week_eff_filtered.append(eff_val)
            
                avg_eff_2 = None
                if week_dates_filtered:
                    ax.plot(week_dates_filtered, week_eff_filtered, 'r-s', linewidth=2, markersize=8,
                            label='+ PWV < 2mm & $N_{\\mathrm{dets}}$/wafer > 10', alpha=0.8)
                    # Average for curve 2
                    avg_eff_2 = np.mean(week_eff_filtered)
                    ax.axhline(y=avg_eff_2, color='red', linestyle='--', linewidth=1.5, alpha=0.5,
                               label=f'Average ({avg_eff_2:.2f}%)')
            
                # Weekly average - Curve 3: etio with cuts
                week_dates_etio = []
                week_eff_etio = []
                for week_key in sorted(weekly_eff_etio.keys()):
                    dt, eff_val = weekly_eff_etio[week_key]
                    week_dates_etio.append(dt)
                    week_eff_etio.append(eff_val)
            
                avg_eff_3 = None
                if week_dates_etio:
                    ax.plot(week_dates_etio, week_eff_etio, 'g-^', linewidth=2, markersize=8,
                            label='+ excluding flagged samples', alpha=0.8)
                    # Average for curve 3
                    avg_eff_3 = np.mean(week_eff_etio)
                    ax.axhline(y=avg_eff_3, color='green', linestyle='--', linewidth=1.5, alpha=0.5,
                               label=f'Average ({avg_eff_3:.2f}%)')

                plot_data['efficiency'][band] = {
                    'curve1': {'dates': week_dates, 'eff': week_eff, 'avg': avg_eff_1},
                    'curve2': {'dates': week_dates_filtered, 'eff': week_eff_filtered, 'avg': avg_eff_2},
                    'curve3': {'dates': week_dates_etio, 'eff': week_eff_etio, 'avg': avg_eff_3},
                }

                # Reference line: 20% benchmark
                ax.axhline(y=20, color='black', linestyle=':', linewidth=2, label='20% Benchmark')
            
                ax.set_xlabel('Time', fontsize=12)
                ax.set_ylabel('Observing Efficiency (%)', fontsize=12)
                ax.set_title(f'Observing Efficiency: {telescope.upper()} - {band.lower()} - {version_dir}',  
                            fontsize=14, fontweight='bold')
            
                # Format x-axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
                # Grid and legend
                ax.grid(True, alpha=0.3, linestyle=':')
                ax.set_axisbelow(True)
                ax.legend(loc='lower right', fontsize=10)
            
                plt.tight_layout()
            
                # Save per-band
                add_formula(fig, 'efficiency')
                output_path = os.path.join(output_dir, f'{telescope}_observing_efficiency_{band}.png')
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"✓ Created: {output_path}")
                plt.close(fig)
            
                # Print summary
                eff_values = [eff for _, eff in weekly_eff.values()] if weekly_eff else []
                if eff_values:
                    print(f"\n=== Observing Efficiency Summary ({telescope.upper()} - {band.upper()}) ===")
                    print(f"Efficiency (min/max/mean): {min(eff_values):.2f}% / {max(eff_values):.2f}% / {np.mean(eff_values):.2f}%")
         
        # ===== PLOT 3: WHITE NOISE MEDIAN =====
        # First try to load from HDF5 as fallback if database doesn't have good data
        h5_wn_data = {}     # {(obs_id, band): per-detector median, uK*rt(s)}
        h5_array_net = {}   # {(obs_id, band): array NET of the obs, uK*rt(s)}
         
        for band in bands:
            agg_dir = os.path.join(output_dir, '..', '..', 'aggregation')
            try:
                per_obs, h5_path, calibrated = load_noise_aggregation(agg_dir, band)
                for obs_id, rec in per_obs.items():
                    # Convert: divide by 2, multiply by 1e6 for K->uK
                    h5_wn_data[(obs_id, band)] = (rec['median'] / 2.0) * 1e6
                    h5_array_net[(obs_id, band)] = (rec['array_net'] / 2.0) * 1e6
                if h5_path:
                    print(f"  {band}: {len(per_obs):,} observations from "
                          f"{os.path.basename(h5_path)}"
                          f"{'' if calibrated else '  (NOT abscal-calibrated)'}")
            except Exception as e:
                print(f"  {band}: could not read aggregation HDF5: {e}")
         
        for band in bands:
            # Collect white noise data per observation
            obs_times_wn = []
            obs_wn_median = []
            obs_array_net = []
            obs_ids_wn = []
            debug_skipped_empty = 0

            for obs_id in sorted(obs_band_data.keys()):
                if band not in obs_band_data[obs_id]:
                    continue

                # The aggregation HDF5 is the primary source now, not the
                # database's noise_Q_array_noise column. It is per detector,
                # it covers every wafer, and when a calibrated copy exists it
                # is in K_CMB. The database column stays as a fallback for
                # archives with no aggregation file.
                an_list = obs_band_data[obs_id][band]['array_noise']
                if (obs_id, band) in h5_wn_data:
                    wn_net = h5_wn_data[(obs_id, band)]
                elif an_list:
                    # Median of array_noise * sqrt(n_array_noise) over the wafers
                    # of this obs-band. Zeros are excluded upstream now, but guard
                    # anyway: a zero here is a failed estimate, not a measurement.
                    an_good = [v for v in an_list if np.isfinite(v) and v > 0]
                    if not an_good:
                        debug_skipped_empty += 1
                        continue
                    an_median = np.median(an_good)
                    # Divide by 2 (sqrt(2) NEQ->NET, sqrt(2) /rt(Hz)->/rt(s)), then 1e6 for K->uK
                    wn_net = (an_median / 2.0) * 1e6
                else:
                    debug_skipped_empty += 1
                    continue
                
                timestamp_extracted = extract_timestamp_from_obs_id(obs_id)
                if timestamp_extracted is None:
                    continue
                
                try:
                    dt = datetime.fromtimestamp(timestamp_extracted)
                except (ValueError, OSError):
                    continue
                
                obs_times_wn.append(dt)
                obs_wn_median.append(wn_net)
                obs_array_net.append(h5_array_net.get((obs_id, band), np.nan))
                obs_ids_wn.append(obs_id)

            if not obs_times_wn:
                print(f"⚠ No white noise data for {band} (skipped {debug_skipped_empty} obs with empty array_noise)")
                continue
             
            print(f"✓ Collected {len(obs_times_wn)} observations with white noise data for {band} (skipped {debug_skipped_empty} obs)")
            
            # Sort by time
            sorted_indices = np.argsort(obs_times_wn)
            obs_times_wn_sorted = [obs_times_wn[i] for i in sorted_indices]
            obs_wn_median_sorted = [obs_wn_median[i] for i in sorted_indices]
            obs_array_net_sorted = [obs_array_net[i] for i in sorted_indices]

            # Array NET per observation: 1/sqrt(sum(1/NET_det^2)) over every
            # valid detector of every wafer. Only available from the
            # aggregation HDF5, so it is NaN where the database fallback was
            # used; the weekly series and the plot drop those.
            # Summarised with a median, not a mean. Array NET scales as
            # 1/sqrt(N_dets), so an observation that kept only a handful of
            # detectors reports a huge value -- satp2 f220 ranges from 1 to
            # 4621 detectors per observation, and the few sparse ones pull a
            # mean from 13 to 35 uK*rt(s). The median tracks the typical
            # observation, which is what the benchmark line describes.
            _an = np.asarray(obs_array_net_sorted, dtype=float)
            weekly_an = defaultdict(list)
            week_dates_an = {}
            for dt, an in zip(obs_times_wn_sorted, _an):
                if not np.isfinite(an):
                    continue
                ws = dt - timedelta(days=dt.weekday())
                weekly_an[ws.strftime('%Y-%m-%d')].append(an)
                week_dates_an[ws.strftime('%Y-%m-%d')] = ws
            an_week_dates = [week_dates_an[k] for k in sorted(weekly_an)]
            an_week_vals = [float(np.median(weekly_an[k])) for k in sorted(weekly_an)]
            if np.isfinite(_an).any():
                plot_data['array_net'][band] = {
                    'obs_times': [t for t, a in zip(obs_times_wn_sorted, _an)
                                  if np.isfinite(a)],
                    'obs_an': [float(a) for a in _an if np.isfinite(a)],
                    'week_dates': an_week_dates,
                    'week_an': an_week_vals,
                    'avg': float(np.nanmedian(_an)),
                }

            # Calculate weekly averages
            weekly_wn = defaultdict(list)
            week_dates_wn = {}
            
            for dt, wn in zip(obs_times_wn_sorted, obs_wn_median_sorted):
                week_start = dt - timedelta(days=dt.weekday())
                week_key = week_start.strftime('%Y-%m-%d')
                weekly_wn[week_key].append(wn)
                week_dates_wn[week_key] = week_start
            
            # Weekly averages of medians
            week_dates_list = []
            week_wn_list = []
            for week_key in sorted(weekly_wn.keys()):
                week_dates_list.append(week_dates_wn[week_key])
                week_wn_list.append(np.mean(weekly_wn[week_key]))
            
            # Create white noise plot
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Per-observation points
            ax.scatter(obs_times_wn_sorted, obs_wn_median_sorted, s=20, alpha=0.3, 
                       color='#1f77b4', label='Per-observation median')
            
            # Weekly averages
            ax.plot(week_dates_list, week_wn_list, 'o-', linewidth=2, markersize=8,
                    color='darkblue', label='Weekly average')
            
            # Overall average
            overall_avg_wn = np.mean(obs_wn_median_sorted)
            ax.axhline(y=overall_avg_wn, color='gray', linestyle='--', linewidth=2,
                       label=f'Average ({overall_avg_wn:.1f} uK·√s)')

            plot_data['white_noise'][band] = {
                'obs_times': obs_times_wn_sorted,
                'obs_wn': obs_wn_median_sorted,
                'week_dates': week_dates_list,
                'week_wn': week_wn_list,
                'avg': overall_avg_wn,
            }

            # Benchmark lines (only defined for bands with a known NET target)
            if band in NET_BENCHMARKS:
                baseline_wn = NET_BENCHMARKS[band]['baseline']
                goal_wn = NET_BENCHMARKS[band]['goal']

                ax.axhline(y=baseline_wn, color='orange', linestyle='--', linewidth=2,
                           label=f'Baseline ({baseline_wn:.0f} uK·√s)')
                ax.axhline(y=goal_wn, color='green', linestyle='--', linewidth=2,
                           label=f'Goal ({goal_wn:.0f} uK·√s)')

            # Formatting
            ax.set_xlabel('Observation Time', fontsize=12)
            ax.set_ylabel('White Noise (uK·√s)', fontsize=12)
            ax.set_title(f'Median White Noise: {telescope.upper()} - {band.lower()} - {version_dir}', 
                        fontsize=14, fontweight='bold')
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Grid and legend
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.set_axisbelow(True)
            ax.legend(loc='best', fontsize=10)
             
            # Set y-axis range. 100-600 fitted the MF bands; UHF NETs run
            # several times higher, so follow the weekly points and the
            # benchmark lines instead of a fixed window.
            wn_must = [overall_avg_wn]
            if band in NET_BENCHMARKS:
                wn_must += [NET_BENCHMARKS[band]['baseline'],
                            NET_BENCHMARKS[band]['goal']]
            quantile_ylim(ax, obs_wn_median_sorted, wn_must)

            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(output_dir, f'{telescope}_white_noise_median_{band}.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Created: {output_path}")
            plt.close(fig)
            
            # Print summary
            print(f"\n=== White Noise Summary ({telescope.upper()} - {band.upper()}) ===")
            print(f"White Noise (min/max/mean): {min(obs_wn_median_sorted):.1f} / {max(obs_wn_median_sorted):.1f} / {overall_avg_wn:.1f} uK·√s")
        
        # ===== PLOT 4: MAP DEPTH (using HDF5 aggregation data) =====
        for band in bands:
            # Load white noise data from HDF5 file
            h5_file = os.path.join(output_dir, '..', '..', 'aggregation', f'noise_aggregation_{band}.h5')
            if not os.path.exists(h5_file):
                print(f"  HDF5 not found: {h5_file}")
                continue
            
            # Same loader as the white-noise plot: accumulates every (obs,
            # wafer) entry instead of keeping only the last, drops zero
            # white-noise detectors, and prefers the abscal-calibrated file.
            try:
                recs, used_path, calibrated = load_noise_aggregation(
                    os.path.dirname(h5_file), band)
            except Exception as e:
                print(f"  Error reading HDF5: {e}")
                continue
            wn_per_obs = {oid: r['inv_var_mean'] for oid, r in recs.items()}

            print(f"  Loaded {len(wn_per_obs)} observations from "
                  f"{os.path.basename(used_path) if used_path else 'HDF5'} ({band})"
                  f"{'' if calibrated else '  (NOT abscal-calibrated)'}")
            
            # Collect observations by week
            daily_combined = defaultdict(list)  # {date: [obs_data_dicts]}
            daily_dates = {}  # {date: datetime}
            
            for obs_id in sorted(wn_per_obs.keys()):
                # Get timestamp
                timestamp_extracted = extract_timestamp_from_obs_id(obs_id)
                if timestamp_extracted is None:
                    continue
                
                try:
                    dt = datetime.fromtimestamp(timestamp_extracted)
                except (ValueError, OSError):
                    continue
                
                # Get on-sky detector-hours for this observation
                # total_samples in database is the total samples for that wafer-band combination
                # Since we query all rows (one per wafer), total_samples is per-wafer
                # So DH = total_samples / 200 Hz gives the observation time in seconds
                # When we sum across wafers, we get the total observation time
                if obs_id in obs_band_data and band in obs_band_data[obs_id]:
                    total_samples_band = obs_band_data[obs_id][band]['samples']
                    
                    if total_samples_band > 0:
                        # DH is in seconds: total_samples / sampling_rate(200 Hz)
                        # This already accounts for all wafers (summed in obs_band_data)
                        dh_seconds = total_samples_band / 200.0
                    else:
                        continue
                else:
                    continue
                
                # Get NET_det from HDF5
                net_det_raw = wn_per_obs[obs_id]
                net_det = (net_det_raw / 2.0) * 1e6  # Convert to uK√s
                
                day_key = dt.strftime('%Y-%m-%d')
                daily_combined[day_key].append({
                    'net_det': net_det,
                    'dh_seconds': dh_seconds,
                    'timestamp': dt
                })
                daily_dates[day_key] = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            
            if not daily_combined:
                print(f"  No data for {band}")
                continue
            
            # Reorganize by week for inverse-variance weighting
            weekly_data = defaultdict(lambda: {'observations': [], 'week_start': None})
            for day_key in sorted(daily_combined.keys()):
                day_dt = daily_dates[day_key]
                week_start = day_dt - timedelta(days=day_dt.weekday())
                week_key = week_start.strftime('%Y-%m-%d')
                weekly_data[week_key]['week_start'] = week_start
                
                for obs_data in daily_combined[day_key]:
                   weekly_data[week_key]['observations'].append(obs_data)
            
            if not daily_combined:
                continue
            
            # Reorganize by week for inverse-variance weighting of per-observation NET
            weekly_data = defaultdict(lambda: {'observations': [], 'week_start': None})
            for day_key in sorted(daily_combined.keys()):
                day_dt = daily_dates[day_key]
                week_start = day_dt - timedelta(days=day_dt.weekday())
                week_key = week_start.strftime('%Y-%m-%d')
                weekly_data[week_key]['week_start'] = week_start
                
                for obs_data in daily_combined[day_key]:
                    weekly_data[week_key]['observations'].append(obs_data)
            
            # Compute per-week map depths
            week_times = []
            week_values = []
            for week_key in sorted(weekly_data.keys()):
                week_info = weekly_data[week_key]
                observations = week_info['observations']
                
                if len(observations) == 0:
                    continue
                
                # Use general formula: W = Σ(Δtᵢ / NETᵢ²)
                # Then MapDepth = sqrt(8π · f_sky / W) × (10800/π)
                # Each observation contributes: Δt_obs / NET_obs²
                
                w_sum = 0.0
                net_list = []
                total_dh = 0.0
                
                for obs_data in observations:
                    net_det = obs_data['net_det']
                    dh_seconds = obs_data['dh_seconds']
                    
                    if net_det > 0:
                        w_sum += dh_seconds / (net_det ** 2)
                        net_list.append(net_det)
                        total_dh += dh_seconds
                
                if w_sum > 0:
                    f_sky = 0.1
                    map_depth = np.sqrt(8.0 * np.pi * f_sky / w_sum) * (10800.0 / np.pi)
                    
                    # Debug output for first week only
                    if len(week_values) < 1:
                        avg_net = np.mean(net_list) if net_list else 0
                        print(f"\n[DEBUG] Week {week_key} ({band}):")
                        print(f"  Observations: {len(net_list)}")
                        print(f"  Total DH: {total_dh:.2e} seconds")
                        print(f"  Mean NET_det: {avg_net:.2f} uK√s")
                        print(f"  W_sum: {w_sum:.2e}")
                        print(f"  Computed MapDepth: {map_depth:.2f} µK√s")
                    
                    week_times.append(week_info['week_start'])
                    week_values.append(map_depth)
            
            # Compute per-day map depths (independent from each day's observations only)
            daily_times = []
            daily_values = []
            for day_key in sorted(daily_combined.keys()):
                day_obs = daily_combined[day_key]
                
                if len(day_obs) == 0:
                    continue
                
                # Compute map depth for this day alone
                w_sum_day = 0.0
                net_list_day = []
                total_dh_day = 0.0
                
                for obs_data in day_obs:
                    net_det = obs_data['net_det']
                    dh_seconds = obs_data['dh_seconds']
                    
                    if net_det > 0:
                        w_sum_day += dh_seconds / (net_det ** 2)
                        net_list_day.append(net_det)
                        total_dh_day += dh_seconds
                
                if w_sum_day > 0:
                    f_sky = 0.1
                    map_depth_day = np.sqrt(8.0 * np.pi * f_sky / w_sum_day) * (10800.0 / np.pi)
                    # Scale daily map depth to be comparable to weekly average
                    # Daily has 1/7 the observation time, so divide by sqrt(7) to correct
                    map_depth_day_scaled = map_depth_day / np.sqrt(7)
                    daily_times.append(daily_dates[day_key])
                    daily_values.append(map_depth_day_scaled)
            
            if not week_values:
                continue
            
            # Overall average map depth
            overall_avg = np.mean(week_values)

            plot_data['map_depth'][band] = {
                'daily_times': daily_times,
                'daily_values': daily_values,
                'week_times': week_times,
                'week_values': week_values,
                'avg': overall_avg,
            }

            # Compute benchmark map depths
            # General formula benchmarks: W_target = yield_target × 6020 × 0.2 × (7 × 24 × 3600)
            # But for uniform NET, this reduces to: DH_target_week / NET_target²
            # So: MapDepth_target = sqrt(8π · f_sky / (DH_target_week / NET_target²)) × (10800/π)
            #                      = sqrt(8π · f_sky · NET_target² / DH_target_week) × (10800/π)
            seconds_per_week = 7 * 24 * 3600
            f_sky = 0.1
            
            # Only defined for bands with a known NET target (see NET_BENCHMARKS).
            map_depth_baseline = None
            map_depth_goal = None
            if band in NET_BENCHMARKS:
                net_baseline = NET_BENCHMARKS[band]['baseline']
                net_goal = NET_BENCHMARKS[band]['goal']

                dh_baseline_seconds = NUM_DETECTORS_PER_BAND * 0.7 * 0.2 * seconds_per_week
                dh_goal_seconds = NUM_DETECTORS_PER_BAND * 0.8 * 0.2 * seconds_per_week

                # For uniform NET benchmarks, use W_target = DH_target / NET_target²
                w_baseline = dh_baseline_seconds / (net_baseline ** 2)
                w_goal = dh_goal_seconds / (net_goal ** 2)

                map_depth_baseline = np.sqrt(8.0 * np.pi * f_sky / w_baseline) * (10800.0 / np.pi)
                map_depth_goal = np.sqrt(8.0 * np.pi * f_sky / w_goal) * (10800.0 / np.pi)

                # Debug: print benchmark values
                print(f"\n[DEBUG] Benchmarks ({band}):")
                print(f"  DH_baseline/week: {dh_baseline_seconds:.2e} seconds ({dh_baseline_seconds/3600:.0f} hours)")
                print(f"  DH_goal/week: {dh_goal_seconds:.2e} seconds ({dh_goal_seconds/3600:.0f} hours)")
                print(f"  NET_baseline: {net_baseline} µK√s")
                print(f"  NET_goal: {net_goal} µK√s")
                print(f"  W_baseline: {w_baseline:.2e}")
                print(f"  W_goal: {w_goal:.2e}")
                print(f"  MapDepth_baseline: {map_depth_baseline:.2f} µK√s")
                print(f"  MapDepth_goal: {map_depth_goal:.2f} µK√s")

            # Create plot
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Daily points
            ax.scatter(daily_times, daily_values, s=20, alpha=0.3, 
                      color='#1f77b4', label='Daily Average/√7')
            
            # Weekly curve
            ax.plot(week_times, week_values, 'o-', linewidth=2, markersize=8,
                   color='darkblue', label='Weekly average')
            
            # Overall average line
            ax.axhline(y=overall_avg, color='gray', linestyle='--', linewidth=2,
                       label=f'Average ({overall_avg:.2f} µK√s)')
            
            # Benchmark lines (only when a NET target is known for this band)
            if map_depth_baseline is not None:
                ax.axhline(y=map_depth_baseline, color='orange', linestyle='--', linewidth=2,
                           label=f'Baseline ({map_depth_baseline:.2f} µK√s)')
                ax.axhline(y=map_depth_goal, color='green', linestyle='--', linewidth=2,
                           label=f'Goal ({map_depth_goal:.2f} µK√s)')


            # Formatting
            ax.set_xlabel('Observation Time', fontsize=12)
            ax.set_ylabel('Inferred Weekly Map Depth (µK·arcmin)', fontsize=12)
            ax.set_title(f'Weekly Map Depth: {telescope.upper()} - {band.lower()} - {version_dir}', 
                        fontsize=14, fontweight='bold')
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Grid and legend
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.set_axisbelow(True)
            ax.legend(loc='best', fontsize=10)
            
            # Set y-axis limits. Same reason as the white-noise plot: 25-250
            # was an MF window, UHF map depths sit far above it.
            md_must = [overall_avg]
            if map_depth_baseline is not None:
                md_must += [map_depth_baseline, map_depth_goal]
            quantile_ylim(ax, list(daily_values) + list(week_values), md_must)

            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(output_dir, f'{telescope}_map_depth_{band}.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Created: {output_path}")
            plt.close(fig)
            
            # Print summary
            print(f"\n=== Map Depth Summary ({telescope.upper()} - {band.upper()}) ===")
            print(f"Map Depth (min/max/mean): {min(week_values):.2f} / {max(week_values):.2f} / {overall_avg:.2f} µK√s")
            if map_depth_baseline is not None:
                print(f"Baseline: {map_depth_baseline:.2f}, Goal: {map_depth_goal:.2f} µK√s")

        # ===== PLOT 5: MAPPING EFFICIENCY =====
        # Detector-hours of unflagged data from uncut detectors that actually went
        # into the map, as a fraction of the theoretical maximum (NUM_DETECTORS_PER_BAND
        # detectors observing continuously for the whole week). Comparable to the
        # yield*duty-cycle baseline/goal benchmarks (0.7*0.2, 0.8*0.2) used elsewhere.
        #
        # total_samples (obs_band_data[obs_id][band]['samples']) is already a sum,
        # over uncut detectors, of each detector's unflagged sample count (see
        # count_ranges_matrix_samples in track_cuts_and_stats.py), summed over wafers
        # via the query above -- so dividing by the sample rate gives detector-seconds
        # of good data for that observation+band directly.
        SAMPLE_RATE_HZ = 200.0
        seconds_per_week = 7 * 24 * 3600
        baseline_me = 0.7 * 0.2 * 100.0
        goal_me = 0.8 * 0.2 * 100.0

        for band in bands:
            weekly_det_seconds = defaultdict(float)
            week_dates_me = {}

            for obs_id, band_data in obs_band_data.items():
                if band not in band_data:
                    continue
                timestamp_extracted = extract_timestamp_from_obs_id(obs_id)
                if timestamp_extracted is None:
                    continue
                try:
                    dt = datetime.fromtimestamp(timestamp_extracted)
                except (ValueError, OSError):
                    continue

                det_seconds = band_data[band]['samples'] / SAMPLE_RATE_HZ
                week_start = dt - timedelta(days=dt.weekday())
                week_key = week_start.strftime('%Y-%m-%d')
                weekly_det_seconds[week_key] += det_seconds
                week_dates_me[week_key] = week_start

            if not weekly_det_seconds:
                print(f"No data for mapping efficiency ({band})")
                continue

            sorted_weeks = sorted(weekly_det_seconds.keys())
            week_dates_list = [week_dates_me[w] for w in sorted_weeks]
            week_eff_list = [
                weekly_det_seconds[w] / (seconds_per_week * NUM_DETECTORS_PER_BAND) * 100.0
                for w in sorted_weeks
            ]
            avg_me = np.mean(week_eff_list)

            plot_data['mapping_efficiency'][band] = {
                'week_dates': week_dates_list,
                'week_eff': week_eff_list,
                'avg': avg_me,
            }

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(week_dates_list, week_eff_list, 'o-', linewidth=2, markersize=8,
                    color='darkblue', label='Weekly mapping efficiency')
            ax.axhline(y=avg_me, color='gray', linestyle='--', linewidth=2,
                       label=f'Average ({avg_me:.2f}%)')
            ax.axhline(y=baseline_me, color='red', linestyle='--', linewidth=2,
                       label=f'Baseline (0.7×0.2 = {baseline_me:.0f}%)')
            ax.axhline(y=goal_me, color='orange', linestyle='--', linewidth=2,
                       label=f'Goal (0.8×0.2 = {goal_me:.0f}%)')

            ax.set_xlabel('Week Starting', fontsize=12)
            ax.set_ylabel('Mapping Efficiency (%)', fontsize=12)
            ax.set_title(f'Mapping Efficiency: {telescope.upper()} - {band.lower()} - {version_dir}',
                        fontsize=14, fontweight='bold')

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            ax.grid(True, alpha=0.3, linestyle=':')
            ax.set_axisbelow(True)
            ax.legend(loc='best', fontsize=10)

            plt.tight_layout()

            add_formula(fig, 'mapping_efficiency')
            output_path = os.path.join(output_dir, f'{telescope}_mapping_efficiency_{band}.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Created: {output_path}")
            plt.close(fig)

            print(f"\n=== Mapping Efficiency Summary ({telescope.upper()} - {band.upper()}) ===")
            print(f"Mapping efficiency (min/max/mean): {min(week_eff_list):.2f}% / {max(week_eff_list):.2f}% / {avg_me:.2f}%")
            print(f"Baseline: {baseline_me:.0f}%, Goal: {goal_me:.0f}%")

        return plot_data

    finally:
        conn.close()

def combined_bands(all_data, key):
    """Bands present under plot_data[key] for any platform, in BAND_ORDER.

    The combined plots used to assume f090/f150 because satp1 and satp3 are
    both MF. satp2 is UHF, so a satp2-vs-satp3 run shares no band at all and
    the panels have to come from the data instead of from a literal.
    """
    found = set()
    for tel_data in all_data.values():
        found.update(tel_data.get(key, {}).keys())
    ordered = [b for b in BAND_ORDER if b in found]
    ordered += sorted(b for b in found if b not in BAND_ORDER)
    return ordered


def combined_telescopes(all_data):
    """Platforms to draw, in a stable order."""
    known = [t for t in ('satp1', 'satp2', 'satp3') if t in all_data]
    return known + sorted(t for t in all_data if t not in known)


def combined_grid(bands):
    """One panel per band, keeping each panel the shape the 1x2 figures had."""
    fig, axes = plt.subplots(1, len(bands), figsize=(10 * len(bands), 6), squeeze=False)
    return fig, list(axes[0])


# Upper y-limit for every noise-like plot, as a quantile of the plotted
# per-observation values.
#
# Chosen to reproduce the limits asked for by eye: 10 uK*rt(s) for MF array
# NET and 50 uK*rt(s) for UHF. Those correspond to the 0.8445 and 0.8470
# quantiles of their distributions -- the same cut to within 0.3%, which is
# why one number serves both. Expressed as a quantile rather than a fixed
# value so the same rule carries to per-detector noise and map depth, where
# the natural scale differs by an order of magnitude between MF and UHF.
#
# The distributions have very long tails -- MF array NET has a median of 5.8
# and a 99th percentile of 221 -- because an observation that kept only a
# handful of detectors reports a huge value. Clipping there keeps the bulk of
# the data legible instead of compressing it against the axis.
NOISE_YLIM_QUANTILE = 0.845


# Which bands share a y-scale. MF and UHF sit an order of magnitude apart in
# every noise quantity, so one limit across all four panels would flatten the
# MF ones; within a group the panels must share, or f090 and f150 cannot be
# compared by eye.
BAND_GROUP = {'f090': 'MF', 'f150': 'MF', 'f220': 'UHF', 'f280': 'UHF'}


def apply_grouped_ylims(axes, bands, values_by_band, must_by_band,
                        q=NOISE_YLIM_QUANTILE):
    """One y-limit per band group, from the pooled distribution of that group.

    Pooling is what makes the limits reproduce the asked-for values exactly:
    the 0.845 quantile of all MF array-NET observations together is 10
    uK*rt(s), and of all UHF together is 50. Taking the quantile panel by
    panel instead gives each band its own number (f150 12.1, f280 66) and
    leaves the panels within a group incomparable.

    A band whose benchmark or average would fall outside its group's limit
    gets that panel widened; hiding a target line is worse than one panel
    differing from its neighbour.
    """
    pooled = defaultdict(list)
    for band in bands:
        pooled[BAND_GROUP.get(band, band)].extend(values_by_band.get(band, []))

    caps = {}
    for grp, vals in pooled.items():
        arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)],
                         dtype=float)
        if arr.size:
            caps[grp] = (float(np.quantile(arr, 1.0 - q)),
                         float(np.quantile(arr, q)))

    for ax, band in zip(axes, bands):
        cap = caps.get(BAND_GROUP.get(band, band))
        if cap is None:
            continue
        lo, hi = cap
        extra = [v for v in must_by_band.get(band, ())
                 if v is not None and np.isfinite(v)]
        if extra:
            hi = max(hi, max(extra))
            lo = min(lo, min(extra))
        if not np.isfinite(hi) or hi <= lo:
            continue
        pad = 0.08 * (hi - lo)
        ax.set_ylim(max(0.0, lo - pad), hi + pad)


def quantile_ylim(ax, values, must_show=(), q=NOISE_YLIM_QUANTILE):
    """Clip the y-axis to a quantile of the data, never hiding a benchmark.

    ``must_show`` values (benchmark lines, platform averages) always stay
    inside the limits: a plot whose target sits off-screen is worse than one
    with a visible tail.
    """
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                      dtype=float)
    extra = np.asarray([v for v in must_show if v is not None and np.isfinite(v)],
                       dtype=float)
    if vals.size == 0:
        return
    hi = float(np.quantile(vals, q))
    lo = float(np.quantile(vals, 1.0 - q)) if vals.size > 1 else float(vals.min())
    if extra.size:
        hi = max(hi, float(extra.max()))
        lo = min(lo, float(extra.min()))
    if not np.isfinite(hi) or hi <= lo:
        return
    pad = 0.08 * (hi - lo)
    ax.set_ylim(max(0.0, lo - pad), hi + pad)


def unify_xlim(axes, times):
    """Give every panel the same time axis.

    Panels otherwise autoscale independently, so a platform observing over a
    few months and one observing over two years fill their panels equally and
    look like comparable amounts of data. Sharing the range makes the
    difference in coverage visible.
    """
    ts = [t for t in times if t is not None]
    if not ts:
        return
    lo, hi = min(ts), max(ts)
    if lo == hi:
        return
    pad = (hi - lo) * 0.02
    for ax in axes:
        ax.set_xlim(lo - pad, hi + pad)


def autoscale_with_benchmarks(ax, values):
    """Set y-limits from the data plus benchmark lines.

    The MF-only figures could hard-code 100-600 (white noise) and 25-250 (map
    depth). UHF NETs are several times larger, so fixed limits would push the
    benchmark lines off the axes; follow the values actually drawn instead.
    """
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return
    lo, hi = min(values), max(values)
    pad = 0.35 * (hi - lo) if hi > lo else 0.35 * max(hi, 1.0)
    ax.set_ylim(max(0.0, lo - pad), hi + pad)


def generate_combined_yield_plot(all_data, version_number, output_dir):
    """Combined detector yield: one panel per band, all platforms overlaid."""
    bands = combined_bands(all_data, 'yield')
    telescopes = combined_telescopes(all_data)
    if not bands:
        return
    fig, axes = combined_grid(bands)
    any_data = False

    all_times = []
    for ax, band in zip(axes, bands):
        for telescope in telescopes:
            band_data = all_data.get(telescope, {}).get('yield', {}).get(band)
            if not band_data or not band_data['obs_times']:
                continue
            any_data = True
            all_times.extend(band_data['week_dates'])
            style = TELESCOPE_STYLE[telescope]

            ax.scatter(band_data['obs_times'], band_data['obs_yield'], s=20, alpha=0.3,
                       color=style['scatter'], label=f'{telescope.upper()} per-observation')
            ax.plot(band_data['week_dates'], band_data['week_yield'], 'o-', linewidth=2, markersize=8,
                    color=style['line'], label=f'{telescope.upper()} weekly average')
            ax.axhline(y=band_data['avg'], color=style['line'], linestyle='--', linewidth=2, alpha=0.6,
                       label=f'{telescope.upper()} average ({band_data["avg"]:.2f})')

        # Reference lines (telescope-independent, same as per-platform plots)
        ax.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='Baseline (0.7)')
        ax.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, label='Goal (0.8)')

        ax.set_xlabel('Observation Time', fontsize=12)
        ax.set_ylabel('Detector Yield (uncut detectors / 6020)', fontsize=12)
        ax.set_title(f'{band.lower()}', fontsize=14, fontweight='bold')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_axisbelow(True)
        ax.legend(loc='lower right', fontsize=9)

    if not any_data:
        plt.close(fig)
        return

    unify_xlim(axes, all_times)

    fig.suptitle(f'Detector Yield: {" vs ".join(telescopes)} - {version_number}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    add_formula(fig, 'yield')
    output_path = os.path.join(output_dir, f'combined_platforms_detector_yield_{version_number}.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Created: {output_path}")
    plt.close(fig)

def generate_combined_efficiency_plot(all_data, version_number, output_dir):
    """Combined observing efficiency: one panel per band, all platforms overlaid."""
    bands = combined_bands(all_data, 'efficiency')
    telescopes = combined_telescopes(all_data)
    if not bands:
        return
    fig, axes = combined_grid(bands)
    any_data = False

    all_times = []
    for ax, band in zip(axes, bands):
        for telescope in telescopes:
            band_data = all_data.get(telescope, {}).get('efficiency', {}).get(band)
            if not band_data:
                continue
            tstyle = TELESCOPE_STYLE[telescope]

            for curve_key in ['curve1', 'curve2', 'curve3']:
                curve = band_data[curve_key]
                if not curve['dates']:
                    continue
                any_data = True
                all_times.extend(curve['dates'])
                cstyle = EFFICIENCY_CURVE_STYLE[curve_key]
                avg_suffix = f" ({curve['avg']:.1f}%)" if curve['avg'] is not None else ""
                ax.plot(curve['dates'], curve['eff'], marker=cstyle['marker'], linestyle=cstyle['linestyle'],
                        color=tstyle['line'], linewidth=2, markersize=7, alpha=0.85,
                        label=f"{telescope.upper()}: {cstyle['label']}{avg_suffix}")

        # Reference line: 20% benchmark (telescope-independent, same as per-platform plots)
        ax.axhline(y=20, color='black', linestyle=':', linewidth=2, label='20% Benchmark')

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Observing Efficiency (%)', fontsize=12)
        ax.set_title(f'{band.lower()}', fontsize=14, fontweight='bold')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_axisbelow(True)
        ax.legend(loc='lower right', fontsize=8)

    if not any_data:
        plt.close(fig)
        return

    unify_xlim(axes, all_times)

    fig.suptitle(f'Observing Efficiency: {" vs ".join(telescopes)} - {version_number}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    add_formula(fig, 'efficiency')
    output_path = os.path.join(output_dir, f'combined_platforms_observing_efficiency_{version_number}.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Created: {output_path}")
    plt.close(fig)

def generate_combined_white_noise_plot(all_data, version_number, output_dir):
    """Combined white noise median: one panel per band, all platforms overlaid."""
    bands = combined_bands(all_data, 'white_noise')
    telescopes = combined_telescopes(all_data)
    if not bands:
        return
    fig, axes = combined_grid(bands)
    any_data = False

    all_times = []
    values_by_band, must_by_band = {}, {}
    for ax, band in zip(axes, bands):
        panel_values, panel_must, panel_times = [], [], []
        for telescope in telescopes:
            band_data = all_data.get(telescope, {}).get('white_noise', {}).get(band)
            if not band_data or not band_data['obs_times']:
                continue
            any_data = True
            panel_values.extend(band_data['obs_wn'])
            panel_must.append(band_data['avg'])
            panel_times.extend(band_data['obs_times'])
            style = TELESCOPE_STYLE[telescope]

            ax.scatter(band_data['obs_times'], band_data['obs_wn'], s=20, alpha=0.3,
                       color=style['scatter'], label=f'{telescope.upper()} per-observation median')
            ax.plot(band_data['week_dates'], band_data['week_wn'], 'o-', linewidth=2, markersize=8,
                    color=style['line'], label=f'{telescope.upper()} weekly average')
            ax.axhline(y=band_data['avg'], color=style['line'], linestyle='--', linewidth=2, alpha=0.6,
                       label=f'{telescope.upper()} average ({band_data["avg"]:.1f} uK·√s)')

        # Benchmark lines (band-dependent, telescope-independent, same as
        # per-platform plots). Read from NET_BENCHMARKS so every band this run
        # finds gets its own target rather than an f090-or-else branch.
        bench = NET_BENCHMARKS.get(band)
        if bench:
            ax.axhline(y=bench['baseline'], color='orange', linestyle='--', linewidth=2,
                       label=f'Baseline ({bench["baseline"]:.0f} uK·√s)')
            ax.axhline(y=bench['goal'], color='green', linestyle='--', linewidth=2,
                       label=f'Goal ({bench["goal"]:.0f} uK·√s)')
            panel_must.extend([bench['baseline'], bench['goal']])

        ax.set_xlabel('Observation Time', fontsize=12)
        ax.set_ylabel('White Noise (uK·√s)', fontsize=12)
        ax.set_title(f'{band.lower()}', fontsize=14, fontweight='bold')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_axisbelow(True)
        ax.legend(loc='best', fontsize=9)

        values_by_band[band] = panel_values
        must_by_band[band] = panel_must
        all_times.extend(panel_times)

    apply_grouped_ylims(axes, bands, values_by_band, must_by_band)
    unify_xlim(axes, all_times)

    if not any_data:
        plt.close(fig)
        return

    fig.suptitle(f'Median White Noise: {" vs ".join(telescopes)} - {version_number}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'combined_platforms_white_noise_{version_number}.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Created: {output_path}")
    plt.close(fig)

def generate_combined_array_net_plot(all_data, version_number, output_dir):
    """Combined array NET: one panel per band, all platforms overlaid.

    Array NET is 1/sqrt(sum(1/NET_det^2)) over every valid detector of an
    observation. The benchmark is the same per-detector NET target divided by
    sqrt(NUM_DETECTORS_PER_BAND * yield) -- the array NET a single platform
    would reach at the baseline/goal detector yield, and exactly the quantity
    whose square sets the map-depth weight used in the map-depth plot.
    """
    bands = combined_bands(all_data, 'array_net')
    telescopes = combined_telescopes(all_data)
    if not bands:
        return
    fig, axes = combined_grid(bands)
    any_data = False

    all_times = []
    values_by_band, must_by_band = {}, {}
    for ax, band in zip(axes, bands):
        panel_values, panel_must, panel_times = [], [], []
        for telescope in telescopes:
            band_data = all_data.get(telescope, {}).get('array_net', {}).get(band)
            if not band_data or not band_data['week_an']:
                continue
            any_data = True
            panel_values.extend(band_data['obs_an'])
            panel_must.append(band_data['avg'])
            panel_times.extend(band_data['obs_times'])
            style = TELESCOPE_STYLE[telescope]
            ax.scatter(band_data['obs_times'], band_data['obs_an'], s=20, alpha=0.3,
                       color=style['scatter'],
                       label=f'{telescope.upper()} per-observation array NET')
            ax.plot(band_data['week_dates'], band_data['week_an'], 'o-',
                    linewidth=2, markersize=8, color=style['line'],
                    label=f'{telescope.upper()} weekly median')
            ax.axhline(y=band_data['avg'], color=style['line'], linestyle='--',
                       linewidth=2, alpha=0.6,
                       label=f'{telescope.upper()} median ({band_data["avg"]:.2f} uK·√s)')

        bench = NET_BENCHMARKS.get(band)
        if bench:
            an_base = bench['baseline'] / np.sqrt(NUM_DETECTORS_PER_BAND * 0.7)
            an_goal = bench['goal'] / np.sqrt(NUM_DETECTORS_PER_BAND * 0.8)
            ax.axhline(y=an_base, color='orange', linestyle='--', linewidth=2,
                       label=f'Baseline ({an_base:.2f} uK·√s)')
            ax.axhline(y=an_goal, color='green', linestyle='--', linewidth=2,
                       label=f'Goal ({an_goal:.2f} uK·√s)')
            panel_must.extend([an_base, an_goal])

        ax.set_xlabel('Observation Time', fontsize=12)
        ax.set_ylabel('Array NET (uK·√s)', fontsize=12)
        ax.set_title(f'{band.lower()}', fontsize=14, fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_axisbelow(True)
        ax.legend(loc='best', fontsize=9)
        values_by_band[band] = panel_values
        must_by_band[band] = panel_must
        all_times.extend(panel_times)

    apply_grouped_ylims(axes, bands, values_by_band, must_by_band)
    unify_xlim(axes, all_times)

    if not any_data:
        plt.close(fig)
        return

    fig.suptitle(f'Array NET: {" vs ".join(telescopes)} - {version_number}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'combined_platforms_array_net_{version_number}.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Created: {output_path}")
    plt.close(fig)


def generate_combined_map_depth_plot(all_data, version_number, output_dir):
    """Combined weekly map depth: one panel per band, all platforms overlaid."""
    bands = combined_bands(all_data, 'map_depth')
    telescopes = combined_telescopes(all_data)
    if not bands:
        return
    fig, axes = combined_grid(bands)
    any_data = False

    all_times = []
    values_by_band, must_by_band = {}, {}
    for ax, band in zip(axes, bands):
        panel_values, panel_must, panel_times = [], [], []
        for telescope in telescopes:
            band_data = all_data.get(telescope, {}).get('map_depth', {}).get(band)
            if not band_data or not band_data['week_values']:
                continue
            any_data = True
            panel_values.extend(band_data['daily_values'])
            panel_values.extend(band_data['week_values'])
            panel_must.append(band_data['avg'])
            panel_times.extend(band_data['week_times'])
            panel_times.extend(band_data['daily_times'])
            style = TELESCOPE_STYLE[telescope]

            ax.scatter(band_data['daily_times'], band_data['daily_values'], s=20, alpha=0.3,
                       color=style['scatter'], label=f'{telescope.upper()} daily average/√7')
            ax.plot(band_data['week_times'], band_data['week_values'], 'o-', linewidth=2, markersize=8,
                    color=style['line'], label=f'{telescope.upper()} weekly average')
            ax.axhline(y=band_data['avg'], color=style['line'], linestyle='--', linewidth=2, alpha=0.6,
                       label=f'{telescope.upper()} average ({band_data["avg"]:.2f} µK√s)')

        # Benchmark lines (band-dependent, telescope-independent, same as
        # per-platform plots), from NET_BENCHMARKS so every band gets its own.
        bench = NET_BENCHMARKS.get(band)
        if bench:
            map_depth_baseline = map_depth_from_net(bench['baseline'], 0.7)
            map_depth_goal = map_depth_from_net(bench['goal'], 0.8)

            ax.axhline(y=map_depth_baseline, color='orange', linestyle='--', linewidth=2,
                       label=f'Baseline ({map_depth_baseline:.2f} µK√s)')
            ax.axhline(y=map_depth_goal, color='green', linestyle='--', linewidth=2,
                       label=f'Goal ({map_depth_goal:.2f} µK√s)')
            panel_must.extend([map_depth_baseline, map_depth_goal])

        ax.set_xlabel('Observation Time', fontsize=12)
        ax.set_ylabel('Inferred Weekly Map Depth (µK·arcmin)', fontsize=12)
        ax.set_title(f'{band.lower()}', fontsize=14, fontweight='bold')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_axisbelow(True)
        ax.legend(loc='best', fontsize=9)

        values_by_band[band] = panel_values
        must_by_band[band] = panel_must
        all_times.extend(panel_times)

    apply_grouped_ylims(axes, bands, values_by_band, must_by_band)
    unify_xlim(axes, all_times)

    if not any_data:
        plt.close(fig)
        return

    fig.suptitle(f'Weekly Map Depth: {" vs ".join(telescopes)} - {version_number}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'combined_platforms_map_depth_{version_number}.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Created: {output_path}")
    plt.close(fig)

def generate_combined_mapping_efficiency_plot(all_data, version_number, output_dir):
    """Combined weekly mapping efficiency: one panel per band, all platforms overlaid."""
    bands = combined_bands(all_data, 'mapping_efficiency')
    telescopes = combined_telescopes(all_data)
    if not bands:
        return
    fig, axes = combined_grid(bands)
    any_data = False

    baseline_me = 0.7 * 0.2 * 100.0
    goal_me = 0.8 * 0.2 * 100.0

    all_times = []
    for ax, band in zip(axes, bands):
        for telescope in telescopes:
            band_data = all_data.get(telescope, {}).get('mapping_efficiency', {}).get(band)
            if not band_data or not band_data['week_eff']:
                continue
            any_data = True
            all_times.extend(band_data['week_dates'])
            style = TELESCOPE_STYLE[telescope]

            ax.plot(band_data['week_dates'], band_data['week_eff'], 'o-', linewidth=2, markersize=8,
                    color=style['line'], label=f'{telescope.upper()} weekly mapping efficiency')
            ax.axhline(y=band_data['avg'], color=style['line'], linestyle='--', linewidth=2, alpha=0.6,
                       label=f'{telescope.upper()} average ({band_data["avg"]:.2f}%)')

        ax.axhline(y=baseline_me, color='red', linestyle='--', linewidth=2,
                   label=f'Baseline (0.7×0.2 = {baseline_me:.0f}%)')
        ax.axhline(y=goal_me, color='orange', linestyle='--', linewidth=2,
                   label=f'Goal (0.8×0.2 = {goal_me:.0f}%)')

        ax.set_xlabel('Week Starting', fontsize=12)
        ax.set_ylabel('Mapping Efficiency (%)', fontsize=12)
        ax.set_title(f'{band.lower()}', fontsize=14, fontweight='bold')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_axisbelow(True)
        ax.legend(loc='best', fontsize=9)

    if not any_data:
        plt.close(fig)
        return

    unify_xlim(axes, all_times)

    fig.suptitle(f'Mapping Efficiency: {" vs ".join(telescopes)} - {version_number}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    add_formula(fig, 'mapping_efficiency')
    output_path = os.path.join(output_dir, f'combined_platforms_mapping_efficiency_{version_number}.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Created: {output_path}")
    plt.close(fig)

def generate_combined_platform_plots(version_number, all_data, output_dir):
    """
    Generate combined satp1-vs-satp3 plots for a given version number.

    Parameters
    ----------
    version_number : str
        Version suffix shared by both platform directories (e.g. '20260417').
    all_data : dict
        {telescope: plot_data} as returned by generate_detector_yield_and_efficiency_plots,
        for whichever of satp1/satp3 were available for this version.
    output_dir : str
        Directory to write combined plots into.
    """
    if not all_data:
        return

    os.makedirs(output_dir, exist_ok=True)

    generate_combined_yield_plot(all_data, version_number, output_dir)
    generate_combined_efficiency_plot(all_data, version_number, output_dir)
    generate_combined_white_noise_plot(all_data, version_number, output_dir)
    generate_combined_array_net_plot(all_data, version_number, output_dir)
    generate_combined_map_depth_plot(all_data, version_number, output_dir)
    generate_combined_mapping_efficiency_plot(all_data, version_number, output_dir)
    write_combined_benchmark_table(all_data, version_number, output_dir)

def write_combined_benchmark_table(all_data, version_number, output_dir):
    """Write every benchmark drawn, its provenance, and the measured mean beside it.

    The benchmark lines are now band-dependent and come from a choosable NET
    set, so a figure alone no longer says which model produced its targets.
    This CSV is the record of what each run actually drew.
    """
    import csv

    bands = sorted(
        set(combined_bands(all_data, 'white_noise'))
        | set(combined_bands(all_data, 'map_depth'))
        | set(combined_bands(all_data, 'mapping_efficiency')),
        key=lambda b: BAND_ORDER.index(b) if b in BAND_ORDER else 99)
    if not bands:
        return

    path = os.path.join(output_dir, f'combined_platforms_benchmarks_{version_number}.csv')
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['band', 'platform',
                    'net_baseline_uK_rts', 'net_goal_uK_rts', 'net_source',
                    'mapdepth_baseline_uK_arcmin', 'mapdepth_goal_uK_arcmin',
                    'mapeff_baseline_pct', 'mapeff_goal_pct',
                    'arraynet_baseline_uK_rts', 'arraynet_goal_uK_rts',
                    'measured_net_mean_uK_rts', 'measured_arraynet_median_uK_rts',
                    'measured_mapdepth_mean_uK_arcmin',
                    'measured_mapeff_mean_pct'])
        for band in bands:
            bench = NET_BENCHMARKS.get(band)
            for telescope in combined_telescopes(all_data):
                td = all_data.get(telescope, {})
                wn = td.get('white_noise', {}).get(band)
                md = td.get('map_depth', {}).get(band)
                me = td.get('mapping_efficiency', {}).get(band)
                an = td.get('array_net', {}).get(band)
                if not (wn or md or me):
                    continue
                w.writerow([
                    band, telescope,
                    f"{bench['baseline']:.1f}" if bench else '',
                    f"{bench['goal']:.1f}" if bench else '',
                    bench.get('source', '') if bench else '',
                    f"{map_depth_from_net(bench['baseline'], 0.7):.2f}" if bench else '',
                    f"{map_depth_from_net(bench['goal'], 0.8):.2f}" if bench else '',
                    '14.0', '16.0',
                    f"{bench['baseline']/np.sqrt(NUM_DETECTORS_PER_BAND*0.7):.3f}" if bench else '',
                    f"{bench['goal']/np.sqrt(NUM_DETECTORS_PER_BAND*0.8):.3f}" if bench else '',
                    f"{wn['avg']:.1f}" if wn else '',
                    f"{an['avg']:.3f}" if an else '',
                    f"{md['avg']:.2f}" if md else '',
                    f"{me['avg']:.2f}" if me else '',
                ])
    print(f"✓ Created: {path}")

def find_stats_db(db_dir, telescope):
    """The cuts/stats sqlite for a platform, not its neighbours.

    databases/ also holds loaded_dets_<tel>.sqlite and ns_study_<tel>.sqlite,
    and a bare *.sqlite glob returns them in arbitrary order, so name the
    archive explicitly and fall back to "anything that isn't one of those".
    """
    named = sorted(Path(db_dir).glob(f'iso_v4_{telescope}_*.sqlite'))
    if named:
        return str(named[0])
    others = [p for p in sorted(Path(db_dir).glob('*.sqlite'))
              if not p.name.startswith(('loaded_dets', 'ns_study'))]
    return str(others[0]) if others else None


def main():
    """Generate per-platform plots and the combined cross-platform figures."""
    parser = argparse.ArgumentParser(
        description='Detector yield, observing efficiency, white noise, map depth '
                    'and mapping efficiency plots for the ISO-SAT cuts/stats archives.')
    parser.add_argument('--base-dir', default='/scratch/gpfs/SIMONSOBS/sat-iso/v4/cuts_stats')
    parser.add_argument('--versions', nargs='+', default=None,
                        help="version suffixes to process, e.g. 260702 or 20260520_paperlist. "
                             "Default: every version directory except the 260702 archives.")
    parser.add_argument('--telescopes', nargs='+', default=None,
                        help="restrict to these platforms, e.g. satp2 satp3. "
                             "Default: whichever are present.")
    parser.add_argument('--net-source', choices=sorted(NET_BENCHMARK_SETS), default='deck',
                        help="'deck' keeps the f090/f150 NET targets printed in "
                             "f2f_data_retention.pdf and takes f220/f280 from V3r6; "
                             "'v3r6' puts all four bands on the V3r6 model.")
    parser.add_argument('--out-dir', default=None,
                        help='combined-plot output directory (default <base-dir>/summary_plots)')
    args = parser.parse_args()

    base_dir = args.base_dir
    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        return

    # Bind the chosen NET set before anything draws, so the per-platform plots
    # and the combined figures quote the same targets.
    global NET_BENCHMARKS
    NET_BENCHMARKS = NET_BENCHMARK_SETS[args.net_source]
    print(f"NET benchmark set: {args.net_source}")

    # Collected per-platform plot data, keyed by version number, used afterwards
    # to build the combined cross-platform plots.
    results_by_version = defaultdict(dict)

    for version_dir in sorted(os.listdir(base_dir)):
        version_path = os.path.join(base_dir, version_dir)
        if not os.path.isdir(version_path):
            continue
        if '_' not in version_dir:
            continue

        telescope = version_dir.split('_')[0]
        # 'satp1_20260520_paperlist' -> '20260520_paperlist', not 'paperlist'.
        version_number = version_dir.split('_', 1)[1]

        if args.versions is None:
            # Default pass skips the 260702 archives, which are driven
            # explicitly with --versions 260702.
            if '260702' in version_dir:
                print(f"Skipping {version_dir}")
                continue
        elif version_number not in args.versions:
            continue

        if args.telescopes and telescope not in args.telescopes:
            continue

        db_dir = os.path.join(version_path, 'databases')
        if not os.path.exists(db_dir):
            continue

        db_path = find_stats_db(db_dir, telescope)
        if db_path is None:
            continue

        print(f"\nProcessing {version_dir} ({os.path.basename(db_path)})...")
        plot_data = generate_detector_yield_and_efficiency_plots(db_path, telescope, version_dir)
        if plot_data:
            results_by_version[version_number][telescope] = plot_data

    # Build combined cross-platform plots for every version where we have data
    combined_output_dir = args.out_dir or os.path.join(base_dir, 'summary_plots')
    for version_number, all_data in sorted(results_by_version.items()):
        print(f"\nGenerating combined platform plots for version {version_number}...")
        generate_combined_platform_plots(version_number, all_data, combined_output_dir)

if __name__ == '__main__':
    main()
