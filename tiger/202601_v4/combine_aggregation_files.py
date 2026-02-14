#!/usr/bin/env python3
"""
Combine per-task aggregation HDF5 files into per-band combined files.
Run after the parallel tracking job completes.
"""

import os
import glob
import h5py
import numpy as np
import argparse
from pathlib import Path


def combine_band_files(aggregation_dir, band, output_dir):
    """
    Combine all per-task HDF5 files for a specific band into a single file.
    
    Parameters
    ----------
    aggregation_dir : str
        Directory containing per-task HDF5 files
    band : str
        Band name (e.g., 'f090', 'f150')
    output_dir : str
        Directory to write combined file
    """
    # Find all files for this band
    pattern = os.path.join(aggregation_dir, f'*_task*_{band}.h5')
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No files found for band {band} in {aggregation_dir}")
        return None
    
    print(f"Combining {len(files)} files for band {band}...")
    
    # Read all data from per-task files
    all_white_noise = []
    all_det_ids = []
    all_wafer_slots = []
    all_obs_ids = []
    all_obs_starts = []
    all_obs_ends = []
    obs_ranges = {}
    
    current_det_idx = 0
    
    for task_file in files:
        print(f"  Reading {os.path.basename(task_file)}...")
        with h5py.File(task_file, 'r') as f:
            white_noise = f['white_noise'][:]
            det_ids = [d.decode('utf-8') if isinstance(d, bytes) else d for d in f['det_ids'][:]]
            wafer_slots = f['wafer_slots'][:]
            obs_ids = [o.decode('utf-8') if isinstance(o, bytes) else o for o in f['obs_ids'][:]]
            obs_starts = f['obs_starts'][:]
            obs_ends = f['obs_ends'][:]
            
            n_dets = len(det_ids)
            
            all_white_noise.append(white_noise)
            all_det_ids.extend(det_ids)
            all_wafer_slots.append(wafer_slots)
            
            # Adjust obs_ranges indices for the combined array offset
            for obs_id, obs_start, obs_end in zip(obs_ids, obs_starts, obs_ends):
                if obs_id not in obs_ranges:
                    obs_ranges[obs_id] = (current_det_idx + obs_start, current_det_idx + obs_end)
                    all_obs_ids.append(obs_id)
                else:
                    # Update range to include new data
                    old_start, old_end = obs_ranges[obs_id]
                    obs_ranges[obs_id] = (min(old_start, current_det_idx + obs_start),
                                         max(old_end, current_det_idx + obs_end))
            
            current_det_idx += n_dets
    
    # Concatenate all data
    combined_white_noise = np.concatenate(all_white_noise)
    combined_wafer_slots = np.concatenate(all_wafer_slots)
    
    # Write combined file
    output_file = os.path.join(output_dir, f'noise_aggregation_{band}.h5')
    print(f"Writing combined file: {output_file}")
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('white_noise', data=combined_white_noise, compression='gzip')
        
        # Use fixed-length strings for detector names
        dt = h5py.string_dtype(encoding='utf-8', length=24)
        f.create_dataset('det_ids', data=np.array(all_det_ids, dtype=str), dtype=dt, compression='gzip')
        
        f.create_dataset('wafer_slots', data=combined_wafer_slots, compression='gzip')
        
        # Write observation metadata
        obs_ids_list = list(obs_ranges.keys())
        obs_starts_array = np.array([obs_ranges[oid][0] for oid in obs_ids_list], dtype=np.int64)
        obs_ends_array = np.array([obs_ranges[oid][1] for oid in obs_ids_list], dtype=np.int64)
        
        dt_obs = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('obs_ids', data=np.array(obs_ids_list, dtype=str), dtype=dt_obs)
        f.create_dataset('obs_starts', data=obs_starts_array)
        f.create_dataset('obs_ends', data=obs_ends_array)
        
        # Add metadata
        f.attrs['band'] = band
        f.attrs['n_total_dets'] = len(all_det_ids)
        f.attrs['n_obs'] = len(obs_ids_list)
    
    print(f"  Created {output_file} with {len(all_det_ids)} detectors from {len(obs_ids_list)} observations")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Combine per-task aggregation files into per-band files')
    parser.add_argument('--aggregation-dir', required=True, help='Directory with per-task HDF5 files')
    parser.add_argument('--output-dir', help='Output directory (defaults to aggregation-dir)')
    parser.add_argument('--bands', default='f090,f150', help='Bands to combine (comma-separated)')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.aggregation_dir
    os.makedirs(output_dir, exist_ok=True)
    
    bands = args.bands.split(',')
    
    print(f"\nCombining aggregation files from: {args.aggregation_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Bands: {bands}\n")
    
    for band in bands:
        band = band.strip()
        combine_band_files(args.aggregation_dir, band, output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
