#!/usr/bin/env python3
"""
Generate time series plots matching the expected format:
- *_cuts_timeseries_daily_grid.png (15×7 grid: rows=steps, cols=wafers)
- *_cuts_timeseries_daily_by_wafer.png (stacked: rows=steps, each with wafers overlaid)

These match the format shown in the reference PDF.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import sys

def extract_timestamp_from_obsid(obs_id):
    """Extract Unix timestamp from obs_id like obs_1742934989_satp1_1111111"""
    try:
        parts = str(obs_id).split('_')
        if len(parts) > 1:
            timestamp = int(parts[1])
            return pd.to_datetime(timestamp, unit='s')
    except:
        pass
    return None

def load_observations(db_path, band=None):
    """Load observation data from database."""
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM obs"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Extract date from obs_id
    df['date'] = df['obs_id'].apply(extract_timestamp_from_obsid)
    df = df.dropna(subset=['date'])
    
    # Filter by band if specified
    if band and 'band' in df.columns:
        df = df[df['band'] == band]
    
    return df

def get_detector_cut_columns(df):
    """Extract all detector cut step columns."""
    # Pattern: det_<step_num>_<step_name>_ndets
    cut_cols = [c for c in df.columns if c.startswith('det_') and c.endswith('_ndets')]
    
    # Parse step names
    steps = {}
    for col in cut_cols:
        parts = col.split('_')
        if len(parts) >= 4:
            step_num = int(parts[1])
            step_name = '_'.join(parts[2:-2])  # Join middle parts
            steps[step_num] = {'name': step_name, 'col': col}
    
    return steps

def create_grid_plot(df, steps, band, telescope, output_path, wafers=None):
    """Create grid plot: rows=steps, cols=wafers."""
    
    if wafers is None:
        wafers = sorted(df['wafer'].dropna().unique())
    
    # Sort by step number
    sorted_steps = sorted(steps.keys())
    n_steps = len(sorted_steps)
    n_wafers = len(wafers)
    
    fig, axes = plt.subplots(n_steps, n_wafers, figsize=(n_wafers*2, n_steps*1.5))
    
    if n_steps == 1:
        axes = axes.reshape(1, -1)
    if n_wafers == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'SATP1 {band} cut fraction vs. date (daily median) — newer steps, cstr: wafers', 
                 fontsize=14, y=0.995)
    
    for row, step_num in enumerate(sorted_steps):
        step_info = steps[step_num]
        step_name = step_info['name']
        col = step_info['col']
        
        # Get starting ndets
        start_col = None
        for c in df.columns:
            if c.startswith('det_0_') and c.endswith('_ndets'):
                start_col = c
                break
        
        if start_col is None:
            continue
        
        for col_idx, wafer in enumerate(wafers):
            ax = axes[row, col_idx]
            
            wafer_data = df[df['wafer'] == wafer].sort_values('date')
            
            if len(wafer_data) > 0 and start_col in wafer_data.columns and col in wafer_data.columns:
                # Daily aggregation
                wafer_data['date_daily'] = wafer_data['date'].dt.date
                daily = wafer_data.groupby('date_daily').agg({
                    start_col: 'mean',
                    col: 'mean'
                }).reset_index()
                
                if len(daily) > 0:
                    daily['fraction_cut'] = 1.0 - (daily[col] / daily[start_col])
                    daily['date_daily'] = pd.to_datetime(daily['date_daily'])
                    
                    ax.plot(daily['date_daily'], daily['fraction_cut'], linewidth=0.8, color='steelblue')
                    ax.set_ylim(-0.05, 1.05)
            
            # Labels
            if row == 0:
                ax.set_title(wafer, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(step_name, fontsize=9)
            
            ax.tick_params(axis='both', labelsize=8)
            if row < n_steps - 1:
                ax.set_xticklabels([])
            else:
                ax.tick_params(axis='x', rotation=45)
            
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")

def create_by_wafer_plot(df, steps, band, telescope, output_path, wafers=None):
    """Create by-wafer plot: stacked rows, each with wafers overlaid."""
    
    if wafers is None:
        wafers = sorted(df['wafer'].dropna().unique())
    
    sorted_steps = sorted(steps.keys())
    n_steps = len(sorted_steps)
    
    fig, axes = plt.subplots(n_steps, 1, figsize=(14, n_steps*1.2))
    
    if n_steps == 1:
        axes = [axes]
    
    fig.suptitle(f'SATP1 {band} cut fraction vs. date (daily median) by step', fontsize=14)
    
    # Color palette for wafers
    colors = plt.cm.tab10(np.linspace(0, 1, len(wafers)))
    
    # Get starting ndets column
    start_col = None
    for c in df.columns:
        if c.startswith('det_0_') and c.endswith('_ndets'):
            start_col = c
            break
    
    for row, step_num in enumerate(sorted_steps):
        step_info = steps[step_num]
        step_name = step_info['name']
        col = step_info['col']
        ax = axes[row]
        
        for wafer_idx, wafer in enumerate(wafers):
            wafer_data = df[df['wafer'] == wafer].sort_values('date')
            
            if len(wafer_data) > 0 and start_col and col in wafer_data.columns:
                wafer_data['date_daily'] = wafer_data['date'].dt.date
                daily = wafer_data.groupby('date_daily').agg({
                    start_col: 'mean',
                    col: 'mean'
                }).reset_index()
                
                if len(daily) > 0:
                    daily['fraction_cut'] = 1.0 - (daily[col] / daily[start_col])
                    daily['date_daily'] = pd.to_datetime(daily['date_daily'])
                    
                    ax.plot(daily['date_daily'], daily['fraction_cut'], 
                           label=wafer, linewidth=1.5, color=colors[wafer_idx], alpha=0.7)
        
        ax.set_ylabel(step_name, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelsize=8)
        
        if row == 0:
            ax.legend(loc='upper right', fontsize=8, ncol=len(wafers))
        
        if row < n_steps - 1:
            ax.set_xticklabels([])
        else:
            ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_timeseries_plots.py <db_path> <output_dir> [telescope]")
        return 1
    
    db_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    telescope = sys.argv[3] if len(sys.argv) > 3 else 'satp1'
    
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create telescope subdirectory if needed
    tel_dir = output_dir / telescope
    tel_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating time series plots...")
    print(f"Database: {db_path}")
    print(f"Output: {tel_dir}\n")
    
    for band in ['f090', 'f150']:
        print(f"Processing {band}...")
        
        try:
            df = load_observations(db_path, band=band)
            
            if len(df) == 0:
                print(f"  WARNING: No observations found for {band}")
                continue
            
            steps = get_detector_cut_columns(df)
            
            if not steps:
                print(f"  WARNING: No detector cut columns found")
                continue
            
            # Grid plot
            grid_path = tel_dir / f"{telescope}_{band}_cuts_timeseries_daily_grid.png"
            create_grid_plot(df, steps, band, telescope, grid_path)
            
            # By-wafer plot
            by_wafer_path = tel_dir / f"{telescope}_{band}_cuts_timeseries_daily_by_wafer.png"
            create_by_wafer_plot(df, steps, band, telescope, by_wafer_path)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ Done! Plots saved to: {tel_dir}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
