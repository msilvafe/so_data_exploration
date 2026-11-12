#!/usr/bin/env python3
"""
Generate survey efficiency plots showing detector-hours per week relative to theoretical maximum.
"""

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
from pathlib import Path
from collections import defaultdict

def extract_timestamp_from_obs_id(obs_id):
    """Extract Unix timestamp from observation ID.
    
    Format: obs_<timestamp>_<sat>_<wafer_config>
    """
    parts = obs_id.split('_')
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None

def generate_survey_efficiency_plot(db_path, telescope, version_dir, output_dir=None):
    """
    Generate survey efficiency plot for a given database.
    
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
        # Get all observations with their sample counts
        cursor.execute("SELECT obs_id, total_samples FROM obs WHERE total_samples > 0")
        rows = cursor.fetchall()
        
        if not rows:
            print(f"No observations with samples found in {db_path}")
            return False
        
        # Group by week
        weekly_samples = defaultdict(float)
        week_dates = {}
        
        for obs_id, total_samples in rows:
            timestamp = extract_timestamp_from_obs_id(obs_id)
            if timestamp is None:
                continue
            
            # Convert timestamp to datetime
            try:
                dt = datetime.fromtimestamp(timestamp)
            except (ValueError, OSError):
                continue
            
            # Get Monday of the week (ISO week)
            week_start = dt - timedelta(days=dt.weekday())
            week_key = week_start.strftime('%Y-%m-%d')
            
            # Convert samples to detector-hours
            # total_samples is for all detectors, 200 Hz sampling
            detector_hours = total_samples / (200 * 3600)
            
            weekly_samples[week_key] += detector_hours
            if week_key not in week_dates:
                week_dates[week_key] = week_start
        
        if not weekly_samples:
            print(f"No valid observations found in {db_path}")
            return False
        
        # Sort by week
        sorted_weeks = sorted(weekly_samples.keys())
        dates = [week_dates[w] for w in sorted_weeks]
        surveyed_hours = [weekly_samples[w] for w in sorted_weeks]
        
        # Calculate theoretical and goal targets per week
        # 0.7 × 6048 detectors × 0.2 duty cycle × seconds per week
        num_detectors = 6048
        duty_cycle = 0.2
        seconds_per_week = 7 * 24 * 3600
        
        theoretical_target = 0.7 * num_detectors * duty_cycle * seconds_per_week / 3600  # Convert to hours
        goal_target = 0.8 * num_detectors * duty_cycle * seconds_per_week / 3600  # Convert to hours
        
        theoretical_hours = [theoretical_target] * len(dates)
        goal_hours = [goal_target] * len(dates)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot surveyed vs targets
        ax.plot(dates, surveyed_hours, 'o-', linewidth=2, markersize=8, 
                label='Surveyed Detector-Hours', color='#1f77b4')
        ax.axhline(y=theoretical_target, color='red', linestyle='--', linewidth=2, 
                   label=f'Theoretical (0.7 × 6048 × 0.2 × T = {theoretical_target:.0f} hr/week)')
        ax.axhline(y=goal_target, color='orange', linestyle='--', linewidth=2, 
                   label=f'Goal (0.8 × 6048 × 0.2 × T = {goal_target:.0f} hr/week)')
        
        # Formatting
        ax.set_xlabel('Week Starting', fontsize=12)
        ax.set_ylabel('Surveyed Detector-Hours', fontsize=12)
        ax.set_title(f'Survey Efficiency: {telescope.upper()} - {version_dir}\n'
                    f'Weekly Detector-Hours vs Targets', 
                    fontsize=14, fontweight='bold')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_axisbelow(True)
        
        # Legend
        ax.legend(loc='upper left', fontsize=10)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, 
                                   f'{telescope}_survey_efficiency_weekly.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Created: {output_path}")
        plt.close()
        
        # Print summary statistics
        print(f"\n=== Survey Efficiency Summary ({telescope}) ===")
        print(f"Theoretical target (0.7 × 6048 × 0.2 × T): {theoretical_target:.0f} detector-hours/week")
        print(f"Goal target (0.8 × 6048 × 0.2 × T): {goal_target:.0f} detector-hours/week")
        print(f"Surveyed hours (min/max/mean): {min(surveyed_hours):.0f} / {max(surveyed_hours):.0f} / {np.mean(surveyed_hours):.0f}")
        
        return True
        
    finally:
        conn.close()

def main():
    """Generate survey efficiency plots for all databases."""
    base_dir = '/scratch/gpfs/SIMONSOBS/sat-iso/v4/cuts_stats'
    
    # Find all database directories
    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        return
    
    for version_dir in sorted(os.listdir(base_dir)):
        version_path = os.path.join(base_dir, version_dir)
        if not os.path.isdir(version_path):
            continue
        
        db_dir = os.path.join(version_path, 'databases')
        if not os.path.exists(db_dir):
            continue
        
        # Find sqlite database
        sqlite_files = list(Path(db_dir).glob('*.sqlite'))
        if not sqlite_files:
            continue
        
        db_path = str(sqlite_files[0])
        
        # Extract telescope name from directory (e.g., 'satp1' from 'satp1_20260417')
        telescope = version_dir.split('_')[0]
        
        print(f"\nProcessing {version_dir}...")
        generate_survey_efficiency_plot(db_path, telescope, version_dir)

if __name__ == '__main__':
    main()
