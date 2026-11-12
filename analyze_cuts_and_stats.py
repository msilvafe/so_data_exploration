#!/usr/bin/env python3
"""
Analysis script for ISO-V4 cuts and stats data.

Generates comprehensive plots and tables for detector cuts and sample flag analysis,
including PWV effects and yield metrics.

Usage:
    python3 analyze_cuts_and_stats.py --db-path <path> --output-dir <path> [--obsdb-path <path>]
"""

import argparse
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import sotodlib for PWV data
try:
    from sotodlib.core.metadata import obsdb as obsdb_module
    SOTODLIB_AVAILABLE = True
except ImportError:
    SOTODLIB_AVAILABLE = False
    print("Warning: sotodlib not available, PWV analysis will be skipped")


class CutsStatsAnalyzer:
    """Analyzes cuts and stats data from ISO-V4 preprocessing."""

    def __init__(self, db_path: str, output_dir: str, obsdb_path: Optional[str] = None):
        """
        Initialize the analyzer.
        
        Parameters
        ----------
        db_path : str
            Path to the SQLite database
        output_dir : str
            Output directory for plots and tables
        obsdb_path : str, optional
            Path to ObsDb context file for PWV data
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.obsdb_path = obsdb_path
        
        # Load data
        self.obs_df = self._load_database()
        self.pwv_data = self._load_pwv_data() if obsdb_path and SOTODLIB_AVAILABLE else None
        
        # Define process names (extracted from database column names)
        self.process_names = self._extract_process_names()
        self.flag_process_names = self._extract_flag_process_names()
        
        # Color palette
        self.colors = self._get_color_palette()
        
    def _load_database(self) -> pd.DataFrame:
        """Load data from SQLite database."""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM obs WHERE success=1"
        df = pd.read_sql_query(query, conn)
        conn.close()
        print(f"Loaded {len(df)} successful observations")
        return df
    
    def _load_pwv_data(self) -> Dict[str, float]:
        """Load PWV data from ObsDb."""
        try:
            if not self.obsdb_path:
                return {}
            
            odb = obsdb_module.ObsDb(self.obsdb_path)
            pwv_data = {}
            
            for obs_id in self.obs_df['obs_id'].unique():
                try:
                    obs_info = odb.query(f"obs_id == '{obs_id}'")
                    if len(obs_info) > 0:
                        pwv = obs_info.iloc[0].get('pwv', np.nan)
                        pwv_data[obs_id] = pwv
                except Exception as e:
                    pass
            
            print(f"Loaded PWV data for {len(pwv_data)} observations")
            return pwv_data
        except Exception as e:
            print(f"Warning: Could not load PWV data: {e}")
            return {}
    
    def _extract_process_names(self) -> List[str]:
        """Extract process names from detector columns."""
        process_names = []
        for col in self.obs_df.columns:
            if col.startswith('det_') and '_ndets' in col:
                # Parse det_<step>_<process>_ndets
                parts = col.split('_')
                if len(parts) >= 4:
                    process_name = '_'.join(parts[2:-1])
                    if process_name and process_name not in process_names:
                        process_names.append(process_name)
        return sorted(set(process_names))
    
    def _extract_flag_process_names(self) -> List[str]:
        """Extract flag process names from sample columns."""
        flag_names = []
        for col in self.obs_df.columns:
            if col.startswith('samp_') and '_nsamps' in col:
                parts = col.split('_')
                if len(parts) >= 4:
                    flag_name = '_'.join(parts[2:-1])
                    if flag_name and flag_name not in flag_names:
                        flag_names.append(flag_name)
        return sorted(set(flag_names))
    
    def _get_color_palette(self) -> Dict[str, str]:
        """Get color palette for plots."""
        return {
            'f090': '#1f77b4',
            'f150': '#ff7f0e',
            'ws1': '#2ca02c',
            'ws2': '#d62728',
            'ws3': '#9467bd',
            'ws4': '#8c564b',
        }
    
    def _extract_timestamp_from_obsid(self, obs_id: str) -> float:
        """Extract Unix timestamp from obs_id string."""
        try:
            parts = obs_id.split('_')
            if len(parts) > 0 and parts[0].isdigit():
                return float(parts[0])
        except (ValueError, IndexError):
            pass
        return np.nan
    
    def _analyze_timeseries_grid(self):
        """Generate daily aggregated grid plots for detector cuts."""
        print("Generating time series grid plots...")
        
        # Get detector count columns
        det_cols = sorted([col for col in self.obs_df.columns if col.startswith('det_') and col.endswith('_ndets')])
        if not det_cols:
            return
        
        # Extract timestamps
        self.obs_df['timestamp_unix'] = self.obs_df['obs_id'].apply(self._extract_timestamp_from_obsid)
        self.obs_df['timestamp_date'] = pd.to_datetime(self.obs_df['timestamp_unix'], unit='s').dt.date
        
        # Get unique wafers (limited to ~7 for grid layout)
        wafers = sorted(self.obs_df['wafer'].unique())[:7]
        
        # Get unique cut types (limited to ~15 for grid layout)
        cut_types = []
        for col in det_cols:
            parts = col.split('_')
            if len(parts) >= 3:
                cut_name = '_'.join(parts[2:-1])
                if cut_name not in cut_types:
                    cut_types.append(cut_name)
        cut_types = cut_types[:15]
        
        # Analyze by band
        for band in sorted(self.obs_df['band'].unique()):
            band_df = self.obs_df[self.obs_df['band'] == band].copy()
            
            if len(band_df) == 0:
                continue
            
            # Create grid plot (15 rows x 7 columns for cut types x wafers)
            fig, axes = plt.subplots(len(cut_types), len(wafers), figsize=(16, 20))
            if len(cut_types) == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle(f'Daily Detector Cuts Time Series - {band}', fontsize=14, fontweight='bold', y=0.995)
            
            for i, cut_type in enumerate(cut_types):
                # Find column for this cut type
                col = None
                for dc in det_cols:
                    if cut_type in dc:
                        col = dc
                        break
                
                if col is None or col not in band_df.columns:
                    continue
                
                for j, wafer in enumerate(wafers):
                    ax = axes[i, j]
                    
                    wafer_data = band_df[band_df['wafer'] == wafer].copy()
                    if len(wafer_data) == 0:
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if i == 0:
                            ax.set_title(wafer, fontsize=10, fontweight='bold')
                        if j == 0:
                            ax.set_ylabel(cut_type, fontsize=9, fontweight='bold')
                        continue
                    
                    # Sort by timestamp and aggregate by date
                    wafer_data = wafer_data.sort_values('timestamp_unix')
                    daily = wafer_data.groupby('timestamp_date')[col].mean()
                    
                    if len(daily) > 0 and not daily.isna().all():
                        daily_clean = daily.dropna()
                        if len(daily_clean) > 0:
                            ax.plot(range(len(daily_clean)), daily_clean.values, 'o-', linewidth=1.5, markersize=4)
                            ax.fill_between(range(len(daily_clean)), daily_clean.values, alpha=0.3)
                            ax.set_ylim(bottom=0)
                    
                    ax.set_xticks([])
                    if i == 0:
                        ax.set_title(wafer, fontsize=10, fontweight='bold')
                    if j == 0:
                        ax.set_ylabel(cut_type, fontsize=9, fontweight='bold')
                    
                    # Only show y-axis for leftmost column
                    if j > 0:
                        ax.set_yticks([])
            
            plt.tight_layout()
            version = self._extract_version_from_db_path()
            filename = f'{version}_timeseries_{band}_grid.png'
            plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Generated {filename}")
    
    def _analyze_timeseries_by_wafer(self):
        """Generate by-wafer step-wise plots showing fraction of detectors cut over time."""
        print("Generating time series by-wafer plots...")
        
        # Get detector count columns
        det_cols = sorted([col for col in self.obs_df.columns if col.startswith('det_') and col.endswith('_ndets')])
        if not det_cols:
            return
        
        # Extract timestamps
        self.obs_df['timestamp_unix'] = self.obs_df['obs_id'].apply(self._extract_timestamp_from_obsid)
        self.obs_df['timestamp_date'] = pd.to_datetime(self.obs_df['timestamp_unix'], unit='s')
        
        # Get unique cut types
        cut_types = []
        for col in det_cols:
            parts = col.split('_')
            if len(parts) >= 3:
                cut_name = '_'.join(parts[2:-1])
                if cut_name not in cut_types:
                    cut_types.append(cut_name)
        
        # Analyze by band
        for band in sorted(self.obs_df['band'].unique()):
            band_df = self.obs_df[self.obs_df['band'] == band].copy()
            
            if len(band_df) == 0:
                continue
            
            # Create one plot per cut type
            for cut_type in cut_types[:15]:  # Limit to first 15
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Find column for this cut type
                col = None
                for dc in det_cols:
                    if cut_type in dc:
                        col = dc
                        break
                
                if col is None or col not in band_df.columns:
                    plt.close()
                    continue
                
                # Plot for each wafer
                wafers = sorted(band_df['wafer'].unique())
                colors = plt.cm.Set3(np.linspace(0, 1, len(wafers)))
                
                has_data = False
                for idx, wafer in enumerate(wafers):
                    wafer_data = band_df[band_df['wafer'] == wafer].copy()
                    if len(wafer_data) == 0:
                        continue
                    
                    # Sort by timestamp
                    wafer_data = wafer_data.sort_values('timestamp_unix')
                    
                    # Calculate fraction remaining (cut fraction)
                    starting = wafer_data[det_cols[0]].values
                    current = wafer_data[col].values
                    
                    # Handle division by zero and NaN values
                    with np.errstate(divide='ignore', invalid='ignore'):
                        fraction_cut = 1 - (current / starting)
                    
                    # Replace inf and nan with 0
                    fraction_cut = np.nan_to_num(fraction_cut, nan=0, posinf=1, neginf=0)
                    fraction_cut = np.clip(fraction_cut, 0, 1)
                    
                    if len(fraction_cut) > 0 and not np.all(np.isnan(fraction_cut)):
                        has_data = True
                        ax.fill_between(range(len(fraction_cut)), fraction_cut, alpha=0.3, color=colors[idx], label=wafer)
                        ax.plot(range(len(fraction_cut)), fraction_cut, 'o-', color=colors[idx], linewidth=1.5, markersize=4)
                
                ax.set_xlabel('Observation Index (time)', fontsize=11, fontweight='bold')
                ax.set_ylabel('Fraction of Detectors Cut', fontsize=11, fontweight='bold')
                ax.set_title(f'Detector Cuts Over Time - {cut_type.replace("_", " ").title()} ({band})', fontsize=12, fontweight='bold')
                ax.legend(loc='best', fontsize=9, ncol=2)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1])
                
                plt.tight_layout()
                version = self._extract_version_from_db_path()
                cut_name_safe = cut_type.replace('_', '-')
                filename = f'{version}_timeseries_{band}_by_wafer_{cut_name_safe}.png'
                plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Generated {filename}")
    
    def _extract_version_from_db_path(self) -> str:
        """Extract version string from database path (e.g., iso_v4_satp1_20260520)."""
        db_name = Path(self.db_path).stem
        # Remove .sqlite extension if present
        if db_name.endswith('.sqlite'):
            db_name = db_name[:-7]
        return db_name
    
    def analyze(self):
        """Run all analyses."""
        print("\n=== Starting Analysis ===\n")
        
        # Check if we have any data
        if len(self.obs_df) == 0:
            print("WARNING: No successful observations in database")
            print("Skipping analysis")
            return
        
        # PWV analysis (if data available)
        if self.pwv_data:
            self._analyze_pwv()
        
        # Detector cuts analysis
        self._analyze_detector_cuts()
        self._generate_detailed_cut_tables()
        
        # Yield analysis
        self._analyze_yield()
        
        # White noise analysis
        self._analyze_white_noise()
        
        # Sample flags analysis
        self._analyze_sample_flags()
        self._generate_detailed_flag_tables()
        
        # Time series analysis
        self._analyze_timeseries_grid()
        self._analyze_timeseries_by_wafer()
        
        print("\n=== Analysis Complete ===\n")
        print(f"Outputs saved to: {self.output_dir}")
    
    def _analyze_pwv(self):
        """Analyze PWV effects on noise and cuts."""
        print("Analyzing PWV effects...")
        
        # Merge PWV data
        pwv_series = pd.Series(self.pwv_data, name='pwv')
        df = self.obs_df.merge(pwv_series, left_on='obs_id', right_index=True, how='left')
        df = df.dropna(subset=['pwv'])
        
        if len(df) == 0:
            print("No PWV data available for analysis")
            return
        
        # Convert to μK√s units
        df['noise_Q_white_noise_q50_uK'] = df['noise_Q_white_noise_q50'] * 1e6
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        colors = [self.colors[band] for band in df['band']]
        ax.scatter(df['pwv'], df['noise_Q_white_noise_q50_uK'], alpha=0.6, c=colors, s=30)
        
        # Add trend line
        z = np.polyfit(df['pwv'], df['noise_Q_white_noise_q50_uK'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df['pwv'].min(), df['pwv'].max(), 100)
        ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')
        
        # Mark PWV > 2mm cutoff
        pwv_cut = 2.0
        cut_count = (df['pwv'] > pwv_cut).sum()
        cut_pct = 100 * cut_count / len(df)
        ax.axvline(pwv_cut, color='green', linestyle='--', linewidth=2, 
                   label=f'PWV > {pwv_cut}mm: {cut_count} obs ({cut_pct:.1f}%)')
        
        # Labels and formatting
        ax.set_xlabel('PWV (mm)', fontsize=11, fontweight='bold')
        ax.set_ylabel('White Noise Q (μK√s)', fontsize=11, fontweight='bold')
        ax.set_title('White Noise (Q) vs PWV', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add band legend
        band_handles = [mpatches.Patch(color=self.colors[band], label=band) 
                       for band in sorted(df['band'].unique())]
        ax.legend(handles=band_handles + ax.get_lines(), loc='upper left', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'white_noise_q_vs_pwv.png', dpi=150)
        plt.close()
        
        # Save PWV statistics
        stats = {
            'mean_pwv_mm': df['pwv'].mean(),
            'median_pwv_mm': df['pwv'].median(),
            'max_pwv_mm': df['pwv'].max(),
            'obs_above_2mm': cut_count,
            'obs_above_2mm_pct': cut_pct,
            'correlation_noise_q_pwv': df['noise_Q_white_noise_q50_uK'].corr(df['pwv']),
        }
        
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(self.output_dir / 'pwv_statistics.csv', index=False)
        print(f"  PWV statistics: {cut_count} obs above {pwv_cut}mm ({cut_pct:.1f}%)")
    
    def _analyze_detector_cuts(self):
        """Analyze detector cuts by type."""
        print("Analyzing detector cuts...")
        
        # Get detector counts at each step
        det_cols = sorted([col for col in self.obs_df.columns if col.startswith('det_') and col.endswith('_ndets')])
        
        if not det_cols:
            print("  No detector columns found")
            return
        
        # Extract step number and process name
        steps = []
        for col in det_cols:
            parts = col.split('_')
            step = int(parts[1])
            process = '_'.join(parts[2:-1])
            steps.append((step, process, col))
        
        steps.sort()
        
        # Calculate losses per process (aggregate over all observations)
        losses = {}
        for i in range(len(steps) - 1):
            step1, process1, col1 = steps[i]
            step2, process2, col2 = steps[i + 1]
            
            # Detectors lost at this process (median over observations)
            loss = (self.obs_df[col1].median() - self.obs_df[col2].median())
            
            if process2 not in losses:
                losses[process2] = loss
            else:
                losses[process2] += loss
        
        # Plot: Stacked bar chart - overall
        fig, ax = plt.subplots(figsize=(10, 6))
        
        starting = self.obs_df[det_cols[0]].median()
        final = self.obs_df[det_cols[-1]].median()
        
        # Sort losses by magnitude
        sorted_losses = sorted(losses.items(), key=lambda x: -x[1])
        
        # Group small losses
        threshold = starting * 0.01  # Top 1% threshold
        major_losses = [(p, l) for p, l in sorted_losses if l >= threshold]
        minor_loss = sum([l for p, l in sorted_losses if l < threshold])
        
        if minor_loss > 0:
            major_losses.append(('Other (minor cuts)', minor_loss))
        
        # Create stacked bar
        categories = ['Total']
        bottom = 0
        colors_cuts = plt.cm.Set3(np.linspace(0, 1, len(major_losses)))
        
        y_pos = 0
        for i, (process, loss) in enumerate(major_losses):
            ax.barh(y_pos, loss, left=bottom, label=process, color=colors_cuts[i], height=0.5)
            bottom += loss
        
        ax.set_xlim(0, starting * 1.05)
        ax.set_ylabel('', fontsize=11)
        ax.set_xlabel('Detectors Lost', fontsize=11, fontweight='bold')
        ax.set_title(f'Detector Cut Breakdown\n(Starting: {starting:.0f} → Final: {final:.0f})', 
                    fontsize=12, fontweight='bold')
        ax.set_yticks([])
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add text showing final count
        ax.text(bottom + starting * 0.01, y_pos, f'Final: {final:.0f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'detector_cuts_breakdown.png', dpi=150)
        plt.close()
        
        # Per-band breakdown
        self._plot_cuts_by_band_wafer('band', 'Band')
        self._plot_cuts_by_band_wafer('wafer', 'Wafer')
        
        # Top 5 cuts table
        top_n = 5
        top_cuts = sorted_losses[:top_n]
        cuts_data = []
        for process, loss in top_cuts:
            loss_pct = 100 * loss / starting
            cuts_data.append({
                'Cut Type': process,
                'Avg Detectors Lost': f'{loss:.1f}',
                'Percentage': f'{loss_pct:.1f}%'
            })
        
        cuts_df = pd.DataFrame(cuts_data)
        cuts_df.to_csv(self.output_dir / 'detector_cuts_summary.csv', index=False)
        print(f"  Top {top_n} detector cuts identified")
    
    def _plot_cuts_by_band_wafer(self, group_by: str, label: str):
        """Plot detector cuts grouped by band or wafer."""
        det_cols = sorted([col for col in self.obs_df.columns if col.startswith('det_') and col.endswith('_ndets')])
        
        groups = self.obs_df[group_by].unique()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = 0
        colors_list = plt.cm.Set3(np.linspace(0, 1, len(groups)))
        
        for i, group in enumerate(sorted(groups)):
            group_data = self.obs_df[self.obs_df[group_by] == group]
            starting = group_data[det_cols[0]].median()
            final = group_data[det_cols[-1]].median()
            
            # Stack bars for each cut
            bottom = 0
            for j in range(len(det_cols) - 1):
                loss = group_data[det_cols[j]].median() - group_data[det_cols[j + 1]].median()
                if loss > 0:
                    ax.bar(x_pos, loss, bottom=bottom, color=colors_list[i], alpha=0.7, width=0.6)
                    bottom += loss
            
            # Add label
            ax.text(x_pos, bottom * 1.02, f'{final:.0f}', ha='center', fontsize=9, fontweight='bold')
            x_pos += 1
        
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(sorted(groups), fontsize=10)
        ax.set_ylabel('Detectors', fontsize=11, fontweight='bold')
        ax.set_xlabel(label, fontsize=11, fontweight='bold')
        ax.set_title(f'Detector Cuts by {label}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'detector_cuts_by_{group_by}.png', dpi=150)
        plt.close()
    
    def _generate_detailed_cut_tables(self):
        """Generate detailed per-wafer breakdown tables for cuts."""
        det_cols = sorted([col for col in self.obs_df.columns if col.startswith('det_') and col.endswith('_ndets')])
        
        if not det_cols:
            return
        
        # Calculate losses per process
        losses_by_process = {}
        for i in range(len(det_cols) - 1):
            col1, col2 = det_cols[i], det_cols[i + 1]
            parts = col2.split('_')
            if len(parts) >= 3:
                process = '_'.join(parts[2:-1])
                loss = (self.obs_df[col1].median() - self.obs_df[col2].median())
                losses_by_process[process] = loss
        
        sorted_losses = sorted(losses_by_process.items(), key=lambda x: -x[1])
        top_processes = [p for p, _ in sorted_losses[:5]]
        
        # Per-wafer breakdown
        wafers = sorted(self.obs_df['wafer'].unique())
        per_wafer_data = []
        
        for wafer in wafers:
            wafer_df = self.obs_df[self.obs_df['wafer'] == wafer]
            row = {'Wafer': wafer}
            row['Starting Detectors'] = f"{wafer_df[det_cols[0]].median():.0f}"
            row['Final Detectors'] = f"{wafer_df[det_cols[-1]].median():.0f}"
            
            for process in top_processes:
                total_loss = 0
                for i in range(len(det_cols) - 1):
                    col1, col2 = det_cols[i], det_cols[i + 1]
                    parts = col2.split('_')
                    if len(parts) >= 3:
                        proc = '_'.join(parts[2:-1])
                        if proc == process:
                            total_loss += (wafer_df[col1].median() - wafer_df[col2].median())
                row[process] = f'{total_loss:.0f}'
            per_wafer_data.append(row)
        
        per_wafer_df = pd.DataFrame(per_wafer_data)
        per_wafer_df.to_csv(self.output_dir / 'detector_cuts_per_wafer.csv', index=False)
    
    def _generate_detailed_flag_tables(self):
        """Generate detailed per-wafer breakdown tables for flags."""
        flag_cols = {
            'smurfgaps': 'smurfgaps',
            'turnaround_flags': 'turnaround_flags',
            'jumps_slow': 'jumps_slow',
            'jumps_2pi': 'jumps_2pi',
            'glitches_pre_hwpss': 'glitches_pre_hwpss',
            'source_flags': 'source_flags',
            'glitches_post_hwpss': 'glitches_post_hwpss',
            'noisy_subscan_flags': 'noisy_subscan_flags',
        }
        
        available_flags = {k: v for k, v in flag_cols.items() if v in self.obs_df.columns}
        
        if not available_flags:
            return
        
        # Calculate losses
        losses = {}
        for name, col in available_flags.items():
            loss = self.obs_df[col].median()
            losses[name] = loss
        
        sorted_losses = sorted(losses.items(), key=lambda x: -x[1])
        top_flags = [p for p, _ in sorted_losses[:5]]
        
        # Per-wafer breakdown
        wafers = sorted(self.obs_df['wafer'].unique())
        per_wafer_data = []
        
        for wafer in wafers:
            wafer_df = self.obs_df[self.obs_df['wafer'] == wafer]
            row = {'Wafer': wafer}
            row['Total Samples'] = f"{wafer_df['total_samples'].median():.0f}"
            
            for flag in top_flags:
                if flag in available_flags:
                    loss = wafer_df[available_flags[flag]].median()
                    row[flag] = f'{loss:.0f}'
            per_wafer_data.append(row)
        
        per_wafer_df = pd.DataFrame(per_wafer_data)
        per_wafer_df.to_csv(self.output_dir / 'sample_flags_per_wafer.csv', index=False)
    
    def _analyze_yield(self):
        """Analyze yield metrics."""
        print("Analyzing yield...")
        
        det_cols = sorted([col for col in self.obs_df.columns if col.startswith('det_') and col.endswith('_ndets')])
        
        if not det_cols:
            return
        
        # Calculate yield
        df = self.obs_df.copy()
        df['yield'] = df[det_cols[-1]] / df[det_cols[0]]
        df['noise_Q_white_noise_q50_uK'] = df['noise_Q_white_noise_q50'] * 1e6
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot by band
        for band in sorted(df['band'].unique()):
            band_data = df[df['band'] == band]
            ax.scatter(band_data['noise_Q_white_noise_q50_uK'], band_data['yield'], 
                      alpha=0.6, label=band, color=self.colors.get(band, 'gray'), s=40)
        
        # Add trend line
        z = np.polyfit(df['noise_Q_white_noise_q50_uK'], df['yield'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df['noise_Q_white_noise_q50_uK'].min(), df['noise_Q_white_noise_q50_uK'].max(), 100)
        ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, alpha=0.7)
        
        # Labels and formatting
        ax.set_xlabel('White Noise Q (μK√s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Yield (Final/Starting Detectors)', fontsize=11, fontweight='bold')
        ax.set_title('Detector Yield vs White Noise', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'white_noise_q_vs_yield.png', dpi=150)
        plt.close()
        
        print(f"  Mean yield: {df['yield'].mean():.3f}, Median: {df['yield'].median():.3f}")
    
    def _analyze_white_noise(self):
        """Analyze white noise distribution."""
        print("Analyzing white noise distribution...")
        
        df = self.obs_df.copy()
        
        # Convert to μK√s units (data is in K√s, multiply by 1e6)
        df['noise_Q_white_noise_q50_uK'] = df['noise_Q_white_noise_q50'] * 1e6
        
        # Create figure with subplots for each band
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        bands = sorted(df['band'].unique())
        
        for idx, band in enumerate(bands):
            band_data = df[df['band'] == band]['noise_Q_white_noise_q50_uK'].dropna()
            
            ax = axes[idx]
            ax.hist(band_data, bins=30, alpha=0.7, color=self.colors.get(band, 'gray'), edgecolor='black')
            
            ax.set_xlabel('White Noise Q (μK√s)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Number of Observations', fontsize=11, fontweight='bold')
            ax.set_title(f'{band} Band\n(μ={band_data.mean():.2f}, σ={band_data.std():.2f})', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'white_noise_q_histogram.png', dpi=150)
        plt.close()
        
        print(f"  White noise statistics by band:")
        for band in bands:
            band_data = df[df['band'] == band]['noise_Q_white_noise_q50_uK'].dropna()
            print(f"    {band}: μ={band_data.mean():.2f}, σ={band_data.std():.2f}, n={len(band_data)}")
    
    def _analyze_sample_flags(self):
        """Analyze sample flags."""
        print("Analyzing sample flags...")
        
        # Sample flags are stored as individual columns (not per-step)
        flag_cols = {
            'smurfgaps': 'smurfgaps',
            'turnaround_flags': 'turnaround_flags',
            'jumps_slow': 'jumps_slow',
            'jumps_2pi': 'jumps_2pi',
            'glitches_pre_hwpss': 'glitches_pre_hwpss',
            'source_flags': 'source_flags',
            'glitches_post_hwpss': 'glitches_post_hwpss',
            'noisy_subscan_flags': 'noisy_subscan_flags',
        }
        
        # Check which columns exist
        available_flags = {k: v for k, v in flag_cols.items() if v in self.obs_df.columns}
        
        if not available_flags:
            print("  No sample flag columns found")
            return
        
        # Calculate average samples lost per flag type
        df = self.obs_df.copy()
        
        # Get total samples (starting point)
        total_samps = df['total_samples'].median()
        
        if total_samps == 0:
            print("  No sample data available")
            return
        
        # Calculate losses
        losses = {}
        for name, col in available_flags.items():
            loss = df[col].median()
            losses[name] = loss
        
        # Plot stacked bar
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sorted_losses = sorted(losses.items(), key=lambda x: -x[1])
        
        # Group small losses
        threshold = total_samps * 0.01
        major_losses = [(p, l) for p, l in sorted_losses if l >= threshold]
        minor_loss = sum([l for p, l in sorted_losses if l < threshold])
        
        if minor_loss > 0:
            major_losses.append(('Other (minor flags)', minor_loss))
        
        bottom = 0
        colors_flags = plt.cm.Set2(np.linspace(0, 1, len(major_losses)))
        
        for i, (process, loss) in enumerate(major_losses):
            ax.barh(0, loss, left=bottom, label=process, color=colors_flags[i], height=0.5)
            bottom += loss
        
        ax.set_xlim(0, total_samps * 1.05)
        ax.set_ylabel('', fontsize=11)
        ax.set_xlabel('Samples Flagged', fontsize=11, fontweight='bold')
        ax.set_title(f'Sample Flags Breakdown\n(Total: {total_samps:.0f})', 
                    fontsize=12, fontweight='bold')
        ax.set_yticks([])
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3, axis='x')
        ax.text(bottom + total_samps * 0.01, 0, f'Total: {total_samps:.0f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_flags_breakdown.png', dpi=150)
        plt.close()
        
        # Top flags table
        top_n = 5
        top_flags = sorted_losses[:top_n]
        flags_data = []
        for process, loss in top_flags:
            loss_pct = 100 * loss / total_samps if total_samps > 0 else 0
            flags_data.append({
                'Flag Type': process,
                'Avg Samples Flagged': f'{loss:.0f}',
                'Percentage': f'{loss_pct:.1f}%'
            })
        
        flags_df = pd.DataFrame(flags_data)
        flags_df.to_csv(self.output_dir / 'sample_flags_summary.csv', index=False)
        print(f"  Top {top_n} sample flags identified")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze ISO-V4 cuts and stats data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python3 analyze_cuts_and_stats.py \\
    --db-path /path/to/iso_v4_satp1_20260520.sqlite \\
    --output-dir /path/to/analysis/
    
  python3 analyze_cuts_and_stats.py \\
    --db-path /path/to/iso_v4_satp1_20260520.sqlite \\
    --output-dir /path/to/analysis/ \\
    --obsdb-path /path/to/context.yaml
        '''
    )
    
    parser.add_argument('--db-path', required=True,
                       help='Path to SQLite database')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for plots and tables')
    parser.add_argument('--obsdb-path', default=None,
                       help='Path to ObsDb context file for PWV data (optional)')
    
    args = parser.parse_args()
    
    # Check files exist
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        return 1
    
    # Run analysis
    analyzer = CutsStatsAnalyzer(
        db_path=str(db_path),
        output_dir=args.output_dir,
        obsdb_path=args.obsdb_path
    )
    
    analyzer.analyze()
    return 0


if __name__ == '__main__':
    exit(main())
