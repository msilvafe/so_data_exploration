import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Optional, List, Dict
import os
import yaml
import json


class CutsStatsAnalyzerV4:
    """Analyzer for cuts and statistics database with v4 schema."""
    
    def __init__(self, db_path: str, config_path: Optional[str] = None):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        if config_path:
            self.step_mapping = self._parse_config(config_path)
        else:
            # Fallback to schema detection if no config provided
            self.column_offset = self._detect_schema()
            self.step_mapping = self._build_default_mapping()
    
    def _parse_config(self, config_path: str) -> Dict[str, Dict[str, int]]:
        """Parse processing configuration to build step mappings."""
        try:
            # Determine config format based on file extension
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                # Try to parse as YAML first, then JSON
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                except:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
            
            # Extract processing steps and build column mapping
            step_mapping = {
                'detector_cuts': {},
                'sample_cuts': {},
                'processing_order': []
            }
            
            # Look for processing pipeline in various possible locations in config
            pipeline = None
            if 'preprocess' in config:
                pipeline = config['preprocess']
            elif 'pipeline' in config:
                pipeline = config['pipeline']
            elif 'processing' in config:
                pipeline = config['processing']
            elif isinstance(config, list):
                pipeline = config
            
            if pipeline:
                step_num = 0
                for step in pipeline:
                    step_name = None
                    
                    # Handle different config formats
                    if isinstance(step, dict):
                        # Get the operation name (could be key or 'name' field)
                        if 'name' in step:
                            step_name = step['name']
                        elif len(step) == 1:
                            step_name = list(step.keys())[0]
                        else:
                            # Look for common operation keys
                            for key in ['operation', 'op', 'func', 'function']:
                                if key in step:
                                    step_name = step[key]
                                    break
                    elif isinstance(step, str):
                        step_name = step
                    
                    if step_name:
                        column_name = f"{step_num}_{step_name}"
                        step_mapping['processing_order'].append(column_name)
                        
                        # Categorize steps as detector cuts or sample cuts
                        detector_cut_keywords = [
                            'dark_dets', 'fp_flags', 'detcal_nan', 'det_bias_flags',
                            'trends', 'jumps', 'glitches', 'ptp_flags', 
                            'noisy_subscan_flags', 'estimate_t2p', 'subtract_t2p',
                            'inv_var_flags', 'cut_bad_dist'
                        ]
                        
                        sample_cut_keywords = [
                            'smurfgaps_flags', 'turnarounds', 'source_flags'
                        ]
                        
                        if any(keyword in step_name.lower() for keyword in detector_cut_keywords):
                            step_mapping['detector_cuts'][step_name] = step_num
                        elif any(keyword in step_name.lower() for keyword in sample_cut_keywords):
                            step_mapping['sample_cuts'][step_name] = step_num
                    
                    step_num += 1
            
            return step_mapping
            
        except Exception as e:
            print(f"Error parsing config file: {e}")
            print("Falling back to schema detection...")
            self.column_offset = self._detect_schema()
            return self._build_default_mapping()
    
    def _build_default_mapping(self) -> Dict[str, Dict[str, int]]:
        """Build default step mapping based on schema detection."""
        # This creates a mapping based on the hardcoded knowledge but adjusted for schema
        offset = getattr(self, 'column_offset', 0)
        
        return {
            'detector_cuts': {
                'dark_dets': 6 + offset,
                'fp_flags': 7 + offset,
                'detcal_nan_cuts': 8 + offset,
                'det_bias_flags': 10 + offset,
                'trends': 11 + offset,
                'jumps_slow': 12 + offset,
                'jumps_2pi': 15 + offset,
                'glitches_pre': 19 + offset,
                'glitches_post': 27 + offset,
                'ptp_flags': 33 + offset,
                'noisy_subscan_flags': 46 + offset,
                'estimate_t2p': 56 + offset,
                'inv_var_flags': 80 + offset
            },
            'sample_cuts': {
                'smurfgaps_flags': 5 + offset,
                'turnarounds': 9 + offset,
                'source_flags': 20 + offset
            }
        }
    
    def _detect_schema(self) -> int:
        """Detect if this is SATP1 (has 1_move) or SATP3 (no 1_move) schema."""
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA table_info(detector_counts);")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        # Check if 1_move column exists (SATP1 schema)
        if '1_move' in column_names:
            return 0  # No offset needed
        else:
            return -1  # Subtract 1 from all column numbers
    
    def _get_column_name(self, step_name: str, category: str = 'detector_cuts') -> str:
        """Get the actual column name for a processing step."""
        if hasattr(self, 'step_mapping') and category in self.step_mapping:
            if step_name in self.step_mapping[category]:
                step_num = self.step_mapping[category][step_name]
                return f"{step_num}_{step_name}"
        
        # Fallback to old method
        if hasattr(self, 'column_offset'):
            # This is a fallback for the old hardcoded approach
            step_mapping = {
                'dark_dets': 6,
                'fp_flags': 7,
                'detcal_nan_cuts': 8,
                'det_bias_flags': 10,
                'trends': 11,
                'jumps': 12,
                'glitchfill': 14,
                'jumps2': 15,
                'detrend': 18,
                'glitches': 19,
                'glitchfill2': 26,
                'glitches2': 27,
                'detrend2': 32,
                'ptp_flags': 33,
                'tod_stats': 45,
                'noisy_subscan_flags': 46,
                'estimate_t2p': 56,
                'subtract_t2p': 57,
                'scan_freq_cut': 79,
                'inv_var_flags': 80
            }
            if step_name in step_mapping:
                actual_number = step_mapping[step_name] + self.column_offset
                return f"{actual_number}_{step_name}"
        
        # If all else fails, return the step name as-is (might cause errors)
        print(f"Warning: Could not find column mapping for {step_name}")
        return step_name
    
    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def get_sample_cuts_summary(self) -> pd.DataFrame:
        """Get summary of sample cuts across all observations."""
        query = """
        SELECT wafer, band,
               AVG(total_samples) as avg_total_samples,
               AVG(smurfgaps_cuts) as avg_smurfgaps_cuts,
               AVG(turnarounds_cuts) as avg_turnarounds_cuts,
               AVG(jumps_slow_cuts) as avg_jumps_slow_cuts,
               AVG(jumps_2pi_cuts) as avg_jumps_2pi_cuts,
               AVG(glitches_pre_hwpss_cuts) as avg_glitches_pre_cuts,
               AVG(glitches_post_hwpss_cuts) as avg_glitches_post_cuts,
               AVG(source_moon_cuts) as avg_moon_cuts,
               COUNT(*) as n_obs
        FROM sample_cuts
        GROUP BY wafer, band
        ORDER BY wafer, band
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_detector_cuts_summary(self) -> pd.DataFrame:
        """Get summary of detector cuts from detector_counts table."""
        try:
            # Build query with schema-aware column names
            dark_dets = self._get_column_name('dark_dets')
            fp_flags = self._get_column_name('fp_flags')
            detcal_nan = self._get_column_name('detcal_nan_cuts')
            det_bias = self._get_column_name('det_bias_flags')
            trends = self._get_column_name('trends')
            jumps = self._get_column_name('jumps')
            glitchfill1 = self._get_column_name('glitchfill')
            jumps2 = self._get_column_name('jumps2')
            detrend1 = self._get_column_name('detrend')
            glitches1 = self._get_column_name('glitches')
            glitchfill2 = self._get_column_name('glitchfill2')
            glitches2 = self._get_column_name('glitches2')
            detrend2 = self._get_column_name('detrend2')
            ptp_flags = self._get_column_name('ptp_flags')
            tod_stats = self._get_column_name('tod_stats')
            noisy_subscan = self._get_column_name('noisy_subscan_flags')
            estimate_t2p = self._get_column_name('estimate_t2p')
            subtract_t2p = self._get_column_name('subtract_t2p')
            scan_freq_cut = self._get_column_name('scan_freq_cut')
            inv_var_flags = self._get_column_name('inv_var_flags')
            
            query = f"""
            SELECT wafer, band,
                   AVG(`0_starting`) as avg_total_detectors,
                   AVG(`0_starting` - `{dark_dets}`) as avg_darks_cut,
                   AVG(`{dark_dets}` - `{fp_flags}`) as avg_fp_flags_cut,
                   AVG(`{fp_flags}` - `{detcal_nan}`) as avg_detcal_nan_cut,
                   AVG(`{detcal_nan}` - `{det_bias}`) as avg_det_bias_cut,
                   AVG(`{det_bias}` - `{trends}`) as avg_trends_cut,
                   AVG(`{trends}` - `{jumps}`) as avg_jumps_slow_cut,
                   AVG(`{glitchfill1}` - `{jumps2}`) as avg_jumps_2pi_cut,
                   AVG(`{detrend1}` - `{glitches1}`) as avg_glitches_pre_cut,
                   AVG(`{glitchfill2}` - `{glitches2}`) as avg_glitches_post_cut,
                   AVG(`{detrend2}` - `{ptp_flags}`) as avg_ptp_cut,
                   AVG(`{tod_stats}` - `{noisy_subscan}`) as avg_noisy_subscan_cut,
                   AVG(`{estimate_t2p}` - `{subtract_t2p}`) as avg_t2p_cut,
                   AVG(`{scan_freq_cut}` - `{inv_var_flags}`) as avg_inv_var_cut,
                   COUNT(*) as n_obs
            FROM detector_counts
            GROUP BY wafer, band
            ORDER BY wafer, band
            """
            return pd.read_sql_query(query, self.conn)
        except Exception as e:
            print(f"Error in detector cuts summary: {e}")
            return pd.DataFrame()
    
    def get_noise_stats_summary(self) -> pd.DataFrame:
        """Get summary of noise statistics."""
        query = """
        SELECT wafer, band, polarization as noise_type,
               AVG(white_noise_avg) as avg_white_noise,
               AVG(white_noise_q50) as median_white_noise,
               AVG(fknee_avg) as avg_fknee,
               AVG(fknee_q50) as median_fknee,
               AVG(alpha_avg) as avg_alpha,
               AVG(alpha_q50) as median_alpha,
               COUNT(*) as n_obs
        FROM noise_stats
        GROUP BY wafer, band, polarization
        ORDER BY wafer, band, polarization
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_failed_obs_summary(self) -> pd.DataFrame:
        """Get summary of failed observations."""
        try:
            query = """
            SELECT wafer, band,
                   COUNT(*) as n_failed_obs
            FROM failed_observations
            GROUP BY wafer, band
            ORDER BY wafer, band
            """
            return pd.read_sql_query(query, self.conn)
        except Exception:
            return pd.DataFrame()
    
    def plot_sample_cuts_by_wafer_band(self, output_dir: str = "."):
        """Plot sample cuts breakdown by wafer and band."""
        df = pd.read_sql_query("SELECT * FROM sample_cuts", self.conn)
        
        if df.empty:
            print("No sample cuts data found.")
            return
        
        # Calculate cut fractions
        cut_columns = ['smurfgaps_cuts', 'turnarounds_cuts', 'jumps_slow_cuts', 
                      'jumps_2pi_cuts', 'glitches_pre_hwpss_cuts', 
                      'glitches_post_hwpss_cuts', 'source_moon_cuts']
        
        for col in cut_columns:
            df[f'{col}_frac'] = df[col] / df['total_samples']
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sample Cuts by Cut Type, Wafer, and Band', fontsize=16)
        
        cut_types = [
            (['smurfgaps_cuts_frac', 'turnarounds_cuts_frac'], 'Instrumental Cuts'),
            (['jumps_slow_cuts_frac', 'jumps_2pi_cuts_frac'], 'Jump Cuts'),
            (['glitches_pre_hwpss_cuts_frac', 'glitches_post_hwpss_cuts_frac'], 'Glitch Cuts'),
            (['source_moon_cuts_frac'], 'Source Cuts')
        ]
        
        for i, (cols, title) in enumerate(cut_types):
            ax = axes[i//2, i%2]
            
            # Melt data for seaborn
            plot_data = []
            for col in cols:
                temp_df = df[['wafer', 'band', col]].copy()
                temp_df['cut_type'] = col.replace('_frac', '').replace('_cuts', '')
                temp_df['cut_fraction'] = temp_df[col]
                plot_data.append(temp_df[['wafer', 'band', 'cut_type', 'cut_fraction']])
            
            if plot_data:
                plot_df = pd.concat(plot_data, ignore_index=True)
                sns.boxplot(data=plot_df, x='wafer', y='cut_fraction', hue='band', ax=ax)
                ax.set_title(title)
                ax.set_ylabel('Cut Fraction')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_cuts_by_wafer_band.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_detector_cuts_by_wafer_band(self, output_dir: str = "."):
        """Plot detector cuts breakdown by wafer and band."""
        # Get detector counts and calculate cuts
        df = pd.read_sql_query("SELECT * FROM detector_counts", self.conn)
        
        if df.empty:
            print("No detector counts data found.")
            return
        
        # Calculate number of detectors cut at each step
        df['darks_cut'] = df['0_starting'] - df['6_dark_dets']
        df['fp_flags_cut'] = df['6_dark_dets'] - df['7_fp_flags'] 
        df['detcal_nan_cut'] = df['7_fp_flags'] - df['8_detcal_nan_cuts']
        df['det_bias_cut'] = df['8_detcal_nan_cuts'] - df['10_det_bias_flags']
        df['trends_cut'] = df['10_det_bias_flags'] - df['11_trends']
        df['jumps_slow_cut'] = df['11_trends'] - df['12_jumps']
        df['jumps_2pi_cut'] = df['14_glitchfill'] - df['15_jumps']
        df['glitches_pre_cut'] = df['18_detrend'] - df['19_glitches']
        df['glitches_post_cut'] = df['26_glitchfill'] - df['27_glitches']
        df['ptp_cut'] = df['32_detrend'] - df['33_ptp_flags']
        df['noisy_subscan_cut'] = df['45_tod_stats'] - df['46_noisy_subscan_flags']
        df['t2p_cut'] = df['56_estimate_t2p'] - df['57_subtract_t2p']
        df['inv_var_cut'] = df['79_scan_freq_cut'] - df['80_inv_var_flags']
        
        # Calculate cut fractions
        cut_columns = ['darks_cut', 'fp_flags_cut', 'detcal_nan_cut', 'det_bias_cut', 
                      'trends_cut', 'jumps_slow_cut', 'jumps_2pi_cut', 'glitches_pre_cut',
                      'glitches_post_cut', 'ptp_cut', 'noisy_subscan_cut', 't2p_cut', 'inv_var_cut']
        
        for col in cut_columns:
            df[f'{col}_frac'] = df[col] / df['0_starting']
        
        # Group cuts by type
        cut_groups = {
            'Calibration': ['darks_cut_frac', 'fp_flags_cut_frac', 'detcal_nan_cut_frac', 'det_bias_cut_frac'],
            'Data Quality': ['trends_cut_frac', 'jumps_slow_cut_frac', 'jumps_2pi_cut_frac'],
            'Glitch/Noise': ['glitches_pre_cut_frac', 'glitches_post_cut_frac', 
                            'ptp_cut_frac', 'noisy_subscan_cut_frac'],
            'Analysis': ['t2p_cut_frac', 'inv_var_cut_frac']
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detector Cuts by Cut Type, Wafer, and Band', fontsize=16)
        
        for i, (group_name, cols) in enumerate(cut_groups.items()):
            ax = axes[i//2, i%2]
            
            # Melt data for seaborn
            plot_data = []
            for col in cols:
                if col in df.columns:
                    temp_df = df[['wafer', 'band', col]].copy()
                    temp_df['cut_type'] = col.replace('_cut_frac', '').replace('_frac', '')
                    temp_df['cut_fraction'] = temp_df[col]
                    plot_data.append(temp_df[['wafer', 'band', 'cut_type', 'cut_fraction']])
            
            if plot_data:
                plot_df = pd.concat(plot_data, ignore_index=True)
                sns.boxplot(data=plot_df, x='wafer', y='cut_fraction', hue='band', ax=ax)
                ax.set_title(group_name)
                ax.set_ylabel('Cut Fraction')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detector_cuts_by_wafer_band.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_detector_survival(self, output_dir: str = "."):
        """Plot detector survival through processing steps."""
        df = pd.read_sql_query("SELECT * FROM detector_counts", self.conn)
        
        if df.empty:
            print("No detector counts data found.")
            return
        
        # Get processing steps (columns that start with a number and underscore)
        step_cols = [col for col in df.columns if col[0].isdigit() and '_' in col]
        step_cols = sorted(step_cols, key=lambda x: int(x.split('_')[0]))
        
        # Create survival plot for each wafer/band combination
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Detector Survival Through Processing Steps', fontsize=16)
        
        wafer_band_combos = df.groupby(['wafer', 'band']).size().reset_index()[['wafer', 'band']]
        
        for idx, (_, row) in enumerate(wafer_band_combos.iterrows()):
            if idx >= 6:  # Only plot first 6 combinations
                break
                
            ax = axes[idx//3, idx%3]
            subset = df[(df['wafer'] == row['wafer']) & (df['band'] == row['band'])]
            
            # Calculate mean and std for each step
            steps = []
            means = []
            stds = []
            
            for step_col in step_cols:
                step_num = int(step_col.split('_')[0])
                steps.append(step_num)
                means.append(subset[step_col].mean())
                stds.append(subset[step_col].std())
            
            ax.errorbar(steps, means, yerr=stds, marker='o', capsize=3)
            ax.set_title(f'{row["wafer"]} {row["band"]}')
            ax.set_xlabel('Processing Step')
            ax.set_ylabel('Detector Count')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for idx in range(len(wafer_band_combos), 6):
            fig.delaxes(axes[idx//3, idx%3])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detector_survival.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_noise_stats(self, output_dir: str = "."):
        """Plot noise statistics distributions."""
        df = pd.read_sql_query("SELECT * FROM noise_stats", self.conn)
        
        if df.empty:
            print("No noise statistics data found.")
            return
        
        # Create plots for each polarization
        polarizations = df['polarization'].unique()
        
        for polarization in polarizations:
            noise_df = df[df['polarization'] == polarization]
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'{polarization} Polarization Noise Statistics', fontsize=16)
            
            # White noise
            sns.boxplot(data=noise_df, x='wafer', y='white_noise_q50', hue='band', ax=axes[0])
            axes[0].set_title('White Noise (Median)')
            axes[0].set_ylabel('White Noise')
            axes[0].set_yscale('log')
            
            # Fknee (if available)
            if not noise_df['fknee_q50'].isna().all():
                sns.boxplot(data=noise_df, x='wafer', y='fknee_q50', hue='band', ax=axes[1])
                axes[1].set_title('Fknee (Median)')
                axes[1].set_ylabel('Fknee [Hz]')
                axes[1].set_yscale('log')
            else:
                axes[1].text(0.5, 0.5, 'No fknee data', transform=axes[1].transAxes, 
                           ha='center', va='center', fontsize=14)
                axes[1].set_title('Fknee (Not Available)')
            
            # Alpha (if available) 
            if not noise_df['alpha_q50'].isna().all():
                sns.boxplot(data=noise_df, x='wafer', y='alpha_q50', hue='band', ax=axes[2])
                axes[2].set_title('Alpha (Median)')
                axes[2].set_ylabel('Alpha')
            else:
                axes[2].text(0.5, 0.5, 'No alpha data', transform=axes[2].transAxes,
                           ha='center', va='center', fontsize=14)
                axes[2].set_title('Alpha (Not Available)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'noise_stats_{polarization.lower()}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_detector_cut_analysis(self, output_file: str = "detector_cut_analysis.txt"):
        """Generate detailed detector cut analysis with fractions."""
        with open(output_file, 'w') as f:
            f.write("=== DETAILED DETECTOR CUT ANALYSIS ===\n\n")
            
            # Get raw detector counts data  
            df = pd.read_sql_query("SELECT * FROM detector_counts", self.conn)
            if df.empty:
                f.write("No detector counts data available.\n")
                return
                
            # Calculate cuts for each step
            cuts_data = []
            
            for _, row in df.iterrows():
                row_data = {'obs_id': row['obs_id'], 'wafer': row['wafer'], 'band': row['band']}
                
                # Get all step columns in order
                step_cols = [col for col in df.columns if col[0].isdigit() and '_' in col]
                step_cols = sorted(step_cols, key=lambda x: int(x.split('_')[0]))
                
                prev_count = None
                total_starting = row[step_cols[0]]  # First column is starting count
                row_data['total_starting'] = total_starting
                
                for i, col in enumerate(step_cols):
                    current_count = row[col]
                    if prev_count is not None:
                        cuts_this_step = prev_count - current_count
                        step_name = col.split('_', 1)[1]  # Remove number prefix
                        row_data[f'cuts_{step_name}'] = cuts_this_step
                    prev_count = current_count
                    
                cuts_data.append(row_data)
            
            cuts_df = pd.DataFrame(cuts_data)
            
            # Get cut column names
            cut_cols = [col for col in cuts_df.columns if col.startswith('cuts_')]
            
            # Generate the 4 types of summaries
            self._write_detector_summary(f, cuts_df, cut_cols, "ALL DATA", None, None)
            
            # By band only
            for band in cuts_df['band'].unique():
                band_df = cuts_df[cuts_df['band'] == band]
                self._write_detector_summary(f, band_df, cut_cols, f"BAND: {band}", None, band)
            
            # By wafer only  
            for wafer in cuts_df['wafer'].unique():
                wafer_df = cuts_df[cuts_df['wafer'] == wafer]
                self._write_detector_summary(f, wafer_df, cut_cols, f"WAFER: {wafer}", wafer, None)
                
            # By wafer and band
            for wafer in cuts_df['wafer'].unique():
                for band in cuts_df['band'].unique():
                    wb_df = cuts_df[(cuts_df['wafer'] == wafer) & (cuts_df['band'] == band)]
                    if not wb_df.empty:
                        self._write_detector_summary(f, wb_df, cut_cols, f"WAFER: {wafer}, BAND: {band}", wafer, band)
                        
        print(f"Detector cut analysis written to: {output_file}")
    
    def _write_detector_summary(self, f, df, cut_cols, title, wafer, band):
        """Write detector cut summary for a subset of data."""
        f.write(f"\n--- {title} ---\n")
        f.write(f"Number of observations: {len(df)}\n")
        
        if df.empty:
            f.write("No data available.\n")
            return
            
        total_starting = df['total_starting'].sum()
        f.write(f"Total starting detectors: {total_starting}\n")
        
        # Calculate total cuts and fractions
        cut_totals = {}
        for col in cut_cols:
            cut_totals[col] = df[col].sum()
            
        total_cuts = sum(cut_totals.values())
        f.write(f"Total detector cuts: {total_cuts}\n")
        f.write(f"Overall cut fraction: {total_cuts/total_starting:.4f} ({100*total_cuts/total_starting:.2f}%)\n\n")
        
        f.write("FRACTION OF TOTAL DETECTORS:\n")
        f.write("Step                     | Cuts      | Fraction  | Percentage\n")
        f.write("-" * 60 + "\n")
        for col in cut_cols:
            step_name = col.replace('cuts_', '')
            cuts = cut_totals[col]
            fraction = cuts / total_starting if total_starting > 0 else 0
            f.write(f"{step_name:<24} | {cuts:>8} | {fraction:>8.4f} | {100*fraction:>8.2f}%\n")
        
        f.write("\nFRACTION OF CUT DETECTORS:\n") 
        f.write("Step                     | Cuts      | Fraction  | Percentage\n")
        f.write("-" * 60 + "\n")
        for col in cut_cols:
            step_name = col.replace('cuts_', '')
            cuts = cut_totals[col]
            fraction = cuts / total_cuts if total_cuts > 0 else 0
            f.write(f"{step_name:<24} | {cuts:>8} | {fraction:>8.4f} | {100*fraction:>8.2f}%\n")
        f.write("\n")
    
    def generate_sample_cut_analysis(self, output_file: str = "sample_cut_analysis.txt"):
        """Generate detailed sample cut analysis with fractions.""" 
        with open(output_file, 'w') as f:
            f.write("=== DETAILED SAMPLE CUT ANALYSIS ===\n\n")
            
            # Get sample cuts data
            df = pd.read_sql_query("SELECT * FROM sample_cuts", self.conn)
            if df.empty:
                f.write("No sample cuts data available.\n")
                return
                
            cut_cols = ['smurfgaps_cuts', 'turnarounds_cuts', 'jumps_slow_cuts', 'jumps_2pi_cuts',
                       'glitches_pre_hwpss_cuts', 'glitches_post_hwpss_cuts', 'source_moon_cuts']
            
            # Generate the 4 types of summaries
            self._write_sample_summary(f, df, cut_cols, "ALL DATA", None, None)
            
            # By band only
            for band in df['band'].unique():
                band_df = df[df['band'] == band]
                self._write_sample_summary(f, band_df, cut_cols, f"BAND: {band}", None, band)
            
            # By wafer only
            for wafer in df['wafer'].unique():
                wafer_df = df[df['wafer'] == wafer]  
                self._write_sample_summary(f, wafer_df, cut_cols, f"WAFER: {wafer}", wafer, None)
                
            # By wafer and band
            for wafer in df['wafer'].unique():
                for band in df['band'].unique():
                    wb_df = df[(df['wafer'] == wafer) & (df['band'] == band)]
                    if not wb_df.empty:
                        self._write_sample_summary(f, wb_df, cut_cols, f"WAFER: {wafer}, BAND: {band}", wafer, band)
                        
        print(f"Sample cut analysis written to: {output_file}")
    
    def _write_sample_summary(self, f, df, cut_cols, title, wafer, band):
        """Write sample cut summary for a subset of data."""
        f.write(f"\n--- {title} ---\n")
        f.write(f"Number of observations: {len(df)}\n")
        
        if df.empty:
            f.write("No data available.\n")
            return
            
        total_samples = df['total_samples'].sum()
        f.write(f"Total samples: {total_samples}\n")
        
        # Calculate total cuts and fractions
        cut_totals = {}
        for col in cut_cols:
            cut_totals[col] = df[col].sum()
            
        total_cuts = sum(cut_totals.values())
        f.write(f"Total sample cuts: {total_cuts}\n")
        f.write(f"Overall cut fraction: {total_cuts/total_samples:.4f} ({100*total_cuts/total_samples:.2f}%)\n\n")
        
        f.write("FRACTION OF TOTAL SAMPLES:\n")
        f.write("Step                     | Cuts        | Fraction  | Percentage\n")
        f.write("-" * 65 + "\n")
        for col in cut_cols:
            step_name = col.replace('_cuts', '')
            cuts = cut_totals[col]
            fraction = cuts / total_samples if total_samples > 0 else 0
            f.write(f"{step_name:<24} | {cuts:>10} | {fraction:>8.4f} | {100*fraction:>8.2f}%\n")
        
        f.write("\nFRACTION OF CUT SAMPLES:\n")
        f.write("Step                     | Cuts        | Fraction  | Percentage\n") 
        f.write("-" * 65 + "\n")
        for col in cut_cols:
            step_name = col.replace('_cuts', '')
            cuts = cut_totals[col]
            fraction = cuts / total_cuts if total_cuts > 0 else 0
            f.write(f"{step_name:<24} | {cuts:>10} | {fraction:>8.4f} | {100*fraction:>8.2f}%\n")
        f.write("\n")
        """Generate a text summary report."""
        with open(output_file, 'w') as f:
            f.write("=== CUTS AND STATISTICS SUMMARY REPORT (V4 Schema) ===\n\n")
            
            # Sample cuts summary
            sample_summary = self.get_sample_cuts_summary()
            if not sample_summary.empty:
                f.write("SAMPLE CUTS SUMMARY (by wafer/band):\n")
                f.write(sample_summary.to_string(index=False))
                f.write("\n\n")
            
            # Detector cuts summary
            detector_summary = self.get_detector_cuts_summary()
            if not detector_summary.empty:
                f.write("DETECTOR CUTS SUMMARY (by wafer/band):\n")
                f.write(detector_summary.to_string(index=False))
                f.write("\n\n")
            
            # Noise statistics summary
            noise_summary = self.get_noise_stats_summary()
            if not noise_summary.empty:
                f.write("NOISE STATISTICS SUMMARY (by wafer/band/polarization):\n")
                f.write(noise_summary.to_string(index=False))
                f.write("\n\n")
            
            # Failed observations
            failed_summary = self.get_failed_obs_summary()
            if not failed_summary.empty:
                f.write("FAILED OBSERVATIONS SUMMARY (by wafer/band):\n")
                f.write(failed_summary.to_string(index=False))
                f.write("\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS:\n")
            try:
                total_obs = pd.read_sql_query("SELECT COUNT(DISTINCT obs_id) as total FROM sample_cuts", self.conn)
                f.write(f"Total observations processed: {total_obs['total'].iloc[0] if not total_obs.empty else 'N/A'}\n")
            except:
                f.write("Total observations processed: N/A\n")
            
            try:
                total_failed = pd.read_sql_query("SELECT COUNT(DISTINCT obs_id) as total FROM failed_observations", self.conn)
                total_failed_count = total_failed['total'].iloc[0] if not total_failed.empty else 0
                f.write(f"Total observations with failures: {total_failed_count}\n")
            except:
                f.write("Total observations with failures: N/A\n")
                
            # Detector yield statistics
            try:
                yield_query = """
                SELECT wafer, band, 
                       AVG(`0_starting`) as avg_starting,
                       AVG(`85_noise`) as avg_final,
                       AVG(CAST(`85_noise` AS FLOAT) / `0_starting`) as avg_yield
                FROM detector_counts
                GROUP BY wafer, band
                ORDER BY wafer, band
                """
                yield_stats = pd.read_sql_query(yield_query, self.conn)
                if not yield_stats.empty:
                    f.write("\nDETECTOR YIELD STATISTICS (by wafer/band):\n")
                    f.write(yield_stats.to_string(index=False))
                    f.write("\n")
            except Exception as e:
                f.write(f"\nDetector yield statistics: Error - {e}\n")
        
        print(f"Summary report written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze cuts and statistics database (v4 schema)")
    parser.add_argument('db_path', help="Path to cuts and statistics database")
    parser.add_argument('--config', help="Path to processing configuration file (YAML or JSON)")
    parser.add_argument('--output-dir', default='.', help="Output directory for plots")
    parser.add_argument('--report', help="Generate summary report to file")
    parser.add_argument('--no-plots', action='store_true', help="Skip generating plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db_path):
        print(f"Error: Database file not found: {args.db_path}")
        return
    
    if args.config and not os.path.exists(args.config):
        print(f"Warning: Config file not found: {args.config}, using schema detection")
        args.config = None
    
    analyzer = CutsStatsAnalyzerV4(args.db_path, args.config)
    
    # Generate plots unless disabled
    if not args.no_plots:
        print("Generating plots...")
        os.makedirs(args.output_dir, exist_ok=True)
        
        try:
            analyzer.plot_sample_cuts_by_wafer_band(args.output_dir)
        except Exception as e:
            print(f"Error generating sample cuts plot: {e}")
        
        try:
            analyzer.plot_detector_cuts_by_wafer_band(args.output_dir)
        except Exception as e:
            print(f"Error generating detector cuts plot: {e}")
            
        try:
            analyzer.plot_detector_survival(args.output_dir)
        except Exception as e:
            print(f"Error generating detector survival plot: {e}")
        
        try:
            analyzer.plot_noise_stats(args.output_dir)
        except Exception as e:
            print(f"Error generating noise stats plot: {e}")
    
    # Generate summary report
    report_file = args.report if args.report else os.path.join(args.output_dir, "cuts_stats_summary.txt")
    try:
        analyzer.generate_summary_report(report_file)
    except Exception as e:
        print(f"Error generating summary report: {e}")
    
    # Generate detailed cut analyses
    try:
        detector_analysis_file = os.path.join(args.output_dir, "detector_cut_analysis.txt")
        analyzer.generate_detector_cut_analysis(detector_analysis_file)
    except Exception as e:
        print(f"Error generating detector cut analysis: {e}")
        
    try:
        sample_analysis_file = os.path.join(args.output_dir, "sample_cut_analysis.txt") 
        analyzer.generate_sample_cut_analysis(sample_analysis_file)
    except Exception as e:
        print(f"Error generating sample cut analysis: {e}")


if __name__ == '__main__':
    main()