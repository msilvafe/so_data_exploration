import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Optional, List
import os


class CutsStatsAnalyzer:
    """Analyzer for cuts and statistics database."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
    
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
        """Get summary of detector cuts across all observations."""
        query = """
        SELECT wafer, band,
               AVG(total_detectors) as avg_total_detectors,
               AVG(darks_cut) as avg_darks_cut,
               AVG(fp_nans_cut) as avg_fp_nans_cut,
               AVG(det_bias_flags_cut) as avg_det_bias_cut,
               AVG(trends_cut) as avg_trends_cut,
               AVG(jumps_slow_cut) as avg_jumps_slow_cut,
               AVG(jumps_2pi_cut) as avg_jumps_2pi_cut,
               AVG(glitches_pre_hwpss_cut) as avg_glitches_pre_cut,
               AVG(glitches_post_hwpss_cut) as avg_glitches_post_cut,
               AVG(ptp_flags_cut) as avg_ptp_cut,
               AVG(noisy_subscan_flags_cut) as avg_noisy_subscan_cut,
               AVG(t2p_cut) as avg_t2p_cut,
               AVG(inv_var_flags_cut) as avg_inv_var_cut,
               COUNT(*) as n_obs
        FROM detector_cuts
        GROUP BY wafer, band
        ORDER BY wafer, band
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_noise_stats_summary(self) -> pd.DataFrame:
        """Get summary of noise statistics."""
        query = """
        SELECT wafer, band, noise_type,
               AVG(white_noise_avg) as avg_white_noise,
               AVG(white_noise_q50) as median_white_noise,
               AVG(fknee_avg) as avg_fknee,
               AVG(fknee_q50) as median_fknee,
               AVG(alpha_avg) as avg_alpha,
               AVG(alpha_q50) as median_alpha,
               COUNT(*) as n_obs
        FROM noise_stats
        GROUP BY wafer, band, noise_type
        ORDER BY wafer, band, noise_type
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
        df = pd.read_sql_query("SELECT * FROM detector_cuts", self.conn)
        
        if df.empty:
            print("No detector cuts data found.")
            return
        
        # Calculate cut fractions
        cut_columns = ['darks_cut', 'fp_nans_cut', 'det_bias_flags_cut', 'trends_cut',
                      'jumps_slow_cut', 'jumps_2pi_cut', 'glitches_pre_hwpss_cut',
                      'glitches_post_hwpss_cut', 'ptp_flags_cut', 'noisy_subscan_flags_cut',
                      't2p_cut', 'inv_var_flags_cut']
        
        for col in cut_columns:
            df[f'{col}_frac'] = df[col] / df['total_detectors']
        
        # Group cuts by type
        cut_groups = {
            'Calibration': ['darks_cut_frac', 'fp_nans_cut_frac', 'det_bias_flags_cut_frac'],
            'Data Quality': ['trends_cut_frac', 'jumps_slow_cut_frac', 'jumps_2pi_cut_frac'],
            'Glitch/Noise': ['glitches_pre_hwpss_cut_frac', 'glitches_post_hwpss_cut_frac', 
                            'ptp_flags_cut_frac', 'noisy_subscan_flags_cut_frac'],
            'Analysis': ['t2p_cut_frac', 'inv_var_flags_cut_frac']
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detector Cuts by Cut Type, Wafer, and Band', fontsize=16)
        
        for i, (group_name, cols) in enumerate(cut_groups.items()):
            ax = axes[i//2, i%2]
            
            # Melt data for seaborn
            plot_data = []
            for col in cols:
                if col.replace('_frac', '') in df.columns:
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
    
    def plot_noise_stats(self, output_dir: str = "."):
        """Plot noise statistics distributions."""
        df = pd.read_sql_query("SELECT * FROM noise_stats", self.conn)
        
        if df.empty:
            print("No noise statistics data found.")
            return
        
        # Create plots for each noise type
        noise_types = df['noise_type'].unique()
        
        for noise_type in noise_types:
            noise_df = df[df['noise_type'] == noise_type]
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'{noise_type} Polarization Noise Statistics', fontsize=16)
            
            # White noise
            sns.boxplot(data=noise_df, x='wafer', y='white_noise_q50', hue='band', ax=axes[0])
            axes[0].set_title('White Noise (Median)')
            axes[0].set_ylabel('White Noise')
            
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
            plt.savefig(os.path.join(output_dir, f'noise_stats_{noise_type.lower()}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_summary_report(self, output_file: str = "cuts_stats_summary.txt"):
        """Generate a text summary report."""
        with open(output_file, 'w') as f:
            f.write("=== CUTS AND STATISTICS SUMMARY REPORT ===\n\n")
            
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
                f.write("NOISE STATISTICS SUMMARY (by wafer/band/type):\n")
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
            total_obs = pd.read_sql_query("SELECT COUNT(DISTINCT obsid) as total FROM sample_cuts", self.conn)
            f.write(f"Total observations processed: {total_obs['total'].iloc[0] if not total_obs.empty else 'N/A'}\n")
            
            total_failed = pd.read_sql_query("SELECT COUNT(DISTINCT obsid) as total FROM failed_observations", self.conn)
            total_failed_count = total_failed['total'].iloc[0] if not total_failed.empty else 0
            f.write(f"Total observations with failures: {total_failed_count}\n")
        
        print(f"Summary report written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze cuts and statistics database")
    parser.add_argument('db_path', help="Path to cuts and statistics database")
    parser.add_argument('--output-dir', default='.', help="Output directory for plots")
    parser.add_argument('--report', help="Generate summary report to file")
    parser.add_argument('--no-plots', action='store_true', help="Skip generating plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db_path):
        print(f"Error: Database file not found: {args.db_path}")
        return
    
    analyzer = CutsStatsAnalyzer(args.db_path)
    
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
            analyzer.plot_noise_stats(args.output_dir)
        except Exception as e:
            print(f"Error generating noise stats plot: {e}")
    
    # Generate summary report
    report_file = args.report if args.report else os.path.join(args.output_dir, "cuts_stats_summary.txt")
    try:
        analyzer.generate_summary_report(report_file)
    except Exception as e:
        print(f"Error generating summary report: {e}")


if __name__ == '__main__':
    main()