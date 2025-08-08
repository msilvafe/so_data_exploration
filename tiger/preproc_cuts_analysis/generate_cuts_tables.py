#!/usr/bin/env python3
"""
Standalone Table Generation for Preprocessing Cuts Analysis

This script generates summary tables from the cuts analysis database,
similar to the tables shown in the attached screenshots.
"""

import sqlite3
import pandas as pd
import argparse
from typing import Dict, List, Tuple


def generate_detector_cuts_table(sqlite_path: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """
    Generate detector cuts summary table.
    
    Args:
        sqlite_path: Path to SQLite database
        start_ts: Start timestamp
        end_ts: End timestamp
        
    Returns:
        DataFrame with detector cut statistics
    """
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    
    # Get column information
    cur.execute('PRAGMA table_info(results)')
    columns = [row[1] for row in cur.fetchall()]
    
    # Find detector cut columns (ending with 'cuts')
    cut_keys = [k for k in columns if k.endswith('cuts')]
    
    # Build query
    query = f"""
    SELECT {', '.join(cut_keys)}, ndets FROM results
    WHERE CAST(substr(obsid, 5, 10) AS INTEGER) BETWEEN ? AND ?
    """
    cur.execute(query, (start_ts, end_ts))
    
    # Transpose and sum
    columns_data = list(zip(*cur))
    cut_sums = [sum(x for x in col) for col in columns_data]
    *cut_sums, total_dets = cut_sums
    
    # Calculate fractions
    sum_of_all_cuts = sum(cut_sums)
    total_fractions = [cut_sum / total_dets if total_dets else 0 for cut_sum in cut_sums]
    cut_fractions = [cut_sum / sum_of_all_cuts if sum_of_all_cuts else 0 for cut_sum in cut_sums]
    
    # Create readable cut type names
    cut_names = []
    for key in cut_keys:
        name = key.replace('_cuts', '').replace('_', ' ').title()
        # Special cases for better formatting
        name_mapping = {
            'Fp': 'Focalplane',
            'Det Bias': 'Biased',
            'Ptp': 'Peak-Peak',
            'White Noise': 'Noise',
            'Inv Var': 'Inv Var',
            'Noisy Subscans': 'Noisy Subscans'
        }
        cut_names.append(name_mapping.get(name, name))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Cut Type': cut_names,
        'Fraction of Cut Detectors': [f"{frac*100:.2f}" for frac in cut_fractions],
        'Fraction of Total Data': [f"{frac*100:.2f}" for frac in total_fractions]
    })
    
    conn.close()
    return df


def generate_sample_cuts_table(sqlite_path: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """
    Generate sample cuts summary table.
    
    Args:
        sqlite_path: Path to SQLite database  
        start_ts: Start timestamp
        end_ts: End timestamp
        
    Returns:
        DataFrame with sample cut statistics
    """
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    
    # Get column information
    cur.execute('PRAGMA table_info(results)')
    columns = [row[1] for row in cur.fetchall()]
    
    # Find sample cut columns (ending with 'nsamps' but not 'nsamps' itself)
    cut_keys = [k for k in columns if k.endswith('nsamps') and k != 'nsamps']
    
    # Build query
    query = f"""
    SELECT {', '.join(cut_keys)}, nsamps, end_yield FROM results
    WHERE CAST(substr(obsid, 5, 10) AS INTEGER) BETWEEN ? AND ?
    """
    cur.execute(query, (start_ts, end_ts))
    
    # Fetch all rows and process
    rows = list(cur)
    cut_cols = list(zip(*rows))
    cut_sums = [sum(int(x) for x in col) for col in cut_cols[:-2]]
    nsamps_col = cut_cols[-2]  
    end_yield_col = cut_cols[-1]
    
    # Total data volume (nsamps * end_yield summed over all observations)
    total_data = sum(int(n) * int(e) for n, e in zip(nsamps_col, end_yield_col))
    sum_of_all_cuts = sum(cut_sums)
    
    # Calculate fractions
    total_fractions = [cut_sum / total_data if total_data else 0 for cut_sum in cut_sums]
    cut_fractions = [cut_sum / sum_of_all_cuts if sum_of_all_cuts else 0 for cut_sum in cut_sums]
    
    # Create readable cut type names
    cut_names = []
    for key in cut_keys:
        name = key.replace('_nsamps', '').replace('_', ' ').title()
        # Special cases for better formatting  
        name_mapping = {
            'Turnaround': 'Turnarounds',
            'Jumps Slow': 'Jumps',
            'Jumps 2Pi': 'Jumps',  # Combine jump types
            'Glitch': 'Glitches',
            'Edge': 'Edges',
            'Noisy Subscans': 'Subscans',
            'Source Flags': 'Sources'
        }
        cut_names.append(name_mapping.get(name, name))
    
    # Combine jump types if both present
    combined_names = []
    combined_cut_fractions = []
    combined_total_fractions = []
    
    i = 0
    while i < len(cut_names):
        if cut_names[i] == 'Jumps':
            # Combine all jump types
            jump_cut_frac = 0
            jump_total_frac = 0
            while i < len(cut_names) and cut_names[i] == 'Jumps':
                jump_cut_frac += cut_fractions[i]
                jump_total_frac += total_fractions[i]
                i += 1
            combined_names.append('Jumps')
            combined_cut_fractions.append(jump_cut_frac)
            combined_total_fractions.append(jump_total_frac)
        else:
            combined_names.append(cut_names[i])
            combined_cut_fractions.append(cut_fractions[i])
            combined_total_fractions.append(total_fractions[i])
            i += 1
    
    # Create DataFrame
    df = pd.DataFrame({
        'Cut Type': combined_names,
        'Fraction of Flagged Data': [f"{frac*100:.2f}" for frac in combined_cut_fractions],
        'Fraction of Total Data': [f"{frac*100:.2f}" for frac in combined_total_fractions]
    })
    
    conn.close()
    return df


def print_formatted_table(df: pd.DataFrame, title: str, table_type: str = 'detector'):
    """
    Print a formatted table similar to the attached screenshots.
    
    Args:
        df: DataFrame to print
        title: Title for the table
        table_type: Either 'detector' or 'sample'
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    if table_type == 'detector':
        # Format for detector cuts table
        print(f"{'Cut Type':<20} {'Fraction of':<15} {'Fraction of':<15}")
        print(f"{'':20} {'Cut Detectors':<15} {'Total Data':<15}")
        print("-" * 50)
        
        for _, row in df.iterrows():
            print(f"{row['Cut Type']:<20} {row['Fraction of Cut Detectors']:<15} {row['Fraction of Total Data']:<15}")
            
    else:  # sample cuts
        # Format for sample cuts table  
        print(f"{'Cut Type':<20} {'Fraction of':<15} {'Fraction of':<15}")
        print(f"{'':20} {'Flagged Data':<15} {'Total Data':<15}")
        print("-" * 50)
        
        for _, row in df.iterrows():
            print(f"{row['Cut Type']:<20} {row['Fraction of Flagged Data']:<15} {row['Fraction of Total Data']:<15}")


def main():
    """Main function for standalone table generation."""
    parser = argparse.ArgumentParser(description='Generate cuts analysis summary tables')
    parser.add_argument('sqlite_path', help='Path to cuts analysis SQLite database')
    parser.add_argument('--start-ts', type=int, required=True, help='Start timestamp')
    parser.add_argument('--end-ts', type=int, required=True, help='End timestamp')
    parser.add_argument('--output-csv', action='store_true', help='Save tables as CSV files')
    parser.add_argument('--output-dir', default='.', help='Directory to save CSV files')
    
    args = parser.parse_args()
    
    # Generate detector cuts table
    print("Generating detector cuts table...")
    det_table = generate_detector_cuts_table(args.sqlite_path, args.start_ts, args.end_ts)
    print_formatted_table(det_table, "DETECTOR CUTS SUMMARY", "detector")
    
    # Generate sample cuts table  
    print("\nGenerating sample cuts table...")
    samp_table = generate_sample_cuts_table(args.sqlite_path, args.start_ts, args.end_ts)
    print_formatted_table(samp_table, "SAMPLE CUTS SUMMARY", "sample")
    
    # Save as CSV if requested
    if args.output_csv:
        import os
        det_output = os.path.join(args.output_dir, 'detector_cuts_summary.csv')
        samp_output = os.path.join(args.output_dir, 'sample_cuts_summary.csv')
        
        det_table.to_csv(det_output, index=False)
        samp_table.to_csv(samp_output, index=False)
        
        print(f"\nTables saved:")
        print(f"Detector cuts: {det_output}")
        print(f"Sample cuts: {samp_output}")


if __name__ == '__main__':
    main()
