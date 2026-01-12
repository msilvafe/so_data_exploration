#!/usr/bin/env python3
"""
Utility script for querying and analyzing the detector counts database
created by track_det_counts.py
"""

import sqlite3
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def get_database_info(db_path: str):
    """Get basic information about the database structure and contents."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get table info
    cursor.execute("PRAGMA table_info(detector_counts)")
    columns = cursor.fetchall()
    
    print("Database Structure:")
    print("==================")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    
    # Get summary statistics
    cursor.execute("SELECT COUNT(*) FROM detector_counts")
    total_rows = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT obsid) FROM detector_counts")
    unique_obsids = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT wafer) FROM detector_counts")
    unique_wafers = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT band) FROM detector_counts")
    unique_bands = cursor.fetchone()[0]
    
    print(f"\nDatabase Summary:")
    print(f"================")
    print(f"Total rows: {total_rows}")
    print(f"Unique observations: {unique_obsids}")
    print(f"Unique wafers: {unique_wafers}")
    print(f"Unique bands: {unique_bands}")
    
    conn.close()


def query_obsid(db_path: str, obsid: str):
    """Query all data for a specific observation ID."""
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT * FROM detector_counts 
    WHERE obsid = ?
    ORDER BY wafer, band
    """
    
    df = pd.read_sql_query(query, conn, params=[obsid])
    conn.close()
    
    if df.empty:
        print(f"No data found for observation: {obsid}")
        return None
    
    print(f"Data for observation: {obsid}")
    print("=" * 40)
    
    # Get count columns (all except obsid, wafer, band)
    count_columns = [col for col in df.columns if col not in ['obsid', 'wafer', 'band']]
    
    for _, row in df.iterrows():
        print(f"\nWafer: {row['wafer']}, Band: {row['band']}")
        print("-" * 30)
        
        counts = []
        for col in count_columns:
            if pd.notna(row[col]):
                counts.append(int(row[col]))
        
        if counts:
            print(f"Starting count: {counts[0]}")
            print(f"Final count: {counts[-1]}")
            print(f"Total steps: {len(counts)}")
            print(f"Max reduction: {max(counts) - min(counts)} detectors")
    
    return df


def plot_count_evolution(db_path: str, obsid: str = None, wafer: str = None, 
                        band: str = None, output_path: str = None):
    """Plot the evolution of detector counts through the pipeline."""
    conn = sqlite3.connect(db_path)
    
    # Build query based on parameters
    where_conditions = []
    params = []
    
    if obsid:
        where_conditions.append("obsid = ?")
        params.append(obsid)
    if wafer:
        where_conditions.append("wafer = ?")
        params.append(wafer)
    if band:
        where_conditions.append("band = ?")
        params.append(band)
    
    where_clause = " AND ".join(where_conditions)
    if where_clause:
        where_clause = "WHERE " + where_clause
    
    query = f"SELECT * FROM detector_counts {where_clause}"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df.empty:
        print("No data found for the specified criteria")
        return
    
    # Get count columns
    count_columns = [col for col in df.columns if col not in ['obsid', 'wafer', 'band']]
    
    plt.figure(figsize=(15, 8))
    
    for _, row in df.iterrows():
        label = f"{row['obsid']} {row['wafer']} {row['band']}"
        counts = [row[col] for col in count_columns if pd.notna(row[col])]
        steps = list(range(len(counts)))
        plt.plot(steps, counts, 'o-', label=label, alpha=0.7)
    
    plt.xlabel('Pipeline Step')
    plt.ylabel('Detector Count')
    plt.title('Detector Count Evolution Through Pipeline')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def summary_stats(db_path: str):
    """Calculate summary statistics across all observations."""
    conn = sqlite3.connect(db_path)
    
    df = pd.read_sql_query("SELECT * FROM detector_counts", conn)
    conn.close()
    
    if df.empty:
        print("No data in database")
        return
    
    # Get count columns
    count_columns = [col for col in df.columns if col not in ['obsid', 'wafer', 'band']]
    
    print("Summary Statistics:")
    print("==================")
    
    # Calculate statistics for each step
    for i, col in enumerate(count_columns):
        valid_counts = df[col].dropna()
        if len(valid_counts) > 0:
            print(f"Step {i:2d} ({col}): "
                  f"mean={valid_counts.mean():.1f}, "
                  f"std={valid_counts.std():.1f}, "
                  f"min={valid_counts.min()}, "
                  f"max={valid_counts.max()}")
    
    # Calculate loss statistics
    if len(count_columns) >= 2:
        first_col = count_columns[0]
        last_col = count_columns[-1]
        
        valid_rows = df.dropna(subset=[first_col, last_col])
        if len(valid_rows) > 0:
            total_loss = valid_rows[first_col] - valid_rows[last_col]
            loss_fraction = total_loss / valid_rows[first_col]
            
            print(f"\nDetector Loss Analysis:")
            print(f"======================")
            print(f"Mean total loss: {total_loss.mean():.1f} detectors")
            print(f"Mean loss fraction: {loss_fraction.mean():.3f}")
            print(f"Max total loss: {total_loss.max()} detectors")
            print(f"Max loss fraction: {loss_fraction.max():.3f}")


def export_csv(db_path: str, output_path: str, obsid: str = None):
    """Export data to CSV format."""
    conn = sqlite3.connect(db_path)
    
    if obsid:
        query = "SELECT * FROM detector_counts WHERE obsid = ?"
        df = pd.read_sql_query(query, conn, params=[obsid])
    else:
        query = "SELECT * FROM detector_counts"
        df = pd.read_sql_query(query, conn)
    
    conn.close()
    
    if df.empty:
        print("No data to export")
        return
    
    df.to_csv(output_path, index=False)
    print(f"Data exported to: {output_path}")
    print(f"Exported {len(df)} rows")


def get_parser():
    parser = argparse.ArgumentParser(description="Query and analyze detector counts database")
    parser.add_argument('db_path', help="Path to the SQLite database file")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show database structure and summary')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query data for specific observation')
    query_parser.add_argument('obsid', help='Observation ID to query')
    
    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Plot detector count evolution')
    plot_parser.add_argument('--obsid', help='Specific observation ID')
    plot_parser.add_argument('--wafer', help='Specific wafer (e.g., ws0)')
    plot_parser.add_argument('--band', help='Specific band (e.g., f090)')
    plot_parser.add_argument('--output', help='Output file path for plot')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Calculate summary statistics')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export data to CSV')
    export_parser.add_argument('output', help='Output CSV file path')
    export_parser.add_argument('--obsid', help='Export only specific observation')
    
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    if not Path(args.db_path).exists():
        print(f"Database file not found: {args.db_path}")
        return
    
    if args.command == 'info':
        get_database_info(args.db_path)
    elif args.command == 'query':
        query_obsid(args.db_path, args.obsid)
    elif args.command == 'plot':
        plot_count_evolution(args.db_path, args.obsid, args.wafer, args.band, args.output)
    elif args.command == 'stats':
        summary_stats(args.db_path)
    elif args.command == 'export':
        export_csv(args.db_path, args.output, args.obsid)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()