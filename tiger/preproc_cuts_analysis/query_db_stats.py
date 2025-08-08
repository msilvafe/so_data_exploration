#!/usr/bin/env python3
"""
Simple command-line tool to query basic statistics from the generalized cuts analysis SQLite database.
Shows total number of observations, wafer slots, bands, and sample data.
"""

import argparse
import sqlite3
import sys
from datetime import datetime

def query_db_stats(db_path):
    """Query basic statistics from the cuts analysis database."""
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if results table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='results'
        """)
        
        if not cursor.fetchone():
            print(f"Error: No 'results' table found in {db_path}")
            return False
        
        # Get table schema to see what columns are available
        cursor.execute("PRAGMA table_info(results)")
        columns = [row[1] for row in cursor.fetchall()]
        print(f"Available columns: {', '.join(columns)}")
        print()
        
        # Get total number of observations
        cursor.execute("SELECT COUNT(DISTINCT obsid) FROM results")
        total_obs = cursor.fetchone()[0]
        
        print(f"Database: {db_path}")
        print(f"Total observations: {total_obs}")
        
        # Get number of unique wafer slots and bands
        cursor.execute("SELECT COUNT(DISTINCT ws) FROM results")
        total_ws = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT band) FROM results")
        total_bands = cursor.fetchone()[0]
        
        print(f"Total wafer slots: {total_ws}")
        print(f"Total bands: {total_bands}")
        
        # Get unique wafer slots and bands
        cursor.execute("SELECT DISTINCT ws FROM results ORDER BY ws")
        wafer_slots = [row[0] for row in cursor.fetchall()]
        cursor.execute("SELECT DISTINCT band FROM results ORDER BY band")
        bands = [row[0] for row in cursor.fetchall()]
        
        print(f"Wafer slots: {', '.join(wafer_slots)}")
        print(f"Bands: {', '.join(bands)}")
        
        # Extract ctime from obsid strings (format: obs_<ctime>_satp1_1111111)
        try:
            cursor.execute("SELECT DISTINCT obsid FROM results")
            obsids = [row[0] for row in cursor.fetchall()]
            
            # Extract ctimes from obsids
            ctimes = []
            for obsid in obsids:
                parts = obsid.split('_')
                if len(parts) >= 2 and parts[0] == 'obs':
                    try:
                        ctime = int(parts[1])
                        ctimes.append(ctime)
                    except ValueError:
                        continue
            
            if ctimes:
                min_ctime = min(ctimes)
                max_ctime = max(ctimes)
                
                # Convert unix timestamps to readable dates
                min_date = datetime.fromtimestamp(min_ctime).strftime('%Y-%m-%d %H:%M:%S')
                max_date = datetime.fromtimestamp(max_ctime).strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"\nTime range (extracted from obsids):")
                print(f"  Earliest: {min_date} (ctime: {min_ctime})")
                print(f"  Latest:   {max_date} (ctime: {max_ctime})")
                
                # Calculate time span
                time_span_days = (max_ctime - min_ctime) / (24 * 3600)
                print(f"  Span:     {time_span_days:.1f} days")
            else:
                print("\nCould not extract ctime from obsid format")
                
        except Exception as e:
            print(f"\nError extracting time from obsids: {e}")
        
        # Check if there's also a separate ctime column (unlikely but possible)
        if 'ctime' in columns:
            cursor.execute("SELECT MIN(ctime), MAX(ctime) FROM results")
            min_ctime_col, max_ctime_col = cursor.fetchone()
            
            if min_ctime_col is not None and max_ctime_col is not None:
                min_date_col = datetime.fromtimestamp(min_ctime_col).strftime('%Y-%m-%d %H:%M:%S')
                max_date_col = datetime.fromtimestamp(max_ctime_col).strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"\nTime range (from ctime column):")
                print(f"  Earliest: {min_date_col} (ctime: {min_ctime_col})")
                print(f"  Latest:   {max_date_col} (ctime: {max_ctime_col})")
        
        # Get a few sample obsids
        cursor.execute("SELECT DISTINCT obsid FROM results ORDER BY obsid LIMIT 5")
        sample_obsids = [row[0] for row in cursor.fetchall()]
        print(f"\nSample obsids: {', '.join(sample_obsids)}")
        
        # Show cuts analysis specific information
        cuts_columns = [col for col in columns if col not in ['obsid', 'ws', 'band']]
        if cuts_columns:
            print(f"\nCuts analysis columns ({len(cuts_columns)} total):")
            for i, col in enumerate(cuts_columns):
                if i < 10:  # Show first 10 cuts columns
                    cursor.execute(f"SELECT AVG(CAST({col} AS REAL)) FROM results WHERE {col} IS NOT NULL")
                    avg_val = cursor.fetchone()[0]
                    cursor.execute(f"SELECT MAX({col}) FROM results WHERE {col} IS NOT NULL")
                    max_val = cursor.fetchone()[0]
                    if avg_val is not None:
                        print(f"  {col}: avg={avg_val:.1f}, max={max_val}")
                    else:
                        print(f"  {col}: no data")
                elif i == 10:
                    print(f"  ... and {len(cuts_columns) - 10} more columns")
                    break
        
        # Show some sample data from the first few rows
        print("\nSample data (first 3 rows):")
        cursor.execute("SELECT * FROM results LIMIT 3")
        rows = cursor.fetchall()
        
        # Print header
        print("  " + " | ".join(f"{col:>12}" for col in columns))
        print("  " + "-" * (len(columns) * 15))
        
        # Print sample rows
        for row in rows:
            print("  " + " | ".join(f"{str(val):>12}" for val in row))
        
        conn.close()
        return True
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Query basic statistics from cuts analysis SQLite database"
    )
    parser.add_argument(
        'database', 
        help='Path to the SQLite database file'
    )
    
    args = parser.parse_args()
    
    # Check if database file exists
    import os
    if not os.path.exists(args.database):
        print(f"Error: Database file not found: {args.database}")
        sys.exit(1)
    
    # Query the database
    success = query_db_stats(args.database)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
