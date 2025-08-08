#!/usr/bin/env python3
"""
Simple command-line tool to query basic statistics from the cuts analysis SQLite database.
Shows total number of observations and their time range.
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
        
        # Get total number of observations
        cursor.execute("SELECT COUNT(DISTINCT obsid) FROM results")
        total_obs = cursor.fetchone()[0]
        
        # Get min and max ctime
        cursor.execute("SELECT MIN(ctime), MAX(ctime) FROM results")
        min_ctime, max_ctime = cursor.fetchone()
        
        # Print results
        print(f"Database: {db_path}")
        print(f"Total observations: {total_obs}")
        
        if min_ctime is not None and max_ctime is not None:
            # Convert unix timestamps to readable dates
            min_date = datetime.fromtimestamp(min_ctime).strftime('%Y-%m-%d %H:%M:%S')
            max_date = datetime.fromtimestamp(max_ctime).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"Time range:")
            print(f"  Earliest: {min_date} (ctime: {min_ctime})")
            print(f"  Latest:   {max_date} (ctime: {max_ctime})")
            
            # Calculate time span
            time_span_days = (max_ctime - min_ctime) / (24 * 3600)
            print(f"  Span:     {time_span_days:.1f} days")
        else:
            print("No ctime data found")
        
        # Get a few sample obsids
        cursor.execute("SELECT DISTINCT obsid FROM results LIMIT 5")
        sample_obsids = [row[0] for row in cursor.fetchall()]
        print(f"Sample obsids: {', '.join(sample_obsids)}")
        
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
