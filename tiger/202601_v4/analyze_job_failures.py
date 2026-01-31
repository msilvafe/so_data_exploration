import sqlite3
import argparse
import pandas as pd
from typing import List, Dict, Optional
import numpy as np


def get_failed_observations(jobs_db_path: str, table_name: str = "jobs") -> pd.DataFrame:
    """Extract failed observations from the jobs database."""
    conn = sqlite3.connect(jobs_db_path)
    
    # Query failed jobs
    query = f"""
    SELECT obs_id, task, status, exit_code, exception, traceback
    FROM {table_name}
    WHERE status != 'finished' OR exit_code != 0
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error querying jobs database: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def summarize_failures(failed_df: pd.DataFrame) -> Dict:
    """Summarize failure patterns from failed observations."""
    if failed_df.empty:
        return {"total_failed": 0}
    
    summary = {
        "total_failed": len(failed_df),
        "unique_obs_failed": failed_df['obs_id'].nunique() if 'obs_id' in failed_df.columns else 0,
        "failure_by_status": failed_df['status'].value_counts().to_dict() if 'status' in failed_df.columns else {},
        "failure_by_exit_code": failed_df['exit_code'].value_counts().to_dict() if 'exit_code' in failed_df.columns else {},
    }
    
    # Analyze common exception patterns
    if 'exception' in failed_df.columns:
        exceptions = failed_df['exception'].dropna()
        if not exceptions.empty:
            # Extract exception types (first word before colon)
            exception_types = exceptions.str.extract(r'^([^:]+)')[0]
            summary["common_exceptions"] = exception_types.value_counts().head(10).to_dict()
    
    return summary


def insert_failed_obs_summary_to_cuts_db(cuts_db_path: str, obs_id: str, failure_info: Dict):
    """Insert failed observation summary into the cuts database."""
    conn = sqlite3.connect(cuts_db_path)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs_failures (
        obsid TEXT PRIMARY KEY,
        total_tasks INTEGER,
        failed_tasks INTEGER,
        common_exception TEXT,
        failure_summary TEXT
    )
    """)
    
    cursor.execute("""
    INSERT OR REPLACE INTO jobs_failures 
    (obsid, total_tasks, failed_tasks, common_exception, failure_summary)
    VALUES (?, ?, ?, ?, ?)
    """, (
        obs_id,
        failure_info.get('total_tasks', 0),
        failure_info.get('failed_tasks', 0),
        failure_info.get('common_exception', ''),
        str(failure_info)
    ))
    
    conn.commit()
    conn.close()


def analyze_jobs_database(jobs_db_path: str, cuts_db_path: Optional[str] = None,
                         table_name: str = "jobs", output_csv: Optional[str] = None):
    """Main function to analyze failed jobs and optionally integrate with cuts database."""
    
    print(f"Analyzing jobs database: {jobs_db_path}")
    
    # Get failed observations
    failed_df = get_failed_observations(jobs_db_path, table_name)
    
    if failed_df.empty:
        print("No failed observations found.")
        return
    
    # Summarize failures
    summary = summarize_failures(failed_df)
    
    print(f"\n=== FAILURE SUMMARY ===")
    print(f"Total failed entries: {summary['total_failed']}")
    print(f"Unique obs_ids failed: {summary['unique_obs_failed']}")
    
    if summary.get('failure_by_status'):
        print(f"\nFailure by status:")
        for status, count in summary['failure_by_status'].items():
            print(f"  {status}: {count}")
    
    if summary.get('failure_by_exit_code'):
        print(f"\nFailure by exit code:")
        for code, count in summary['failure_by_exit_code'].items():
            print(f"  {code}: {count}")
    
    if summary.get('common_exceptions'):
        print(f"\nMost common exceptions:")
        for exc_type, count in summary['common_exceptions'].items():
            print(f"  {exc_type}: {count}")
    
    # Save detailed results
    if output_csv:
        failed_df.to_csv(output_csv, index=False)
        print(f"\nDetailed failure data saved to: {output_csv}")
    
    # Integrate with cuts database if provided
    if cuts_db_path:
        print(f"\nIntegrating failure data into cuts database: {cuts_db_path}")
        
        # Group by obs_id and create summaries
        for obs_id, obs_df in failed_df.groupby('obs_id'):
            failure_info = {
                'total_tasks': len(obs_df),
                'failed_tasks': len(obs_df),
                'common_exception': obs_df['exception'].mode().iloc[0] if not obs_df['exception'].isna().all() else '',
                'status_counts': obs_df['status'].value_counts().to_dict(),
                'exit_codes': obs_df['exit_code'].value_counts().to_dict()
            }
            insert_failed_obs_summary_to_cuts_db(cuts_db_path, obs_id, failure_info)
        
        print(f"Inserted failure summaries for {failed_df['obs_id'].nunique()} observations")


def get_table_info(db_path: str):
    """Get information about tables in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"Tables in {db_path}:")
    for table in tables:
        table_name = table[0]
        print(f"\n  Table: {table_name}")
        
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        for col in columns:
            print(f"    {col[1]} ({col[2]})")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"    Rows: {count}")
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze failed observations from jobs database")
    parser.add_argument('jobs_db', help="Path to jobs database file")
    parser.add_argument('--cuts-db', help="Path to cuts database to integrate failures")
    parser.add_argument('--table-name', default='jobs', help="Table name in jobs database")
    parser.add_argument('--output-csv', help="Save detailed failure data to CSV file")
    parser.add_argument('--info', action='store_true', help="Show database schema information")
    
    args = parser.parse_args()
    
    if args.info:
        get_table_info(args.jobs_db)
        if args.cuts_db:
            print("\n" + "="*50)
            get_table_info(args.cuts_db)
        return
    
    analyze_jobs_database(
        jobs_db_path=args.jobs_db,
        cuts_db_path=args.cuts_db,
        table_name=args.table_name,
        output_csv=args.output_csv
    )


if __name__ == '__main__':
    main()