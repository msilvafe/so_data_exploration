#!/usr/bin/env python3
"""
Batch analysis script for all ISO-V4 cuts_stats folders.
Runs both cuts/stats analysis and generates full PDF reports.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_analysis(db_path, output_dir, version_name):
    """Run cuts_and_stats analysis for a single database."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {version_name}")
    print(f"{'='*80}")
    
    cmd = [
        'python3', 
        'analyze_cuts_and_stats.py',
        '--db-path', str(db_path),
        '--output-dir', str(output_dir)
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

def run_report_generation(db_path, output_dir, version_name):
    """Generate PDF report for a single database."""
    print(f"\n{'='*80}")
    print(f"Generating PDF report: {version_name}")
    print(f"{'='*80}")
    
    cmd = [
        'python3',
        'generate_report_wrapper.py',
        '--db-path', str(db_path),
        '--output-dir', str(output_dir),
        '--version-number', version_name
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

def main():
    cuts_stats_root = Path('/scratch/gpfs/SIMONSOBS/sat-iso/v4/cuts_stats')
    
    if not cuts_stats_root.exists():
        print(f"ERROR: {cuts_stats_root} does not exist")
        return 1
    
    # Find all version folders
    versions = sorted([
        d for d in cuts_stats_root.iterdir() 
        if d.is_dir() and d.name not in ['.', '..']
    ])
    
    if not versions:
        print(f"No version folders found in {cuts_stats_root}")
        return 1
    
    print(f"Found {len(versions)} version folders to process:")
    for v in versions:
        print(f"  - {v.name}")
    
    # Process each version
    results = {}
    for version_dir in versions:
        version_name = version_dir.name
        db_path = version_dir / 'databases' / f'iso_v4_{version_name}.sqlite'
        analysis_dir = version_dir / 'analysis'
        
        # Check if database exists
        if not db_path.exists():
            print(f"\nWARNING: Database not found for {version_name}: {db_path}")
            results[version_name] = {'cuts_stats': False, 'report': False, 'reason': 'No database'}
            continue
        
        # Create analysis directory if needed
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Run cuts_and_stats analysis
        cuts_success = run_analysis(db_path, analysis_dir, version_name)
        
        # Run PDF report generation
        report_success = run_report_generation(db_path, analysis_dir, version_name)
        
        results[version_name] = {
            'cuts_stats': cuts_success,
            'report': report_success
        }
    
    # Print summary
    print(f"\n\n{'='*80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    
    for version_name, status in sorted(results.items()):
        cuts = "✓" if status.get('cuts_stats') else "✗"
        report = "✓" if status.get('report') else "✗"
        reason = status.get('reason', '')
        print(f"{version_name:20} | Cuts: {cuts} | Report: {report} {reason}")
    
    # Count successes
    total = len(results)
    cuts_ok = sum(1 for s in results.values() if s.get('cuts_stats'))
    reports_ok = sum(1 for s in results.values() if s.get('report'))
    
    print(f"\n{'='*80}")
    print(f"Results: {cuts_ok}/{total} cuts_stats, {reports_ok}/{total} PDF reports")
    print(f"{'='*80}")
    
    return 0 if (cuts_ok == total and reports_ok == total) else 1

if __name__ == '__main__':
    sys.exit(main())
