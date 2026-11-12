#!/usr/bin/env python3
"""
Wrapper script for generate_iso_v4_report.py that accepts command-line arguments
and adapts the report generation to work with dynamically named output directories.

Usage:
    python3 generate_report_wrapper.py --db-path <path> --hdf5-dir <path> --output-dir <path> [--version-number <version>]
"""

import sys
import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate ISO-V4 analysis report with dynamic paths")
    parser.add_argument("--db-path", required=True, help="Path to the SQLite database file")
    parser.add_argument("--hdf5-dir", required=True, help="Directory containing HDF5 aggregation files")
    parser.add_argument("--output-dir", required=True, help="Output directory for analysis plots and PDF")
    parser.add_argument("--version-number", default="unknown", help="Version number for report naming")
    parser.add_argument("--pdf-output", default=None, help="Explicit path for output PDF (default: output-dir/iso_v4_VERS_report.pdf)")
    
    args = parser.parse_args()
    
    # Validate inputs
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: Database file not found: {db_path}")
        return 1
    
    hdf5_dir = Path(args.hdf5_dir)
    if not hdf5_dir.exists():
        print(f"WARNING: HDF5 directory does not exist: {hdf5_dir}")
        print("Continuing - report generation may have limited aggregation data")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine PDF output path
    if args.pdf_output:
        pdf_path = Path(args.pdf_output)
    else:
        pdf_path = output_dir / f"iso_v4_{args.version_number}_report.pdf"
    
    print(f"[INFO] Database:        {db_path}")
    print(f"[INFO] HDF5 directory:  {hdf5_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] PDF output:      {pdf_path}")
    print()
    
    # Try to dynamically patch the generate_iso_v4_report module
    # by temporarily overriding the hardcoded paths
    
    try:
        # Import the report generation module
        import generate_iso_v4_report as report_module
        
        # Extract SAT tags from database and HDF5 files found
        db_name = db_path.stem  # e.g., iso_v4_satp1_20260720
        version_parts = db_name.split('_')
        
        # Try to infer sat tags
        satp1_db = None
        satp3_db = None
        satp2_db = None

        # Check if this is a satp1, satp2, or satp3 database
        if 'satp1' in db_name:
            satp1_db = db_path
        elif 'satp3' in db_name:
            satp3_db = db_path
        elif 'satp2' in db_name:
            satp2_db = db_path

        print(f"[INFO] Detected database type: {db_name}")
        print()
        
        # Check for matching HDF5 files
        if hdf5_dir.exists():
            h5_files = list(hdf5_dir.glob("*.h5"))
            print(f"[INFO] Found {len(h5_files)} HDF5 files in {hdf5_dir}")
            if h5_files:
                for f in h5_files[:3]:  # Show first 3
                    print(f"       - {f.name}")
                if len(h5_files) > 3:
                    print(f"       ... and {len(h5_files)-3} more")
        
        print()
        print("[INFO] Attempting to patch hardcoded paths in generate_iso_v4_report...")
        
        # Monkey-patch the SQLITE_PATHS dictionary
        original_paths = report_module.SQLITE_PATHS.copy()
        
        if satp1_db:
            report_module.SQLITE_PATHS['satp1'] = satp1_db
            print(f"[INFO] Updated SQLITE_PATHS['satp1'] -> {satp1_db}")
        
        if satp3_db:
            report_module.SQLITE_PATHS['satp3'] = satp3_db
            print(f"[INFO] Updated SQLITE_PATHS['satp3'] -> {satp3_db}")

        if satp2_db:
            report_module.SQLITE_PATHS['satp2'] = satp2_db
            print(f"[INFO] Updated SQLITE_PATHS['satp2'] -> {satp2_db}")

        # Override ANALYSIS_ROOT
        report_module.ANALYSIS_ROOT = output_dir
        print(f"[INFO] Updated ANALYSIS_ROOT -> {output_dir}")
        print()

        # Run the report-building function. --db-path names exactly one
        # telescope, so build the single-telescope report for it. The paired
        # satp1-vs-satp3 report is only correct when both databases come from
        # the same processing version; falling back to it here would silently
        # pair this database against whatever stale default the other
        # telescope's SQLITE_PATHS entry still points at.
        print("[INFO] Starting report generation...")
        telescope = next((t for t, db in (('satp1', satp1_db),
                                          ('satp2', satp2_db),
                                          ('satp3', satp3_db)) if db), None)
        if telescope is None:
            raise SystemExit(
                f"Could not infer a telescope from database name {db_name!r}; "
                "expected it to contain satp1, satp2 or satp3."
            )
        report_module.build_report_single(telescope)
        
        # If successful and PDF path differs, try to move it
        default_pdf = report_module.ANALYSIS_ROOT / 'iso_v4_analysis_report.pdf'
        if default_pdf.exists() and default_pdf != pdf_path:
            print(f"[INFO] Moving generated PDF from {default_pdf} to {pdf_path}")
            default_pdf.rename(pdf_path)
        
        print(f"[INFO] Report generation completed!")
        print(f"[INFO] Output files in: {output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
