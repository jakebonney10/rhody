#!/usr/bin/env python3
"""
batch_nortek2ros.py

Batch processor for Nortek DVL data conversion to ROS2 bags.
Automatically finds directories containing DVL CSV files and processes each one.

Usage:
    python3 batch_nortek2ros.py --root-dir /path/to/data --output-suffix _processed
    python3 batch_nortek2ros.py  # Process current directory
    python3 batch_nortek2ros.py --config config.yaml  # Use YAML configuration
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import shutil
import yaml
import tempfile
from datetime import datetime

# Required CSV files for DVL processing
REQUIRED_DVL_FILES = [
    "Bottom Track.csv",
    "IMU.csv", 
    "INS.csv",
    "Magnetometer.csv"
]

def find_dvl_directories(root_path):
    """
    Find all directories that contain the required DVL CSV files.
    Returns list of (directory_path, missing_files) tuples.
    """
    dvl_dirs = []
    root_path = Path(root_path).resolve()
    
    print(f"🔍 Searching for DVL data in: {root_path}")
    
    # Walk through all subdirectories
    for dirpath, dirnames, filenames in os.walk(root_path):
        dir_path = Path(dirpath)
        
        # Check if this directory has all required CSV files
        missing_files = []
        for required_file in REQUIRED_DVL_FILES:
            if required_file not in filenames:
                missing_files.append(required_file)
        
        if not missing_files:  # All files present
            dvl_dirs.append((dir_path, []))
            print(f"✅ Found complete DVL dataset: {dir_path}")
        elif len(missing_files) < len(REQUIRED_DVL_FILES):  # Some files present
            dvl_dirs.append((dir_path, missing_files))
            print(f"⚠️  Incomplete DVL dataset: {dir_path} (missing: {', '.join(missing_files)})")
    
    return dvl_dirs

def process_dvl_directory(dvl_dir, nortek_script_path, force_process=False):
    """
    Process a single DVL directory by running nortek2ros.py in that directory.
    """
    dvl_dir = Path(dvl_dir)
    original_cwd = os.getcwd()
    
    try:
        # Change to the DVL directory
        os.chdir(dvl_dir)
        print(f"\n📁 Processing: {dvl_dir}")
        
        # Check if output already exists
        output_dir = dvl_dir / "nortek_dvl_bag"
        if output_dir.exists() and not force_process:
            print(f"⏭️  Skipping (output already exists): {output_dir}")
            print(f"   Use --force to overwrite existing outputs")
            return True
        
        # Run the nortek2ros.py script
        cmd = [sys.executable, str(nortek_script_path)]
        print(f"🔄 Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Success: {dvl_dir}")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Error processing {dvl_dir}")
            print(f"   Exit code: {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        print(f"❌ Exception processing {dvl_dir}: {e}")
        return False
    finally:
        # Always return to original directory
        os.chdir(original_cwd)

def main():
    parser = argparse.ArgumentParser(description='Batch process Nortek DVL data directories')
    parser.add_argument('--root-dir', '-r', type=str, default='.',
                       help='Root directory to search for DVL data (default: current directory)')
    parser.add_argument('--script-path', '-s', type=str, 
                       default=None,
                       help='Path to nortek2ros.py script (auto-detected if not specified)')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Overwrite existing output directories')
    parser.add_argument('--incomplete', '-i', action='store_true',
                       help='Attempt to process directories with missing CSV files')
    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='Show what would be processed without actually running')
    
    args = parser.parse_args()
    
    # Auto-detect nortek2ros.py script path if not provided
    if args.script_path is None:
        # Try to find the script relative to this script's location
        current_script_dir = Path(__file__).parent
        nortek_script = current_script_dir / "nortek2ros.py"
        if nortek_script.exists():
            args.script_path = str(nortek_script)
        else:
            print("❌ Could not find nortek2ros.py script")
            print("   Specify path with --script-path")
            return 1
    
    nortek_script_path = Path(args.script_path)
    if not nortek_script_path.exists():
        print(f"❌ Script not found: {nortek_script_path}")
        return 1
    
    print(f"🚀 Batch Nortek DVL Processing")
    print(f"   Root directory: {Path(args.root_dir).resolve()}")
    print(f"   Script: {nortek_script_path}")
    print(f"   Force overwrite: {args.force}")
    print(f"   Process incomplete: {args.incomplete}")
    print(f"   Dry run: {args.dry_run}")
    
    # Find all DVL directories
    dvl_dirs = find_dvl_directories(args.root_dir)
    
    if not dvl_dirs:
        print(f"\n❌ No DVL directories found in {args.root_dir}")
        return 1
    
    # Filter based on completeness
    if not args.incomplete:
        complete_dirs = [(d, m) for d, m in dvl_dirs if not m]
        incomplete_dirs = [(d, m) for d, m in dvl_dirs if m]
        
        if incomplete_dirs:
            print(f"\n⚠️  Found {len(incomplete_dirs)} incomplete directories (use --incomplete to process them):")
            for dir_path, missing in incomplete_dirs:
                print(f"   {dir_path} (missing: {', '.join(missing)})")
        
        dvl_dirs = complete_dirs
    
    if not dvl_dirs:
        print(f"\n❌ No processable DVL directories found")
        return 1
    
    print(f"\n📋 Found {len(dvl_dirs)} directories to process:")
    for dir_path, missing in dvl_dirs:
        status = "incomplete" if missing else "complete"
        print(f"   {dir_path} ({status})")
    
    if args.dry_run:
        print(f"\n🏃 Dry run complete - no files processed")
        return 0
    
    # Process each directory
    print(f"\n🔄 Processing {len(dvl_dirs)} directories...")
    success_count = 0
    
    for dir_path, missing in dvl_dirs:
        if missing and not args.incomplete:
            continue  # Skip incomplete unless explicitly requested
            
        success = process_dvl_directory(dir_path, nortek_script_path, args.force)
        if success:
            success_count += 1
    
    # Summary
    print(f"\n📊 Processing Summary:")
    print(f"   Total directories: {len(dvl_dirs)}")
    print(f"   Successfully processed: {success_count}")
    print(f"   Failed: {len(dvl_dirs) - success_count}")
    
    if success_count == len(dvl_dirs):
        print(f"🎉 All directories processed successfully!")
        return 0
    else:
        print(f"⚠️  Some directories failed to process")
        return 1

if __name__ == "__main__":
    sys.exit(main())