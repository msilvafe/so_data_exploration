#!/usr/bin/env python3
"""
Utility to extract processing pipeline information from SO preprocessing configuration files.
This helps understand the step sequence for the analysis script.
"""

import argparse
import yaml
import json
import os
from typing import Dict, List, Any


def parse_so_config(config_path: str) -> Dict[str, Any]:
    """Parse SO preprocessing configuration file."""
    
    # Determine file format
    try:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Try YAML first, then JSON
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            except:
                with open(config_path, 'r') as f:
                    config = json.load(f)
    except Exception as e:
        print(f"Error reading config file: {e}")
        return {}
    
    # Extract processing pipeline
    pipeline_info = {
        'steps': [],
        'detector_cuts_steps': [],
        'sample_cuts_steps': [],
        'column_mapping': {}
    }
    
    # Look for processing pipeline in various locations
    pipeline = None
    if 'preprocess' in config:
        pipeline = config['preprocess']
    elif 'pipeline' in config:
        pipeline = config['pipeline']
    elif 'processing' in config:
        pipeline = config['processing']
    elif 'subpipelines' in config:
        # Handle subpipelines structure
        pipeline = []
        for subpipe_name, subpipe_config in config['subpipelines'].items():
            if 'preprocess' in subpipe_config:
                pipeline.extend(subpipe_config['preprocess'])
    elif isinstance(config, list):
        pipeline = config
    
    if pipeline:
        step_num = 0
        for step in pipeline:
            step_name = None
            step_info = {}
            
            # Handle different config formats
            if isinstance(step, dict):
                # Get the operation name
                if 'name' in step:
                    step_name = step['name']
                elif len(step) == 1:
                    step_name = list(step.keys())[0]
                    step_info = step[step_name] if isinstance(step[step_name], dict) else {}
                else:
                    # Look for common operation keys
                    for key in ['operation', 'op', 'func', 'function']:
                        if key in step:
                            step_name = step[key]
                            break
                
                # Extract additional info
                step_info.update({k: v for k, v in step.items() if k != step_name and k != 'name'})
                    
            elif isinstance(step, str):
                step_name = step
            
            if step_name:
                column_name = f"{step_num}_{step_name}"
                
                step_entry = {
                    'step_number': step_num,
                    'step_name': step_name,
                    'column_name': column_name,
                    'config': step_info
                }
                
                pipeline_info['steps'].append(step_entry)
                pipeline_info['column_mapping'][step_name] = step_num
                
                # Categorize steps
                detector_cut_keywords = [
                    'dark_dets', 'fp_flags', 'detcal_nan', 'det_bias_flags',
                    'trends', 'jumps', 'glitches', 'ptp_flags', 
                    'noisy_subscan_flags', 'estimate_t2p', 'subtract_t2p',
                    'inv_var_flags', 'cut_bad_dist', 'calibrate',
                    'fourier_filter', 'apodize', 'demodulate', 'azss',
                    'subtract_azss_template', 'psd', 'noise', 'sub_polyf'
                ]
                
                sample_cut_keywords = [
                    'smurfgaps_flags', 'turnarounds', 'source_flags',
                    'flag_turnarounds'
                ]
                
                if any(keyword in step_name.lower() for keyword in detector_cut_keywords):
                    pipeline_info['detector_cuts_steps'].append(step_entry)
                elif any(keyword in step_name.lower() for keyword in sample_cut_keywords):
                    pipeline_info['sample_cuts_steps'].append(step_entry)
            
            step_num += 1
    
    return pipeline_info


def print_pipeline_info(info: Dict[str, Any]):
    """Print pipeline information in a readable format."""
    print("=" * 80)
    print("PREPROCESSING PIPELINE ANALYSIS")
    print("=" * 80)
    
    print(f"\nTotal steps: {len(info['steps'])}")
    print(f"Detector cuts steps: {len(info['detector_cuts_steps'])}")
    print(f"Sample cuts steps: {len(info['sample_cuts_steps'])}")
    
    print("\n" + "=" * 40)
    print("ALL PROCESSING STEPS")
    print("=" * 40)
    
    for step in info['steps']:
        config_str = ""
        if step['config']:
            config_str = f" (config: {step['config']})"
        print(f"{step['step_number']:2d}: {step['step_name']}{config_str}")
    
    print("\n" + "=" * 40)
    print("DETECTOR CUTS STEPS")
    print("=" * 40)
    
    for step in info['detector_cuts_steps']:
        print(f"{step['step_number']:2d}: {step['step_name']} -> {step['column_name']}")
    
    print("\n" + "=" * 40)
    print("SAMPLE CUTS STEPS")
    print("=" * 40)
    
    for step in info['sample_cuts_steps']:
        print(f"{step['step_number']:2d}: {step['step_name']} -> {step['column_name']}")
    
    print("\n" + "=" * 40)
    print("COLUMN MAPPING (for analysis script)")
    print("=" * 40)
    
    print("detector_cuts = {")
    for step in info['detector_cuts_steps']:
        print(f"    '{step['step_name']}': {step['step_number']},")
    print("}")
    
    print("\nsample_cuts = {")
    for step in info['sample_cuts_steps']:
        print(f"    '{step['step_name']}': {step['step_number']},")
    print("}")


def main():
    parser = argparse.ArgumentParser(description="Analyze SO preprocessing configuration")
    parser.add_argument('config_path', help="Path to preprocessing configuration file")
    parser.add_argument('--output', help="Output the analysis to a file instead of printing")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config_path):
        print(f"Error: Config file not found: {args.config_path}")
        return
    
    info = parse_so_config(args.config_path)
    
    if not info['steps']:
        print("No processing pipeline found in config file")
        return
    
    if args.output:
        import sys
        with open(args.output, 'w') as f:
            # Redirect stdout to file
            old_stdout = sys.stdout
            sys.stdout = f
            print_pipeline_info(info)
            sys.stdout = old_stdout
        print(f"Analysis written to {args.output}")
    else:
        print_pipeline_info(info)


if __name__ == '__main__':
    main()