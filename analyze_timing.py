#!/usr/bin/env python3
"""
Timing Analysis Tool for Navigation Server
Analyzes timing logs from navigation sessions
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
import statistics


def load_timing_files(log_dir: str) -> List[Dict]:
    """Load all timing JSON files from a log directory"""
    timing_files = sorted(Path(log_dir).glob('iter_*_timing.json'))
    
    timings = []
    for timing_file in timing_files:
        try:
            with open(timing_file, 'r') as f:
                data = json.load(f)
                timings.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {timing_file}: {e}")
    
    return timings


def analyze_timings(timings: List[Dict]) -> Dict:
    """Compute statistics from timing data"""
    if not timings:
        return {}
    
    stats = {
        'total_iterations': len(timings),
        'total_server_time': {
            'values': [],
            'mean': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'sum': 0.0
        },
        'detection_time': {
            'values': [],
            'mean': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'sum': 0.0
        },
        'vlm_projection_time': {
            'values': [],
            'mean': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'sum': 0.0
        },
        'vlm_inference_time': {
            'values': [],
            'mean': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'sum': 0.0
        },
        'action_type_counts': {},
        'detected_count': 0
    }
    
    # Collect data
    for timing in timings:
        if 'total_server_time' in timing:
            stats['total_server_time']['values'].append(timing['total_server_time'])
        if 'detection_time' in timing:
            stats['detection_time']['values'].append(timing['detection_time'])
        if 'vlm_projection_time' in timing:
            stats['vlm_projection_time']['values'].append(timing['vlm_projection_time'])
        if 'vlm_inference_time' in timing:
            stats['vlm_inference_time']['values'].append(timing['vlm_inference_time'])
        
        # Count action types
        action_type = timing.get('action_type', 'unknown')
        stats['action_type_counts'][action_type] = stats['action_type_counts'].get(action_type, 0) + 1
        
        # Count detections
        if timing.get('detected', False):
            stats['detected_count'] += 1
    
    # Calculate statistics for each timing category
    for key in ['total_server_time', 'detection_time', 'vlm_projection_time', 'vlm_inference_time']:
        values = stats[key]['values']
        if values:
            stats[key]['mean'] = statistics.mean(values)
            stats[key]['median'] = statistics.median(values)
            stats[key]['min'] = min(values)
            stats[key]['max'] = max(values)
            stats[key]['sum'] = sum(values)
        else:
            # Remove empty categories
            stats[key] = {'values': [], 'mean': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0, 'sum': 0.0}
    
    return stats


def print_report(stats: Dict, log_name: str):
    """Print formatted timing report"""
    print("=" * 80)
    print(f"TIMING ANALYSIS REPORT: {log_name}")
    print("=" * 80)
    
    if not stats:
        print("No timing data available.")
        return
    
    print(f"\nTotal Iterations: {stats['total_iterations']}")
    print(f"Target Detected: {stats['detected_count']} times ({stats['detected_count']/stats['total_iterations']*100:.1f}%)")
    
    print("\n" + "-" * 80)
    print("ACTION TYPE DISTRIBUTION:")
    print("-" * 80)
    for action_type, count in sorted(stats['action_type_counts'].items()):
        percentage = count / stats['total_iterations'] * 100
        print(f"  {action_type:20s}: {count:3d} times ({percentage:5.1f}%)")
    
    print("\n" + "-" * 80)
    print("TIMING STATISTICS (seconds):")
    print("-" * 80)
    
    categories = [
        ('Total Server Time', 'total_server_time'),
        ('Detection Time', 'detection_time'),
        ('VLM Projection Time', 'vlm_projection_time'),
        ('VLM Inference Time', 'vlm_inference_time')
    ]
    
    header = f"{'Category':<25} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'Total':>10}"
    print(header)
    print("-" * 80)
    
    for label, key in categories:
        data = stats[key]
        if data['values']:
            print(f"{label:<25} {data['mean']:>10.3f} {data['median']:>10.3f} "
                  f"{data['min']:>10.3f} {data['max']:>10.3f} {data['sum']:>10.3f}")
    
    # Calculate breakdown percentages
    if stats['total_server_time']['mean'] > 0:
        print("\n" + "-" * 80)
        print("TIME BREAKDOWN (percentage of total server time):")
        print("-" * 80)
        
        total_mean = stats['total_server_time']['mean']
        detection_pct = (stats['detection_time']['mean'] / total_mean) * 100
        projection_pct = (stats['vlm_projection_time']['mean'] / total_mean) * 100
        inference_pct = (stats['vlm_inference_time']['mean'] / total_mean) * 100
        other_pct = 100 - (detection_pct + projection_pct + inference_pct)
        
        print(f"  Detection:       {detection_pct:6.2f}%")
        print(f"  VLM Projection:  {projection_pct:6.2f}%")
        print(f"  VLM Inference:   {inference_pct:6.2f}%")
        print(f"  Other/Overhead:  {other_pct:6.2f}%")
    
    print("\n" + "=" * 80)


def save_report(stats: Dict, log_dir: str, output_file: str):
    """Save timing report to JSON file"""
    output_path = os.path.join(log_dir, output_file)
    with open(output_path, 'w') as f:
        # Remove raw values for cleaner output
        clean_stats = stats.copy()
        for key in ['total_server_time', 'detection_time', 'vlm_projection_time', 'vlm_inference_time']:
            if key in clean_stats and 'values' in clean_stats[key]:
                del clean_stats[key]['values']
        
        json.dump(clean_stats, f, indent=2)
    
    print(f"\nDetailed report saved to: {output_path}")


def list_available_logs(base_dir: str):
    """List all available log directories"""
    logs_path = Path(base_dir)
    if not logs_path.exists():
        print(f"Logs directory not found: {base_dir}")
        return []
    
    log_dirs = sorted([d for d in logs_path.iterdir() if d.is_dir() and d.name.startswith('run_')], reverse=True)
    
    print("\nAvailable log directories:")
    print("-" * 60)
    for i, log_dir in enumerate(log_dirs, 1):
        # Count timing files
        timing_files = list(log_dir.glob('iter_*_timing.json'))
        
        # Try to read metadata
        metadata_file = log_dir / 'metadata.json'
        goal = "unknown"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    goal = metadata.get('goal', 'unknown')
            except:
                pass
        
        print(f"  {i}. {log_dir.name} - Goal: {goal}, Iterations: {len(timing_files)}")
    
    return log_dirs


def main():
    parser = argparse.ArgumentParser(description='Analyze navigation timing logs')
    parser.add_argument('--log_dir', type=str, help='Path to specific log directory to analyze')
    parser.add_argument('--list', action='store_true', help='List all available log directories')
    parser.add_argument('--latest', action='store_true', help='Analyze the latest log directory')
    parser.add_argument('--save', action='store_true', help='Save report to JSON file')
    parser.add_argument('--base_dir', type=str, default='../logs', help='Base logs directory (default: ../logs)')
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, args.base_dir)
    
    if args.list:
        list_available_logs(base_dir)
        return
    
    # Determine which log directory to analyze
    log_dir = None
    
    if args.log_dir:
        log_dir = args.log_dir
    elif args.latest:
        log_dirs = list_available_logs(base_dir)
        if log_dirs:
            log_dir = str(log_dirs[0])
            print(f"\nAnalyzing latest log: {log_dir}")
        else:
            print("No log directories found.")
            return
    else:
        print("Please specify --log_dir, --latest, or --list")
        parser.print_help()
        return
    
    if not os.path.exists(log_dir):
        print(f"Error: Log directory not found: {log_dir}")
        return
    
    # Load and analyze timing data
    print(f"\nLoading timing data from: {log_dir}")
    timings = load_timing_files(log_dir)
    
    if not timings:
        print("No timing data found in this directory.")
        return
    
    print(f"Loaded {len(timings)} timing records.\n")
    
    # Analyze and print report
    stats = analyze_timings(timings)
    log_name = os.path.basename(log_dir)
    print_report(stats, log_name)
    
    # Save report if requested
    if args.save:
        save_report(stats, log_dir, 'timing_report.json')


if __name__ == '__main__':
    main()

