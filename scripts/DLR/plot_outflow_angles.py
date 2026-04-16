#!/usr/bin/env python3
"""
Script to read OutflowAngle values from aero-optim.log files in subdirectories
and plot them in order.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def parse_log_file(log_path):
    """
    Parse aero-optim.log file to extract OutflowAngle values.
    Handles two formats:
    1. High fidelity: "post_process g*, c* ADP.." followed by table with OutflowAngle
    2. Low fidelity: "g*, c* ADP outflow angle: (value)"
    
    Returns:
        dict: Dictionary with keys like 'c0_ADP', 'c1_OP1', etc. mapping to OutflowAngle values
    """
    angles = {}
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Format 1: Look for post_process lines (high fidelity)
        match = re.search(r'post_process g\d+, (c\d+) (ADP|OP1|OP2)', line)
        if match:
            candidate = match.group(1)  # e.g., 'c0'
            point_type = match.group(2)  # e.g., 'ADP'
            
            # Look for the OutflowAngle value in the next few lines
            for j in range(i+1, min(i+10, len(lines))):
                if 'OutflowAngle' in lines[j]:
                    # Next line should contain the value
                    if j+1 < len(lines):
                        value_line = lines[j+1].strip()
                        try:
                            # Extract the numeric value (last column)
                            parts = value_line.split()
                            if len(parts) >= 2:
                                angle = float(parts[-1])
                                key = f"{candidate}_{point_type}"
                                angles[key] = angle
                                break
                        except (ValueError, IndexError):
                            pass
        
        # Format 2: Look for low fidelity format "g*, c* ADP outflow angle: (value)"
        match_lf = re.search(r'g\d+, (c\d+) (ADP|OP1|OP2) outflow angle: \(([-\d.]+)\)', line)
        if match_lf:
            candidate = match_lf.group(1)  # e.g., 'c0'
            point_type = match_lf.group(2)  # e.g., 'ADP'
            angle = float(match_lf.group(3))  # e.g., '-1.313983163736734'
            key = f"{candidate}_{point_type}"
            angles[key] = angle
        
        i += 1
    
    return angles

def collect_angles_from_directory(output_dir):
    """
    Collect OutflowAngle values from all relevant subdirectories.
    
    Args:
        output_dir: Path to the output_mf directory
        
    Returns:
        tuple: (high_fidelity_data, low_fidelity_data)
            - high_fidelity_data: dict organized by point_type -> infill_number -> candidate -> angle
            - low_fidelity_data: dict organized by point_type -> infill_number -> candidate -> angle
    """
    output_path = Path(output_dir)
    
    # Data structure: {point_type: {infill_num: {candidate: angle}}}
    high_data = {
        'ADP': {},
        'OP1': {},
        'OP2': {}
    }
    
    low_data = {
        'ADP': {},
        'OP1': {},
        'OP2': {}
    }
    
    # Find all aero-optim.log files recursively
    for log_file in output_path.rglob('aero-optim.log'):
        parent_dir = log_file.parent
        dir_name = parent_dir.name
        
        # Check if it's hf_doe
        if dir_name == 'hf_doe':
            infill_num = 0  # DOE is index 0
            angles = parse_log_file(log_file)
            
            # Store DOE data
            for key, angle in angles.items():
                parts = key.split('_')
                if len(parts) == 2:
                    candidate, point_type = parts
                    if point_type in high_data:
                        if infill_num not in high_data[point_type]:
                            high_data[point_type][infill_num] = {}
                        high_data[point_type][infill_num][candidate] = angle
        
        # Check if it's high_infill_*
        elif dir_name.startswith('high_infill_'):
            # Extract infill number
            match = re.search(r'high_infill_(\d+)', dir_name)
            if match:
                infill_num = int(match.group(1)) + 1  # +1 because DOE is 0
                angles = parse_log_file(log_file)
                
                print(f"Found {dir_name} with angles: {angles}")
                
                # Store infill data (should be only c0)
                for key, angle in angles.items():
                    parts = key.split('_')
                    if len(parts) == 2:
                        candidate, point_type = parts
                        if point_type in high_data:
                            if infill_num not in high_data[point_type]:
                                high_data[point_type][infill_num] = {}
                            high_data[point_type][infill_num][candidate] = angle
        
        # Check if it's low_infill_*
        elif dir_name.startswith('low_infill_'):
            # Extract infill number
            match = re.search(r'low_infill_(\d+)', dir_name)
            if match:
                infill_num = int(match.group(1)) + 1  # +1 because DOE is 0
                angles = parse_log_file(log_file)
                
                print(f"Found {dir_name} with angles: {angles}")
                
                # Store low fidelity infill data (should be only c0)
                for key, angle in angles.items():
                    parts = key.split('_')
                    if len(parts) == 2:
                        candidate, point_type = parts
                        if point_type in low_data:
                            if infill_num not in low_data[point_type]:
                                low_data[point_type][infill_num] = {}
                            low_data[point_type][infill_num][candidate] = angle
    
    return high_data, low_data

def plot_angles(data, output_dir):
    """
    Plot OutflowAngle evolution for each design point (ADP, OP1, OP2).
    
    Args:
        data: Nested dictionary from collect_angles_from_directory
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    colors = {'ADP': 'blue', 'OP1': 'red', 'OP2': 'green'}
    markers = {'ADP': 'o', 'OP1': 's', 'OP2': '^'}
    
    for point_type in ['ADP', 'OP1', 'OP2']:
        if point_type not in data or not data[point_type]:
            continue
        
        # Sort by infill number
        sorted_infills = sorted(data[point_type].keys())
        
        angles_list = []
        infill_list = []
        
        for infill_num in sorted_infills:
            # For DOE (infill_num = 0), we have multiple candidates
            # For infills, we typically have only c0
            candidates = data[point_type][infill_num]
            
            if infill_num == 0:  # DOE
                # Sort candidates (c0, c1, c2, c3, c4)
                sorted_candidates = sorted(candidates.keys(), 
                                          key=lambda x: int(x[1:]))
                for idx, candidate in enumerate(sorted_candidates):
                    angles_list.append(candidates[candidate])
                    infill_list.append(idx)
            else:
                # For infills, typically only c0
                if 'c0' in candidates:
                    angles_list.append(candidates['c0'])
                    # Continue the sequence after DOE
                    doe_size = len(data[point_type].get(0, {}))
                    infill_list.append(doe_size + infill_num - 1)
        
        if angles_list:
            plt.plot(infill_list, angles_list, 
                    marker=markers[point_type], 
                    color=colors[point_type],
                    label=point_type,
                    linewidth=2,
                    markersize=8)
    
    plt.xlabel('Evaluation Number', fontsize=12)
    plt.ylabel('OutflowAngle [deg]', fontsize=12)
    plt.title('OutflowAngle Evolution', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = 'outflow_angles_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()
    
    return angles_list, infill_list

def parse_qoi_convergence(csv_path):
    """
    Parse QoI_convergence.csv file to extract OutflowAngle and MixedoutLossCoef values.
    
    Args:
        csv_path: Path to QoI_convergence.csv file
        
    Returns:
        tuple: (angles, min_angle, max_angle, loss_coefs, min_loss, max_loss)
    """
    angles = []
    loss_coefs = []
    
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            try:
                loss_coef = float(parts[0])
                angle = float(parts[1])
                loss_coefs.append(loss_coef)
                angles.append(angle)
            except ValueError:
                pass
    
    if angles and loss_coefs:
        return (angles, min(angles), max(angles), 
                loss_coefs, min(loss_coefs), max(loss_coefs))
    else:
        return [], None, None, [], None, None

def collect_qoi_convergence_data(output_dir):
    """
    Collect OutflowAngle data from QoI_convergence.csv files in:
    - output_mf/hf_doe/MUSICAA/musicaa_*/ADP,OP1,OP2
    - output_mf/output_mf_*/high_infill_*/MUSICAA/musicaa_g0_c0/ADP,OP1,OP2
    
    Args:
        output_dir: Path to the output_mf directory
        
    Returns:
        dict: {point_type: {infill_num: {'angles': [...], 'min': float, 'max': float}}}
    """
    output_path = Path(output_dir)
    
    data = {
        'ADP': {},
        'OP1': {},
        'OP2': {}
    }
    
    # Pattern 1: hf_doe/MUSICAA/musicaa_g0_c*/ADP,OP1,OP2
    hf_doe_path = output_path / 'hf_doe' / 'MUSICAA'
    if hf_doe_path.exists():
        for musicaa_dir in hf_doe_path.glob('musicaa_g0_c*'):
            if not musicaa_dir.is_dir():
                continue
            
            for point_type in ['ADP', 'OP1', 'OP2']:
                qoi_file = musicaa_dir / point_type / 'QoI_convergence.csv'
                if qoi_file.exists():
                    angles, min_angle, max_angle, loss_coefs, min_loss, max_loss = parse_qoi_convergence(qoi_file)
                    if angles and loss_coefs:
                        # Extract candidate number from musicaa_g0_c*
                        match = re.search(r'musicaa_g0_c(\d+)', musicaa_dir.name)
                        if match:
                            candidate_num = int(match.group(1))
                            # Store in infill_num = 0 (DOE), with candidate key
                            if 0 not in data[point_type]:
                                data[point_type][0] = {}
                            data[point_type][0][f'c{candidate_num}'] = {
                                'angles': angles,
                                'min_angle': min_angle,
                                'max_angle': max_angle,
                                'mean_angle': np.mean(angles),
                                'loss_coefs': loss_coefs,
                                'min_loss': min_loss,
                                'max_loss': max_loss,
                                'mean_loss': np.mean(loss_coefs)
                            }
                            print(f"Found DOE {musicaa_dir.name}/{point_type}: {len(angles)} angles, mean={np.mean(angles):.4f}, loss={np.mean(loss_coefs):.6f}")
    
    # Pattern 2: output_mf_*/high_infill_*/MUSICAA/musicaa_g0_c0/ADP,OP1,OP2
    for output_mf_dir in output_path.glob('output_mf_*'):
        if not output_mf_dir.is_dir():
            continue
        
        for high_infill_dir in output_mf_dir.glob('high_infill_*'):
            if not high_infill_dir.is_dir():
                continue
            
            # Extract infill number
            match = re.search(r'high_infill_(\d+)', high_infill_dir.name)
            if not match:
                continue
            infill_num = int(match.group(1)) + 1  # +1 because DOE is 0
            
            musicaa_path = high_infill_dir / 'MUSICAA' / 'musicaa_g0_c0'
            if not musicaa_path.exists():
                continue
            
            for point_type in ['ADP', 'OP1', 'OP2']:
                qoi_file = musicaa_path / point_type / 'QoI_convergence.csv'
                if qoi_file.exists():
                    angles, min_angle, max_angle, loss_coefs, min_loss, max_loss = parse_qoi_convergence(qoi_file)
                    if angles and loss_coefs:
                        if infill_num not in data[point_type]:
                            data[point_type][infill_num] = {}
                        data[point_type][infill_num]['c0'] = {
                            'angles': angles,
                            'min_angle': min_angle,
                            'max_angle': max_angle,
                            'mean_angle': np.mean(angles),
                            'loss_coefs': loss_coefs,
                            'min_loss': min_loss,
                            'max_loss': max_loss,
                            'mean_loss': np.mean(loss_coefs)
                        }
                        print(f"Found {high_infill_dir.name}/{point_type}: {len(angles)} angles, mean={np.mean(angles):.4f}, loss={np.mean(loss_coefs):.6f}")
    
    return data

def plot_qoi_convergence(data, output_dir):
    """
    Plot OutflowAngle evolution with ranges from QoI_convergence data.
    Similar to plot_angles but includes min/max ranges as shaded regions.
    
    Args:
        data: Data from collect_qoi_convergence_data
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(14, 7))
    
    colors = {'ADP': 'blue', 'OP1': 'red', 'OP2': 'green'}
    markers = {'ADP': 'o', 'OP1': 's', 'OP2': '^'}
    
    for point_type in ['ADP', 'OP1', 'OP2']:
        if point_type not in data or not data[point_type]:
            continue
        
        # Sort by infill number
        sorted_infills = sorted(data[point_type].keys())
        
        mean_list = []
        min_list = []
        max_list = []
        infill_list = []
        
        for infill_num in sorted_infills:
            candidates = data[point_type][infill_num]
            
            if infill_num == 0:  # DOE
                # Sort candidates (c0, c1, c2, c3, c4)
                sorted_candidates = sorted(candidates.keys(), 
                                          key=lambda x: int(x[1:]))
                for idx, candidate in enumerate(sorted_candidates):
                    cand_data = candidates[candidate]
                    mean_list.append(cand_data['mean_angle'])
                    min_list.append(cand_data['min_angle'])
                    max_list.append(cand_data['max_angle'])
                    infill_list.append(idx)
            else:
                # For infills, typically only c0
                if 'c0' in candidates:
                    cand_data = candidates['c0']
                    mean_list.append(cand_data['mean_angle'])
                    min_list.append(cand_data['min_angle'])
                    max_list.append(cand_data['max_angle'])
                    # Continue the sequence after DOE
                    doe_size = len(data[point_type].get(0, {}))
                    infill_list.append(doe_size + infill_num - 1)
        
        if mean_list:
            # Plot mean values with line and markers
            plt.plot(infill_list, mean_list, 
                    marker=markers[point_type], 
                    color=colors[point_type],
                    label=f'{point_type} (mean)',
                    linewidth=2,
                    markersize=8,
                    zorder=3)
            
            # Plot shaded range (min to max)
            plt.fill_between(infill_list, min_list, max_list,
                           color=colors[point_type],
                           alpha=0.2,
                           label=f'{point_type} (range)',
                           zorder=1)
    
    plt.xlabel('Evaluation Number', fontsize=12)
    plt.ylabel('OutflowAngle [deg]', fontsize=12)
    plt.title('OutflowAngle Evolution with Convergence Range', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim([-8, 8])
    plt.tight_layout()
    
    # Save plot
    output_path = 'outflow_angles_qoi_convergence.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"QoI convergence plot saved to: {output_path}")
    plt.show()


def plot_loss_coef_convergence(data, output_dir):
    """
    Plot MixedoutLossCoef evolution with ranges from QoI_convergence data.
    
    Args:
        data: Data from collect_qoi_convergence_data
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(14, 7))
    
    colors = {'ADP': 'blue', 'OP1': 'red', 'OP2': 'green'}
    markers = {'ADP': 'o', 'OP1': 's', 'OP2': '^'}
    
    for point_type in ['ADP', 'OP1', 'OP2']:
        if point_type not in data or not data[point_type]:
            continue
        
        # Sort by infill number
        sorted_infills = sorted(data[point_type].keys())
        
        mean_list = []
        min_list = []
        max_list = []
        infill_list = []
        
        for infill_num in sorted_infills:
            candidates = data[point_type][infill_num]
            
            if infill_num == 0:  # DOE
                # Sort candidates (c0, c1, c2, c3, c4)
                sorted_candidates = sorted(candidates.keys(), 
                                          key=lambda x: int(x[1:]))
                for idx, candidate in enumerate(sorted_candidates):
                    cand_data = candidates[candidate]
                    mean_list.append(cand_data['mean_loss'])
                    min_list.append(cand_data['min_loss'])
                    max_list.append(cand_data['max_loss'])
                    infill_list.append(idx)
            else:
                # For infills, typically only c0
                if 'c0' in candidates:
                    cand_data = candidates['c0']
                    mean_list.append(cand_data['mean_loss'])
                    min_list.append(cand_data['min_loss'])
                    max_list.append(cand_data['max_loss'])
                    # Continue the sequence after DOE
                    doe_size = len(data[point_type].get(0, {}))
                    infill_list.append(doe_size + infill_num - 1)
        
        if mean_list:
            # Plot mean values with line and markers
            plt.plot(infill_list, mean_list, 
                    marker=markers[point_type], 
                    color=colors[point_type],
                    label=f'{point_type} (mean)',
                    linewidth=2,
                    markersize=8,
                    zorder=3)
            
            # Plot shaded range (min to max)
            plt.fill_between(infill_list, min_list, max_list,
                           color=colors[point_type],
                           alpha=0.2,
                           label=f'{point_type} (range)',
                           zorder=1)
    
    plt.xlabel('Evaluation Number', fontsize=12)
    plt.ylabel('MixedoutLossCoef', fontsize=12)
    plt.title('MixedoutLossCoef Evolution with Convergence Range', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = 'loss_coef_qoi_convergence.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Loss coefficient convergence plot saved to: {output_path}")
    plt.show()


def plot_high_vs_low_fidelity(high_data, low_data, output_dir):
    """
    Plot high fidelity vs low fidelity predictions for c0 individual.
    Each infill number gets the same marker but different colors for ADP, OP1, OP2.
    
    Args:
        high_data: High fidelity data from collect_angles_from_directory
        low_data: Low fidelity data from collect_angles_from_directory
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 10))
    
    colors = {'ADP': 'blue', 'OP1': 'red', 'OP2': 'green'}
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    
    # Collect matching high and low fidelity points
    for point_type in ['ADP', 'OP1', 'OP2']:
        # Get infill numbers that have data in both high and low
        high_infills = set(high_data[point_type].keys()) - {0}  # Exclude DOE
        low_infills = set(low_data[point_type].keys()) - {0}
        common_infills = sorted(high_infills & low_infills)
        
        for idx, infill_num in enumerate(common_infills):
            # Get c0 data for this infill
            high_angle = high_data[point_type].get(infill_num, {}).get('c0', None)
            low_angle = low_data[point_type].get(infill_num, {}).get('c0', None)
            
            if high_angle is not None and low_angle is not None:
                marker_idx = (infill_num - 1) % len(markers)
                
                # Only add label for the first point_type of each infill
                label = f'Infill {infill_num-1}' if point_type == 'ADP' else None
                
                plt.scatter(low_angle, high_angle, 
                           marker=markers[marker_idx],
                           color=colors[point_type],
                           s=100,
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=1,
                           label=label)
    
    # Plot bisector (y = x line)
    all_angles = []
    for point_type in ['ADP', 'OP1', 'OP2']:
        for infill_num in high_data[point_type].keys():
            if infill_num > 0 and 'c0' in high_data[point_type][infill_num]:
                all_angles.append(high_data[point_type][infill_num]['c0'])
        for infill_num in low_data[point_type].keys():
            if infill_num > 0 and 'c0' in low_data[point_type][infill_num]:
                all_angles.append(low_data[point_type][infill_num]['c0'])
    
    if all_angles:
        min_angle = min(all_angles)
        max_angle = max(all_angles)
        margin = (max_angle - min_angle) * 0.1
        plot_range = [min_angle - margin, max_angle + margin]
        plt.plot(plot_range, plot_range, 'k--', linewidth=2, label='y = x', alpha=0.5)
        plt.xlim(plot_range)
        plt.ylim(plot_range)
    
    # Create custom legend for colors (point types)
    from matplotlib.lines import Line2D
    color_legend = [Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=colors[pt], markersize=10, 
                          label=pt, markeredgecolor='black')
                   for pt in ['ADP', 'OP1', 'OP2']]
    
    # Get marker legend
    legend1 = plt.legend(handles=color_legend, loc='upper left', title='Design Point')
    plt.gca().add_artist(legend1)
    
    # Add infill legend
    plt.legend(loc='lower right', title='Infill Number')
    
    plt.xlabel('Low Fidelity OutflowAngle [deg]', fontsize=12)
    plt.ylabel('High Fidelity OutflowAngle [deg]', fontsize=12)
    plt.title('Low vs High Fidelity Predictions (c0)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    # Save plot
    output_path = 'hf_vs_lf_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()

def print_summary(data):
    """Print a summary of collected data."""
    print("\n" + "="*60)
    print("SUMMARY OF COLLECTED DATA")
    print("="*60)
    
    for point_type in ['ADP', 'OP1', 'OP2']:
        print(f"\n{point_type}:")
        if point_type not in data or not data[point_type]:
            print("  No data found")
            continue
        
        sorted_infills = sorted(data[point_type].keys())
        for infill_num in sorted_infills:
            if infill_num == 0:
                print(f"  DOE (hf_doe):")
            else:
                print(f"  Infill {infill_num} (high_infill_{infill_num-1}):")
            
            candidates = data[point_type][infill_num]
            for candidate in sorted(candidates.keys()):
                angle = candidates[candidate]
                print(f"    {candidate}: {angle:10.6f} deg")


def main():
    """Main function."""
    # Set the output directory
    output_dir = '/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper'
    
    # Check if directory exists
    if not os.path.exists(output_dir):
        print(f"Error: Directory {output_dir} does not exist!")
        return
    
    print(f"Scanning directory: {output_dir}")
    print("\n" + "="*60)
    print("COLLECTING DATA FROM aero-optim.log FILES")
    print("="*60)
    
    # Collect angles from all subdirectories
    high_data, low_data = collect_angles_from_directory(output_dir)

    print(high_data)
    print(low_data)
    
    # Print summary
    print_summary(high_data)
    
    # # Plot the results
    # if any(high_data.values()):
    #     plot_angles(high_data, output_dir)
    # else:
    #     print("\nNo data found to plot!")
    
    # NEW: Collect and plot QoI convergence data
    print("\n" + "="*60)
    print("COLLECTING DATA FROM QoI_convergence.csv FILES")
    print("="*60)
    qoi_data = collect_qoi_convergence_data(output_dir)

    print("\n" + "="*60)
    print("QoI CONVERGENCE DATA SUMMARY")
    print("="*60)
    for point_type in ['ADP', 'OP1', 'OP2']:
        print(f"\n{point_type}:")
        if point_type not in qoi_data or not qoi_data[point_type]:
            print("  No data found")
            continue
        
        sorted_infills = sorted(qoi_data[point_type].keys())
        for infill_num in sorted_infills:
            if infill_num == 0:
                print(f"  DOE:")
            else:
                print(f"  Infill {infill_num-1}:")
            
            candidates = qoi_data[point_type][infill_num]
            for candidate in sorted(candidates.keys()):
                cdata = candidates[candidate]
                print(f"    {candidate}: mean_angle={cdata['mean_angle']:8.4f}, range=[{cdata['min_angle']:8.4f}, {cdata['max_angle']:8.4f}], mean_loss={cdata['mean_loss']:.6f}")
    
    plot_qoi_convergence(qoi_data, output_dir)
    plot_loss_coef_convergence(qoi_data, output_dir)

    
    # Plot high vs low fidelity comparison
    if any(high_data.values()) and any(low_data.values()):
        plot_high_vs_low_fidelity(high_data, low_data, output_dir)
    else:
        print("\nNo paired high/low fidelity data found for comparison plot!")


if __name__ == '__main__':
    plt.close('all')
    main()
