#!/usr/bin/env python3
"""
Script to plot airfoil profile from ogv1c.dat file.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def read_airfoil_dat(file_path):
    """
    Read airfoil coordinates from .dat file.
    
    Args:
        file_path: Path to the .dat file
        
    Returns:
        tuple: (x_coords, y_coords) arrays
    """
    x_coords = []
    y_coords = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    x_coords.append(x)
                    y_coords.append(y)
                except ValueError:
                    continue
    
    return np.array(x_coords), np.array(y_coords)


def plot_airfoil(x, y, title='Airfoil Profile', save_path='airfoil_profile.png'):
    """
    Plot airfoil profile.
    
    Args:
        x: x-coordinates
        y: y-coordinates
        title: Plot title
        save_path: Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    # Plot the airfoil
    plt.plot(x, y, color='black', linewidth=2, label='Airfoil Profile')
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    # plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # print(f"Plot saved to: {save_path}")
    plt.show()


def main():
    """Main function."""
    # Set the file path
    airfoil_file = Path.home() / 'GPROfficial/beta-aero-optim/Optimization/RANS_bruteForce/ogv1c.dat'
    
    # Check if file exists
    if not airfoil_file.exists():
        print(f"Error: File {airfoil_file} does not exist!")
        return
    
    print(f"Reading airfoil profile from: {airfoil_file}")
    
    # Read airfoil coordinates
    # x1, y1 = read_airfoil_dat('/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_baseline/FFD/ogv1c_g0_c0.dat')
    x1, y1 = read_airfoil_dat('/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/RANS_bruteForce/ogv1c.dat')
    x2, y2 = read_airfoil_dat('/home/mciarlatani/Irene/validation_cases/deltaP_n10/FFD/ogv1c_g0_c0.dat')
    
    print(f"Number of points: {len(x1)}")
    print(f"x range: [{x1.min():.4f}, {x1.max():.4f}]")
    print(f"y range: [{y1.min():.4f}, {y1.max():.4f}]")
    
    # Plot the airfoil
    # plot_airfoil(x1, y1, title='OGV1C Baseline Profile')
    plt.plot(x1, y1, label='Baseline') 
    plt.plot(x2, y2, label='DeltaP')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
