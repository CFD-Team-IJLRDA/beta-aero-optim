#!/usr/bin/env python3
"""
Script to read and explore HDF5 file contents.

This script reads the DB_random_DoF15.h5 file and displays its structure,
datasets, and attributes.
"""

import h5py
import numpy as np
import os


def print_structure(name, obj):
    """Callback function to print HDF5 structure."""
    indent = "  " * name.count('/')
    if isinstance(obj, h5py.Group):
        print(f"{indent}Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}Dataset: {name}")
        print(f"{indent}  Shape: {obj.shape}")
        print(f"{indent}  Dtype: {obj.dtype}")


def explore_h5_file(filepath):
    """
    Open and explore an HDF5 file.
    
    Args:
        filepath: Path to the HDF5 file
    """
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found!")
        return
    
    print(f"Opening HDF5 file: {filepath}")
    print("=" * 80)
    
    with h5py.File(filepath, 'r') as f:
        # Print file attributes
        print("\nFile Attributes:")
        for attr_name, attr_value in f.attrs.items():
            print(f"  {attr_name}: {attr_value}")
        
        # Print structure
        print("\nFile Structure:")
        f.visititems(print_structure)
        
        # Print keys at root level
        print("\nRoot level keys:")
        for key in f.keys():
            print(f"  - {key}")
        
        # Explore each dataset/group
        print("\n" + "=" * 80)
        print("Detailed Information:")
        print("=" * 80)
        
        for key in f.keys():
            print(f"\n[{key}]")
            obj = f[key]
            
            if isinstance(obj, h5py.Dataset):
                print(f"  Type: Dataset")
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
                print(f"  Size: {obj.size} elements")
                
                # Print attributes
                if obj.attrs:
                    print(f"  Attributes:")
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"    {attr_name}: {attr_value}")
                
                # Print sample data for small datasets
                if obj.size > 0:
                    if obj.size <= 20:
                        print(f"  Data:\n{obj[:]}")
                    else:
                        print(f"  First few elements: {obj[:5]}")
                        if len(obj.shape) == 2 and obj.shape[0] > 5:
                            print(f"  Shape of data: {obj.shape}")
                            print(f"  First row: {obj[0, :]}")
                            print(f"  Last row: {obj[-1, :]}")
            
            elif isinstance(obj, h5py.Group):
                print(f"  Type: Group")
                print(f"  Contains {len(obj.keys())} items:")
                for subkey in obj.keys():
                    subobj = obj[subkey]
                    if isinstance(subobj, h5py.Dataset):
                        print(f"    - {subkey}: Dataset {subobj.shape} {subobj.dtype}")
                    else:
                        print(f"    - {subkey}: Group")


def main():
    """Main function to read DB_random_DoF15.h5."""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    h5_file = os.path.join(script_dir, "DB_opt_DoF15.h5")
    # print(h5_file["free_variable"])
    # explore_h5_file(h5_file)
    
    with h5py.File(h5_file, 'r') as f:
        # Print file attributes
        print("\nFile Attributes:")
        for attr_name, attr_value in f.attrs.items():
            print(f"  {attr_name}: {attr_value}")
    
        obj = f['free_variable']

        print(f"  Type: Dataset")
        print(f"  Shape: {obj.shape}")
        print(f"  Dtype: {obj.dtype}")
        print(f"  Size: {obj.size} elements")

        print(f"  Attributes:")
        for attr_name, attr_value in obj.attrs.items():
            print(f"    {attr_name}: {attr_value}")

        for i in range(obj.shape[0]):
            print(f"{attr_value[i]} in {[np.min(obj[i, :]), np.max(obj[i, :])]}")


if __name__ == '__main__':
    main()
