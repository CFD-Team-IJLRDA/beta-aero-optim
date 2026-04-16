#!/usr/bin/env python3
"""
Unsteady post-processing script

This script performs the post-processing of unsteady data saved at "plane" snapshots.
More specifically, it goes through the following steps:
- reads the data saved at "plane" snapshots
- for each specified variable:
    - compute the min/max values of the specified variable
    - saves the corresponding animation
- converts the data to .vtk files readable with paraview.

Notes: the script relies on adjustments of functions and processing methods written 
by J. Liu, C. Matar, and L. Zemmour.
"""

from IPython.display import HTML
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os

from musicaa_utils import get_block_info, get_min_max, read_grid, read_info, read_planes, write_para


# Plot params
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "Times"
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8


def main():
    """Main function to perform unsteady post-processing."""
    
    # ============================================================================
    # 1. Data processing
    # ============================================================================
    # The input variables are:
    # * input_dir: the path to the directory containing the data
    # * output_dir: the path to the folder where processed results should be saved.
    # * var_list: the list of variables to be plotted
    # 
    # Note: it is assumed that we are looking at a unique plane (i.e. "plane_1" 
    # in block_info) and that the folder output_dir exists.
    
    input_dir = "./output/MUSICAA/musicaa_g0_c0/ADP/init_2D"
    output_dir = "temp"
    var_list = ["Mach"]
    
    # Extract the simulation and blocks information
    print("Reading simulation information...")
    dict_info = read_info(input_dir)
    block_info = get_block_info(input_dir)
    
    # For each block, the grid coordinates (x, y, z) and plane data are extracted 
    # and saved in data[bl]
    # Notes: ngh is the number of ghost cells and num_blocks is the total number 
    # of blocks. The blocks are numbered from 1 to num_blocks.
    
    print("Reading grid and plane data...")
    data = {}
    ngh = int(dict_info["ngh"])
    n_block = dict_info["nbloc"]
    
    for bl in range(1, n_block + 1):
        data[bl] = {}
        # read grid
        bl_file = os.path.join(input_dir, f'grid_bl{bl}_ngh{ngh}.bin')
        nx, ny, nz, x, y, z = read_grid(input_dir, bl_file)
        # scale grid
        data[bl]["x"], data[bl]["y"], data[bl]["z"] = x, y, z
        # read data
        data[bl]["planes"], n_ckpt = read_planes(
            os.path.join(input_dir, f"plane_001_bl{bl}.bin"), 
            dict_info, 
            block_info[f"block_{bl}"]["plane_1"]["var_list"]
        )
    
    print(f"The extracted variables: {block_info[f'block_{bl}']['plane_1']['var_list']}, "
          f"the number of checkpoints: {n_ckpt}")
    
    # The sensors coordinates are also extracted for each block
    print("Extracting sensor coordinates...")
    sensor = {}
    for bl in range(1, n_block + 1):
        sensor[bl] = []
        nb_pt = block_info[f"block_{bl}"]["nb_points"]
        nb_li = block_info[f"block_{bl}"]["nb_lines"]
        if nb_pt > 0:
            for pt in range(1, nb_pt + 1):
                xs = block_info[f"block_{bl}"][f"point_{pt}"]["nx1"]
                ys = block_info[f"block_{bl}"][f"point_{pt}"]["ny1"]
                sensor[bl].append([xs, ys])
                print(f"point in block {bl} at indexes {xs, ys}")
        if nb_li > 0:
            for li in range(1, nb_li + 1):
                xs = block_info[f"block_{bl}"][f"line_{li}"]["nx1"]
                ys = block_info[f"block_{bl}"][f"line_{li}"]["ny1"]
                sensor[bl].append([xs, ys])
                print(f"line in block {bl} at indexes {xs, ys}")
    
    # Animations are produced and saved for each variable in var_list
    # Notes: the animation can be parameterized in multiple ways by changing 
    # the figure size, the animation interval, etc.
    
    print("Creating animations...")
    fig, ax = plt.subplots(figsize=(5.2, 3.64))
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    cmap = plt.cm.viridis
    
    for var in var_list:
        print(f"Processing variable: {var}")
        vmin, vmax = get_min_max(var, data, n_block, n_ckpt)
    
        # create pcolormesh objects for each block
        blocks = []
        for bl in range(1, n_block + 1):
            Z_initial = data[bl]["planes"][1][var]
            cmesh = ax.pcolormesh(data[bl]['x'], data[bl]['y'], Z_initial, 
                                 vmin=vmin, vmax=vmax, cmap=cmap)
            blocks.append(cmesh)
    
        # update the plot at each frame
        def update_plot(n):
            for bl in range(1, n_block + 1):
                Z = data[bl]["planes"][n + 1][var]
                blocks[bl - 1].set_array(Z.flatten())
            return blocks
    
        # create the animation
        ani = animation.FuncAnimation(fig, update_plot, frames=range(n_ckpt), 
                                     interval=200, repeat=False)
    
        # plot sensors
        s_id = 1
        epsilon = 1.
        for bl in range(1, n_block + 1):
            for x_id, y_id in sensor[bl]:
                ax.scatter(data[bl]['x'][x_id, y_id], data[bl]['y'][x_id, y_id], 
                          marker="o", s=5, c="k")
                xy = (data[bl]['x'][x_id, y_id] + epsilon, 
                     data[bl]['y'][x_id, y_id] + epsilon)
                ax.annotate(f"{s_id}", xy=xy)
                s_id += 1
    
        fig.tight_layout()
        anim_path = os.path.join(output_dir, f'{var}_field.mp4')
        print(f"Saving {var} animation to {anim_path}")
        ani.save(anim_path)
        
        # Clear the axis for next variable
        ax.clear()
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
    
    plt.show()
    plt.close(fig)
    
    # The plane n°1 results are converted to .vtk format and saved to output_dir
    print("Converting to VTK format...")
    write_para(input_dir, output_dir, plane_nb=1, 
              var_names=block_info[f"block_1"]["plane_1"]["var_list"], 
              plane=True)
    
    print("Post-processing complete!")


if __name__ == '__main__':
    main()
