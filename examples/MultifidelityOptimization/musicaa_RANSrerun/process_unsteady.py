# Converted from process_unsteady.ipynb
# --- code cells below ---


from IPython.display import HTML
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os

from musicaa_utils_orig import get_block_info, get_min_max, read_grid, read_info, read_planes, write_para


# plot params
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "Times"
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8

input_dir = "ADP/init_2D"
output_dir = "ADP_out"
var_list = ["pres", "Mach"]

input_dit = '/examples/LRN-CASCADE/cascade_musicaa_base/output_baseline/MUSICAA/musicaa_g0_c0/ADP/init_2D'
print('baby')

dict_info = read_info(input_dir)
block_info = get_block_info(input_dir)

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
    data[bl]["planes"], n_ckpt = read_planes(os.path.join(input_dir, f"plane_001_bl{bl}.bin"), dict_info, block_info[f"block_{bl}"]["plane_1"]["var_list"])

print(f"The extracted variables: {block_info[f'block_{bl}']['plane_1']['var_list']}, the number of checkpoints: {n_ckpt}")

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

fig, ax = plt.subplots(figsize=(5.2, 3.64))
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
cmap = plt.cm.viridis

for var in var_list:
    vmin, vmax = get_min_max(var, data, n_block, n_ckpt)

    # create pcolormesh objects for each block
    blocks = []
    for bl in range(1, n_block + 1):
        Z_initial = data[bl]["planes"][1][var]
        cmesh = ax.pcolormesh(data[bl]['x'], data[bl]['y'], Z_initial, vmin=vmin,vmax=vmax, cmap=cmap)
        blocks.append(cmesh)

    # update the plot at each frame
    def update_plot(n):
        for bl in range(1, n_block + 1):
            Z =  data[bl]["planes"][n + 1][var]
            blocks[bl - 1].set_array(Z.flatten())
        return blocks

    # create the animation
    ani = animation.FuncAnimation(fig, update_plot, frames=range(n_ckpt), interval=200, repeat=False)

    # plot sensors
    s_id = 1
    epsilon = 1.
    for bl in range(1, n_block + 1):
        for x_id, y_id in sensor[bl]:
            ax.scatter(data[bl]['x'][x_id, y_id], data[bl]['y'][x_id, y_id], marker="o", s=5, c="k")
            xy =(data[bl]['x'][x_id, y_id] + epsilon, data[bl]['y'][x_id, y_id] + epsilon)
            ax.annotate(f"{s_id}", xy=xy)
            s_id += 1

    fig.tight_layout()
    plt.close(fig)
    HTML(ani.to_jshtml())
    anim_path = os.path.join(output_dir, f'{var}_field.mp4')
    print(f"save {var} animation to {anim_path}")
    ani.save(anim_path)


write_para(input_dir, output_dir, plane_nb=1, var_names=block_info[f"block_1"]["plane_1"]["var_list"], plane=True)

# Auto-added save (no plt.show() found)
plt.savefig(r'process_unsteady.png', dpi=150, bbox_inches='tight')
