"""
Stats post-processing

This script performs the post-processing of the statistical data. More specifically, it goes through the following steps:
1. reads the `stats1` and `stats2` files of each block and plots any variable
2. computes the total pressure field across the domain, the measurement planes and the mixed out loss
3. plots the isentropic Mach and loss coefficient distributions
4. computes the inlet/outlet angles of the flow.

Notes: the script relies on adjustments of functions and processing methods written by J. Liu, C. Matar, and L. Zemmour.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from musicaa_utils import get_block_info, line_interp, mixed_out, plot_grid, read_grid, read_info, read_stats

def read_case(input_dir, in_blocks, out_blocks, plot_mesh=False):

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

    # Plot mesh
    if plot_mesh is True:

        plot_grid(input_dir, True, n_bl=n_block, every=5, figsize=figsize)

    # The sensors coordinates are also extracted for each block

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

    # Statistics dictionary are created and filled with each block's data

    stats1 = {}
    stats2 = {}
    for bl in range(1, n_block + 1):
        nx, ny = block_info[f"block_{bl}"]["nx"], block_info[f"block_{bl}"]["ny"]
        stats1[bl] = read_stats(os.path.join(input_dir, f"stats1_bl{bl}.bin"), nx, ny)
        stats2[bl] = read_stats(os.path.join(input_dir, f"stats2_bl{bl}.bin"), nx, ny)

    # union of data
    in_data = {}
    for bl in in_blocks:
        in_data[bl] = data[bl] | stats1[bl] | stats2[bl]

    out_data = {}
    for bl in out_blocks:
        out_data[bl] = data[bl] | stats1[bl] | stats2[bl]

    # The limits along the y axis of both measurement planes are computed 

    # find y corresponding to x1
    x0 = in_data[in_blocks[0]]["x"]
    closest_index = np.argmin(abs(x0[:, 0] - x1))
    y1 = in_data[in_blocks[0]]["y"][closest_index, :].min()
    y2 = y1 + pitch
    # compute interpolation axis
    y_in = np.linspace(y1, y2, 1000)
    # build inlet_lims
    inlet_lims = [x1, y1, x1, y2]
    print(f"inlet_lims: {inlet_lims}")

    # find y corresponding to x1
    x0 = out_data[out_blocks[0]]["x"]
    closest_index = np.argmin(abs(x0[:, 0] - x2))
    y1 = out_data[out_blocks[0]]["y"][closest_index, :].min()
    y2 = y1 + pitch
    # compute interpolation axis
    y_out = np.linspace(y1, y2, 1000)
    outlet_lims = [x2, y1, x2, y2]
    print(f"outlet_lims: {outlet_lims}")


    data_inlet = {}
    data_outlet = {}
    for var in ["uu", "vv", "ww", "rhou", "rhov", "rho*uu", "rho*uv", "rho*uw", "p", "T", "M", "cp", "cv"]:
        data_inlet[f"{var}_interp"] = line_interp(in_data, var, inlet_lims, in_blocks)
        data_outlet[f"{var}_interp"] = line_interp(out_data, var, outlet_lims, out_blocks)

    results = {'data_inlet': data_inlet,
               'data_outlet': data_outlet,
               'stats1': stats1,
               'stats2': stats2,
               'data': data,
               'inlet_lims': inlet_lims,
               'outlet_lims': outlet_lims,
               'y_out': y_out,
               'n_block': n_block,
               'sensor': sensor}

    return results

def compute_total_pressure(data, stats1, stats2, inlet_lims, outlet_lims, n_block):

    # The total Pressure field is first computed

    pres_tot = {bl:[] for bl in range(1, n_block + 1)}

    for bl in range(1, n_block + 1):
        cp = stats2[bl]['cp']
        cv = stats2[bl]['cv']
        gamma = cp / cv
        Mach = stats2[bl]['M']
        pres = stats1[bl]['p']
        pres_tot[bl] = pres * (1 + (gamma - 1) / 2 * Mach**2)**(gamma / (gamma - 1))

    in_data = {}
    for bl in in_blocks:
        in_data[bl] = data[bl] | stats1[bl] | stats2[bl]

    out_data = {}
    for bl in out_blocks:
        out_data[bl] = data[bl] | stats1[bl] | stats2[bl]

    # The data is interpolated along the measurement planes

    data_inlet = {}
    data_outlet = {}
    for var in ["uu", "vv", "ww", "rhou", "rhov", "rho*uu", "rho*uv", "rho*uw", "p", "T", "M", "cp", "cv"]:
        data_inlet[f"{var}_interp"] = line_interp(in_data, var, inlet_lims, in_blocks)
        data_outlet[f"{var}_interp"] = line_interp(out_data, var, outlet_lims, out_blocks)

    # The mixed out loss coefficient is computed

    # add gam and R entries as required by mixed_out
    data_inlet["gam"] = np.nanmean(data_inlet["cp_interp"] / data_inlet["cv_interp"])
    data_inlet["R"] = np.nanmean(data_inlet["cp_interp"] - data_inlet["cv_interp"])
    data_outlet["gam"] = data_inlet["gam"]
    data_outlet["R"] = data_inlet["R"]
    # compute inlet/outlet mixed out states
    inlet_mixed_out_state = mixed_out(data_inlet)
    outlet_mixed_out_state = mixed_out(data_outlet)
    # compute mixed out loss
    mo_loss = (inlet_mixed_out_state["p0_bar"] - outlet_mixed_out_state["p0_bar"]) / (inlet_mixed_out_state["p0_bar"] - inlet_mixed_out_state["p_bar"])
    print(f"Mixed out loss: {mo_loss}")

    # =============================================================================
    # 3. Loss coefficient and isentropic Mach distributions
    # =============================================================================
    # The additional input variables are:
    # * `path_to_loss` the path to the loss measurements
    # * `path_to_mis` the path to the isentropic Mach measurements
    # * `wall_blocks` the list of blocks encompassing the geometry walls.

    in_idx =  np.argwhere(~np.isnan(data_inlet["rhou_interp"]))
    q1 = np.sum(data_inlet["rhou_interp"][in_idx])
    P1 = np.sum(data_inlet["rhou_interp"][in_idx] * data_inlet["p_interp"][in_idx]) / q1
    gamma = np.mean(data_inlet["cp_interp"][in_idx] / data_inlet["cv_interp"][in_idx])
    P01 = np.sum(data_inlet["p_interp"][in_idx] * (1 + (gamma - 1 ) / 2 * data_inlet["M_interp"][in_idx]**2)**(gamma / (gamma - 1)) * data_inlet["rhou_interp"][in_idx]) / q1

    out_idx =  np.argwhere(~np.isnan(data_outlet["rhou_interp"]))
    q2 = np.sum(data_outlet["rhou_interp"][out_idx])
    P2 = np.sum(data_outlet["rhou_interp"][out_idx] * data_outlet["p_interp"][out_idx]) / q2
    gamma = np.mean(data_outlet["cp_interp"][out_idx] / data_outlet["cv_interp"][out_idx])
    print(f"gamma: {gamma}")
    P02 = np.sum(data_outlet["p_interp"][out_idx] * (1 + (gamma - 1 ) / 2 * data_outlet["M_interp"][out_idx]**2)**(gamma / (gamma - 1)) * data_outlet["rhou_interp"][out_idx]) / q2

    print(f"Loss coefficient w: {(P01 - P02) / (P01 - P1)}")

    loss = (P01 - data_outlet["p_interp"][out_idx] * (1 + (gamma - 1 ) / 2 * data_outlet["M_interp"][out_idx]**2)**(gamma / (gamma - 1))) / (P01 - P1)
    delta= 0.0225

    return out_idx, loss, P01, gamma

# plot params
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "Times"
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8
figsize = (5.2, 3.64)

path_to_mis = "cascade_mis.dat"

x1 = -20.108296
x2 = 87.25188
pitch = 40.39
in_blocks = [1, 2]
out_blocks = [8, 9]

wall_blocks = [3, 4, 7, 6]


# =============================================================================
# 1. Data processing
# =============================================================================

# input_dir_ADP1 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_0/high_infill_0/MUSICAA/musicaa_g0_c0/ADP" # "ADP"
# input_dir_OP1 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_0/high_infill_0/MUSICAA/musicaa_g0_c0/OP1" # "OP1"
# input_dir_OP2 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_0/high_infill_0/MUSICAA/musicaa_g0_c0/OP2" # "OP2"

# input_dir_ADP2 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_6/high_infill_6/MUSICAA/musicaa_g0_c0/OP1" # "ADP"
# input_dir_OP1 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_6/high_infill_6/MUSICAA/musicaa_g0_c0/OP1" # "OP1"
# input_dir_OP2 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_6/high_infill_6/MUSICAA/musicaa_g0_c0/OP2" # "OP2"

# input_dir_ADP3 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_3/high_infill_3/MUSICAA/musicaa_g0_c0/ADP" # "ADP"
# input_dir_OP1 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_3/high_infill_3/MUSICAA/musicaa_g0_c0/OP1" # "OP1"
# input_dir_OP2 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_3/high_infill_3/MUSICAA/musicaa_g0_c0/OP2" # "OP2"

# input_dir_ADP3 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_3/high_infill_3/MUSICAA/musicaa_g0_c0/ADP" # "ADP"
# input_dir_ADPb = "/home/mciarlatani/Hilbert/aero-optim/examples/LRN-CASCADE/cascade_musicaa_base/output_baseline/MUSICAA/musicaa_g0_c0/ADP" # "ADP"

input_dir_ADP2 = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_0/high_infill_0/MUSICAA/musicaa_g0_c0/OP1" # "ADP"

# ADP_1 = read_case(input_dir_ADP1, in_blocks, out_blocks)
ADP_2 = read_case(input_dir_ADP2, in_blocks, out_blocks)
# ADP_3 = read_case(input_dir_ADP3, in_blocks, out_blocks)
# ADP_b = read_case(input_dir_ADPb, in_blocks, out_blocks)

# out_idx_ADP1, loss_ADP1, _, _ = compute_total_pressure(ADP_1['data'], ADP_1['stats1'], ADP_1['stats2'], ADP_1['inlet_lims'], ADP_1['outlet_lims'], ADP_1['n_block'])
out_idx_ADP2, loss_ADP2, P01_ADP2, gamma = compute_total_pressure(ADP_2['data'], ADP_2['stats1'], ADP_2['stats2'], ADP_2['inlet_lims'], ADP_2['outlet_lims'], ADP_2['n_block'])
# out_idx_ADP3, loss_ADP3, _, _ = compute_total_pressure(ADP_3['data'], ADP_3['stats1'], ADP_3['stats2'], ADP_3['inlet_lims'], ADP_3['outlet_lims'], ADP_3['n_block'])
# out_idx_ADPb, loss_ADPb, _, _ = compute_total_pressure(ADP_b['data'], ADP_b['stats1'], ADP_b['stats2'], ADP_b['inlet_lims'], ADP_b['outlet_lims'], ADP_b['n_block'])

fig, ax = plt.subplots(figsize=figsize)
# ax.plot(ADP_1['y_out'][out_idx_ADP1] / 1000, loss_ADP1, label="LES 1")
ax.plot(ADP_2['y_out'][out_idx_ADP2] / 1000, loss_ADP2, label="LES 2")
# ax.plot(ADP_3['y_out'][out_idx_ADP3] / 1000, loss_ADP3, label="LES 3")
# ax.plot(ADP_b['y_out'][out_idx_ADPb] / 1000, loss_ADPb, label="Baseline")
ax.legend()
ax.set_xlabel("$\\bar{y}$")
ax.set_ylabel("$w$ [-]")
plt.show()

# # The loss data is saved

# outfile = f"loss_{output_key}.csv"
# loss_df = pd.DataFrame(np.column_stack([y_out[out_idx] / 1000, loss]), columns=["y", "loss"])
# # loss_df.to_csv(outfile, index=False)

# The data is extracted along the wall

pres_wall_list = []
x_list = []
y_list = []

for bl in wall_blocks:
    new_pres_value = ADP_2['stats1'][bl]['p'][:, 0]
    pres_wall_list.append(new_pres_value)
    
    new_x_value = ADP_2['data'][bl]['x'][:, 0]
    x_list.append(new_x_value)

    new_y_value = ADP_2['data'][bl]['y'][:, 0]
    y_list.append(new_y_value)

    
pres_wall = np.concatenate(pres_wall_list)
x_wall = np.concatenate(x_list) / 1000.
y_wall = np.concatenate(y_list) / 1000.
# x_tilde = np.cos(np.arctan(y_wall / x_wall) - (106.04 - 90) / 180 * np.pi) * np.sqrt(x_wall**2 + y_wall**2)

# The isentropic Mach is computed

Mach_is = np.sqrt(((P01_ADP2 / pres_wall)**((gamma - 1) / gamma) - 1) * 5)

exp_mis = np.loadtxt(path_to_mis, skiprows=1)

fig, ax = plt.subplots(figsize=figsize)
ax.plot(x_wall / 0.067, Mach_is, color="blue", label="LES")
ax.scatter(exp_mis[:, 0] / 0.067, exp_mis[:, 1], color="k", label="DLR exp.")
ax.legend()
ax.set_ylim(0.3, 1.)
ax.set_xlabel('$\\bar{x}$ [-]')
ax.set_ylabel('$Mis$ [-]')

plt.show()

# The isentropic Mach data is saved

# outfile = f"mis_{output_key}.csv"
# mis_df = pd.DataFrame(np.column_stack([x_wall, y_wall, x_wall / 0.067, Mach_is]), columns=["x", "y", "x/cax", "mis"])
# # mis_df.to_csv(outfile, index=False)

# =============================================================================
# 4. Outflow/inflow angles
# =============================================================================

u_mean = np.nanmean(data_outlet["rhou_interp"])
v_mean = np.nanmean(data_outlet["rhov_interp"])
print(f"Outflow angle: {np.atan(v_mean / u_mean) / np.pi * 180} deg.")

u_mean = np.nanmean(data_inlet["rhou_interp"])
v_mean = np.nanmean(data_inlet["rhov_interp"])
print(f"Inflow angle: {np.atan(v_mean / u_mean) / np.pi * 180} deg.")

angle = np.nanmean(np.arctan(data_outlet["rhov_interp"] / data_outlet["rhou_interp"]))
print(f"Outflow angle: {angle / np.pi * 180} deg.")

# =============================================================================
# Paraview stats (extra step for paper)
# =============================================================================

from musicaa_utils import write_para
para_dir = "ADP_stats_para"
os.makedirs(os.path.join(input_dir, para_dir, "par_planes"))
write_para(input_dir, para_dir, plane_nb=-1, var_names="", stats=True)

plt.show()
