import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import pandas as pd

from scipy.interpolate import splprep, splev
from scipy.interpolate import Rbf

from musicaa_utils import get_block_info, line_interp, mixed_out, plot_grid, read_grid, read_info, read_stats

def rearrange(x,d):
    len_ = d['nx_bl3']+d['nx_bl4']
    # save bl6 coords
    x_ = np.zeros(d['nx_bl6'])
    x_ = x[len_:len_+d['nx_bl6']].copy()
    # rotate array
    x[len_:len_+d['nx_bl7']] = x[-d['nx_bl7']:].copy()
    x[-d['nx_bl6']:] = x_.copy()

    return x

def read_case(input_dir, dict_input, plot_mesh=False):

    pitch      = dict_input['pitch']
    x1_samp    = dict_input['x1']
    x2_samp    = dict_input['x2']
    in_blocks  = dict_input['in_blocks']
    out_blocks = dict_input['out_blocks']
    wall_blocks= dict_input['wall_blocks']

    dict_info = read_info(input_dir)
    block_info = get_block_info(input_dir)
    ngh = int(dict_info["ngh"])
    n_block = dict_info["nbloc"]
    Re_in  = dict_info['Reref']
    u_in   = dict_info['Uref']
    p_in   = dict_info['Pref']
    rho_in = dict_info['Roref']
    mu_in  = dict_info['Muref']
    c_in   = dict_info['cref']
    nz     = dict_info['nz_bl1']

    print('Input case info: Re = ', Re_in, ', U = ', u_in, ', p = ', p_in, ', rho = ', rho_in, ', mu = ', mu_in, ', c = ', c_in)

    data = {}
    stats1 = {}
    stats2 = {}
    x_flat   = []
    y_flat   = []

    for bl in range(1, n_block + 1):
        data[bl] = {}
        # read grid
        bl_file = os.path.join(input_dir, f'grid_bl{bl}_ngh{ngh}.bin')
        nx, ny, nz, x, y, z = read_grid(input_dir, bl_file)
        # scale grid
        data[bl]["x"], data[bl]["y"], data[bl]["z"] = x, y, z

        if bl in wall_blocks:
            block_info[f'block_{bl}']['wall'] = True
        else:
            block_info[f'block_{bl}']['wall'] = False

        # Wall location
        block_info[f'block_{bl}']['jmin'] = True
        block_info[f'block_{bl}']['imax'] = False

    # Plot mesh
    if plot_mesh is True:
        plot_grid(input_dir, True, n_bl=n_block, every=5, figsize=(5.2, 3.64))
    
    nw = sum([dict_info[f'nx_bl{bl}'] for bl in wall_blocks])
    
    x1,xw,x1_     = np.zeros((nw)),np.zeros((nw)),np.zeros((nw))
    y1,yw,y1_     = np.zeros((nw)),np.zeros((nw)),np.zeros((nw))
    duxw,duyw,dxw = np.zeros((nw)),np.zeros((nw)),np.zeros((nw))
    dvxw,dvyw,dyw = np.zeros((nw)),np.zeros((nw)),np.zeros((nw))
    rhow,muw,pw   = np.zeros((nw)),np.zeros((nw)),np.zeros((nw))
    tau = np.zeros((nw,2,2))

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
    
    nw_loc = 0
    for bl in range(1, n_block + 1):

        nx, ny = block_info[f"block_{bl}"]["nx"], block_info[f"block_{bl}"]["ny"]
        stats1[bl] = read_stats(os.path.join(input_dir, f"stats1_bl{bl}.bin"), nx, ny)
        stats2[bl] = read_stats(os.path.join(input_dir, f"stats2_bl{bl}.bin"), nx, ny)

        # # isentropic Mach number
        # data[f'block_{bl}']['M_is'] = np.sqrt(2/(gam_in-1)*\
        #       ((p0_in/stats1[bl] ['p'])**((gam_in-1)/gam_in)-1))
        # M_is_flat.append(data[f'block_{bl}']['M_is'].flatten())
        
        data[bl]['rho'] = stats1[bl]['rho']
        data[bl]['x_flat'] = data[bl]['x'].flatten()
        data[bl]['y_flat'] = data[bl]['y'].flatten()

        x_flat.append(data[bl]['x'].flatten())
        y_flat.append(data[bl]['y'].flatten())

        if block_info[f'block_{bl}']['wall']:
            if block_info[f'block_{bl}']['jmin']:
                # Wall coordinates
                xw[nw_loc:nw_loc+nx]  = data[bl]['x'][:,0]
                yw[nw_loc:nw_loc+nx]  = data[bl]['y'][:,0]
                # Compute first cell height
                x1_[nw_loc:nw_loc+nx] = data[bl]['x'][:,1]
                y1_[nw_loc:nw_loc+nx] = data[bl]['y'][:,1]
                # Compute wall tangential distance from cell to cell
                dxw[nw_loc:nw_loc+nx] = np.hstack((data[bl]['x'][1:,0]-data[bl]['x'][:-1,0],data[bl]['x'][-2,0]-data[bl]['x'][-1,0]))
                dyw[nw_loc:nw_loc+nx] = np.hstack((data[bl]['y'][1:,0]-data[bl]['y'][:-1,0],data[bl]['y'][-2,0]-data[bl]['y'][-1,0]))
                if bl==6:
                    yw[nw_loc:nw_loc+nx] +=-pitch
                    y1_[nw_loc:nw_loc+nx]+=-pitch
                y1[nw_loc:nw_loc+nx]    = (np.sqrt((x1_-xw)**2+(y1_-yw)**2))[nw_loc:nw_loc+nx]

                # # Cf
                # rhow[nw_loc:nw_loc+nx] = stats1[bl]['rho'][:,0]
                # muw[nw_loc:nw_loc+nx]  = stats1[bl]['mu'][:,0]
                # duxw[nw_loc:nw_loc+nx] = stats2[bl]['rho*dux'][:,0]
                # duyw[nw_loc:nw_loc+nx] = stats2[bl]['rho*duy'][:,0]
                # dvxw[nw_loc:nw_loc+nx] = stats2[bl]['rho*dvx'][:,0]
                # dvyw[nw_loc:nw_loc+nx] = stats2[bl]['rho*dvy'][:,0]

                # # In cartesian coordinates
                # tau[nw_loc:nw_loc+nx,0,0] = (muw*2*duxw/rhow)[nw_loc:nw_loc+nx]
                # tau[nw_loc:nw_loc+nx,0,1] = (muw*(duyw+dvxw)/rhow)[nw_loc:nw_loc+nx]
                # tau[nw_loc:nw_loc+nx,1,0] = (muw*(duyw+dvxw)/rhow)[nw_loc:nw_loc+nx]
                # tau[nw_loc:nw_loc+nx,1,1] = (muw*2*dvyw/rhow)[nw_loc:nw_loc+nx]

                # Cf
                muw[nw_loc:nw_loc+nx]  = stats1[bl]['mu'][:,0]
                duxw[nw_loc:nw_loc+nx] = np.sqrt(stats2[bl]['dux**2'][:,0])
                duyw[nw_loc:nw_loc+nx] = np.sqrt(stats2[bl]['duy**2'][:,0])
                dvxw[nw_loc:nw_loc+nx] = np.sqrt(stats2[bl]['dvx**2'][:,0])
                dvyw[nw_loc:nw_loc+nx] = np.sqrt(stats2[bl]['dvy**2'][:,0])

                # In cartesian coordinates
                tau[nw_loc:nw_loc+nx,0,0] = (muw*2*duxw)[nw_loc:nw_loc+nx]
                tau[nw_loc:nw_loc+nx,0,1] = (muw*(duyw+dvxw))[nw_loc:nw_loc+nx]
                tau[nw_loc:nw_loc+nx,1,0] = (muw*(duyw+dvxw))[nw_loc:nw_loc+nx]
                tau[nw_loc:nw_loc+nx,1,1] = (muw*2*dvyw)[nw_loc:nw_loc+nx]

                pw[nw_loc:nw_loc+nx] = stats1[bl]['p'][:,0]

                nw_loc+=nx



    xw = rearrange(xw,dict_info)
    yw = rearrange(yw,dict_info)
    pw = rearrange(pw,dict_info)
    tau = rearrange(tau,dict_info)

    xPlot = copy.deepcopy(xw)
    yPlot = copy.deepcopy(yw)

    # Valutazione spline
    tck, u = splprep([xPlot, yPlot], s=0, k=2, per = True)
    
    # Calcolo normali (rispetto alla curva originale xPlot, yPlot)
    dx, dy = splev(u, tck, der=1)
    norm = np.sqrt(dx**2 + dy**2)
    tx, ty = dx/norm, dy/norm
    nx, ny = -dy/norm, dx/norm

    # u_fine = np.linspace(0, 1, 10000)
    # x_s, y_s = splev(u_fine, tck)
    # # Plot
    # plt.figure(figsize=(8, 6))
    # plt.plot(xPlot[10:20], yPlot[10:20], '.', label='Punti')
    # plt.plot(x_s, y_s, '-', label='Spline')
    # plt.quiver(xPlot[10:20], yPlot[10:20], nx[10:20], ny[10:20], color='red', label='Normali')
    # plt.quiver(xPlot[10:20], yPlot[10:20], ny[10:20], -nx[10:20], color='green', label='Normali')
    # plt.axis('equal')
    # plt.show()

    # print(tau.shape)
    # input()
    tauw = []
    for idx in range(len(tau)):
        val = (tau[idx,0,0]*nx[idx]+tau[idx,0,1]*ny[idx])*tx[idx] + (tau[idx,1,0]*nx[idx]+tau[idx,1,1]*ny[idx])*ty[idx]
        tauw.append(np.sqrt(val*val))

    data['cf'] = np.array(tauw)/(0.5*rho_in*(u_in**2))

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
    closest_index = np.argmin(abs(x0[:, 0] - x1_samp))
    y1_samp = in_data[in_blocks[0]]["y"][closest_index, :].min()
    y2_samp = y1_samp + pitch
    # compute interpolation axis
    y_in = np.linspace(y1_samp, y2_samp, 1000)
    # build inlet_lims
    inlet_lims = [x1_samp, y1_samp, x1_samp, y2_samp]
    print(f"inlet_lims: {inlet_lims}")

    # find y corresponding to x1
    x0 = out_data[out_blocks[0]]["x"]
    closest_index = np.argmin(abs(x0[:, 0] - x2_samp))
    y1_samp = out_data[out_blocks[0]]["y"][closest_index, :].min()
    y2_samp = y1_samp + pitch
    # compute interpolation axis
    y_out = np.linspace(y1_samp, y2_samp, 1000)
    outlet_lims = [x2_samp, y1_samp, x2_samp, y2_samp]
    print(f"outlet_lims: {outlet_lims}")

    data_inlet = {}
    data_outlet = {}
    for var in ["uu", "vv", "ww", "rhou", "rhov", "rho*uu", "rho*uv", "rho*uw", "p", "T", "M", "cp", "cv"]:
        data_inlet[f"{var}_interp"] = line_interp(in_data, var, inlet_lims, in_blocks)
        data_outlet[f"{var}_interp"] = line_interp(out_data, var, outlet_lims, out_blocks)

    data['gamma']  = np.nanmean(data_inlet["cp_interp"] / data_inlet["cv_interp"])
    # gam1 = gam - 1.0

    data['xw'] = (xw-xw.min())/(xw.max()-xw.min())
    data['yw'] = copy.deepcopy(yw)

    in_idx =  np.argwhere(~np.isnan(data_inlet["rhou_interp"]))
    q1 = np.sum(data_inlet["rhou_interp"][in_idx])
    P1 = np.sum(data_inlet["rhou_interp"][in_idx] * data_inlet["p_interp"][in_idx]) / q1
    gamma = np.mean(data_inlet["cp_interp"][in_idx] / data_inlet["cv_interp"][in_idx])
    P01 = np.sum(data_inlet["p_interp"][in_idx] * (1 + (gamma - 1 ) / 2 * data_inlet["M_interp"][in_idx]**2)**(gamma / (gamma - 1)) * data_inlet["rhou_interp"][in_idx]) / q1

    out_idx =  np.argwhere(~np.isnan(data_outlet["rhou_interp"]))
    q2 = np.sum(data_outlet["rhou_interp"][out_idx])
    P2 = np.sum(data_outlet["rhou_interp"][out_idx] * data_outlet["p_interp"][out_idx]) / q2
    gamma = np.mean(data_outlet["cp_interp"][out_idx] / data_outlet["cv_interp"][out_idx])
    P02 = np.sum(data_outlet["p_interp"][out_idx] * (1 + (gamma - 1 ) / 2 * data_outlet["M_interp"][out_idx]**2)**(gamma / (gamma - 1)) * data_outlet["rhou_interp"][out_idx]) / q2

    print(f"Loss coefficient w: {(P01 - P02) / (P01 - P1)}; Gamma: {gamma}")

    data['M_is'] = np.sqrt(2/(data['gamma']-1)*((P01/pw)**((data['gamma']-1)/data['gamma'])-1))
    data['Loss'] = (P01 - data_outlet["p_interp"][out_idx] * (1 + (gamma - 1 ) / 2 * data_outlet["M_interp"][out_idx]**2)**(gamma / (gamma - 1))) / (P01 - P1)
    data['P01']  = P01
    data['yLoss'] = y_out[out_idx]

    # nwall = np.zeros((nw,2))
    # twall = np.zeros((nw,2))

    # with open(os.path.join(input_dir,'norm_surf.dat'),'r') as f:
    #     dat = np.loadtxt(f)
    #     nxwall = dat[:,2]
    #     nywall = dat[:,3]
    #     nwall[:,0],nwall[:,1] = nxwall,nywall
    #     data['nwall'] = nwall

    # # Wall friction
    # twall[:,0] = nwall[:,1]
    # twall[:,1] = -nwall[:,0]
    # tauw = ((tau[...,0,0]*nwall[:,0]+tau[...,0,1]*nwall[:,1])*twall[:,0] + \
    #         (tau[...,1,0]*nwall[:,0]+tau[...,1,1]*nwall[:,1])*twall[:,1]).copy()

    # data['cf'] = tauw/(0.5*rho_in*(u_in**2))

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

# # plot params
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "Times"
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8
figsize = (5.2, 3.64)
plt.close('all')

path_to_mis = "cascade_mis.dat"

dict_input_case = {'pitch': 40.39, 'cax': 1, 'x1': -20.108296, 'x2': 87.25188, 'in_blocks': [1, 2], 'out_blocks': [8, 9], 'wall_blocks': [3, 4, 7, 6]}

# # c_ax = dict_input_case['cax']

# bestLES = [6, 0, 3]
# # input_dir_ADP_1 = f"/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_{bestLES[2]}/high_infill_{bestLES[2]}/MUSICAA/musicaa_g0_c0/OP2" # "ADP"
# input_dir_ADP_1 = "/home/mciarlatani/Hilbert/aero-optim/examples/LRN-CASCADE/cascade_musicaa_base_fine/output_baseline/MUSICAA/musicaa_g0_c0/ADP" # "ADP"

# ADP_1 = read_case(input_dir_ADP_1, dict_input_case)

# to_plot = [ADP_1]

# fig = plt.figure(figsize=(8,2.25))
# for i in range(len(to_plot)):
#     ddd = to_plot[i]
#     idx = i%3+1
#     # ax = plt.subplot(1, 3, idx)

#     if i//3==0:
#         color, line, lab = ['tab:blue', '-', 'Best ADP']
#     if i//3==1:
#         color, line, lab = ['tab:orange', '-', 'Best comp']
#     if i//3==2:
#         color, line, lab = ['tab:green', '-', 'Best OP']
#     if i//3==3:
#         color, line, lab = ['black', '--', 'Baseline']
#     plt.plot(ddd['data']['xw'], ddd['data']['cf'], label=lab, color=color, linestyle=line)

#     # if idx == 1:
#     #     ax.set_ylabel(r'$w_{ADP}$ [-]')
#     # if idx == 2:
#     #     ax.set_ylabel(r'$w_{OP1}$ [-]')
#     #     ax.set_yticklabels([])
#     # if idx == 3:
#     #     ax.set_ylabel(r'$w_{OP2}$ [-]')
#     #     ax.set_yticklabels([])
#         # ax.legend()
    
#     # ax.set_xlabel('$x/c_{ax}$ [-]')
#     # # ax.set_xlim([0.01,0.04])
#     # ax.set_ylim([-0.01,0.48])

# plt.legend()
# plt.show()

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

dict_input_case = {'pitch': 40.39, 'cax': 1, 'x1': -20.108296, 'x2': 87.25188, 'in_blocks': [1, 2], 'out_blocks': [8, 9], 'wall_blocks': [3, 4, 7, 6]}

bestLES = [6, 0, 3]
input_dir_ADP_1 = f"/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_{bestLES[0]}/high_infill_{bestLES[0]}/MUSICAA/musicaa_g0_c0/ADP" # "ADP"
input_dir_OP1_1 = f"/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_{bestLES[0]}/high_infill_{bestLES[0]}/MUSICAA/musicaa_g0_c0/OP1" # "ADP"
input_dir_OP2_1 = f"/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_{bestLES[0]}/high_infill_{bestLES[0]}/MUSICAA/musicaa_g0_c0/OP2" # "ADP"

input_dir_ADP_2 = f"/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_{bestLES[1]}/high_infill_{bestLES[1]}/MUSICAA/musicaa_g0_c0/ADP" # "ADP"
input_dir_OP1_2 = f"/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_{bestLES[1]}/high_infill_{bestLES[1]}/MUSICAA/musicaa_g0_c0/OP1" # "ADP"
input_dir_OP2_2 = f"/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_{bestLES[1]}/high_infill_{bestLES[1]}/MUSICAA/musicaa_g0_c0/OP2" # "ADP"

input_dir_ADP_3 = f"/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_{bestLES[2]}/high_infill_{bestLES[2]}/MUSICAA/musicaa_g0_c0/ADP" # "ADP"
input_dir_OP1_3 = f"/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_{bestLES[2]}/high_infill_{bestLES[2]}/MUSICAA/musicaa_g0_c0/OP1" # "ADP"
input_dir_OP2_3 = f"/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_{bestLES[2]}/high_infill_{bestLES[2]}/MUSICAA/musicaa_g0_c0/OP2" # "ADP"

input_dir_ADP_b = "/home/mciarlatani/Hilbert/aero-optim/examples/LRN-CASCADE/cascade_musicaa_base/output_baseline/MUSICAA/musicaa_g0_c0/ADP" # "ADP"
input_dir_OP1_b = "/home/mciarlatani/Hilbert/aero-optim/examples/LRN-CASCADE/cascade_musicaa_base/output_baseline/MUSICAA/musicaa_g0_c0/OP1" # "OP1"
input_dir_OP2_b = "/home/mciarlatani/Hilbert/aero-optim/examples/LRN-CASCADE/cascade_musicaa_base/output_baseline/MUSICAA/musicaa_g0_c0/OP2" # "OP2"

ADP_1 = read_case(input_dir_ADP_1, dict_input_case)
OP1_1 = read_case(input_dir_OP1_1, dict_input_case)
OP2_1 = read_case(input_dir_OP2_1, dict_input_case)

ADP_2 = read_case(input_dir_ADP_2, dict_input_case)
OP1_2 = read_case(input_dir_OP1_2, dict_input_case)
OP2_2 = read_case(input_dir_OP2_2, dict_input_case)

ADP_3 = read_case(input_dir_ADP_3, dict_input_case)
OP1_3 = read_case(input_dir_OP1_3, dict_input_case)
OP2_3 = read_case(input_dir_OP2_3, dict_input_case)

ADP_b = read_case(input_dir_ADP_b, dict_input_case)
OP1_b = read_case(input_dir_OP1_b, dict_input_case)
OP2_b = read_case(input_dir_OP2_b, dict_input_case)

to_plot = [ADP_1, OP1_1, OP2_1, ADP_2, OP1_2, OP2_2, ADP_3, OP1_3, OP2_3, ADP_b, OP1_b, OP2_b]

fig = plt.figure(figsize=(8,2.25))
for i in range(len(to_plot)):
    ddd = to_plot[i]
    idx = i%3+1
    ax = plt.subplot(1, 3, idx)

    if i//3==0:
        color, line, lab = ['tab:blue', '-', 'Best ADP']
    if i//3==1:
        color, line, lab = ['tab:orange', '-', 'Best comp']
    if i//3==2:
        color, line, lab = ['tab:green', '-', 'Best OP']
    if i//3==3:
        color, line, lab = ['black', '--', 'Baseline']
    plt.plot(ddd['data']['xw'], ddd['data']['cf'], label=lab, color=color, linestyle=line)

plt.legend()
plt.show()

fig = plt.figure(figsize=(8,2.25))
for i in range(len(to_plot)):
    ddd = to_plot[i]
    idx = i%3+1
    ax = plt.subplot(1, 3, idx)

    if i//3==0:
        color, line, lab = ['tab:blue', '-', 'Best ADP']
    if i//3==1:
        color, line, lab = ['tab:orange', '-', 'Best comp']
    if i//3==2:
        color, line, lab = ['tab:green', '-', 'Best OP']
    if i//3==3:
        color, line, lab = ['black', '--', 'Baseline']
    ax.plot(ddd['data']['yLoss'], ddd['data']['Loss'], label=lab, color=color, linestyle=line)

    if idx == 1:
        ax.set_ylabel(r'$w_{ADP}$ [-]')
    if idx == 2:
        ax.set_ylabel(r'$w_{OP1}$ [-]')
        ax.set_yticklabels([])
    if idx == 3:
        ax.set_ylabel(r'$w_{OP2}$ [-]')
        ax.set_yticklabels([])
        ax.legend()
    
    ax.set_xlabel('$x/c_{ax}$ [-]')
    # ax.set_xlim([0.01,0.04])
    ax.set_ylim([-0.01,0.48])

plt.legend()
plt.show()