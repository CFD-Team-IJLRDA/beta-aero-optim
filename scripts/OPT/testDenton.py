import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import pandas as pd
import scipy.interpolate as si

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
    x_flat, y_flat, u_flat, v_flat = [], [], [], []
    p_flat, T_flat, P_flat, s_flat = [], [], [], []
    rho_flat, rhou_flat, rhov_flat = [], [], []
    P_c_flat, P_s_flat, M_flat, M_is_flat = [], [], [], []
    p_wake_flat, k_flat, var_flat = [], [], []

    for bl in range(1, n_block + 1):
        data[f'block_{bl}'] = {}
        # read grid
        bl_file = os.path.join(input_dir, f'grid_bl{bl}_ngh{ngh}.bin')
        nx, ny, nz, x, y, z = read_grid(input_dir, bl_file)
        # scale grid
        data[f'block_{bl}']["x"], data[f'block_{bl}']["y"], data[f'block_{bl}']["z"] = x, y, z

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
        if bl==6:
            data[f'block_{bl}']['y'] = copy.deepcopy(data[f'block_{bl}']['y'])-pitch

        nx, ny = block_info[f"block_{bl}"]["nx"], block_info[f"block_{bl}"]["ny"]
        stats1[bl] = read_stats(os.path.join(input_dir, f"stats1_bl{bl}.bin"), nx, ny)
        stats2[bl] = read_stats(os.path.join(input_dir, f"stats2_bl{bl}.bin"), nx, ny)
        
        cp = stats2[bl]['cp']
        cv = stats2[bl]['cv']
        gamma = cp / cv
        
        data[f'block_{bl}']['rho'] = stats1[bl]['rho']
        data[f'block_{bl}']['x_flat'] = data[f'block_{bl}']['x'].flatten()
        data[f'block_{bl}']['y_flat'] = data[f'block_{bl}']['y'].flatten()
        data[f'block_{bl}']['Mach']   = stats2[bl]['M']
        data[f'block_{bl}']['M']  = stats2[bl]['M']
        data[f'block_{bl}']['P0'] = stats1[bl]['p'] * (1 + (gamma - 1) / 2 * data[f'block_{bl}']['Mach']**2)**(gamma / (gamma - 1))
        data[f'block_{bl}']['u'] = stats1[bl]['u']
        data[f'block_{bl}']['v'] = stats1[bl]['v']
        
        # Thermos
        data[f'block_{bl}']['u'] = stats1[bl]['u']
        data[f'block_{bl}']['v'] = stats1[bl]['v']
        data[f'block_{bl}']['rhou'] = stats1[bl]['rhou']
        data[f'block_{bl}']['rhov'] = stats1[bl]['rhov']
        data[f'block_{bl}']['p'] = stats1[bl]['p']
        data[f'block_{bl}']['T'] = stats1[bl]['T']
        data[f'block_{bl}']['s'] = stats2[bl]['s']
        data[f'block_{bl}']['mu'] = stats1[bl]['mu']

        # Derivatives
        data[f'block_{bl}']['omz'] = -stats2[bl]['b3']
        data[f'block_{bl}']['dux'] = stats2[bl]['rho*dux']/stats1[bl]['rho']
        data[f'block_{bl}']['duy'] = stats2[bl]['rho*duy']/stats1[bl]['rho']
        data[f'block_{bl}']['dvx'] = stats2[bl]['rho*dvx']/stats1[bl]['rho']
        data[f'block_{bl}']['dvy'] = stats2[bl]['rho*dvy']/stats1[bl]['rho']
        data[f'block_{bl}']['uv']  = stats1[bl]['uv']-stats1[bl]['u']*stats1[bl]['v']

        x_flat.append(data[f'block_{bl}']['x'].flatten())
        y_flat.append(data[f'block_{bl}']['y'].flatten())

        if block_info[f'block_{bl}']['wall']:
            if block_info[f'block_{bl}']['jmin']:

                # Wall coordinates
                xw[nw_loc:nw_loc+nx]  = data[f'block_{bl}']['x'][:,0]
                yw[nw_loc:nw_loc+nx]  = data[f'block_{bl}']['y'][:,0]
                # Compute first cell height
                x1_[nw_loc:nw_loc+nx] = data[f'block_{bl}']['x'][:,1]
                y1_[nw_loc:nw_loc+nx] = data[f'block_{bl}']['y'][:,1]
                # Compute wall tangential distance from cell to cell
                dxw[nw_loc:nw_loc+nx] = np.hstack((data[f'block_{bl}']['x'][1:,0]-data[f'block_{bl}']['x'][:-1,0],data[f'block_{bl}']['x'][-2,0]-data[f'block_{bl}']['x'][-1,0]))
                dyw[nw_loc:nw_loc+nx] = np.hstack((data[f'block_{bl}']['y'][1:,0]-data[f'block_{bl}']['y'][:-1,0],data[f'block_{bl}']['y'][-2,0]-data[f'block_{bl}']['y'][-1,0]))

                y1[nw_loc:nw_loc+nx]    = (np.sqrt((x1_-xw)**2+(y1_-yw)**2))[nw_loc:nw_loc+nx]

                # Cf
                rhow[nw_loc:nw_loc+nx] = stats1[bl]['rho'][:,0]
                muw[nw_loc:nw_loc+nx]  = stats1[bl]['mu'][:,0]
                duxw[nw_loc:nw_loc+nx] = stats2[bl]['rho*dux'][:,0]
                duyw[nw_loc:nw_loc+nx] = stats2[bl]['rho*duy'][:,0]
                dvxw[nw_loc:nw_loc+nx] = stats2[bl]['rho*dvx'][:,0]
                dvyw[nw_loc:nw_loc+nx] = stats2[bl]['rho*dvy'][:,0]

                # In cartesian coordinates
                tau[nw_loc:nw_loc+nx,0,0] = (muw*2*duxw/rhow)[nw_loc:nw_loc+nx]
                tau[nw_loc:nw_loc+nx,0,1] = (muw*(duyw+dvxw)/rhow)[nw_loc:nw_loc+nx]
                tau[nw_loc:nw_loc+nx,1,0] = (muw*(duyw+dvxw)/rhow)[nw_loc:nw_loc+nx]
                tau[nw_loc:nw_loc+nx,1,1] = (muw*2*dvyw/rhow)[nw_loc:nw_loc+nx]

                pw[nw_loc:nw_loc+nx] = stats1[bl]['p'][:,0]

                nw_loc+=nx
                
        # Operations for grid interpolation and streamlines
        u_flat.append(stats1[bl]['u'].flatten())
        v_flat.append(stats1[bl]['v'].flatten())
        rho_flat.append(stats1[bl]['rho'].flatten())
        p_flat.append(stats1[bl]['p'].flatten())
        T_flat.append(stats1[bl]['T'].flatten())
        rhou_flat.append(stats1[bl]['rhou'].flatten())
        rhov_flat.append(stats1[bl]['rhov'].flatten())
        s_flat.append(stats2[bl]['s'].flatten())
        var_flat.append(data[f'block_{bl}']['M'].flatten())
        
        # Per block
        data[f'block_{bl}']['u_flat'] = stats1[bl]['u'].flatten()
        data[f'block_{bl}']['v_flat'] = stats1[bl]['v'].flatten()
        data[f'block_{bl}']['rhou_flat'] = stats1[bl]['rhou'].flatten()
        data[f'block_{bl}']['rhov_flat'] = stats1[bl]['rhov'].flatten()
        data[f'block_{bl}']['T_flat'] = stats1[bl]['T'].flatten()
        data[f'block_{bl}']['rho_flat'] = stats1[bl]['rho'].flatten()
        data[f'block_{bl}']['dux_flat'] = data[f'block_{bl}']['dux'].flatten()
        data[f'block_{bl}']['duy_flat'] = data[f'block_{bl}']['duy'].flatten()
        data[f'block_{bl}']['dvx_flat'] = data[f'block_{bl}']['dvx'].flatten()
        data[f'block_{bl}']['dvy_flat'] = data[f'block_{bl}']['dvy'].flatten()
        data[f'block_{bl}']['uv_flat'] = data[f'block_{bl}']['uv'].flatten()
        data[f'block_{bl}']['mu_flat'] = stats1[bl]['mu'].flatten()
        data[f'block_{bl}']['s_flat'] = stats2[bl]['s'].flatten()
        data[f'block_{bl}']['omz_flat'] = -stats2[bl]['b3'].flatten()
        data[f'block_{bl}']['p_flat'] = stats1[bl]['p'].flatten()
        data[f'block_{bl}']['var_flat'] = data[f'block_{bl}']['M'].flatten()

        # Operations for Mach number and speed of sound
        data[f'block_{bl}']['mu'] = stats1[bl]['mu']
        data[f'block_{bl}']['vort']  = -stats2[bl]['b3']/u_in

    xw = rearrange(xw,dict_info)
    yw = rearrange(yw,dict_info)
    pw = rearrange(pw,dict_info)
    rhow = rearrange(rhow,dict_info)
    muw = rearrange(muw,dict_info)
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

    # # print(tau.shape)
    # # input()
    # tauw = []
    # for idx in range(len(tau)):
    #     val = (tau[idx,0,0]*nx[idx]+tau[idx,0,1]*ny[idx])*tx[idx] + (tau[idx,1,0]*nx[idx]+tau[idx,1,1]*ny[idx])*ty[idx]
    #     # tauw.append(np.sqrt(val*val))
    #     tauw.append(val)

    # data['cf'] = np.array(tauw)/(0.5*rho_in*(u_in**2))

    # union of data
    in_data = {}
    for bl in in_blocks:
        in_data[bl] = data[f'block_{bl}'] | stats1[bl] | stats2[bl]

    out_data = {}
    for bl in out_blocks:
        out_data[bl] = data[f'block_{bl}'] | stats1[bl] | stats2[bl]
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

    data['xw_min'] = xw.min()
    data['xw_max'] = xw.max()

    data['xw'] = (xw-data['xw_min'])/(data['xw_max']-data['xw_min'])
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

    nwall = np.zeros((nw,2))
    twall = np.zeros((nw,2))

    nwall[:,0], nwall[:,1] = nx, ny
    twall[:,0], twall[:,1] = tx, ty

    data['nwall'] = nwall
    data['twall'] = twall

    # Wall friction
    twall[:,0] = nwall[:,1]
    twall[:,1] = -nwall[:,0]
    tauw = ((tau[...,0,0]*nwall[:,0]+tau[...,0,1]*nwall[:,1])*twall[:,0] + \
            (tau[...,1,0]*nwall[:,0]+tau[...,1,1]*nwall[:,1])*twall[:,1]).copy()

    data['cf'] = tauw/(0.5*rho_in*(u_in**2))

    utau = np.sqrt(abs(tauw)/rhow)
    
    data['nw'] = nw
    data['tauw'] = tauw
    data['utau'] = utau
    data['rhow'] = rhow
    data['muw']  = muw
    data['pw']   = pw


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

# plot params
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "Times"
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8
figsize = (5.2, 3.64)
lw = 0.8

dict_input_case = {'pitch': 40.4, 'cax': 1, 'x1': -20.108296, 'x2': 87.25188, 'in_blocks': [1, 2], 'out_blocks': [8, 9], 'wall_blocks': [3, 4, 7, 6]}

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

input_dir_ADP_b = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_baseline/MUSICAA/musicaa_g0_c0/ADP" # "ADP"
input_dir_OP1_b = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_baseline/MUSICAA/musicaa_g0_c0/OP1" # "OP1"
input_dir_OP2_b = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_baseline/MUSICAA/musicaa_g0_c0/OP2" # "OP2"

ADP_1 = read_case(input_dir_ADP_1, dict_input_case)

comp_data = ADP_1['data']

var_list = ['u','v','rhou','rhov','p','rho','omz','x','y']
var_list_w = ['utau','muw','rhow','pw','tauw']

# Suction side
# ============
for var in var_list:
    comp_data = ADP_1['data']
    comp_data[f'{var}_flat'] = np.hstack((comp_data['block_3'][f'{var}_flat'],\
                                                    comp_data['block_4'][f'{var}_flat'],\
                                                    comp_data['block_6'][f'{var}_flat'],\
                                                    comp_data['block_7'][f'{var}_flat']))

nxwall = comp_data['nwall'][:,0]
nywall = comp_data['nwall'][:,1]
txwall = comp_data['twall'][:,0]
tywall = comp_data['twall'][:,1]
nw = comp_data['nw']
xw = comp_data['xw']*(comp_data['xw_max']-comp_data['xw_min'])+comp_data['xw_min']
yw = comp_data['yw']

LE_idx = np.argmin(comp_data['xw'])

s = np.hstack((np.array([0]),np.sqrt((xw[1:]-xw[:-1])**2+(yw[1:]-yw[:-1])**2)))
s = np.cumsum(s)

theta = np.arctan(abs(nywall/nxwall))
for i,thet in enumerate(theta):
    if nywall[i]<=0:
        theta[i] = 2*np.pi-theta[i]
    if nxwall[i]>=0 and nywall[i]>=0:
        theta[i] = np.pi-theta[i]
    if nxwall[i]>=0 and nywall[i]<=0:
        theta[i] = 3*np.pi-theta[i]
theta = np.pi-theta

# Set up stretching rate
nn = 75
maxn = 0.1
# n1 = 3.5e-5 # Baumgartner (chaussette)
n1 = 3.0e-3 # LS89 Mach17
g = 1.06
dn = np.array([n1*g**i for i in range(nn)])
n = np.cumsum(dn)-n1

# Polar grid
theta,n = np.meshgrid(theta,n)
theta = theta.T
n = n.T
xw = np.repeat(xw[:,np.newaxis],nn,axis=1)
yw = np.repeat(yw[:,np.newaxis],nn,axis=1)

# Cartesian cordinates
x_new = xw+n/np.sqrt(1+np.tan(theta)**2)
y_new = yw+np.sqrt(n**2-(x_new-xw)**2)
for i in range(nw):
    if nxwall[i]<=0 and nywall[i]>=0:
        x_new[i] = xw[i]-n[i]/np.sqrt(1+np.tan(theta[i])**2)
    if nxwall[i]<=0 and nywall[i]<=0:
        x_new[i] = xw[i]-n[i]/np.sqrt(1+np.tan(theta[i])**2)
        y_new[i] = yw[i]-np.sqrt(n[i]**2-(x_new[i]-xw[i])**2)
    if nxwall[i]>=0 and nywall[i]<=0:
        y_new[i] = yw[i]-np.sqrt(n[i]**2-(x_new[i]-xw[i])**2)
        
# Rearrange so new mesh starts at front stagnation point
x_new = np.vstack((x_new[LE_idx:],x_new[:LE_idx]))
y_new = np.vstack((y_new[LE_idx:],y_new[:LE_idx]))
theta = np.vstack((theta[LE_idx:],theta[:LE_idx]))
n = np.vstack((n[LE_idx:],n[:LE_idx]))
comp_data['theta'] = theta
s = np.zeros((x_new.shape))
s[1:] = np.sqrt((x_new[1:]-x_new[:-1])**2 + (y_new[1:]-y_new[:-1])**2)
s = np.cumsum(s,axis=0)[:,0]
s_ps = np.zeros((x_new.shape))
s_ps[1:] = np.sqrt((x_new[1:][::-1]-x_new[:-1][::-1])**2 + (y_new[1:][::-1]-y_new[:-1][::-1])**2)
s_ps = np.cumsum(s_ps,axis=0)[:,0]

font=25
markerscale=2
markersize=6
markevery=20
lw=1.5

TE_idx = np.argmax(x_new[:,0])

# plt.figure(figsize=(7,5))
# plt.rcParams['text.usetex'] = True
# plt.plot(comp_data['x_flat'],comp_data['y_flat'],'.r',markersize=0.5)
# plt.plot(x_new,y_new,'.c',markersize=0.5)
# for i in range(20):
#     plt.plot(x_new[int(i*1),:],y_new[int(i*1),:],'k',linewidth=3)

# plt.plot(x_new[TE_idx,:],y_new[TE_idx,:],'g',linewidth=3)
# plt.axis('equal')
# plt.show()

for var in var_list[:-2]:
    comp_data[f'{var}_interp'] = si.griddata((comp_data['x_flat'],comp_data['y_flat']),\
                                              comp_data[f'{var}_flat'],(x_new,y_new),method='cubic')
    print(var)

thetat = comp_data['theta']-np.pi/2

# # Tangential velocity
# comp_data['rhou_interp_'] = comp_data['rhou_interp'].copy()
# comp_data['rhov_interp_'] = comp_data['rhov_interp'].copy()
# rhout = comp_data['rhou_interp_']*np.cos(thetat)+\
#         comp_data['rhov_interp_']*np.sin(thetat)
# comp_data['rhout'] = rhout
# comp_data['ut'] = rhout/comp_data['rho_interp']

# Tangential velocity
comp_data['rhou_interp_'] = comp_data['rhou_interp'].copy()
comp_data['rhov_interp_'] = comp_data['rhov_interp'].copy()
rhout = comp_data['rhou_interp_']*np.cos(thetat)+comp_data['rhov_interp_']*np.sin(thetat)

comp_data['ut'] = rhout/comp_data['rho_interp']
comp_data['ut'][LE_idx:] = -comp_data['ut'][LE_idx:]

f = np.abs(np.nan_to_num(comp_data['omz_interp'],0))
comp_data['yomz'] = n*f/np.repeat((n*f).max(axis=1)[:,np.newaxis],nn,axis=1)

# plt.figure()
# # plt.pcolormesh(x_new,y_new,np.sqrt(comp_data['rhou_interp_']**2+comp_data['rhov_interp_']**2),cmap='RdBu_r')
# plt.pcolormesh(x_new,y_new,comp_data['rho_interp'],cmap='RdBu_r')
# plt.plot(x_new[0],y_new[0],'k')
# plt.ylim(-1.25,-1.15)
# plt.colorbar()
# plt.axis('scaled')
# plt.show()

# Based on vorticity (Griffin 2021)
C_om = 0.02
eps = 0.1

pos = 50

# Region of BL not too close to the wall
eps = 0.1
eps_ = comp_data['ut'][pos,:]/comp_data['ut'][pos,:].max()
ind_start = np.where(eps_>=eps)[0][0]
# print(ind_start)

bl = np.where(comp_data['yomz'][pos,ind_start:]<=C_om)[0][0]+ind_start

x = comp_data['block_7']['x'][0,:]
y = comp_data['block_7']['y'][0,:]
n_old = np.sqrt((x[43]-x[0])**2 + (x[43]-x[0])**2)
print(bl,n[pos,:][bl],n_old)
print("BL height =",n[pos,:][bl])

plt.plot(comp_data['ut'][pos,:],n[pos,:],markerfacecolor='none',markevery=20)
plt.plot(comp_data['ut'][pos,:][bl],n[pos,:][bl],'*g')
plt.show()

# Baumgartner (chaussette)
# ========================
# starts = LE_idx+20
# stops = TE_idx

# for ii,dir_data in enumerate(dirs_data):
    
#     ut = comp_data['ut']
#     rhout = comp_data['rhout']
#     rho = comp_data['rho_interp']
#     p = comp_data['p_interp']
#     x = x_new.copy()

#     # Storage
#     delta_ = []
#     delta_locs_ = []
#     u_delta_ = []
#     rhou_delta_ = []
#     rho_delta_ = []
#     p_delta_ = []
#     x_delta_ = []
#     start,stop = starts[ii],stops[ii]
#     poss = np.arange(start,stop,1)
#     for pos in poss:
#         # print(pos)
        
#         # Region of BL not too close to the wall
#         eps_ = comp_data['ut'][pos,:]/comp_data['ut'][pos,:].max()
#         ind_start = np.where(eps_>=eps)[0][0]

#         delta_loc = np.where(comp_data['yomz'][pos,ind_start:]<=C_om)[0][0]+ind_start
#         delta_.append(n[pos,:][delta_loc])
#         u_delta_.append(ut[pos,:][delta_loc])
#         rhou_delta_.append(rhout[pos,:][delta_loc])
#         rho_delta_.append(rho[pos,:][delta_loc])
#         p_delta_.append(p[pos,:][delta_loc])
#         x_delta_.append(x[pos,:][delta_loc])
#         delta_locs_.append(delta_loc)

#     # Get arrays
#     comp_data['delta'] = np.asarray(delta_)
#     comp_data['delta_locs'] = np.asarray(delta_locs_)
#     comp_data['u_delta'] = np.asarray(u_delta_)
#     comp_data['rhou_delta'] = np.asarray(rhou_delta_)
#     comp_data['rho_delta'] = np.asarray(rho_delta_)
#     comp_data['p_delta'] = np.asarray(p_delta_)
#     comp_data['x_delta'] = np.asarray(x_delta_)

print('done')