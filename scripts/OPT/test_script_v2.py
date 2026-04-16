import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

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
    cax        = dict_input['cax']
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

    data = {}
    stats1 = {}
    stats2 = {}
    x_flat   = []
    y_flat   = []
    M_is_flat = []

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
        data[bl]['x_flat'] = data[bl]['x'].flatten()/cax
        data[bl]['y_flat'] = data[bl]['y'].flatten()/cax

        x_flat.append(data[bl]['x'].flatten()/cax)
        y_flat.append(data[bl]['y'].flatten()/cax)

        if block_info[f'block_{bl}']['wall']:
            if block_info[f'block_{bl}']['jmin']:
                # Wall coordinates
                xw[nw_loc:nw_loc+nx]  = data[bl]['x'][:,0]/cax
                yw[nw_loc:nw_loc+nx]  = data[bl]['y'][:,0]/cax
                # Compute first cell height
                x1_[nw_loc:nw_loc+nx] = data[bl]['x'][:,1]/cax
                y1_[nw_loc:nw_loc+nx] = data[bl]['y'][:,1]/cax
                # Compute wall tangential distance from cell to cell
                dxw[nw_loc:nw_loc+nx] = np.hstack((data[bl]['x'][1:,0]-data[bl]['x'][:-1,0],data[bl]['x'][-2,0]-data[bl]['x'][-1,0]))/cax
                dyw[nw_loc:nw_loc+nx] = np.hstack((data[bl]['y'][1:,0]-data[bl]['y'][:-1,0],data[bl]['y'][-2,0]-data[bl]['y'][-1,0]))/cax
                if bl==6:
                    yw[nw_loc:nw_loc+nx] +=-pitch
                    y1_[nw_loc:nw_loc+nx]+=-pitch
                y1[nw_loc:nw_loc+nx]    = (np.sqrt((x1_-xw)**2+(y1_-yw)**2))[nw_loc:nw_loc+nx]
                # Cp
                rhow[nw_loc:nw_loc+nx] = stats1[bl]['rho'][:,0]
                
                # Cf
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

    xw = rearrange(xw,dict_info)
    pw = rearrange(pw,dict_info)

    data['xw'] = (xw-xw.min())/(xw.max()-xw.min())
    data['pw'] = pw
    
    nwall = np.zeros((nw,2))
    twall = np.zeros((nw,2))
    with open(os.path.join(input_dir,'norm_surf.dat'),'r') as f:
        dat = np.loadtxt(f)
        nxwalldl = dat[:,0]
        nywalldl = dat[:,1]
        nxwall = dat[:,2]
        nywall = dat[:,3]
        # dl     = dat[:,4]
        ijacob = dat[:,4]
        nwall[:,0],nwall[:,1] = nxwall,nywall
        nwalldl = np.sqrt(nxwalldl**2+nywalldl**2)
        data['nwall'] = nwall

    # Wall friction
    twall[:,0] = nwall[:,1]
    twall[:,1] = -nwall[:,0]
    tauw = ((tau[...,0,0]*nwall[:,0]+tau[...,0,1]*nwall[:,1])*twall[:,0] + \
            (tau[...,1,0]*nwall[:,0]+tau[...,1,1]*nwall[:,1])*twall[:,1]).copy()

    data['cf'] = -tauw/(0.5*rho_in*u_in**2)

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

def compute_total_pressure(data, stats1, stats2, inlet_lims, outlet_lims, n_block, dict_input):

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
    for bl in dict_input["in_blocks"]:
        in_data[bl] = data[bl] | stats1[bl] | stats2[bl]

    out_data = {}
    for bl in dict_input["out_blocks"]:
        out_data[bl] = data[bl] | stats1[bl] | stats2[bl]

    # The data is interpolated along the measurement planes

    data_inlet = {}
    data_outlet = {}
    for var in ["uu", "vv", "ww", "rhou", "rhov", "rho*uu", "rho*uv", "rho*uw", "p", "T", "M", "cp", "cv"]:
        data_inlet[f"{var}_interp"] = line_interp(in_data, var, inlet_lims, dict_input["in_blocks"])
        data_outlet[f"{var}_interp"] = line_interp(out_data, var, outlet_lims, dict_input["out_blocks"])

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

dict_input_case = {'pitch': 40.39, 'cax': 0.067, 'x1': -20.108296, 'x2': 87.25188, 'in_blocks': [1, 2], 'out_blocks': [8, 9], 'wall_blocks': [3, 4, 7, 6]}

bestLES = [6, 0, 3]
input_dir_ADP_1 = f"/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_{bestLES[0]}/high_infill_{bestLES[0]}/MUSICAA/musicaa_g0_c0/ADP" # "ADP"
ADP_1 = read_case(input_dir_ADP_1, dict_input_case)
out_idx_ADP_1, loss_ADP_1, P01_ADP_1, gamma_ADP_1 = compute_total_pressure(ADP_1['data'], ADP_1['stats1'], ADP_1['stats2'], ADP_1['inlet_lims'], ADP_1['outlet_lims'], ADP_1['n_block'], dict_input_case)

Mis_to_plot = [[ADP_1, P01_ADP_1, gamma_ADP_1]]

fig = plt.figure(figsize=(8,2.25))

for i in range(len(Mis_to_plot)):
    case, P01, gamma = elem = Mis_to_plot[i]
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

    pres_wall_list = []
    Cf_wall_list = []
    x_list = []
    y_list = []

    for bl in dict_input_case['wall_blocks']:
        new_tau = (muw*2*duxw/rhow)[nw_loc:nw_loc+nx]
        tau[nw_loc:nw_loc+nx,0,1] = (muw*(duyw+dvxw)/rhow)[nw_loc:nw_loc+nx]
        tau[nw_loc:nw_loc+nx,1,0] = (muw*(duyw+dvxw)/rhow)[nw_loc:nw_loc+nx]
        tau[nw_loc:nw_loc+nx,1,1] = (muw*2*dvyw/rhow)[nw_loc:nw_loc+nx]
        print(case['stats1'][bl].keys())
        print(case['stats2'][bl].keys())
        new_pres_value = case['stats1'][bl]['p'][:, 0]
        pres_wall_list.append(new_pres_value)
        
        new_x_value = case['data'][bl]['x'][:, 0]
        x_list.append(new_x_value)

        new_y_value = case['data'][bl]['y'][:, 0]
        y_list.append(new_y_value)

        
    pres_wall = np.concatenate(pres_wall_list)
    x_wall = np.concatenate(x_list) / 1000.
    y_wall = np.concatenate(y_list) / 1000.
    # x_tilde = np.cos(np.arctan(y_wall / x_wall) - (106.04 - 90) / 180 * np.pi) * np.sqrt(x_wall**2 + y_wall**2)

    # The isentropic Mach is computed

    Mach_is = np.sqrt(((P01 / pres_wall)**((gamma - 1) / gamma) - 1) * 5)
    
    ax.plot(x_wall / dict_input_case['cax'], Mach_is, label=lab, color=color, linestyle=line)

    if idx == 1:
        ax.set_ylabel(r'$M_{is}^{ADP}$ [-]')
    if idx == 2:
        ax.set_ylabel(r'$M_{is}^{OP1}$ [-]')
        ax.set_yticklabels([])
    if idx == 3:
        ax.set_ylabel(r'$M_{is}^{OP2}$ [-]')
        ax.set_xlabel('$c_{ax}$ [-]')
        ax.set_yticklabels([])
        ax.legend()

    ax.set_ylim(-0.05, 1.27)
    # ax.set_xlabel('$\\bar{x}$ [-]')
    ax.set_xlabel('$x/c$ [-]')
    ax.set_xlabel('$x/c$ [-]')

plt.show()


# font = 20
# markerscale=2
# lw=1.3
# markersize=5
# markevery=20

# # All
# legends = [r'$Tu=0\%$',r'$Tu=4\%$',r'$Tu=8\%$',r'$Tu=0\%$  Air']
# styles  = ['-b','-^r','--k','-or']
# legends = [r'Baseline',r'Selected',r'LE1']
# styles = ['-b','-.g',':m']

# # Suction side
# start = 0
# end   = -1

# plt.figure(figsize=(7,4))
# # plt.figure(figsize=(5,4))
# # fig, ax = plt.subplots(figsize=(5,4))
# plt.rcParams['text.usetex'] = True
# # dirs_data = [dirs_data[0],dirs_data[-1]]
# plt.plot(ADP_1['data']['pw'][start:end],ADP_1['data']['cf'][start:end])

# plt.plot([-0.1,1.1],[0,0],'--k')
# plt.xlabel(r'$x/c$',fontsize=font)
# plt.ylabel(r'$C_f$',fontsize=font)
# plt.xticks(fontsize=font)
# plt.yticks(fontsize=font)
# plt.yticks(fontsize=font)
# # plt.ylim(-0.05,0.2)
# # plt.xlim(-0.1,1.1)
# plt.legend(prop={'size':font-5},markerscale=markerscale)
# plt.tight_layout()

# # plt.savefig('Figures/ETMM_presentation/Cf_suction_1.pdf',format='pdf')
# plt.show()

# input_dir_ADP_1 = f"/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_{bestLES[0]}/high_infill_{bestLES[0]}/MUSICAA/musicaa_g0_c0/ADP" # "ADP"
# input_dir_OP1_1 = f"/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_{bestLES[0]}/high_infill_{bestLES[0]}/MUSICAA/musicaa_g0_c0/OP1" # "OP1"
# input_dir_OP2_1 = f"/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_{bestLES[0]}/high_infill_{bestLES[0]}/MUSICAA/musicaa_g0_c0/OP2" # "OP2"

# ADP_1 = read_case(input_dir_ADP_1, dict_input_case)
# OP1_1 = read_case(input_dir_OP1_1, dict_input_case)
# OP2_1 = read_case(input_dir_OP2_1, dict_input_case)

# out_idx_ADP_1, loss_ADP_1, P01_ADP_1, gamma_ADP_1 = compute_total_pressure(ADP_1['data'], ADP_1['stats1'], ADP_1['stats2'], ADP_1['inlet_lims'], ADP_1['outlet_lims'], ADP_1['n_block'], dict_input_case)
# out_idx_OP1_1, loss_OP1_1, P01_OP1_1, gamma_OP1_1 = compute_total_pressure(OP1_1['data'], OP1_1['stats1'], OP1_1['stats2'], OP1_1['inlet_lims'], OP1_1['outlet_lims'], OP1_1['n_block'], dict_input_case)
# out_idx_OP2_1, loss_OP2_1, P01_OP2_1, gamma_OP2_1 = compute_total_pressure(OP2_1['data'], OP2_1['stats1'], OP2_1['stats2'], OP2_1['inlet_lims'], OP2_1['outlet_lims'], OP2_1['n_block'], dict_input_case)

# loss_to_plot = [[ADP_1['y_out'], out_idx_ADP_1, loss_ADP_1],[OP1_1['y_out'], out_idx_OP1_1, loss_OP1_1],[OP2_1['y_out'], out_idx_OP2_1, loss_OP2_1]]

# fig = plt.figure(figsize=(8,2.25))
# for i in range(len(loss_to_plot)):
#     y_out, out_idx, loss = loss_to_plot[i]
#     idx = i%3+1
#     ax = plt.subplot(1, 3, idx)

#     if i//3==0:
#         color, line, lab = ['tab:blue', '-', 'Best ADP']
#     if i//3==1:
#         color, line, lab = ['tab:orange', '-', 'Best comp']
#     if i//3==2:
#         color, line, lab = ['tab:green', '-', 'Best OP']
#     if i//3==3:
#         color, line, lab = ['black', '--', 'Baseline']
#     ax.plot(y_out[out_idx] / 1000, loss, label=lab, color=color, linestyle=line)

#     if idx == 1:
#         ax.set_ylabel(r'$w_{ADP}$ [-]')
#     if idx == 2:
#         ax.set_ylabel(r'$w_{OP1}$ [-]')
#         ax.set_yticklabels([])
#     if idx == 3:
#         ax.set_ylabel(r'$w_{OP2}$ [-]')
#         ax.set_yticklabels([])
#         ax.legend()
    
#     ax.set_xlabel('$x/c_{ax}$ [-]')
#     ax.set_xlim([0.01,0.04])
#     ax.set_ylim([-0.01,0.48])

# plt.legend()
# plt.show()
