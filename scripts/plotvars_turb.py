#========================================#
#== Plot variables read by readvars.py ==#
#========================================#

import matplotlib
import matplotlib.pyplot as plt,mpld3
import numpy as np
import readvars_turb as rv
import os
from collections import OrderedDict
import scipy.signal as ss
import scipy.interpolate as si
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import interp2d
import experimental_data as exp
import matplotlib.animation as animation
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.integrate import trapezoid



# Useful stuff
#=============

def d():
    x = OrderedDict()
    return x

def vars_levels(vmin,vmax,levels):
    var_levels = [j*(vmax-vmin)/levels+vmin for j in range(levels)]
    return var_levels

font=25
markerscale=2
markersize=4
markevery=10
lw=1


# Which
#======

def which(check,solver,method,case,geom,fluid,mesh):
    dir_data = '/home/matar/Codes/MUSICAA/RANS/Chaussette/'+\
               geom+'_'+mesh+'/'
    
    if method=='LES':
        dir_data = '/home/matar/Codes/MUSICAA/PFG/Chaussette/'+\
               geom+'_'+mesh+'/'

    if check:
        return dir_data
    else:
        if fluid=='novec':
            fluid = 'PRS'
        else:
            fluid = 'PFG'
        dir_data+= f'Postproc_{fluid.lower()}/{case}/'

    return dir_data


def check_info(dir_data):
    dict_info = rv.read_info(dir_data)
    Re_in  = dict_info['Reref']
    rho_in = dict_info['Roref']
    u_in   = dict_info['Uref']
    mu_in  = dict_info['Muref']
    L_ref  = dict_info['L_ref']
    nz     = dict_info['nz_bl1']
    if nz==1:
        dz = 0.
    else:
        dz = 2*np.pi*L_ref/(nz-1)
    print("nz =",nz,"\nL_ref =",L_ref,"\ndz =",dz)
    print(dir_data)


# Plot grids
#===========

# Quadrilateral elements
def create_elm(nb,line):
    quad = []
    for i in range(nb):
        bg = i       # bas-gauche
        bd = bg+1    # bas-droite
        hd = bd+line # haut-droite
        hg = bg+line # haut-gauche
    
        if bd%line!=0:
            quad.append([bg,bd,hd,hg]) # liste des coins des elements
    return quad

# Plotting function for mesh
def plot_elm(nod_x,nod_y,qd,edgecolor):
    for cell in qd:
        vertices_x = [nod_x[cell[i]] for i in range(len(cell))]
        vertices_y = [nod_y[cell[i]] for i in range(len(cell))]
        plt.fill(vertices_x, vertices_y, edgecolor=edgecolor, linewidth=0.5, fill=False)

# Block to be plotted
def plot_grid(dir_data,is_curv,bl_num,sensors):

    # General information
    dict_info = rv.read_info(dir_data)
    n_bl  = dict_info['nbloc']
    L_ref = dict_info['L_ref']#*1.805
    print(L_ref)
    L_ref=0.08358
    nz  = dict_info['nz_bl1']
    L_z = np.pi*L_ref
    # dz = L_z/(nz-1)
    every = 1

    # plt.figure(figsize=(9,4))
    for bl_num in range(n_bl):
        bl_num+=1
        bl_file = dir_data+f'grid_bl{bl_num}.bin'
        nx,ny,nz,x,y,z = rv.read_grid(bl_file,is_curv)
        x,y,z = x[::every]/L_ref,y[::every]/L_ref,z/L_ref
        node_x,node_y = x.flat[::every],y.flat[::every]

        nb_cells = (nx//every-1)*ny//every

        # prop = (x.max()-x.min())/(y.max()-y.min())
        # plt.figure(figsize=(6*prop,6))

        if is_curv:
            quad = create_elm(nb_cells,ny//every)
            plot_elm(node_x,node_y,quad,'black')

        else:
            plt.vlines(x[0],  *y[[0,-1],0],'black',linewidth=lw)
            plt.hlines(y[:,0],*x[0, [0,-1]],'black',linewidth=lw)

    #    if sensors:
    #        # Coarse mesh sensors
    #        #====================
    #        ind_i = np.array([53,53,53,53,53,53,28,40,43,45,46,78,66,63,61,60,21,36,37,\
    #                          40,41,85,70,69,66,65])-1
    #        ind_j = np.array([38,21,114,165,198,232,29,114,165,198,232,29,114,165,198, \
    #                          232,33,114,165,198,232,33,114,165,198,232])-1
    #        if bl_num==1:
    #            ind_i = np.array([77,27])-1
    #            ind_j = np.array([30,30])-1
    #
    #        if bl_num==2:
    #            ind_i = np.array([3])-1
    #            ind_j = np.array([27])-1
    #
    #        if bl_num==4:
    #            ind_i = np.array([38])-1
    #            ind_j = np.array([27])-1

            # # Coarse HW mesh sensors
            # #======================= 
            # if bl_num==1:
            #     ind_i = np.array([60,21])-1
            #     ind_j = np.array([15,15])-1

            # if bl_num==2:
            #     ind_i = np.array([3])-1
            #     ind_j = np.array([14])-1

            # if bl_num==4:
            #     ind_i = np.array([38])-1
            #     ind_j = np.array([14])-1

            # if bl_num==5:
            #     ind_i = np.array([40,40,40,40,40,40])-1
            #     ind_j = np.array([13,54,78,98,114,127])-1

            #if bl_num==10:
            #    # ind_j = np.arange(1,100)-1
            #    # ind_i = np.zeros(ind_j.size,dtype=int)+299
            #    # ind_i = 300-np.arange(1,20)-1
            #    ind_i = np.array([0])
            #    ind_j = np.array([0])


            # Fine HW mesh sensors
            #===================== 
            # if bl_num==1:
            #     ind_i = np.array([58,23])-1
            #     ind_j = np.array([26,26])-1

            # if bl_num==2:
            #     ind_i = np.array([3])-1
            #     ind_j = np.array([28])-1

            # if bl_num==4:
            #     ind_i = np.array([32])-1
            #     ind_j = np.array([28])-1

            # if bl_num==5:
            #     ind_i = np.array([40,40,40,40,40,40])-1
            #     ind_j = np.array([23,116,155,181,204,224])-1

            # Medium mesh sensors
            # ind_i = np.array([53,53,53,53,53,28,40,43,45,46,78,66,63,61,60,20,36,37,\
            #                   40,41,86,70,69,66,65])-1
            # ind_j = np.array([41,112,151,190,248,47,112,151,190,248,47,112,151,190,248, \
            #                   50,112,151,190,248,50,112,151,190,248])-1

            # Coarse NEW mesh sensors
            #========================
            # ind_i = np.array([41,41,41,41,41,41,\
            #                   23,31,33,34,35,\
            #                   58,50,49,48,47,\
            #                   18,28,29,30,31,\
            #                   63,53,53,52,51])-1
            # ind_j = np.array([16,42,127,202,268,313,\
            #                   24,128,202,268,313,\
            #                   24,128,202,268,313,\
            #                   29,128,202,268,313,\
            #                   29,128,202,268,313])-1
            # if bl_num==1:
            #     ind_i = np.array([59,20])
            #     ind_j = np.array([54,54])

            # if bl_num==2:
            #     ind_i = np.array([2])
            #     ind_j = np.array([51])

            # if bl_num==4:
            #     ind_i = np.array([37])
            #     ind_j = np.array([51])

            # Medium NEW mesh sensors
            #========================
            # ind_i = np.array([81,81,81,81,81,\
            #                   49,63,67,69,70,\
            #                   113,99,95,93,92,\
            #                   38,56,57,60,62,\
            #                   124,105,105,102,100])-1
            # ind_j = np.array([4,127,202,268,313,\
            #                   14,128,202,268,313,\
            #                   14,128,202,268,313,\
            #                   21,128,202,268,313,\
            #                   21,128,202,268,313])-1

            # Fine NEW mesh sensors
            #======================
            # Block 5
            # ind_i = np.array([101,101,101,101,\
            #                   77,82,84,86,\
            #                   124,119,117,115,\
            #                   52,69,70,74,77,\
            #                   149,132,131,127,124])-1
            # ind_j = np.array([160,253,335,391,\
            #                   161,253,335,391,\
            #                   161,253,335,391,\
            #                   3,161,253,335,391,\
            #                   3,161,253,335,391])-1
            # Block 1
            #if bl_num==1:
            #    ind_i = np.array([101,62,137])
            #    ind_j = np.array([130,142,142])

            # high Re mesh
            #if bl_num==2:
            #    ind_i = np.array([18,150,150,99])
            #    ind_j = np.array([134,205,80,133])

            #for i in range(ind_i.size):
            #    plt.plot(x[ind_i[i],ind_j[i]],y[ind_i[i],ind_j[i]],'or',markersize=5)
            #    print(x[ind_i[i],ind_j[i]],y[ind_i[i],ind_j[i]])

    # plt.xlim(-2,4)
    # plt.ylim(0,3)
    # plt.xlabel(r'$x/D$')
    # plt.ylabel(r'$y/D$')
    plt.show()


# Plot time evolving quantites
#=============================

def plot_time_evol(dir_data,iter_avg,coeffs,Nusselt,psd,hilb):

    niter,every,time,tstar,cl,cd,cl_avg,cd_avg,cl_rms,cd_rms \
                  = rv.read_coeffs(False,dir_data+'coeff.dat',iter_avg)
    iter_avg = iter_avg//every

    # cases = ['M07_Re2e5_1155e3it/','M08_Re2e5_11_stats/','M09_Re2e5_9_stats/']
    # styles = ['-b','-g','m']
    # labels = [r'$M=0.7$',r'$M=0.8$',r'$M=0.9$']

    # plt.figure(figsize=(9,6))
    # plt.rcParams['text.usetex'] = True
    # niter,every,time,tstar,cl,cd,cl_avg,cd_avg,cl_rms,cd_rms \
    #           = rv.read_coeffs(False,dir_data+'coeff.dat',iter_avg)
    # plt.plot(cl,'',linewidth=0.5,label=labels[i])
    # plt.ylabel(r'$C_l$',fontsize=font)
    # plt.xticks(fontsize=font)
    # plt.yticks(fontsize=font)
    # # # plt.grid()()
    # plt.legend(fontsize=font)
    # plt.show()

    # Aero coeffs
    if coeffs:
        plt.figure(figsize=(7,4))
        plt.rcParams['text.usetex'] = True
        ax1 = plt.subplot(211)
        plt.plot(niter[500000:],cl[500000:],'-k',linewidth=lw)
        if hilb:
            envelope = np.abs(ss.hilbert(cl))
            ax1.plot(niter,envelope,'b',linewidth=lw)
            ax1.plot(niter,-envelope,'b',linewidth=lw)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.ylabel(r'$C_l$',fontsize=font)
        plt.xticks(fontsize=font)
        plt.yticks(fontsize=font)
        # # plt.grid()()

        ax2 = plt.subplot(212,sharex=ax1)
        plt.plot(niter[500000:],cd[500000:],'-k',linewidth=lw)
        plt.xlabel(r'$n_{iter}$',fontsize=font)
        plt.ylabel(r'$C_d$',fontsize=font)
        plt.xticks(fontsize=font)
        plt.yticks(fontsize=font)
        plt.tight_layout()
        # # plt.grid()()

        # Drag polar
        plt.figure(figsize=(7.5,5))
        plt.plot(cl,cd,'-k',linewidth=lw)
        plt.xlabel(r'$C_l$',fontsize=font)
        plt.ylabel(r'$C_d$',fontsize=font)
        plt.xticks(fontsize=font-5)
        plt.yticks(fontsize=font-5)
        plt.title('Drag polar')
        # # plt.grid()()

        print("Cl_avg =",cl_avg,"\nCd_avg =",cd_avg)
        print("Cl_rms =",cl_rms,"\nCd_rms =",cd_rms)

        if psd:
            plt.figure(figsize=(7.5,5))
            St_list,spec,St = plot_psd(niter,iter_avg,tstar,cl,'r')
            print("St =",St)
            plt.xscale('log')
            plt.yscale('log')
            if hilb:
                plt.figure(figsize=(7.5,5))
                St_list,spec,St = plot_psd(niter,iter_avg,tstar,envelope,'b')
                plt.xscale('log')
                plt.yscale('log')

    # Nusselt
    if Nusselt:
        niter,time,tstar,nus,nus_avg,nus_rms \
                  = rv.read_nusselt(dir_data+'nusselt.dat',iter_avg)
        plt.figure(figsize=(9,6))
        plt.rcParams['text.usetex'] = True

        ax1 = plt.subplot(211)
        plt.plot(niter,nus, '-k',linewidth=lw)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.ylabel(r'$Nusselt$')
        plt.ylim(0,20)
        # # plt.grid()()

        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(niter,cl, '-k',linewidth=lw)
        plt.xlabel(r'$n_{iter}$')
        plt.ylabel(r'$C_l$')
        # # plt.grid()()

        print("Nusselt_avg =",nus_avg,"\nNusselt_rms =",nus_rms)

        if psd:
            plt.figure(figsize=(7.5,5))
            St_list,spec,St = plot_psd(niter,iter_avg,tstar,nus,'r')
            print("St =",St)
            # plt.ylim(0,0.05)
            plt.xscale('log')
            plt.yscale('log')

    plt.show()

# PSD
#====

def plot_psd(niter,iter_avg,tstar,var,style):

    N = niter.shape[0]-iter_avg
    A = 1./N
    dt_star = tstar[1]-tstar[0]
    spec = np.fft.fft(var[iter_avg:],N)
    spec = A*np.abs(spec)[:N//2]**2
    St_list = np.fft.fftfreq(N,dt_star)[:N//2]
    St = round(St_list[np.argmax(spec[1:])+1],3)

    plt.rcParams['text.usetex'] = True
    plt.plot(St_list,spec,style,linewidth=lw)
    plt.xlabel(r'$f/f_S$')
    plt.ylabel(r'$PSD$')
    plt.title('Lift coefficient PSD')
    # # plt.grid()()

    return St_list,spec,St


# Plot residuals
#===============

def plot_residuals(dir_data,is_RANS,var):
    res = rv.read_residuals(f'{dir_data}residuals.bin',is_RANS)

    plt.figure()
    plt.plot(res[f'Rho{var}'],'.k',linewidth=lw)
    plt.title(rf'$\rho\times {var}$ residuals')
    plt.xlabel('Iteration')
    plt.ylabel(rf'$\log(\rho\times {var})$')
    plt.ylim(-7,1)
    # # plt.grid()()
    plt.show()


# Plot wall distance (RANS)
#==========================

def wall_dist(dir_data,nx,ny):

    data  = d()
    f8_dtype = np.dtype('f8')
    i8_dtype = np.dtype('i4')
    dir_file  = f'{dir_data}Wall_distance_check'
    procs     = len(os.listdir(dir_file))
    data_list = ['d','x','y']
    vmax = []

    # Collect data
    for iproc in range(procs):
        f = open(f'{dir_file}/wall_dist_proc{iproc:02}.bin','rb')
        arg = np.fromfile(f,dtype=i8_dtype,count=1)
        #print(len(np.fromfile(f,dtype=i8_dtype)))
        for i,data_ in enumerate(data_list):
            data[f'{iproc}_{data_}'] = np.fromfile(f,dtype=f8_dtype,count=nx*ny).reshape((nx,ny),order='F')
        f.close()
        vmax.append(data[f'{iproc}_d'].max())

    vmax = max(vmax)
    # Plot data
    plt.figure()
    for iproc in range(procs):
        plt.pcolormesh(data[f'{iproc}_x'],data[f'{iproc}_y'],data[f'{iproc}_d'], \
                       vmin=0.,vmax=vmax)
                       # norm=matplotlib.colors.LogNorm(vmin=0.01,vmax=0.5))
        # plt.contour(data[f'{iproc}_x'],data[f'{iproc}_y'],data[f'{iproc}_d'],\
        #               levels=[vmax/12.,vmax/4.,vmax/2.,3*vmax/4.],colors=['black'])
    zero = np.zeros((1,1))
    plt.pcolormesh(zero,zero,zero,vmin=0.,vmax=vmax)

    plt.colorbar()
    plt.show()


# Plot planes
#============

def plot_planes(dir_data,is_curv,var,plane_nb,chkpnt_nb,yz,Lx,wall_dat,wall_plane,anim,save):

    plane_nb = str(plane_nb).rjust(3,'0')

    # General information
    dict_info = rv.read_info(dir_data)
    n_bl   = dict_info['nbloc']
    tscale = dict_info['Tscale']
    u_in   = dict_info['Uref']
    L_ref  = dict_info['L_ref']
    print(L_ref)
    mu_ref = dict_info['Muref']
    ro_ref = dict_info['Roref']
    p_ref  = dict_info['Pref']
    nz = dict_info['nz_bl1']
    # L_ref=1
    L_z = np.pi*L_ref
    # dz = L_z/(nz-1)
    if var=='vort':
        minmax = [(i+1)*0.05 for i in range(10)]

    # Extract wall data
    nw = sum([dict_info['nx_bl{}'.format(i+1)] for i in range(n_bl)])
    if n_bl>=4:
        nw = sum([dict_info['nx_bl{}'.format(i+1)] for i in range(4)])
    locs = 5 # nb of points to investigate from wall
    nw_loc = 0
    xw,yw   = np.zeros((nw)),np.zeros((nw))
    pw,Twa  = np.zeros((nw)),np.zeros((nw))
    theta,r = np.zeros((nw)),np.zeros((nw))
    rhow    = np.zeros((nw)),np.zeros((nw))

    # For regular grid interpolation/streamlines
    x_flat   = []
    y_flat   = []
    var_flat = []
    # Storage
    data = d()
    data['wall'] = d()
    big_data = d()
    vmin,vmax = [],[]

    # Plane normal to z-axis
    if not wall_plane:
        for bl in range(n_bl):
            bl+=1
            # Gather individual blocks variables #
            data[f'block_{bl}'],var_names,big_data[f'block_{bl}'] = \
            rv.read_planes(dir_data+f'plane_{plane_nb}_sol_bl{bl}.bin',chkpnt_nb,wall_plane,anim)
            if not is_curv:
                for item in data[f'block_{bl}'].keys():
                    data[f'block_{bl}'][item] = data[f'block_{bl}'][item].T
            vmin.append(data[f'block_{bl}'][var].min())
            vmax.append(data[f'block_{bl}'][var].max())


            # print(np.argmax(data[f'block_{bl}'][var]))
            # Gather individual blocks coordinates
            nx,ny,nz,x,y,z = rv.read_grid(dir_data+'grid_bl{0}.bin'.format(bl),is_curv)
            # L_ref=0.08358
            L_ref=0.015
            data[f'block_{bl}']['x'] = x/L_ref
            data[f'block_{bl}']['y'] = y/L_ref
            data[f'block_{bl}']['z'] = z/L_ref
            if anim:
                big_data[f'block_{bl}']['x'] = x/L_ref
                big_data[f'block_{bl}']['y'] = y/L_ref
                big_data[f'block_{bl}']['z'] = z/L_ref

            # Operations for grid interpolation and streamlines
            if yz:
                var_flat.append(data[f'block_{bl}'][var].flatten())
                x_flat.append(data[f'block_{bl}']['x'].flatten())
                y_flat.append(data[f'block_{bl}']['y'].flatten())

            # Wall data
            if wall_dat:
                if bl<=4:
                    if bl==1:
                        for var_name in var_names:
                            data['wall'][var_name] = np.zeros((nw,locs))
                    for var_name in var_names:
                        for i in range(locs):
                            xw[nw_loc:nw_loc+nx]    = x[:nx,i]/L_ref
                            yw[nw_loc:nw_loc+nx]    = y[:nx,i]/L_ref
                            r[nw_loc:nw_loc+nx]     = np.sqrt(xw**2+yw**2)[nw_loc:nw_loc+nx]
                            theta[nw_loc:nw_loc+nx] = np.arccos(x[:nx,i]/L_ref/r[nw_loc:nw_loc+nx])*180./np.pi
                            data['wall'][var_name][nw_loc:nw_loc+nx,i]  = \
                                                    data[f'block_{bl}'][var_name][:nx,i]
                    nw_loc+=nx

        # Get min/max values for each variables
        var_names_ = ['u','v','w','p','T','rho','M','s','h','vort']
        vmins,vmaxs = np.zeros((n_bl,len(var_names_))),np.zeros((n_bl,len(var_names_)))
        for bl in range(n_bl):
            bl+=1
            for i,var_ in enumerate(var_names_):
                vmins[bl-1,i] = data[f'block_{bl}'][var_].min()
                vmaxs[bl-1,i] = data[f'block_{bl}'][var_].max()

        #########################
        # Plot individual figure
        plt.figure(figsize=(9,6))
        # plt.figure(figsize=(6,6))
        plt.rcParams['text.usetex'] = True
        cmap='RdBu_r'
        # cmap='plasma'
        var_levels = vars_levels(min(vmin),max(vmax),10)
        for bl in range(n_bl):
            bl+=1

            # Plot individual blocks for ONE variable
            plt.contour(data[f'block_{bl}']['x'], \
                       data[f'block_{bl}']['y'], \
                       data[f'block_{bl}'][var], \
                       levels=[1],colors=['k'],linewidths=2)
            # plt.contour(data[f'block_{bl}']['x'], \
            #            data[f'block_{bl}']['y'], \
            #            data[f'block_{bl}'][var], \
            #            levels=var_levels,colors=['grey'],linewidths=1)
            plt.pcolormesh(data[f'block_{bl}']['x'], \
                           data[f'block_{bl}']['y'], \
                           # np.arctan(data[f'block_{bl}']['v']/data[f'block_{bl}']['u'])*180/np.pi, \
                           data[f'block_{bl}'][var], \
                           # vmin=1.25e5,vmax=1.26e5,\
                           vmin=min(vmin),vmax=max(vmax),\
                           cmap=cmap)#,shading='gouraud')

            pitch = data['block_2']['y'][0,-1]-data['block_1']['y'][0,0]
            corrmin = (data['block_1']['y'][0,1]-data['block_1']['y'][0,0])/2
            corrmax = abs((data['block_2']['y'][0,-1]-data['block_2']['y'][0,-2])/2)
            corr = corrmin+corrmax
            # print(corr)
            # pitch=0.71430119+0.13519881
            # pitch=pitch*1.8055
            plt.pcolormesh(data[f'block_{bl}']['x'], \
                           data[f'block_{bl}']['y']+pitch+corr, \
                           data[f'block_{bl}'][var], \
                           # vmin=1.25e5,vmax=1.26e5,\
                           vmin=min(vmin),vmax=max(vmax),\
                           cmap=cmap)#,shading='gouraud')

        if yz:
            var_flat = np.hstack(var_flat)
            x_flat   = np.hstack(x_flat)
            y_flat   = np.hstack(y_flat)

        # plt.colorbar()
        # xlimd,xlimf = -1,3.5
        # ylimd,ylimf = -1.5,1.5
        xlimd,xlimf = -0.1,1.1
        ylimd,ylimf = -0.1,1.1
        # xlimd,xlimf = -1.1,1.5
        # ylimd,ylimf = -1.25,1.25
        # plt.xlim(xlimd,xlimf)
        # plt.ylim(ylimd,ylimf)

        if yz:
            plt.figure(figsize=(6,6))
            norm = u_in
            if var=='mut':
                norm = mu_ref
            plot_yz(is_curv,x_flat,y_flat,var_flat,Lx,norm,var,'k','')

        plt.xlabel(rf'$x/D$',fontsize=font)
        plt.ylabel(rf'$y/D$',fontsize=font)
        plt.xticks(fontsize=font)
        plt.yticks(fontsize=font)
        zero = np.zeros((1,1))
        im = plt.pcolormesh(zero,zero,zero, \
                       vmin=min(vmin),vmax=max(vmax), \
                       cmap='RdBu_r',shading='gouraud')
        cb = plt.colorbar(im)
        cb.set_label(label=f'${var}$', size=font, weight='bold')
        cb.ax.tick_params(labelsize=font)
        plt.tight_layout()
        # plt.title(rf'${var}$')
        # plt.show()

        # Extract along line x
        #data['block_1'][var]=np.arctan(data[f'block_1']['v']/data[f'block_1'][var])*180/np.pi
        plt.figure()
        plt.plot(data['block_1']['x'][:,-1],data['block_1'][var][:,-1],'b',\
                 label=f'{var} along x')
        plt.legend()

        plt.figure()
        plt.plot(data['block_8']['x'][:,-1],data['block_8'][var][:,-1],'b',\
                 label=f'{var} along x')
        plt.legend()
        # Extract along line y
        plt.figure()
        i=0
        plt.plot(data['block_1']['y'][i,:],data['block_1'][var][0,:],'k',\
                 label=f'{var} along y @ x={i}')
        plt.plot(data['block_2']['y'][i,:],data['block_2'][var][0,:],'r',\
                 label=f'{var} along y @ x={i}')
        i=20
        plt.plot(data['block_1']['y'][i,:],data['block_1'][var][0,:],'--k',\
                 label=f'{var} along y @ x={i}')
        plt.plot(data['block_2']['y'][i,:],data['block_2'][var][0,:],'--r',\
                 label=f'{var} along y @ x={i}')
        i=49
        plt.plot(data['block_1']['y'][i,:],data['block_1'][var][0,:],'*k',\
                 label=f'{var} along y @ x={i}')
        plt.plot(data['block_2']['y'][i,:],data['block_2'][var][0,:],'*r',\
                 label=f'{var} along y @ x={i}')
        plt.legend()

        # Get pitch-wise averages
        # chord = 0.021991935171912772 # true chord
        chord = 0.015
        plt.figure()
        for ind_ in range(20):
            ind_=-ind_
            M_avg = 0.5*(np.mean(data['block_8']['M'][ind_,:])+\
                         np.mean(data['block_9']['M'][ind_,:]))
            rho_avg = 0.5*(np.mean(data['block_8']['rho'][ind_,:])+\
                         np.mean(data['block_9']['rho'][ind_,:]))
            u_avg = 0.5*(np.mean(data['block_8']['u'][ind_,:])+\
                         np.mean(data['block_9']['u'][ind_,:]))
            v_avg = 0.5*(np.mean(data['block_8']['v'][ind_,:])+\
                         np.mean(data['block_9']['v'][ind_,:]))
            U_avg = np.sqrt(u_avg**2+v_avg**2)
            theta_avg = 0.5*(np.mean(np.arctan(data['block_8']['v'][ind_,:]/\
                                             data['block_8']['u'][ind_,:]))+\
                             np.mean(np.arctan(data['block_9']['v'][ind_,:]/\
                                             data['block_9']['u'][ind_,:])))

            theta_avg = theta_avg*180/np.pi
            Re_avg = rho_avg*U_avg*chord/1.3e-5
            plt.plot(ind_,theta_avg,'*b')
        plt.show()


        print('M out avg =',M_avg)
        print('theta out avg =',theta_avg)
        print('Re out avg =',Re_avg)


        # Recalc PRSV
        # T = data[f'block_{bl}']['T'][:,ny//2]
        # rho = data[f'block_{bl}']['rho'][:,ny//2]
        # Pc = 1.41e6
        # Tc = 564
        # R = 35.1518
        # a = 0.457235*R**2*Tc**2/Pc
        # b = 0.077796*R*Tc/Pc
        # om = 0.528
        # m = 0.378893+1.4897153*om-0.17131848*om**2+0.0196554*om**3
        # alpha = lambda T: (1+m*(1-(T/Tc)**0.5))**2
        # p = rho*R*T/(1-rho*b)-a*alpha(T)*rho**2/(1+2*b*rho-b**2*rho**2)
        # plt.plot(data[f'block_{bl}']['x'][:,-1],p,'--g',label='recalc')
        
        # Add literature data
        # x_D2_exp,x_D2_SU2,p_D2_exp,p_D2_SU2 = exp.TROVA()
        # plt.plot(x_D2_exp,p_D2_exp,'sk',markerfacecolor='none',label='TROVA D2 exp')
        # plt.plot(x_D2_SU2,p_D2_SU2,'--k',label='TROVA D2 SU2')
        #
        # plt.xlabel('x [mm]',fontsize=font)
        # plt.ylabel(var,fontsize=font)
        # plt.xticks(fontsize=font)
        # plt.yticks(fontsize=font)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        #########################
        # Plot multiple figures
        fig,axs = plt.subplots(3,3,sharex=True,sharey=True)
        plt.rcParams['text.usetex'] = True
        cmap='RdBu_r'
        count=0
        for i in range(3):
            for j in range(3):
                var_levels = vars_levels(vmins[:,count].min(axis=0),vmaxs[:,count].max(axis=0),10)
                for bl in range(n_bl):
                    bl+=1
                    
                    # Plot individual blocks for ALL variable
                    # axs[i,j].contour(data[f'block_{bl}']['x'], \
                    #            data[f'block_{bl}']['y'], \
                    #            data[f'block_{bl}'][var_names_[count]], \
                    #            levels=[1],colors=['k'],linewidths=2)
                    # plt.contour(data[f'block_{bl}']['x'], \
                    #            data[f'block_{bl}']['y'], \
                    #            data[f'block_{bl}'][var], \
                    #            levels=var_levels,colors=['grey'],linewidths=1)
                    axs[i,j].pcolormesh(data[f'block_{bl}']['x'], \
                                   data[f'block_{bl}']['y'], \
                                   data[f'block_{bl}'][var_names_[count]], \
                                   vmin=vmins[:,count].min(axis=0),\
                                   vmax=vmaxs[:,count].max(axis=0),\
                                   cmap=cmap)
                    # Turbine blade pitch
                    data['block_2']['y'][0,-1]-data['block_1']['y'][0,0]
                    corrmin = (data['block_1']['y'][0,1]-data['block_1']['y'][0,0])/2
                    corrmax = abs((data['block_2']['y'][0,-1]-data['block_2']['y'][0,-2])/2)
                    corr = corrmin+corrmax
                    axs[i,j].pcolormesh(data[f'block_{bl}']['x'], \
                                   data[f'block_{bl}']['y']+pitch+corr, \
                                   data[f'block_{bl}'][var_names_[count]], \
                                   vmin=vmins[:,count].min(axis=0),\
                                   vmax=vmaxs[:,count].max(axis=0),\
                                   cmap=cmap)

                    axs[i,j].set_title(var_names_[count])
                count+=1
        plt.tight_layout()
        plt.show()

        # Plot wall data
        if wall_dat:
            plt.rcParams['text.usetex'] = True
            for i,var_name in enumerate(var_names):
                plt.figure(figsize=(11.5,6))
                for j in range(locs):
                    if var_name=='p':
                        p  = data['wall'][var_name][:,j]
                        data['wall']['p'][:,j] = (p-p_ref)/(p[0]-p_ref)
                    plt.plot(180.-theta,data['wall'][var_name][:,j],linewidth=lw,label=f'j={j+1}')
                plt.xlabel(r'$\theta$')
                plt.ylabel(rf'{var_name}')
                plt.xlim(-10,190)
                plt.title(rf'{var_name} vs. $\theta$')
                plt.legend()
                # # plt.grid()()
                plt.show() # Don't move around otherwise computer crashes

        # Animation
        if anim:
            interval = 10
            frames = big_data[f'block_1']['nb_chkpnts']
            anim_field(x,y,big_data,var,vmin,vmax,n_bl,interval,frames,save,False)
            anim_line(x,y,big_data,var,vmin,vmax,n_bl,interval,frames,save)


    # Wall surface normal to j-dir
    else:
        # Friction
        if var=='Frhou':
            tau   = np.zeros((nw,nz,3,3))
            tauw  = np.zeros((nw,nz))
            nwall = np.zeros((nw,3))
        for bl in range(4):
            bl+=1
            # Gather individual blocks variables
            data[f'block_{bl}'],var_names,big_data[f'block_{bl}'] = \
            rv.read_planes(dir_data+f'plane_{plane_nb}_sol_bl{bl}.bin',chkpnt_nb,wall_plane,anim)
            if not is_curv:
                for item in data[f'block_{bl}'].keys():
                    data[f'block_{bl}'][item] = data[f'block_{bl}'][item].T
            vmin.append(data[f'block_{bl}'][var].min())
            vmax.append(data[f'block_{bl}'][var].max())
            # Gather individual blocks coordinates
            nx,ny,nz,x,y,z = rv.read_grid(dir_data+'/grid_bl{0}.bin'.format(bl),is_curv)
            z = z/L_ref+0.5
            # Get wall coordinates
            xw[nw_loc:nw_loc+nx]    = x[:nx,0]/L_ref
            yw[nw_loc:nw_loc+nx]    = y[:nx,0]/L_ref
            r[nw_loc:nw_loc+nx]     = np.sqrt(xw**2+yw**2)[nw_loc:nw_loc+nx]
            theta[nw_loc:nw_loc+nx] = np.arccos(x[:nx,0]/L_ref/r[nw_loc:nw_loc+nx])
            # Concatenate data over whole surface
            if bl==1:
                data[var] = data[f'block_{bl}'][var]
            else:
                data[var] = np.vstack((data[var],data[f'block_{bl}'][var]))

            # Friction
            if var=='Frhou':
                # Wall data
                tau[nw_loc:nw_loc+nx,:,0,0] = data[f'block_{bl}']['Frhou'][:nx,:]
                tau[nw_loc:nw_loc+nx,:,0,1] = data[f'block_{bl}']['Frhov'][:nx,:]
                tau[nw_loc:nw_loc+nx,:,0,2] = data[f'block_{bl}']['Frhow'][:nx,:]
                tau[nw_loc:nw_loc+nx,:,1,0] = data[f'block_{bl}']['Frhov'][:nx,:]
                tau[nw_loc:nw_loc+nx,:,1,1] = data[f'block_{bl}']['Grhov'][:nx,:]
                tau[nw_loc:nw_loc+nx,:,1,2] = data[f'block_{bl}']['Grhow'][:nx,:]
                tau[nw_loc:nw_loc+nx,:,2,0] = data[f'block_{bl}']['Frhow'][:nx,:]
                tau[nw_loc:nw_loc+nx,:,2,1] = data[f'block_{bl}']['Grhov'][:nx,:]
                tau[nw_loc:nw_loc+nx,:,2,2] = data[f'block_{bl}']['Hrhow'][:nx,:]

            nw_loc+=nx

        adjust=1
        ind_start  = dict_info['nx_bl1']//2
        ind_finish = dict_info['nx_bl1']+dict_info['nx_bl2']+ \
                     dict_info['nx_bl3']//2+1-adjust

        # Friction
        if var=='Frhou':
            with open(dir_data+'norm_surf.dat','r') as f:
                dat = np.loadtxt(f)
                nxwalldl = dat[:,0]
                nywalldl = dat[:,1]
                nxwall = dat[:,2]
                nywall = dat[:,3]
                dl     = dat[:,4]
                ijacob = dat[:,5]
                nwall[:,0],nwall[:,1] = nxwall,nywall
                nwalldl = np.sqrt(nxwalldl**2+nywalldl**2)
                data['nwall'] = nwall[ind_start:ind_finish]
            for i in range(nw):
                for j in range(nz):
                    tauw[i,j] = (tau[i,j]@nwall[i])[0]*ijacob[i]*nwalldl[i]

        # Concatenate all
        theta = np.hstack((np.pi-theta[ind_start:ind_finish][::-1], \
                np.hstack((np.pi+theta[:ind_start][::-1],np.pi+theta[ind_finish:][::-1]))))
        Theta,Z = np.meshgrid(theta,z)
        data    = np.vstack((data[var][ind_start:ind_finish][::-1], \
                  np.vstack((data[var][ind_finish:],data[var][:ind_start]))[::-1])).T
        if var=='Frhou':
            data = np.vstack((tauw[ind_start:ind_finish][::-1], \
                   np.vstack((tauw[ind_finish:],tauw[:ind_start]))[::-1])).T

        # Plot
        plt.figure(figsize=(12,12/2/np.pi))
        plt.rcParams['text.usetex'] = True
        cmap='RdBu_r'

        plt.contour(Theta,Z,data,vmin=min(vmin),vmax=max(vmax), \
                    levels=50,colors=['k'],linewidths=0.7)
        plt.pcolormesh(Theta,Z,data,vmin=min(vmin),vmax=max(vmax),cmap=cmap)

        plt.xlabel(rf'$\theta$')
        plt.ylabel(rf'$z/D$')
        plt.title(rf'${var}$')
        plt.colorbar()
        plt.show()

        # Animation
        if anim:
            vmin,vmax = [],[]
            big_data2 = d()
            frames = big_data['block_1']['nb_chkpnts']
            # Concatenate all for each snapshot
            for frame in range(frames):
                frame+=1
                big_data2[f'{frame}'] = d()
                for bl in range(4):
                    bl+=1
                    if bl==1:
                        big_data2[f'{frame}'][var] = big_data[f'block_{bl}'][f'{frame}'][var]
                    else:
                        big_data2[f'{frame}'][var] = np.vstack((big_data2[f'{frame}'][var], \
                            big_data[f'block_{bl}'][f'{frame}'][var]))
                vmin.append(big_data2[f'{frame}'][var].min())
                vmax.append(big_data2[f'{frame}'][var].max())
                big_data2[f'{frame}'][var] = \
                np.vstack((big_data2[f'{frame}'][var][ind_start:ind_finish][::-1], \
                np.vstack((big_data2[f'{frame}'][var][ind_finish:],                \
                           big_data2[f'{frame}'][var][:ind_start]))[::-1])).T

            interval = 20
            anim_field(Theta,Z,big_data2,var,min(vmin),max(vmax),4,interval,frames,save,True)


# Animate field/line
#===================

def anim_field(x,y,big_data,var,vmin,vmax,n_bl,interval,frames,save,wall_plane):

    fig,ax = plt.subplots(figsize=(11.5,6))
    xlimd = -2.5
    xlimf =  5
    ylimd = -2.
    ylimf =  2.
    # fig,ax = plt.subplots(figsize=(12,12/2/np.pi))
    # xlimd = 0
    # xlimf = 2*np.pi
    # ylimd = 0
    # ylimf = 1
    #ax.set_xlim(xlimd,xlimf)
    #ax.set_ylim(ylimd,ylimf)
    ax.set_xlabel(r'$x/D$')
    ax.set_ylabel(r'$y/D$')

    quad_list = []
    plot_list = []

    # Plane normal to z-axis
    if not wall_plane:
        var_levels1 = vars_levels(min(vmin),25,levels=20)
        var_levels2 = vars_levels(25,max(vmax),levels=70)
        var_levels = var_levels1+var_levels2
        var_levels = vars_levels(min(vmin),max(vmax),levels=100)
        surface = []
        for bl in range(n_bl):
            bl+=1
            surface.append(plt.pcolormesh(big_data[f'block_{bl}']['x'], \
                                     big_data[f'block_{bl}']['y'], \
                                     big_data[f'block_{bl}']['1'][var],    \
                                     # vmin=34.7,vmax=35.5,\
                                     vmin=min(vmin),vmax=max(vmax),\
                                     cmap='RdBu_r'))
                                     # vmin=-20,vmax=80,cmap='RdBu_r'))
        
        def animate(i):
            for bl in range(n_bl):
                # Plot new field
                bl+=1
                surface[bl-1].set_array(big_data[f'block_{bl}'][f'{i+1}'][var].flatten())
    
    # Wall surface normal to j-dir
    else:
        # surface = [plt.pcolormesh(x,y,big_data['1'][var], \
        #           vmin=vmin,vmax=vmax,cmap='RdBu_r')]
        surface = [plt.contour(x,y,big_data['1'][var],    \
                      vmin=vmin,vmax=vmax,levels=50,colors=['k'],linewidths=0.7)]

        def animate(i):
            # Remove previous field
            for field in surface[0].collections:
                field.remove()
            # Plot new field
            surface[0] = plt.contour(x,y,big_data[f'{i+1}'][var],\
                      vmin=vmin,vmax=vmax,levels=20,colors=['k'],linewidths=0.7)
            return surface[0].collections
        
    anim = animation.FuncAnimation(fig,animate,interval=interval,frames=frames,repeat=True)
    if save:
        anim.save('M_turb_mdm.mp4', writer=animation.FFMpegWriter())
    else:
        plt.show()


def anim_line(x,y,big_data,var,vmin,vmax,n_bl,interval,frames,save):

    fig = plt.figure()
    xlimd,xlimf = big_data['block_1']['x'][:,-1].min(),big_data['block_1']['x'][:,-1].max()
    ylimd,ylimf = 0.9e6,1.35e6
    plt.xlim(xlimd,xlimf)
    plt.ylim(ylimd,ylimf)
    line, = plt.plot([],[],'-b',linewidth=1.5)
    def animate(i):
        line.set_data(big_data['block_1']['x'][:,-1],big_data['block_1'][f'{i+1}'][var][:,-1])
        return line,

    anim = animation.FuncAnimation(fig,animate,interval=interval,frames=frames,repeat=True)
    if save:
        anim.save('p_in_air_RANS.mp4', writer=animation.FFMpegWriter())
    else:
        plt.show()



# Combine planes
#===============

def comb_planes(dir_data1,dir_data2,plane_nb):

    rv.write_planes(dir_data1,dir_data2,plane_nb)


# Plot restart files (fields)
#============================

def plot_restart(dir_data,var,tstar,is_2D,is_RANS,k):

    if tstar!=None:
        unit = str(tstar).split('.')[0].rjust(4,'0')
        deci = str(tstar).split('.')[1].ljust(4,'0')

    # General information
    dict_info = rv.read_info(dir_data)
    n_bl   = dict_info['nbloc']
    tscale = dict_info['Tscale']
    u_in   = dict_info['Uref']
    L_ref  = dict_info['L_ref']
    L_z = np.pi*L_ref
    nz = 48.
    dz = L_z/(nz-1)

    data = d()
    vmin,vmax = [],[]

    for bl in range(n_bl):
        bl+=1
        # Retrieve data from readvars
        if tstar==None:
            rest_file = dir_data+'restart_bl{}.bin'.format(bl)
        else:
            rest_file = dir_data+'restart{0}_{1}_bl{2}.bin'.format(unit,deci,bl)
        nx,ny,nz = dict_info['nx_bl{}'.format(bl)], \
        	        dict_info['ny_bl{}'.format(bl)], \
           	        dict_info['nz_bl{}'.format(bl)]
        if is_RANS: 
            ro,rou,rov,row,roe,nutil = rv.read_restart(rest_file,nx,ny,nz,is_2D,is_RANS)
        else:
            ro,rou,rov,row,roe = rv.read_restart(rest_file,nx,ny,nz,is_2D,is_RANS)
        # Store in dict
        nx,ny,nz,x,y,z = rv.read_grid(dir_data+'grid_bl{}.bin'.format(bl),True)
        data[f'block_{bl}'] = d()
        data[f'block_{bl}']['ro']  = ro
        data[f'block_{bl}']['rou'] = rou
        data[f'block_{bl}']['rov'] = rov
        data[f'block_{bl}']['row'] = row
        data[f'block_{bl}']['roe'] = roe
        if is_RANS:
            data[f'block_{bl}']['nutil'] = nutil
        data[f'block_{bl}']['x'],data[f'block_{bl}']['y'],data[f'block_{bl}']['z'] = x/L_ref,y/L_ref,z/L_ref
        if is_2D:
            vmin.append(data[f'block_{bl}'][var].min())
            vmax.append(data[f'block_{bl}'][var].max())
        else:
            vmin.append(data[f'block_{bl}'][var][:,:,k].min())
            vmax.append(data[f'block_{bl}'][var][:,:,k].max())


    plt.figure(figsize=(11.5,6))
    for bl in range(n_bl):
        bl+=1
        # Plot
        if is_2D:
            plt.pcolormesh(data[f'block_{bl}']['x'], \
                           data[f'block_{bl}']['y'], \
                           data[f'block_{bl}'][var], \
                           vmin=min(vmin),vmax=max(vmax),    \
                           cmap='plasma')#,shading='gouraud')
        else:
            plt.pcolormesh(data[f'block_{bl}']['x'], \
                           data[f'block_{bl}']['y'], \
                           data[f'block_{bl}'][var][:,:,k], \
                           vmin=min(vmin),vmax=max(vmax),    \
                           cmap='plasma')#,shading='gouraud')
    plt.colorbar()
    plt.show()


# Plot statistic total quantities
#================================

def plot_turb_total_q(dir_data,is_curv,is_RANS,var,model,field,expe,):

    data = rv.turb_total_q(dir_data,is_curv,is_RANS,var,model)

    # Plot total or post shock field
    if field:
        plt.figure()
        plt.rcParams['text.usetex'] = True
        cmap='RdBu_r'
        for bl in range(data['n_bl']):
            bl+=1

            plt.pcolormesh(data[f'block_{bl}']['x'], \
                           data[f'block_{bl}']['y'], \
                           data[f'block_{bl}'][var], \
                           vmin=data['vmin'],vmax=data['vmax'],
                           cmap=cmap)#,shading='gouraud')

            # Add contour
            plt.contour(data[f'block_{bl}']['x'], \
                        data[f'block_{bl}']['y'], \
                        data[f'block_{bl}'][var], \
                        levels=[300],colors=['k'])

        zero = np.zeros((1,1))
        plt.pcolormesh(zero,zero,zero, \
                       vmin=data['vmin'],vmax=data['vmax'],
                       cmap='RdBu_r',shading='gouraud')
        plt.xlabel(r'$x/D$',fontsize=font)
        plt.ylabel(r'$y/D$',fontsize=font)
        plt.xticks(fontsize=font)
        plt.yticks(fontsize=font)
        im = plt.pcolormesh(zero,zero,zero, \
                       vmin=data['vmin'],vmax=data['vmax'],
                       cmap='RdBu_r',shading='gouraud')
        cb = plt.colorbar(im)
        cb.set_label(label=f'${var}$', size=font, weight='bold')
        cb.ax.tick_params(labelsize=font)
        plt.tight_layout()

    # Extract traverse (Baumgartner)
    N = 1000
    line = np.linspace(-1,0,N)
    plt.plot(np.zeros(N)+1.1,line,'-k',linewidth=1.5)

    #########################
    # Plot multiple figures
    var_names_ = ['p0','T0','ro0','p2','T2','ro2','p02','T02','ro02']
    n_bl = data['n_bl']
    vmins,vmaxs = np.zeros((n_bl,len(var_names_))),np.zeros((n_bl,len(var_names_)))
    for bl in range(n_bl):
        bl+=1
        for i,var_ in enumerate(var_names_):
            vmins[bl-1,i] = data[f'block_{bl}'][var_].min()
            vmaxs[bl-1,i] = data[f'block_{bl}'][var_].max()
    fig,axs = plt.subplots(3,3,sharex=True,sharey=True)
    plt.rcParams['text.usetex'] = True
    count=0
    for i in range(3):
        for j in range(3):
            var_levels = vars_levels(vmins[:,count].min(axis=0),vmaxs[:,count].max(axis=0),10)
            for bl in range(n_bl):
                bl+=1
                
                axs[i,j].pcolormesh(data[f'block_{bl}']['x'], \
                               data[f'block_{bl}']['y'], \
                               data[f'block_{bl}'][var_names_[count]], \
                               vmin=vmins[:,count].min(axis=0),\
                               vmax=vmaxs[:,count].max(axis=0),\
                               cmap=cmap)

                axs[i,j].set_title(var_names_[count])
            count+=1
    plt.tight_layout()
    plt.show()

# Plot statistic fields
#======================

def plot_stats(dir_data,is_curv,is_RANS,var,field,stream,contours,levels,xz,yz,Lx,Cp,Prms,Cf,\
               Nu,expe,yp,BL,BL_scal,k,sl):

    data = rv.extr_stats(dir_data,var,is_curv,is_RANS,stream,contours,xz,yz,Lx,Cp,\
                         Prms,Cf,Nu,yp,BL,BL_scal,k,sl)

    # Inlet data
    u_in = data['u_in']
    p_in = data['p_in']
    rho_in = data['rho_in']
    norm = data['norm']
    normvar = data['normvar']
    vmin = data['vmin']
    vmax = data['vmax']

    # Wall coordinates
    xw = data['xw']
    yw = data['yw']
    nwall = data['nwall']

    if stream or yz or xz or BL or BL_scal:
        # Flattened data
        u_flat = data['u_flat']
        v_flat = data['v_flat']
        x_flat = data['x_flat']
        y_flat = data['y_flat']
        var_flat = data['var_flat']
        xw_r = data['xw_round']
        yw_r = data['yw_round']

    if contours:
        var_levels = vars_levels(vmin,vmax,levels)

    # Plot field
    if field:
        plt.figure(figsize=(9.5,3.5)) # half cylinder
        # plt.figure(figsize=(5,4)) # square figure
        plt.rcParams['text.usetex'] = True
        for bl in range(data['n_bl']):
            bl+=1

            # Additional contours
            # plt.contour(data[f'block_{bl}']['x'], \
            #             data[f'block_{bl}']['y'], \
            #             data[f'block_{bl}']['u'], \
            #             [0],colors=['black'],linestyles=['dashed'], \
            #             linewidths=2.0)
            #plt.contour(data[f'block_{bl}']['x'], \
            #            data[f'block_{bl}']['y'], \
            #            data[f'block_{bl}']['M'], \
            #            [1.45],colors=['black'], \
            #            linewidths=2.0)

            # Plot
            plt.pcolormesh(data[f'block_{bl}']['x'], \
                           data[f'block_{bl}']['y'], \
                           data[f'block_{bl}'][var], \
                           vmin=vmin,vmax=vmax,    \
                           cmap='RdBu_r')#,shading='gouraud')


            if contours:
                plt.contour(data[f'block_{bl}']['x'], \
                            data[f'block_{bl}']['y'], \
                            data[f'block_{bl}'][var], \
                            var_levels,colors=['grey'], \
                            linewidths=lw)
        xlim1,xlim2 = -5.,14.
        ylim1,ylim2 = -3.5,3.5
        if stream:
            lim = [xlim1,xlim2,ylim1,ylim2]
            plot_stream(lim,u_flat,v_flat,x_flat,y_flat,xw_r,yw_r)
        zero = np.zeros((1,1))
        plt.pcolormesh(zero,zero,zero, \
                       vmin=vmin,vmax=vmax, \
                       cmap='RdBu_r',shading='gouraud')
        plt.xlabel(r'$x/D$',fontsize=font)
        plt.ylabel(r'$y/D$',fontsize=font)
        plt.xticks(fontsize=font)
        plt.yticks(fontsize=font)
        im = plt.pcolormesh(zero,zero,zero, \
                       vmin=vmin,vmax=vmax, \
                       cmap='RdBu_r',shading='gouraud')
        cb = plt.colorbar(im)
        cb.set_label(label=f'${var}$', size=font, weight='bold')
        cb.ax.tick_params(labelsize=font)
        plt.tight_layout()

    # Centerline (Baumgartner)
    if var=='M_is':
        plot_centerline(data,'k','musica2 r134a LES')
        # Exp data
        x_pfg_exp,x_pfg_num,x_r134a_exp,x_r134a_num,\
        Mis_pfg_exp,Mis_pfg_num,Mis_r134a_exp,Mis_r134a_num = exp.baumgartner_Mis()
        plt.plot(x_pfg_exp,Mis_pfg_exp,'sr',markerfacecolor='none',markersize=5,label='Baum air exp')
        plt.plot(x_pfg_num,Mis_pfg_num,'-r',linewidth=1.5,label='Baum air RANS')
        plt.plot(x_r134a_exp,Mis_r134a_exp,'sb',markerfacecolor='none',markersize=5,label='Baum r134a exp')
        plt.plot(x_r134a_num,Mis_r134a_num,'-b',linewidth=1.5,label='Baum air RANS')
        plt.legend()
        plt.show()

    # Extract along line x
    plt.figure()
    plt.plot(data['block_1']['x'][:,-1],data['block_1'][var][:,-1],'b',\
             label=f'{var} along x')
    plt.legend()

    plt.figure()
    plt.plot(data['block_8']['x'][:,-1],data['block_8'][var][:,-1],'b',\
             label=f'{var} along x')
    plt.legend()

    if BL:
        plt.figure(figsize=(5,4))
        plt.rcParams['text.usetex'] = True
        # pos = [90,90,90] # 30 degrees
        pos = [675,600,450,381] # 75 degrees
        pos = [370] # 75 degrees
        styles = ['-.b','-^k','-og','-r']
        legends = [r'$\theta=30$  \textdegree',r'$\theta=60$  \textdegree',\
        r'$\theta=90$  \textdegree',r'$\theta=103.74$  \textdegree']
        # pos = [50,52,54] # separation
        for i in range(len(pos)):
            plot_bl_curv(data,styles[i],legends[i],pos[i])
        plt.legend(prop={'size':font},markerscale=markerscale)
        plt.tight_layout()
        # plt.xlim(-0.6,0.6)
        # plt.ylim(0,0.8)

    if BL_scal:
        plt.figure(figsize=(5,4))
        plt.rcParams['text.usetex'] = True
        # pos = [90,90,90] # 30 degrees
        pos = [675,600,450,381] # 75 degrees
        pos = [370] # 75 degrees
        styles = ['-.b','-^k','-og','-r']
        legends = [r'$\theta=30$  \textdegree',r'$\theta=60$  \textdegree',\
        r'$\theta=90$  \textdegree',r'$\theta=103.74$  \textdegree']
        # pos = [50,52,54] # separation
        for i in range(len(pos)):
            plot_bl_curv_scal(data,styles[i],legends[i],pos[i])
        plt.tight_layout()
        # plt.xlim(-0.6,0.6)
        # plt.ylim(0,0.8)

    if sl:
        # plt.figure(figsize=(12,2))
        plt.rcParams['text.usetex'] = True
        plot_sl(data,'k','')
        plt.tight_layout()

    if k:
        plt.figure(figsize=(12,8))
        plt.rcParams['text.usetex'] = True
        plot_k(data,'k','')
        plt.tight_layout()


    if yz:
        plt.figure(figsize=(6,2))
        plt.rcParams['text.usetex'] = True
        plot_yz(is_curv,x_flat,y_flat,var_flat,Lx,norm,var,'k','')
        # plt.legend(prop={'size':font},markerscale=markerscale)
        plt.tight_layout()


    if xz:
        plt.figure(figsize=(6,2))
        plt.rcParams['text.usetex'] = True
        plot_xz(x_flat,y_flat,var_flat,u_in,var,'k','')
        plt.legend(prop={'size':font},markerscale=markerscale)
        plt.tight_layout()


    if Cp:
        p_p0 = data['p/p0']
        M_is = data['M_is']
        xw = data['xw']
        plt.figure(figsize=(7,4))
        plt.rcParams['text.usetex'] = True
        plot_cp(xw,M_is,'b','')
        if expe: 
            x_cinn,Mis_cinn = exp.cinnella_LS59_Mis()
            x_kioc,Mis_kioc = exp.kiock_LS59()
            
            plt.plot(x_cinn,Mis_cinn,'-k',linewidth=1,label='Cinnella')
            plt.plot(x_kioc,Mis_kioc,'ok',markersize=5,label='Kiock')
        # plt.legend(prop={'size':font},markerscale=markerscale)
        plt.tight_layout()

    if Prms:
        prms = data['prms']
        plt.figure(figsize=(7,4))
        plt.rcParams['text.usetex'] = True
        plot_prms(theta,prms,'k','')
        plt.legend(prop={'size':font},markerscale=markerscale)
        plt.tight_layout()

    if Cf:
        plt.figure(figsize=(7,4))
        plt.rcParams['text.usetex'] = True
        plot_cf(theta,data,'k','')
        plt.legend(prop={'size':font},markerscale=markerscale)
        plt.tight_layout()

    if Nu:
        Nus  = data['Nus']
        Nusw = data['Nusw']
        plt.figure(figsize=(7,4))
        plt.rcParams['text.usetex'] = True
        plot_nus(theta,Nusw,'k','')
        plt.tight_layout()
        print('Nu =',Nus)

    if yp:
        yp_ = data['yp']
        xp_ = data['xp']
        xw  = data['xw']
        plt.rcParams['text.usetex'] = True
        fig,ax = plt.subplots(figsize=(7,4))
        plot_yp(xw,yp_,xp_,ax,'k','')
        plt.tight_layout()
        print('y+_avg =',np.mean(yp_))
        print('x+_avg =',np.mean(xp_))

    plt.show()


# Plot shear layers
# =================

def plot_sl(data,style,legend):

    # Interpolation grid
    nx,ny = 500,100
    x1,x2 = -0.35,1.
    y1,y2 = 0.3,0.8
    lim = [x1,x2,y1,y2]
    X,Y = create_cart(nx,ny,lim)
    X,Y = X.T,Y.T
    # Mask
    xw_r,yw_r = data['xw_round'],data['yw_round']
    coord_w = np.vstack((xw_r,yw_r)).T
    poly = matplotlib.patches.Polygon(coord_w,closed=True)
    samp_points = np.vstack((X.flatten(),Y.flatten())).T
    mask = poly.get_path().contains_points(samp_points)

    # Shear layer in blocks 1, 2, 3, 5
    x_flat = np.hstack((data[f'block_1']['x_flat'],data[f'block_2']['x_flat'],\
        data[f'block_3']['x_flat'],data[f'block_5']['x_flat']))
    y_flat = np.hstack((data[f'block_1']['y_flat'],data[f'block_2']['y_flat'],\
        data[f'block_3']['y_flat'],data[f'block_5']['y_flat']))
    # P
    P_k_flat = np.hstack((data[f'block_1']['P_k_flat'],data[f'block_2']['P_k_flat'],\
        data[f'block_3']['P_k_flat'],data[f'block_5']['P_k_flat']))
    P_k_flat = si.griddata((x_flat,y_flat),P_k_flat,(X,Y),method='cubic')
    P_k = np.ma.array(np.reshape(P_k_flat,(nx,ny)),mask=mask)
    # P compr
    P_c_flat = np.hstack((data[f'block_1']['P_c_flat'],data[f'block_2']['P_c_flat'],\
        data[f'block_3']['P_c_flat'],data[f'block_5']['P_c_flat']))
    P_c_flat = si.griddata((x_flat,y_flat),P_c_flat,(X,Y),method='cubic')
    P_c = np.ma.array(np.reshape(P_c_flat,(nx,ny)),mask=mask)
    # P shear
    P_s_flat = np.hstack((data[f'block_1']['P_s_flat'],data[f'block_2']['P_s_flat'],\
        data[f'block_3']['P_s_flat'],data[f'block_5']['P_s_flat']))
    P_s_flat = si.griddata((x_flat,y_flat),P_s_flat,(X,Y),method='cubic')
    P_s = np.ma.array(np.reshape(P_s_flat,(nx,ny)),mask=mask)
    # Others
    # dux_flat = np.hstack((data[f'block_1']['dux_flat'],data[f'block_2']['dux_flat'],\
    #     data[f'block_3']['dux_flat'],data[f'block_5']['dux_flat']))
    # dux_flat = si.griddata((x_flat,y_flat),dux_flat,(X,Y),method='cubic')
    # dux = np.ma.array(np.reshape(dux_flat,(nx,ny)),mask=mask)
    # duy_flat = np.hstack((data[f'block_1']['duy_flat'],data[f'block_2']['duy_flat'],\
    #     data[f'block_3']['duy_flat'],data[f'block_5']['duy_flat']))
    # duy_flat = si.griddata((x_flat,y_flat),duy_flat,(X,Y),method='cubic')
    # duy = np.ma.array(np.reshape(duy_flat,(nx,ny)),mask=mask)

    # Extract shear layer center
    ysl = np.zeros(nx,dtype=int)
    ysl[:] = np.argmax(P_k[:],axis=1)
    Ysl = Y[0,ysl]
    # Smooth out curve
    N = 20
    start,finish = N,nx-N
    Ysl = np.convolve(Ysl,np.ones(N)/N,mode='same')

    # Plot: shear layer center
    # plt.rcParams['text.usetex'] = True
    # plt.pcolormesh(X,Y,P_k,cmap='RdBu_r')
    # plt.xlim(x1,x2)
    # plt.ylim(y1,y2)
    plt.plot(X[start:finish,0],Ysl[start:finish],'k',linewidth=lw)
    # plt.xlabel(r'$x/D$',fontsize=font)
    # plt.ylabel(r'$y/D$',fontsize=font)
    # plt.xticks(fontsize=font)
    # plt.yticks(fontsize=font)

    # # Interpolate on shear layer center
    # P_k = P_k.flatten()
    # P_c = P_c.flatten()
    # P_s = P_s.flatten()
    # # dux = dux.flatten()
    # # duy = duy.flatten()
    # Ysl = Ysl[start:finish]
    # sl_P_k = si.griddata((X.flatten(),Y.flatten()),P_k,(X[start:finish,0],Ysl),method='cubic')
    # sl_P_c = si.griddata((X.flatten(),Y.flatten()),P_c,(X[start:finish,0],Ysl),method='cubic')
    # sl_P_s = si.griddata((X.flatten(),Y.flatten()),P_s,(X[start:finish,0],Ysl),method='cubic')
    # # sl_dux = si.griddata((X.flatten(),Y.flatten()),dux,(X[start:finish,0],Ysl),method='cubic')
    # # sl_duy = si.griddata((X.flatten(),Y.flatten()),duy,(X[start:finish,0],Ysl),method='cubic')
    # # Plot: evolution inside shear layer
    # plt.figure(figsize=(8,4))
    # plt.rcParams['text.usetex'] = True
    # plt.plot(X[start:finish,0],sl_P_k,'k',label=r'$P$')
    # plt.plot(X[start:finish,0],sl_P_c,':k',label=r'$P$ compr')
    # plt.plot(X[start:finish,0],sl_P_s,'--k',label=r'$P$ shear')
    # # plt.plot(X[start:finish,0],sl_dux/abs(sl_dux).max(),'-.k',\
    # #          label=r'$\frac{\partial u}{\partial x}$')
    # # plt.plot(X[start:finish,0],sl_duy/abs(sl_duy).max(),'-.r',\
    # #          label=r'$\frac{\partial u}{\partial y}$')
    # # plt.xlabel(r'$\eta/D$',fontsize=font))
    # plt.ylabel(r'$\phi L_{ref}/U^3$',fontsize=font)
    # plt.xticks(fontsize=font)
    # plt.yticks(fontsize=font)
    # # # plt.grid()
    # plt.legend(fontsize=font)
    # # plt.title('Termes de production de TKE')
    # plt.show()

# Plot TKE
# ========

def plot_k(data,style,legend):

    # Get data
    vmin,vmax,vminP,vmaxP,x_flat,y_flat,k_flat,P_flat = \
              data['vmink'] ,data['vmaxk'] ,data['vminP'] ,data['vmaxP'],   \
              data['x_flat'],data['y_flat'],data['k_flat'],data['P_flat']
    vminPc,vmaxPc,P_c_flat = data['vminPc'] ,data['vmaxPc'],data['P_c_flat']
    vminPs,vmaxPs,P_s_flat = data['vminPs'] ,data['vmaxPs'],data['P_s_flat']
    # TKE field
    var_levels = vars_levels(vmin,vmax,30)
    for bl in range(data['n_bl']):
        bl+=1
        plt.contour(data[f'block_{bl}']['x'], \
                    data[f'block_{bl}']['y'], \
                    data[f'block_{bl}']['k'], \
                    var_levels,colors=['black'], \
                    linewidths=0.8)
        # if bl==1:
        #     plt.plot(data[f'block_{bl}']['x'][566,155], \
        #              data[f'block_{bl}']['y'][566,155],'or')
        #     plt.plot(data[f'block_{bl}']['x'][573,167], \
        #              data[f'block_{bl}']['y'][573,167],'or')
        #     plt.plot(data[f'block_{bl}']['x'][578,174], \
        #              data[f'block_{bl}']['y'][578,174],'or')

        #     plt.plot(data[f'block_{bl}']['x'][300,262], \
        #              data[f'block_{bl}']['y'][300,262],'or')
        # if bl==2:
        #     plt.plot(data[f'block_{bl}']['x'][111,50], \
        #              data[f'block_{bl}']['y'][111,50],'or')
        #     plt.plot(data[f'block_{bl}']['x'][111,74], \
        #              data[f'block_{bl}']['y'][111,74],'or')
        #     plt.plot(data[f'block_{bl}']['x'][112,91], \
        #              data[f'block_{bl}']['y'][112,91],'or')

        #     plt.plot(data[f'block_{bl}']['x'][67,83], \
        #              data[f'block_{bl}']['y'][67,83],'or')
        #     plt.plot(data[f'block_{bl}']['x'][69,106], \
        #              data[f'block_{bl}']['y'][69,106],'or')
        #     plt.plot(data[f'block_{bl}']['x'][71,122], \
        #              data[f'block_{bl}']['y'][71,122],'or')

        #     plt.plot(data[f'block_{bl}']['x'][15,121], \
        #              data[f'block_{bl}']['y'][15,121],'or')
        #     plt.plot(data[f'block_{bl}']['x'][20,140], \
        #              data[f'block_{bl}']['y'][20,140],'or')
        #     plt.plot(data[f'block_{bl}']['x'][23,151], \
        #              data[f'block_{bl}']['y'][23,151],'or')
        # if bl==5:
        #     plt.plot(data[f'block_{bl}']['x'][300,7], \
        #              data[f'block_{bl}']['y'][300,7],'or')
        #     plt.plot(data[f'block_{bl}']['x'][300,54], \
        #              data[f'block_{bl}']['y'][300,54],'or')
        #     plt.plot(data[f'block_{bl}']['x'][300,89], \
        #              data[f'block_{bl}']['y'][300,89],'or')
        #     plt.plot(data[f'block_{bl}']['x'][300,115], \
        #              data[f'block_{bl}']['y'][300,115],'or')
        #     plt.plot(data[f'block_{bl}']['x'][300,136], \
        #              data[f'block_{bl}']['y'][300,136],'or')
        #     plt.plot(data[f'block_{bl}']['x'][300,154], \
        #              data[f'block_{bl}']['y'][300,154],'or')


    # plt.colorbar()
    xlim1,xlim2 = -1,5
    ylim1,ylim2 = -2,2
    plt.xlim(xlim1,xlim2)
    plt.ylim(ylim1,ylim2)
    plt.xlabel(r'$x/D$')
    plt.ylabel(r'$y/D$')
    plt.title(rf'Champ de TKE')
    # P_TKE field
    plt.figure(figsize=(12,8))
    var_levels = np.array(vars_levels(vminP,vmaxP,30))
    for bl in range(data['n_bl']):
        bl+=1
        plt.pcolormesh(data[f'block_{bl}']['x'], \
                       data[f'block_{bl}']['y'], \
                       data[f'block_{bl}']['P_k'], \
                       vmin=vminP,vmax=vmaxP, \
                       cmap='RdBu_r')
        plt.contour(data[f'block_{bl}']['x'], \
                    data[f'block_{bl}']['y'], \
                    data[f'block_{bl}']['P_k'], \
                    var_levels,colors=['black'], \
                    linewidths=0.8)

    plt.xlim(xlim1,xlim2)
    plt.ylim(ylim1,ylim2)
    plt.xlabel(r'$x/D$')
    plt.ylabel(r'$y/D$')
    plt.title(rf'Champ de production de TKE')

    # Profile
    plt.figure(figsize=(7,3))
    plt.rcParams['text.usetex'] = True
    plot_xz(x_flat,y_flat,k_flat,1,'TKE and Production',style,'TKE')
    # Profile
    # plt.figure(figsize=(7,3))
    plt.rcParams['text.usetex'] = True
    plot_xz(x_flat,y_flat,P_flat,1,'','--r','P')
    plot_xz(x_flat,y_flat,P_c_flat,1,'','--b','P compr')
    plot_xz(x_flat,y_flat,P_s_flat,1,'','--g','P shear')
    plt.legend()

# Plot Baumgartner centerline M_is
#=================================

def plot_centerline(data,style,legend):

    # Flattened data
    M_is = data['M_is_flat']
    # M = data['M_flat']
    x_flat = data['x_flat']
    y_flat = data['y_flat']

    # Get interp line
    L_ref  = 0.015 #dict_info['L_ref']
    xl,yl = exp.baumgartner_coords()
    xl=xl/L_ref
    yl=yl/L_ref
    plt.plot(xl,yl,':r',linewidth=5)
    line = np.vstack((xl,yl)).T

    # Interpolate
    M_is_interp = si.griddata((x_flat,y_flat),M_is,(line[:,0],line[:,1]),\
                              method='cubic')
    # M_interp = si.griddata((x_flat,y_flat),M,(line[:,0],line[:,1]),\
    #                           method='cubic')

    # Plot
    s = np.sqrt((xl[1:]-xl[:-1])**2+(yl[1:]-yl[:-1])**2) # curvilinear abscissae
    s = np.cumsum(s)
    s = (s-s.min()) # start s form 0
    s = s/s.max() # normalize by length
    s = s*0.877+0.037
    plt.figure()
    plt.plot(s,M_is_interp[:-1],style,label=legend)
    # plt.plot(s,M_interp[:-1],style,label=legend)
    plt.xlabel(r'$x$',fontsize=font)
    plt.ylabel(r'$M_{is}$',fontsize=font)
    # plt.ylabel(r'$M$',fontsize=font))
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    # plt.title(rf'Profil de ${var}$')

    plt.tight_layout()


# Plot BL profile
#================

def plot_bl_curv_turb(x_flat,y_flat,u_flat,v_flat,xw,yw,nwall,norm,style,legend):
    var_flat = np.sqrt(u_flat**2+v_flat**2)

    # # Get angles
    # x_ = np.array([1,0])
    # thetas = np.arccos(nwall@x_)
    # thet = -np.pi/2.
    # R = np.array([[np.cos(thet),-np.sin(thet)],[np.sin(thet),np.cos(thet)]])
    # twall = np.matmul(nwall,R)

    # # Create interp line
    # poss = [500,600,700] # indices of stations
    # n = 200   # nb of points in BL
    # l = 0.05  # non-dimensional length

    # for pos in poss:
    #     # Rotate line
    #     line = np.linspace(0.,l,n)
    #     line = np.vstack((line,np.zeros((n))))
    #     thet = np.pi-thetas[pos] # angle by which to rotate line
    #     R = np.array([[np.cos(thet),-np.sin(thet)],[np.sin(thet),np.cos(thet)]]) # rotation
    #     line = np.matmul(R,line) # rotated line
    #     line[0]+=xw[pos]
    #     line[1]+=yw[pos]
    
    #     # Velocity components
    #     u_interp = si.griddata((x_flat,y_flat),u_flat,(line[0,:],line[1,:]),method='cubic')
    #     v_interp = si.griddata((x_flat,y_flat),v_flat,(line[0,:],line[1,:]),method='cubic')

    #     # Get wall tangent velocity
    #     var_interp = np.vstack((u_interp,v_interp)).T@twall[pos]

    #     # Rotate BL profile
    #     thet = thet-np.pi/2.
    #     R = np.array([[np.cos(thet),-np.sin(thet)],[np.sin(thet),np.cos(thet)]]) # rotation
    #     var_interp = var_interp/norm*l*0.5
    #     var_interp = np.matmul(R,np.vstack((var_interp,np.linspace(0.,l,n))))
    #     # Plot over field
    #     plt.plot(line[0],line[1],'Grey',linewidth=lw)
    #     plt.plot(xw[pos]+var_interp[0],yw[pos]+var_interp[1],style,linewidth=lw)

    ## For comparison between computations ##
    # Create interp line
    pos = 600 # index of station
    n = 500   # nb of points in BL
    l = 0.05  # non-dimensional length

    # Rotate line
    line = np.linspace(0.,l,n)
    line = np.vstack((line,np.zeros((n))))
    thet = np.pi-thetas[pos] # angle by which to rotate line
    R = np.array([[np.cos(thet),-np.sin(thet)],[np.sin(thet),np.cos(thet)]]) # rotation
    line = np.matmul(R,line) # rotated line
    line[0]+=xw[pos]
    line[1]+=yw[pos]

    # Velocity components
    u_interp = si.griddata((x_flat,y_flat),u_flat,(line[0,:],line[1,:]),method='cubic')
    v_interp = si.griddata((x_flat,y_flat),v_flat,(line[0,:],line[1,:]),method='cubic')

    # Get wall tangent velocity
    var_interp = np.vstack((u_interp,v_interp)).T@twall[pos]/norm

    # Plot
    plt.plot(line[0],line[1],'Grey',linewidth=lw)
    plt.plot(var_interp,np.linspace(0.,l,n),style,linewidth=lw,label=legend)
    plt.xlim(-0.1,1.5)
    plt.ylim(0,l)
    plt.xlabel(r'$u/U_{in}$')
    plt.ylabel(r'$y/D$')
    plt.title(rf'Profil de vitesse $u$')
    # plt.grid()(which='minor',color='grey',alpha=0.5)
    # plt.grid()(which='major',color='black',alpha=0.7)

def plot_bl_curv(data,style,legend,pos):

    # Inlet data
    u_in   = data['u_in']
    rho_in = data['rho_in']
    T_in   = data['T_in']
    p_in   = data['p_in']
    mu_in  = data['mu_in']
    norm   = data['norm']
    L_ref  = data['L_ref']
    print(u_in,L_ref)

    # Wall coordinates
    theta = data['theta']
    r  = data['r']
    xw = data['xw']
    yw = data['yw']
    nwall = data['nwall']

    # Flattened data
    bl = 3
    u_flat = data[f'block_{bl}']['u_flat']
    v_flat = data[f'block_{bl}']['v_flat']
    rho_flat = data[f'block_{bl}']['rho_flat']
    T_flat = data[f'block_{bl}']['T_flat']
    p_flat = data[f'block_{bl}']['p_flat']
    x_flat = data[f'block_{bl}']['x_flat']
    y_flat = data[f'block_{bl}']['y_flat']
    xw_r = data['xw_round']
    yw_r = data['yw_round']
    var_flat = np.sqrt(u_flat**2+v_flat**2)

    # Get angles
    x_ = np.array([1,0])
    thetas = np.arccos(nwall@x_)
    thet = -np.pi/2.
    R = np.array([[np.cos(thet),-np.sin(thet)],[np.sin(thet),np.cos(thet)]])
    twall = np.matmul(nwall,R)

    # # Create interp line
    # poss = [500,600,650] # indices of stations
    # n = 200   # nb of points in BL
    # l = 0.05  # non-dimensional length

    # for pos in poss:
    #     # Rotate line
    #     line = np.linspace(0.,l,n)
    #     line = np.vstack((line,np.zeros((n))))
    #     thet = np.pi-thetas[pos] # angle by which to rotate line
    #     R = np.array([[np.cos(thet),-np.sin(thet)],[np.sin(thet),np.cos(thet)]]) # rotation
    #     line = np.matmul(R,line) # rotated line
    #     line[0]+=xw[pos]
    #     line[1]+=yw[pos]
    # 
    #     # Velocity components
    #     u_interp = si.griddata((x_flat,y_flat),u_flat,(line[0,:],line[1,:]),method='cubic')
    #     v_interp = si.griddata((x_flat,y_flat),v_flat,(line[0,:],line[1,:]),method='cubic')

    #     # Get wall tangent velocity
    #     var_interp = np.vstack((u_interp,v_interp)).T@twall[pos]

    #     # Rotate BL profile
    #     thet = thet-np.pi/2.
    #     R = np.array([[np.cos(thet),-np.sin(thet)],[np.sin(thet),np.cos(thet)]]) # rotation
    #     var_interp = var_interp/norm*l*0.5
    #     var_interp = np.matmul(R,np.vstack((var_interp,np.linspace(0.,l,n))))
    #     # Plot over field
    #     plt.plot(line[0],line[1],'Grey',linewidth=lw)
    #     plt.plot(xw[pos]+var_interp[0],yw[pos]+var_interp[1],style,linewidth=lw)

    ## For comparison between computations ##
    # Create interp line
    #pos = 52 # index of station
    n = 200   # nb of points in BL
    l = 0.01   # non-dimensional length

    # Rotate line
    line = np.linspace(0.,l,n)
    line = np.vstack((line,np.zeros((n))))
    thet = np.pi-thetas[pos] # angle by which to rotate line
    R = np.array([[np.cos(thet),-np.sin(thet)],[np.sin(thet),np.cos(thet)]]) # rotation
    line = np.matmul(R,line) # rotated line
    line[0]+=xw[pos]
    line[1]+=yw[pos]
    theta_pos = -np.arctan(yw[pos]/xw[pos]) # angular position
    x_bl = theta_pos*0.5/L_ref # dimensional equivalent FP BL location
    # print('Re_x =',rho_in*u_in*x_bl/mu_in)

    # Velocity components
    u_interp = si.griddata((x_flat,y_flat),u_flat,(line[0,:],line[1,:]),method='cubic')
    v_interp = si.griddata((x_flat,y_flat),v_flat,(line[0,:],line[1,:]),method='cubic')
    rho_interp = si.griddata((x_flat,y_flat),rho_flat,(line[0,:],line[1,:]),method='cubic')
    T_interp = si.griddata((x_flat,y_flat),T_flat,(line[0,:],line[1,:]),method='cubic')
    p_interp = si.griddata((x_flat,y_flat),p_flat,(line[0,:],line[1,:]),method='cubic')

    # Get wall tangent velocity
    #norm = data['utau'][pos]
    var_interp = np.vstack((u_interp,v_interp)).T@twall[pos]/norm
    y_bl = np.linspace(0.,l,n)

    # Scalings
    # print(mu_in,x_bl,rho_in,u_in,theta_pos,yw[pos],xw[pos],l)
    y_bl = y_bl/np.sqrt((mu_in*x_bl/rho_in/u_in)) # laminar BL scaling (Gloerfelt et al. stab lam BL)
    # y_bl = y_bl*trapezoid(np.sqrt(p_interp/p_in)*T_in/T_interp,y_bl) # Howarth 1947
    # Van Driest scaling
    #uvd = trapezoid(np.sqrt(rho_interp/rho_interp[0]),var_interp)
    #var_interp = var_interp/uvd
    #y_bl = y_bl*rho_interp*data['utau'][pos]/data['muw'][pos]

    # Plot
    #plt.plot(line[0],line[1],'Grey',linewidth=lw)
    plt.plot(var_interp,y_bl,style,linewidth=lw,markevery=3,\
        markersize=markersize,markerfacecolor='none',label=legend)
    plt.xlim(-0.1,0.9)
    plt.ylim(0,y_bl.max())
    plt.xlabel(r'$u/U_{in}$',fontsize=font)
    plt.ylabel(r'$n^*$',fontsize=font)
    # plt.xticks([0,0.5,1,1.5],fontsize=font)
    plt.xticks([0,0.5,1,],fontsize=font)
    plt.yticks(fontsize=font)
    # plt.title(rf'Profil de vitesse $u$')
    # plt.grid(which='minor',color='grey',alpha=0.5)
    # plt.grid(which='major',color='black',alpha=0.7)


def plot_bl_curv_scal(data,style,legend,pos):

    # Inlet data
    u_in   = data['u_in']
    rho_in = data['rho_in']
    T_in   = data['T_in']
    p_in   = data['p_in']
    mu_in  = data['mu_in']
    norm   = data['norm']
    L_ref  = data['L_ref']
    #print(u_in,L_ref,norm)

    # Wall coordinates
    theta = data['theta']
    r  = data['r']
    xw = data['xw']
    yw = data['yw']
    nwall = data['nwall']

    # Flattened data
    bl = 3
    u_flat = data[f'block_{bl}']['u_flat']
    v_flat = data[f'block_{bl}']['v_flat']
    rho_flat = data[f'block_{bl}']['rho_flat']
    T_flat = data[f'block_{bl}']['T_flat']
    p_flat = data[f'block_{bl}']['p_flat']
    x_flat = data[f'block_{bl}']['x_flat']
    y_flat = data[f'block_{bl}']['y_flat']
    var_flat = data[f'block_{bl}']['var_flat']
    xw_r = data['xw_round']
    yw_r = data['yw_round']

    # Get angles
    x_ = np.array([1,0])
    thetas = np.arccos(nwall@x_)
    thet = -np.pi/2.
    R = np.array([[np.cos(thet),-np.sin(thet)],[np.sin(thet),np.cos(thet)]])
    twall = np.matmul(nwall,R)

    # # Create interp line
    # poss = [500,600,650] # indices of stations
    # n = 200   # nb of points in BL
    # l = 0.05  # non-dimensional length

    # for i,pos in enumerate(poss):
    #     # Rotate line
    #     line = np.linspace(0.,l,n)
    #     line = np.vstack((line,np.zeros((n))))
    #     thet = np.pi-thetas[pos] # angle by which to rotate line
    #     R = np.array([[np.cos(thet),-np.sin(thet)],[np.sin(thet),np.cos(thet)]]) # rotation
    #     line = np.matmul(R,line) # rotated line
    #     line[0]+=xw[pos]
    #     line[1]+=yw[pos]
    #     var_interp = si.griddata((x_flat,y_flat),var_flat,(line[0,:],line[1,:]),\
    #                               method='cubic')
    #     # Rotate BL profile
    #     thet = thet-np.pi/2.
    #     R = np.array([[np.cos(thet),-np.sin(thet)],[np.sin(thet),np.cos(thet)]]) # rotation
    #     var_interp = var_interp/normvar*l*0.5
    #     var_interp = np.matmul(R,np.vstack((var_interp,np.linspace(0.,l,n))))
    #     # Plot over field
    #     plt.plot(line[0],line[1],'Grey',linewidth=lw)
    #     plt.plot(xw[pos]+var_interp[0],yw[pos]+var_interp[1],style,linewidth=lw)

    ## For comparison between computations ##
    # Create interp line
    # pos = 600 # index of station
    n = 200   # nb of points in BL
    l = 0.01  # non-dimensional length

    # Rotate line
    line = np.linspace(0.,l,n)
    line = np.vstack((line,np.zeros((n))))
    thet = np.pi-thetas[pos] # angle by which to rotate line
    R = np.array([[np.cos(thet),-np.sin(thet)],[np.sin(thet),np.cos(thet)]]) # rotation
    line = np.matmul(R,line) # rotated line
    line[0]+=xw[pos]
    line[1]+=yw[pos]
    theta_pos = -np.arctan(yw[pos]/xw[pos]) # angular position
    x_bl = theta_pos*0.5/L_ref # dimensional equivalent FP BL location
    var_interp = si.griddata((x_flat,y_flat),var_flat,(line[0,:],line[1,:]),\
                              method='cubic')
    # norm = var_interp[0] # norm is wall value
    #norm = T_in
    norm = data['Twa'][pos]
    print(norm)
    var_interp = var_interp/norm

    # Scalings
    print(theta_pos)
    y_bl = np.linspace(0.,l,n)
    y_bl = y_bl/np.sqrt((mu_in*x_bl/rho_in/u_in)) # laminar BL scaling (Gloerfelt et al. stab lam BL)

    # Plot
    plt.plot(var_interp,y_bl,style,linewidth=lw,markevery=markevery,\
        markersize=markersize,markerfacecolor='none',label=legend)
    # plt.xlim(0.95,2.2)
    plt.ylim(0,y_bl.max())
    plt.xlabel(r'$T_{rms}/T_w$',fontsize=font)#+r'$/\rho_{in}$'
    plt.ylabel(r'$n^*$',fontsize=font)
    plt.xticks([2e-4,5e-4],fontsize=font)
    # plt.xticks([0.996,0.998,1.],fontsize=font)
    plt.yticks(fontsize=font)
    # plt.title(rf'Profil de ${var}$')
    # plt.grid()(which='minor',color='grey',alpha=0.5)
    # plt.grid()(which='major',color='black',alpha=0.7)


def plot_bl(var,yp,varp,style,legend):
    plt.plot(yp,varp,style,linewidth=lw,markevery=markevery,\
        markersize=markersize,markerfacecolor='none',label=legend)
    plt.ylim(0,)
    plt.xlim(0.2,200)
    plt.xscale('log')
    plt.xlabel(r'$y^+$',fontsize=font)
    plt.ylabel(rf'${var}^+$',fontsize=font)
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    # plt.title(rf'Profil de ${var}^+$')
    # plt.grid()(which='minor',color='grey',alpha=0.5)
    # plt.grid()(which='major',color='black',alpha=0.7)


# Plot yz
#========

def plot_yz(is_curv,x_flat,y_flat,var_flat,Lx,norm,var,style,legend):
    lim = [Lx-1e-6,Lx,-3,3]
    nx  = 2
    ny  = 500

    X,Y = create_cart(nx,ny,lim)
    var_interp = si.griddata((x_flat,y_flat),var_flat,(X,Y),method='cubic')

    if len(var)>1 and (var[0]==var[1] or var=='uv' or var=='uw' or var=='vw'):
        norm = norm**2
        plt.ylabel(fr'${var}$'+r'$/U_{in}^2$',fontsize=font)
        plt.title(fr'Tranche suivant $y$ de ${var}$' \
                 +r'$/U_{in}^2$'+r' à $x/D={0}$'.format(Lx),fontsize=font)
    else:
        plt.ylabel(r'${}$'.format(var)+r'$/U_{in}$',fontsize=font)
        plt.title(r'Tranche suivant $y$ de ${0}$'.format(var) \
             +r'$/U_{in}$'+r' à $x/D={0}$'.format(Lx),fontsize=font)

    plt.plot(Y[:,-1],var_interp[:,-1]/norm,style,linewidth=lw,markevery=markevery,\
        markersize=markersize,markerfacecolor='none',label=legend)
    plt.xlim(Y[:,-1][0],Y[:,-1][-1])
    plt.xlabel(r'$y/D$',fontsize=font)
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    # plt.grid()(visible=True)


# Plot xz
#========

def plot_xz(x_flat,y_flat,var_flat,norm,var,style,legend):
    lim = [0.5,10.,-1e-6,1e-6]
    nx_cart  = 2000
    ny_cart  = 3
    X,Y = create_cart(nx_cart,ny_cart,lim)
    var_interp = si.griddata((x_flat,y_flat),var_flat,(X,Y),method='cubic')
    
    plt.plot(X[1,:],var_interp[1,:]/norm,style,linewidth=lw,markevery=markevery,\
        markersize=markersize,markerfacecolor='none',label=legend)
    plt.xlim(0.5,10) #X[1,:][-1]
    plt.xlabel(r'$x/D$',fontsize=font)
    #plt.ylabel(r'${}$'.format(var)+r'$/U_{in}$',fontsize=font)
    plt.ylabel(f'{var}',fontsize=font)
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    #plt.title(r'Tranche suivant $x$ de ${0}$'.format(var)+r'$/U_{in}$ à $y/D=0$',fontsize=font)
    plt.title(rf'Tranche suivant $x$ de {var} à $y/D=0$',fontsize=font)
    # plt.grid()(visible=True)


# Plot cp
#========

def plot_cp(theta,cp,style,legend):
    x_ticks = [0.2*i for i in range(6)]
    plt.plot(theta,cp,style,linewidth=lw,markevery=markevery,\
        markersize=markersize,markerfacecolor='none',label=legend)
    #plt.xlim(-10,190)
    # plt.ylim(-2,1.5)
    plt.xlabel(r'$x/c$',fontsize=font)
    plt.ylabel(r'$C_p$',fontsize=font)
    plt.xticks(x_ticks,fontsize=font)
    plt.yticks(fontsize=font)
    #plt.title(r'$C_p$ vs. $\theta$',fontsize=font)
    # plt.grid()(visible=True)


# Plot prms
#==========

def plot_prms(theta,prms,style,legend):
    t_ticks = [i*30 for i in range(7)]
    plt.plot(theta,prms,style,linewidth=lw,markevery=markevery,\
        markersize=markersize,markerfacecolor='none',label=legend)
    plt.xlim(-10,190)
    plt.ylim(0,0.025)
    plt.xlabel(r'$\theta$',fontsize=font)
    plt.ylabel(r'$P_{rms}$',fontsize=font)
    plt.xticks(t_ticks,fontsize=font)
    plt.yticks(fontsize=font)
    # plt.title(r'$P_{rms}$ vs. $\theta$',fontsize=font)
    # plt.grid()(visible=True)

# Plot cf
#========

def plot_cf(theta,data,style,legend):
    # Inlet data
    u_in   = data['u_in']
    rho_in = data['rho_in']
    p_in   = data['p_in']
    mu_in  = data['mu_in']
    norm   = data['norm']
    L_ref  = data['L_ref']
    # Scaling
    xw = data['xw']
    yw = data['yw']
    # theta_pos = np.pi-np.arctan(yw[pos]/xw[pos]) # angular position
    x_bl = theta*0.5/L_ref # dimensional equivalent FP BL location

    # cf = data['cf']*np.sqrt(rho_in*u_in*x_bl/mu_in)
    cf = data['cf']
    theta_sep = data['theta_sep']

    t_ticks = [i*30 for i in range(7)]
    print('Theta_sep =',theta_sep)
    plt.plot(theta,cf,style,linewidth=lw,markevery=markevery,\
        markersize=markersize,markerfacecolor='none',label=legend)
    plt.xlim(theta[-1]-10,theta[0]+10)
    plt.xlabel(r'$\theta$',fontsize=font)
    plt.ylabel(r'$C_f$',fontsize=font)
    plt.xticks(t_ticks,fontsize=font)
    plt.yticks(fontsize=font)
    plt.plot([-10,190],[0,0],'--k')
    #plt.title(r'$C_f$ vs. $\theta$',fontsize=font)
    # # plt.grid()(visible=True)


# Plot Nusselt
#=============

def plot_nus(theta,Nus,style,legend):
    t_ticks = [i*30 for i in range(7)]
    plt.plot(theta,Nus,style,linewidth=lw,markevery=markevery,\
        markersize=markersize,markerfacecolor='none',label=legend)
    plt.xlim(theta[-1]-10,theta[0]+10)
    plt.ylim(0,)
    plt.xlabel(r'$\theta$',fontsize=font)
    plt.ylabel(r'$Nu$',fontsize=font)
    plt.xticks(t_ticks,fontsize=font)
    plt.yticks(fontsize=font)
    plt.title(r'$Nu$ vs. $\theta$',fontsize=font)
    # plt.grid()(visible=True)


# Plot y^+
#=========

def plot_yp(theta,yp,xp,ax,style,legend):
    # t_ticks = [i*30 for i in range(7)]
    # plt.plot(theta,yp,style,linewidth=lw,label=legend)
    # plt.plot(theta,xp,'-'+style,linewidth=lw,label=legend)
    # plt.xlim(theta[-1]-10,theta[0]+10)
    # plt.ylim(0,)
    # plt.xlabel(rf'$\theta$',fontsize=font)
    # plt.ylabel(rf'$y^+$',fontsize=font)
    # plt.xticks(t_ticks,fontsize=font)
    # plt.yticks(fontsize=font)
    # plt.title(rf'$y^+$ vs. $\theta$',fontsize=font)
    # # plt.grid()(visible=True)


    ax.plot(theta,yp,style,linewidth=lw,label=r'$n^+$')
    # ax.set_xlabel(rf'$\theta$ \textdegree ',fontsize=font)


    plt.ylabel(rf'$n^+$',fontsize=font)
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    plt.ylim(0,)

    ax2=ax.twinx()
    ax2.plot(theta,xp,'-.r',linewidth=lw,label=r'$t^+$')
    # plt.xlim(theta[-1]-10,theta[0]+10)
    plt.ylim(0,)
    plt.ylabel(rf'$t^+$',fontsize=font)
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    # plt.title(rf'$y^+$ vs. $\theta$',fontsize=font)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2,prop={'size':font},markerscale=markerscale)
    # ax2.legend(prop={'size':font},markerscale=markerscale)
    ## plt.grid()(visible=True)




# Plot streamlines
#=================

def plot_stream(lim,u_flat,v_flat,x_flat,y_flat,xw_r,yw_r):

    # Create cartesian grid
    X,Y = create_cart(50,50,lim)

    # Interpolate data on regular grid
    u_interp = si.griddata((x_flat,y_flat),u_flat,(X,Y),method='cubic')
    v_interp = si.griddata((x_flat,y_flat),v_flat,(X,Y),method='cubic')

    # Mask
    coord_w = np.vstack((xw_r,yw_r)).T
    poly = matplotlib.patches.Polygon(coord_w,closed=True)
    samp_points = np.vstack((X.flatten(),Y.flatten())).T
    mask = poly.get_path().contains_points(samp_points)
    
    u_mask = np.ma.array(u_interp,mask=mask)
    v_mask = np.ma.array(v_interp,mask=mask)

    # Stream plot
    plt.streamplot(X,Y,u_mask,v_mask,density=[3,2], \
                   integration_direction="forward",maxlength=5, \
                   arrowstyle="->",linewidth=0.5,color='k')


# Interpolate data on regular rid
#================================

def interp_data(lim,var_flat,x_flat,y_flat,xw_r,yw_r,nX,nY):

    # Create cartesian grid
    X,Y = create_cart(nX,nY,lim)

    # Interpolate data on regular grid
    var_interp = si.griddata((x_flat,y_flat),var_flat,(X,Y),method='cubic')

    # Mask
    coord_w = np.vstack((xw_r,yw_r)).T
    poly = matplotlib.patches.Polygon(coord_w,closed=True)
    samp_points = np.vstack((X.flatten(),Y.flatten())).T
    mask = poly.get_path().contains_points(samp_points)
    var_mask = np.ma.array(var_interp,mask=mask)

    return var_mask


# Plot vorticity criterias
#=========================

def plot_vort(dir_data,crit,nX,nY,py,pa):

    # Specific modules
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage import measure
    from pyevtk.hl import gridToVTK

    vort = rv.read_vort(dir_data,crit)
    n_bl = vort['n_bl']
    vmin = vort['vmin']
    vmax = vort['vmax']
    u_in = vort['u_in']
    L_ref = vort['L_ref']
    nz = vort['nz']
    lim = [-1,15,-6,6] # [xmin,xmax,ymin,ymax]

    x = vort['x']
    y = vort['y']
    xw = vort['xw']
    yw = vort['yw']

    if pa:
        for bl in range(n_bl):
            bl+=1
            z_3d = np.repeat(vort['block_{}'.format(bl)]['z'][np.newaxis,:], \
                             vort['block_{}'.format(bl)]['ny'],axis=0)
            z_3d = np.repeat(z_3d[np.newaxis,...], \
                      vort['block_{}'.format(bl)]['nx'],axis=0)
            del vort[f'block_{bl}']['nx']
            del vort[f'block_{bl}']['ny']
            del vort[f'block_{bl}']['z']
            gridToVTK(dir_data+'Paraview/salut_bl{}'.format(bl),vort['block_{}'.format(bl)]['x'], \
                      vort[f'block_{bl}']['y'],z_3d,      \
                      pointData=vort[f'block_{bl}'])
                                 # 'vort': vort['block_{}'.format(bl)]['vort']})
            # gridToVTK(dir_data+'Paraview/vort_bl{}'.format(bl),vort['block_{}'.format(bl)]['x'], \
            #           vort['block_{}'.format(bl)]['y'],z_3d,      \
            #           pointData={'vort': vort['block_{}'.format(bl)]['vort']})
    
    if py:     
        isoval = .5
        vort_3d = np.zeros((nY,nX,nz))
        crit_3d = np.zeros((nY,nX,nz))
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111, projection='3d')
        for z in range(nz):
           vort_3d[:,:,z] = interp_data(lim,vort['vort'][z],x,y,xw,yw,nX,nY)*L_ref/u_in
           crit_3d[:,:,z] = interp_data(lim,vort[crit][z],x,y,xw,yw,nX,nY)*L_ref/u_in
        verts,faces,normals,values = measure.marching_cubes(crit_3d.T,   \
                                    isoval,gradient_direction='ascent', \
                                    allow_degenerate=True,method='lewiner')

        mesh = Poly3DCollection(verts[faces],alpha=0.5)
        ax.add_collection3d(mesh)
        ax.view_init(elev=20,azim=-15)
        ax.dist = 6
        ax.axes.set_xlim3d(left=0,right=nX) 
        ax.axes.set_ylim3d(bottom=-20,top=nX) 
        ax.axes.set_zlim3d(bottom=0,top=nX)
        plt.show()


# Create cartesian grid
#======================

def create_cart(nx,ny,lim):

    xmin,xmax,ymin,ymax = lim[0],lim[1],lim[2],lim[3]
    x   = np.linspace(xmin,xmax,nx)
    y   = np.linspace(ymin,ymax,ny)
    X,Y = np.meshgrid(x,y)

    return X,Y


# Write to paraview files
#========================
def write_para(dir_data,tstar,rest,plane,stats,plane_nb,chkpnt_nb,anim):

    # Specific modules
    from pyevtk.hl import gridToVTK

    plane_nb = str(plane_nb).rjust(3,'0')
    if tstar!=None:
        unit = str(tstar).split('.')[0].rjust(4,'0')
        deci = str(tstar).split('.')[1].ljust(4,'0')

    # General information
    dict_info = rv.read_info(dir_data)
    n_bl   = dict_info['nbloc']
    tscale = dict_info['Tscale']
    u_in   = dict_info['Uref']
    L_ref  = dict_info['L_ref']
    L_z = L_ref
    nz = dict_info['nz_bl1']
    dz = L_z/(nz-1)

    data = d()
    vmin,vmax = [],[]

    for bl in range(n_bl):
        bl+=1
        # Retrieve data from readvars
        if tstar==None:
            rest_file = dir_data+f'restart_bl{bl}.bin'
        else:
            rest_file = dir_data+f'restart{unit}_{deci}_bl{bl}.bin'
        nx,ny,nz,x,y,z = rv.read_grid(dir_data+'grid_bl{}.bin'.format(bl),True)
        nx,ny,nz = dict_info[f'nx_bl{bl}'], \
                   dict_info[f'ny_bl{bl}'], \
                   dict_info[f'nz_bl{bl}']
        if rest:
            ro,rou,rov,row,roe = rv.read_restart(rest_file,nx,ny,nz,False,False)
            # Store in dict
            data[f'block_{bl}'] = d()
            data[f'block_{bl}']['ro']  = ro
            data[f'block_{bl}']['rou'] = rou
            data[f'block_{bl}']['rov'] = rov
            data[f'block_{bl}']['row'] = row
            data[f'block_{bl}']['roe'] = roe
            data[f'block_{bl}']['x'],data[f'block_{bl}']['y'] = x,y

            # Make coordinates 3D for Paraview
            data[f'block_{bl}']['x'] = np.repeat(x[:,:,np.newaxis],nz,axis=2)/L_ref
            data[f'block_{bl}']['y'] = np.repeat(y[:,:,np.newaxis],nz,axis=2)/L_ref
            z_3d = np.repeat(z   [np.newaxis,...],ny,axis=0)
            z_3d = np.repeat(z_3d[np.newaxis,...],nx,axis=0)/L_ref

            # Save to file
            gridToVTK(dir_data+f'par_restart_bl{bl}',data[f'block_{bl}']['x'], \
                      data[f'block_{bl}']['y'],z_3d, \
                      pointData=data[f'block_{bl}'])

        if plane:
            # Make coordinates 3D for Paraview
            x = np.repeat(x[:,:,np.newaxis],1,axis=2)/L_ref
            y = np.repeat(y[:,:,np.newaxis],1,axis=2)/L_ref
            z = np.zeros((1))
            z_3d = np.repeat(z   [np.newaxis,...],ny,axis=0)
            z_3d = np.repeat(z_3d[np.newaxis,...],nx,axis=0)
            if not anim:
                plane_file= dir_data+f'plane_{plane_nb}_sol_bl{bl}.bin'
                plane,var_names,planes = rv.read_planes(plane_file,chkpnt_nb,False,anim)
                for var in var_names:
                    plane[var] = np.repeat(plane[var][:,:,np.newaxis],1,axis=2)

                # Save to file
                gridToVTK(dir_data+f'par_planes/par_plane_{plane_nb}_{chkpnt_nb}_bl{bl}',x,y,z_3d, \
                          pointData=plane)
            
            else:
                plane_file= dir_data+f'plane_{plane_nb}_sol_bl{bl}.bin'
                plane,var_names,planes = rv.read_planes(plane_file,chkpnt_nb,False,anim)
                nb_chkpnts = planes['nb_chkpnts']
                planes_ = d()
                for i in range(nb_chkpnts):
                    i+=1
                    planes_[f'{i}'] = d()
                    for var in var_names:
                        planes_[f'{i}'][var] = np.repeat(planes[f'{i}'][var][:,:,np.newaxis],\
                                                                                    1,axis=2)
                    # Save to file
                    gridToVTK(dir_data+f'par_planes/BL{bl}/par_plane_{plane_nb}_{i}_bl{bl}',\
                              x,y,z_3d,pointData=planes_[f'{i}'])

                rv.write_pvd(dir_data+f'par_planes/BL{bl}/',plane_nb,bl,nb_chkpnts)

        if stats:
            stats = d()
            no_   = d()
            stats1,no_ = rv.extr_dict(False,1,bl,nx,ny,dir_data,no_)
            stats2,no_ = rv.extr_dict(False,2,bl,nx,ny,dir_data,no_)

            # Make coordinates 3D for Paraview
            x = np.repeat(x[:,:,np.newaxis],1,axis=2)/L_ref
            y = np.repeat(y[:,:,np.newaxis],1,axis=2)/L_ref
            z = np.zeros((1))
            z_3d = np.repeat(z   [np.newaxis,...],ny,axis=0)
            z_3d = np.repeat(z_3d[np.newaxis,...],nx,axis=0)

            for var in stats1.keys():
                stats[var] = np.repeat(stats1[var][:,:,np.newaxis],1,axis=2)
            for var in stats2.keys():
                stats[var] = np.repeat(stats2[var][:,:,np.newaxis],1,axis=2)

            # Save to file
            gridToVTK(dir_data+f'par_stats1_bl{bl}',x,y,z_3d,pointData=stats)


# Plot plane sensor
#==================

def plot_sensor_plane(dir_data,var,plane_nb,bl,ind_i,ind_j,wall_plane):
    dict_info = rv.read_info(dir_data)
    u_in = dict_info['Uref']
    L_ref = dict_info['L_ref']
    every = 100
    dtstar = dict_info['dt']*u_in/L_ref*every
    sensor,sensor2,N = rv.sensor_plane(dir_data,var,plane_nb,bl,ind_i,ind_j,wall_plane)
    plane_file = dir_data+f'plane_{plane_nb}_sol_bl{bl}.bin'
    # PSD
    spec,St = plt.psd(sensor,NFFT=N,pad_to=2*N,Fs=1/dtstar,scale_by_freq=True,detrend='mean')
    spec2,St = plt.psd(sensor2,NFFT=N,pad_to=2*N,Fs=1/dtstar,scale_by_freq=True,detrend='mean')
    plt.figure(figsize=(7.,4.))
    plt.rcParams['text.usetex'] = True
    plt.plot([0.1835,0.1835],[1e-5,1.21e10],'--k',linewidth=lw)
    plt.plot([0.3858,0.3858],[1e-5,8e4],'--k',linewidth=lw)
    plt.plot(St,spec/1e2,'-r',linewidth=lw)
    plt.plot(St,spec2,'-g',linewidth=lw)
    plt.xlabel(r'$S_t$',fontsize=font)
    plt.ylabel(r'$PSD$',fontsize=font)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(5e-3,20,)
    plt.ylim(1e-5,)
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    # # plt.grid()()
    plt.tight_layout()
    plt.show()


# Plot sensor PSD
#================

def plot_sensors(dir_data,sens_nb,sens_bl,iter_start,iter_stop,\
                 var_str,n_planes,auto,every):

    # Computation info
    dict_info = rv.read_info(dir_data)
    u_in   = dict_info['Uref']
    tscale = dict_info['Tscale']
    dt     = dict_info['dt']
    L_ref  = dict_info['L_ref']
    nz     = dict_info['nz_bl1']

    # Get sensor signal
    sensor = rv.read_sensors(dir_data,var_str,str(sens_nb).rjust(3,'0'),\
                             sens_bl,iter_start,iter_stop,n_planes,auto,every)
    var = sensor[var_str]
    dtstar = sensor['dtstar']
    tstar = sensor['tstar']
    niter = sensor['niter']
    tau   = sensor['tau']

    # Physical time resolution
    dt_min   = 1e-1*L_ref/u_in/2
    freq_min = 1/dt_min
    freq_line = dt_min/dt
    print('dt_min =',dt_min,', freq_min =',freq_min,', freq_line =',freq_line)

    # plt.figure(figsize=(7.5,5))
    plt.rcParams['text.usetex'] = True
    if auto:
        # Get spectrum
        E = np.mean(np.array([sensor[var_str]['E'][f'plane_{i+1}'] \
            for i in range(n_planes)]),axis=0)
        K = np.mean(np.array([sensor[var_str]['K'][f'plane_{i+1}'] \
            for i in range(n_planes)]),axis=0)
        # print("time integral scale t =",sensor[var_str]['T']['plane_1'])
        k = lambda x : x**(-5./3.)*1e3
        K_= np.linspace(1,100,5)

        # # Plot autocorrelation in time
        # plt.plot(sensor['tau'],sensor[var_str]['auto_corr']['plane_1'],'k',linewidth=lw)
        # plt.xlabel(rf'$\tau$')
        # plt.ylabel(rf'$R_{var_str}(t)$')
        # plt.title(rf'Autocorrelation of ${var_str}$')
        # # # plt.grid()()
        # Plot autocorrelation in space in spanwise direction
        plt.figure(figsize=(7.,4))
        l = np.linspace(0,0.5,nz//2)
        plt.plot(l,sensor['u']['auto_corr_z'][:nz//2],'-k',linewidth=lw,label=r'$u$')
        plt.plot(l,sensor['v']['auto_corr_z'][:nz//2],'-.k',linewidth=lw,label=r'$v$')
        plt.plot(l,sensor['w']['auto_corr_z'][:nz//2],'--k',linewidth=lw,label=r'$w$')
        #plt.plot(sensor['p']['auto_corr_z'],':k',linewidth=lw,label=r'$p$')
        #plt.plot(sensor['u']['auto_corr_z2'],'-r',linewidth=lw,label=r'$u$')
        #plt.plot(sensor['v']['auto_corr_z2'],'-.r',linewidth=lw,label=r'$v$')
        #plt.plot(sensor['w']['auto_corr_z2'],'--r',linewidth=lw,label=r'$w$')
        #plt.plot(sensor['p']['auto_corr_z2'],':r',linewidth=lw,label=r'$p$')
        plt.xlabel(rf'$z$',fontsize=font)
        # plt.ylabel(rf'$R_{var_str}(z)$')
        plt.ylabel(rf'$R(z)$',fontsize=font)
        plt.xticks(fontsize=font)
        plt.yticks(fontsize=font)
        plt.legend(prop={'size':font},markerscale=markerscale)
        # plt.title(rf'Autocorrelation of ${var_str}$ along $z$')
        ## # plt.grid()()
        plt.tight_layout()

        # Plot spectrum
        plt.figure(figsize=(7.,4))
        plt.plot(K,E,'b',linewidth=lw,\
                 markersize=markersize,markevery=markevery)
        plt.plot(K_,k(K_),'k',linewidth=lw,\
                 markersize=markersize,markevery=markevery)
        plt.text(4,1.0e3,r'$k^{-5/3}$',fontsize=font)

        plt.xlabel(r'$S_t$',fontsize=font)
        plt.ylabel(r'$PSD$',fontsize=font)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(5e-3,5e2)
        plt.ylim(1e-7,)
        plt.xticks(fontsize=font)
        plt.yticks(fontsize=font)
        #plt.title(rf'${var_str}$ energy spectrum')
        ## plt.grid()(which='minor',color='grey',alpha=0.5)
        ## plt.grid()(which='major',color='black',alpha=0.7)
        plt.tight_layout()
        plt.show()

    else:
        plt.plot(sensor['tau'],sensor[var_str]['cross_corr']['plane_1_2'],'k',linewidth=lw)
        plt.xlabel(rf'$\tau$')
        plt.ylabel(rf'$R_{var_str}(t)$')
        plt.title(rf'Crosscorrelation of ${var_str}$ at different spanwise locations')
        # # plt.grid()()
        plt.show()


# Plot sensor PDF
#================

def plot_sensors_PDF(dir_data,sens_nb,sens_bl,ind_i,ind_j,iter_start,iter_stop,n_planes):

    # Computation info
    dict_info = rv.read_info(dir_data)
    u_in   = dict_info['Uref']
    tscale = dict_info['Tscale']
    L_ref  = dict_info['L_ref']

    # Get sensor signal
    sensor_mean,kde_support,kde_density = rv.read_sensors_PDF(dir_data,str(sens_nb).rjust(3,'0'),\
                             sens_bl,ind_i,ind_j,iter_start,iter_stop,n_planes)

    plt.figure(figsize=(7.5,5))
    plt.rcParams['text.usetex'] = True
    # Histogram
    plt.hist(sensor_mean,bins=100,density=True,edgecolor='k',label='Histogram')
    # KDE
    plt.plot(kde_support,kde_density,'r',linewidth=lw,label='KDE')
    plt.xlabel(r'$\phi$ (degrees)')
    plt.ylabel(r'KDE($\phi$)')
    plt.legend()
    plt.title(r'KDE of instantaneous angle between $u$ \& $w$')
    plt.show()

