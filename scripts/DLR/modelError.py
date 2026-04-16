from xml.parsers.expat import model
import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import matplotlib.gridspec as gridspec
import numpy as np
import subprocess
import numpy.random as rng
import pandas as pd
import random
import multiprocessing as mp
from scipy.stats import chi2
from scipy.linalg import eigh
import copy
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.spatial.distance import cdist
from pathlib import Path
from collections import OrderedDict
from typing import Tuple, List, Union
from multiprocessing import Pool, cpu_count
from scipy.stats import qmc
from scipy.interpolate import interp1d
from numpy import linalg as LA

from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.spatial.distance import cdist

from aero_optim.mf_sm.mf_infill import compute_pareto
from aero_optim.ffd.ffd import DLR_POD_2D

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Times",
#     "figure.dpi": 200,
#     "font.size": 4,
#     'legend.fontsize': 4,
#     "axes.titlesize": 4,
#     "axes.labelsize": 4
# })

# Configuration parameters
n_gen = 50
pop_size = 30

beta_ADP_bounds = [-0.67,0.93]  # degrees
beta_OP1_bounds = [-0.22,2.78]  # degrees
beta_OP2_bounds = [-2.67,0.33]  # degrees

hf_bsl_w_ADP = 0.03493
hf_bsl_w_OP2 = 0.05888
hf_bsl_w_OP1 = 0.03765
hf_bsl_w_OP  = (hf_bsl_w_OP1 + hf_bsl_w_OP2) * 0.5

# non-adapted fine mesh mixed-out los
lf_bsl_w_ADP = 0.07827
lf_bsl_w_OP1 = 0.21933
lf_bsl_w_OP2 = 0.07854
lf_bsl_w_OP  = (lf_bsl_w_OP1 + lf_bsl_w_OP2) * 0.5

baseline_file = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/RANS_bruteForce/ogv1c.dat"
baseline = np.loadtxt(baseline_file, skiprows=2) * 1e-3

OD = OrderedDict  # Type alias

def read_hf_results(home_dir, master_dir, n_DOE, n_infill):

    directoryList = [os.path.join(home_dir, 'hf_doe')] + [os.path.join(home_dir, master_dir+f'_{n}/high_infill_{n}') for n in range(n_infill)]

    df_temp = pd.DataFrame({
    "infill": pd.Series(dtype="int"),
    "cand": pd.Series(dtype="int"),
    "w_ADP": pd.Series(dtype="float"),
    "w_OP": pd.Series(dtype="float"),
    "w_OP1": pd.Series(dtype="float"),
    "w_OP2": pd.Series(dtype="float"),
    "beta_ADP": pd.Series(dtype="float"),
    "beta_OP1": pd.Series(dtype="float"),
    "beta_OP2": pd.Series(dtype="float"),
    "x1": pd.Series(dtype="float"),
    "x2": pd.Series(dtype="float"),
    "x3": pd.Series(dtype="float"),
    "x4": pd.Series(dtype="float"),
    "x5": pd.Series(dtype="float"),
    "fName": pd.Series(dtype="str")
    })
    cont = 0
    for dir in directoryList:
        cand = np.genfromtxt(os.path.join(dir, "candidates.txt"))
        if 'doe' in dir:
            for i in range(n_DOE):
                temp_ADP = np.genfromtxt(os.path.join(dir,f'MUSICAA/musicaa_g0_c{i}/ADP/QoI_convergence.csv'),skip_header=1,delimiter=',')[-1,:]
                temp_OP1 = np.genfromtxt(os.path.join(dir,f'MUSICAA/musicaa_g0_c{i}/OP1/QoI_convergence.csv'),skip_header=1,delimiter=',')[-1,:]
                temp_OP2 = np.genfromtxt(os.path.join(dir,f'MUSICAA/musicaa_g0_c{i}/OP2/QoI_convergence.csv'),skip_header=1,delimiter=',')[-1,:]
                df_temp.loc[len(df_temp)] = [0, i+1, temp_ADP[0], (temp_OP1[0]+temp_OP2[0])*0.5, temp_OP1[0], temp_OP2[0], temp_ADP[1], temp_OP1[1], temp_OP2[1],
                                             cand[i,0], cand[i,1], cand[i,2], cand[i,3], cand[i,4],
                                        os.path.join(dir, f"FFD/ogv1c_g0_c{i}.dat")]
        else:
            cont+=1
            temp_ADP = np.genfromtxt(os.path.join(dir,f'MUSICAA/musicaa_g0_c0/ADP/QoI_convergence.csv'),skip_header=1,delimiter=',')[-1,:]
            temp_OP1 = np.genfromtxt(os.path.join(dir,f'MUSICAA/musicaa_g0_c0/OP1/QoI_convergence.csv'),skip_header=1,delimiter=',')[-1,:]
            temp_OP2 = np.genfromtxt(os.path.join(dir,f'MUSICAA/musicaa_g0_c0/OP2/QoI_convergence.csv'),skip_header=1,delimiter=',')[-1,:]
            df_temp.loc[len(df_temp)] = [cont, 1, temp_ADP[0], (temp_OP1[0]+temp_OP2[0])*0.5, temp_OP1[0], temp_OP2[0], temp_ADP[1], temp_OP1[1], temp_OP2[1],
                                            cand[0], cand[1], cand[2], cand[3], cand[4],
                                    os.path.join(dir, f"FFD/ogv1c_g0_c0.dat")]

    return df_temp

def read_model_results(home_dir, n_DOE, n_infill):

    df_temp = pd.DataFrame({
    "infill": pd.Series(dtype="int"),
    "cand": pd.Series(dtype="int"),
    "w_ADP": pd.Series(dtype="float"),
    "std_w_ADP": pd.Series(dtype="float"),
    "w_OP": pd.Series(dtype="float"),
    "std_w_OP": pd.Series(dtype="float"),
    "beta_ADP": pd.Series(dtype="float"),
    "std_beta_ADP": pd.Series(dtype="float"),
    "beta_OP1": pd.Series(dtype="float"),
    "std_beta_OP1": pd.Series(dtype="float"),
    "beta_OP2": pd.Series(dtype="float"),
    "std_beta_OP2": pd.Series(dtype="float"),
    "x1": pd.Series(dtype="float"),
    "x2": pd.Series(dtype="float"),
    "x3": pd.Series(dtype="float"),
    "x4": pd.Series(dtype="float"),
    "x5": pd.Series(dtype="float"),
    "fName": pd.Series(dtype="str")
    })
    
    cand = np.genfromtxt(os.path.join(home_dir, "hf_candidates.txt"))

    for i in range(n_infill):
        idx = i+n_DOE

        x = np.asarray(cand[idx,:]).T.reshape(1, -1)

        with open(os.path.join(home_dir, f"model_{i}.pkl"), "rb") as f:
            model = pickle.load(f)
            
            w_ADP = model.models[0].evaluate(x)[0,0]
            w_OP  = model.models[1].evaluate(x)[0,0]
            beta_ADP = model.models[2].evaluate(x)[0,0]
            beta_OP1  = model.models[3].evaluate(x)[0,0]
            beta_OP2  = model.models[4].evaluate(x)[0,0]
            
            std_w_ADP = model.models[0].evaluate_std(x)[0,0]
            std_w_OP  = model.models[1].evaluate_std(x)[0,0]
            std_beta_ADP = model.models[2].evaluate_std(x)[0,0]
            std_beta_OP1  = model.models[3].evaluate_std(x)[0,0]
            std_beta_OP2  = model.models[4].evaluate_std(x)[0,0]

        df_temp.loc[len(df_temp)] = [i+1, 1, w_ADP, std_w_ADP, w_OP, std_w_OP,
                                     beta_ADP, std_beta_ADP, beta_OP1, std_beta_OP1, beta_OP2, std_beta_OP2,
                                     cand[idx,0], cand[idx,1], cand[idx,2], cand[idx,3], cand[idx,4],
                                     os.path.join(home_dir, f"FFD/ogv1c_g0_c0.dat")]
        
    return df_temp



def main():

    nLFtoHF = 10

    n_DOE = 5
    n_infill = 15
    master_dir_mf = 'output_paper'
    master_dir_hf = 'output_hf'

    selected = [6, 0, 3]  # indices of chosen optimal candidates
    plt.close('all')
    # Load mixed-out loss results
    mf_dir = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/" + master_dir_mf
    hf_dir = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/" + master_dir_hf

    # hf_opt = read_hf_results(hf_dir, master_dir_hf, n_DOE, n_infill)

    mf_hf_opt   = read_hf_results(mf_dir, master_dir_mf, n_DOE, n_infill)
    mf_hf_model = read_model_results(mf_dir, n_DOE, n_infill)

    # mf_hf_opt   = read_hf_results(hf_dir, master_dir_hf, n_DOE, n_infill)
    # mf_hf_model = read_model_results(hf_dir, n_DOE, n_infill)

    mf_hf_infill = mf_hf_opt[mf_hf_opt["infill"] > 0]

    idx = 1
    plt.figure(figsize=(15, 20))

    limits = {'w_ADP': [0.025, 0.06], 'w_OP': [0.025, 0.08], 'beta_ADP': [-3, 1.5], 'beta_OP1': [-1, 4], 'beta_OP2': [-3, 1]}
    
    for QoI in ['w_ADP', 'w_OP', 'beta_ADP', 'beta_OP1', 'beta_OP2']:

        plt.subplot(5,1,idx)
        
        plt.plot(mf_hf_infill['infill'], mf_hf_infill[QoI])
        plt.plot(mf_hf_model['infill'], mf_hf_model[QoI])
        if 'w' in QoI:
            plt.fill_between(mf_hf_infill['infill'], mf_hf_infill[QoI]*0.9, mf_hf_infill[QoI]*1.1, alpha=0.3, label='True ± 10%')
        if 'beta_ADP' in QoI:
            plt.fill_between(mf_hf_infill['infill'], mf_hf_infill[QoI]-0.16, mf_hf_infill[QoI]+0.16, alpha=0.3, label='True ± 0.16°')
        elif 'beta' in QoI:
            plt.fill_between(mf_hf_infill['infill'], mf_hf_infill[QoI]-0.3, mf_hf_infill[QoI]+0.3, alpha=0.3, label='True ± 0.3°')
        if 'beta_ADP' == QoI:
            plt.plot([1, n_infill], [beta_ADP_bounds[0], beta_ADP_bounds[0]], color='tab:grey', linestyle='-.', linewidth=0.8)
            plt.plot([1, n_infill], [beta_ADP_bounds[1], beta_ADP_bounds[1]], color='tab:grey', linestyle='-.', linewidth=0.8)
        if 'beta_OP1' == QoI:
            plt.plot([1, n_infill], [beta_OP1_bounds[0], beta_OP1_bounds[0]], color='tab:grey', linestyle='-.', linewidth=0.8)
            plt.plot([1, n_infill], [beta_OP1_bounds[1], beta_OP1_bounds[1]], color='tab:grey', linestyle='-.', linewidth=0.8)
        if 'beta_OP2' == QoI:
            plt.plot([1, n_infill], [beta_OP2_bounds[0], beta_OP2_bounds[0]], color='tab:grey', linestyle='-.', linewidth=0.8)
            plt.plot([1, n_infill], [beta_OP2_bounds[1], beta_OP2_bounds[1]], color='tab:grey', linestyle='-.', linewidth=0.8)
        
        plt.fill_between(mf_hf_infill['infill'], mf_hf_model[QoI]-2*mf_hf_model['std_' + QoI], mf_hf_model[QoI]+2*mf_hf_model['std_' + QoI], alpha=0.3, label='Model ± 2σ')
        plt.ylim(limits[QoI])

        idx += 1
        plt.legend()
    
    plt.savefig('model_error_analysis.pdf', bbox_inches='tight')

    # plt.show()

#     plt.subplot(5,1,1)
#     plt.plot(mf_hf_infill['infill'], mf_hf_infill['w_ADP'], label='True')
#     plt.plot(mf_hf_infill['infill'], mf_hf_model['w_ADP'], label='Predicted')

#     plt.subplot(5,1,2)
#     plt.plot(mf_hf_infill['infill'], mf_hf_infill['w_OP'], label='True')
#     plt.plot(mf_hf_infill['infill'], mf_hf_model['w_OP'], label='Predicted')

#     plt.subplot(5,1,3)
#     plt.plot(mf_hf_infill['infill'], mf_hf_infill['beta_ADP'], label='True')
#     plt.plot(mf_hf_infill['infill'], mf_hf_model['beta_ADP'], label='Predicted')

#     plt.subplot(5,1,4)
#     plt.plot(mf_hf_infill['infill'], mf_hf_infill['beta_OP1'], label='True')
#     plt.plot(mf_hf_infill['infill'], mf_hf_model['beta_OP1'], label='Predicted')

#     plt.subplot(5,1,5)
#     plt.plot(mf_hf_infill['infill'], mf_hf_infill['beta_OP2'], label='True')
#     plt.plot(mf_hf_infill['infill'], mf_hf_model['beta_OP2'], label='Predicted')
#     plt.legend()
#     plt.show()






if __name__ == "__main__":
    main()
