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
from matplotlib.lines import Line2D
from matplotlib.text import Text
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

fs = 9

plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": fs,
    'legend.fontsize': fs, 
    "axes.titlesize": fs,
    "axes.labelsize": fs
})

lw = 0.8

square_fig = (4.5,4.2)

# Configuration parameters
n_gen = 50
pop_size = 30

beta_ADP_bounds = [-0.67,0.93]  # degrees
beta_OP1_bounds = [-0.22,2.78]  # degrees
beta_OP2_bounds = [-2.67,0.33]  # degrees

hf_bsl_w_ADP = 0.03493
hf_bsl_w_OP1 = 0.05888
hf_bsl_w_OP2 = 0.03765
hf_bsl_w_OP  = (hf_bsl_w_OP1 + hf_bsl_w_OP2) * 0.5

# non-adapted fine mesh mixed-out los
lf_bsl_w_ADP = 0.07827
lf_bsl_w_OP1 = 0.21933
lf_bsl_w_OP2 = 0.07854
lf_bsl_w_OP  = (lf_bsl_w_OP1 + lf_bsl_w_OP2) * 0.5

baseline_file = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/RANS_bruteForce/ogv1c.dat"
baseline = np.loadtxt(baseline_file, skiprows=2) * 1e-3

OD = OrderedDict  # Type alias

def get_best_candidates_idx(fitnesses: np.ndarray, pareto_set: np.ndarray, nb_comp: int = 1) -> tuple[int, int, list[int]]:
    # best ADP
    ADP_cand_idx = np.where(fitnesses[:, 0] == np.min(pareto_set[:, 0]))[0][0]
    print(f"best ADP candidates {ADP_cand_idx}: {fitnesses[ADP_cand_idx]}")

    # best OP
    OP_cand_idx = np.where(fitnesses[:, 1] == np.min(pareto_set[:, 1]))[0][0]
    print(f"best OP candidates {OP_cand_idx}: {fitnesses[OP_cand_idx]}")

    # best compromise
    # matrix made of the distance between each point in the ordered pareto front
    d_matrix = np.linalg.norm(pareto_set[:, np.newaxis] - pareto_set[np.newaxis, :], axis=-1)
    # 1d array with the distance between two consecutive points along the ordered pareto front
    s = np.array([d_matrix[i + 1, i] for i in range(len(d_matrix) - 1)])
    s_length = np.sum(s)
    # index of the closest point to the pareto set center
    cand_idx = []
    for cid in range(nb_comp):
        idx = np.argmin([abs(np.sum(s[:i]) - (cid + 1) * s_length / (nb_comp + 1)) for i in range(len(s))])
        cand_idx.append(np.where(fitnesses == pareto_set[idx])[0][0])
        print(f"best compromise candidates {cand_idx[-1]}: {fitnesses[cand_idx[-1]]}")

    return ADP_cand_idx, OP_cand_idx, cand_idx

def plot_pareto_with_constraints(lf_cand, lf_fit, idx, constraints, figsize=[5.2, 3.64]):
    
    lf_bsl_w_ADP= 0.07826
    lf_bsl_w_OP = 0.1489

    lf_fit = lf_fit[idx, :]
    lf_cand = lf_cand[idx, :]
    lf_pareto = compute_pareto(lf_fit[:, 0], lf_fit[:, 1])

    x_lim_min = 0.025
    x_lim_max = 0.15
    y_lim_min = 0.075
    y_lim_max = 0.20
    
    # plt.subplots(figsize=(figsize[0], figsize[1]))
    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    # Plot 1: beta ADP pareto
    ax = plt.subplot(2,2,1)
    cmap='viridis'
    row_indices = np.arange(len(lf_fit))
    norm = plt.Normalize(row_indices.min() // pop_size, row_indices.max() // pop_size)
    cmap = cm.get_cmap('viridis')
    colors = cmap(norm(row_indices // pop_size))
    plt.scatter(lf_fit[:, 0], lf_fit[:, 1], marker="o", s=10, c=colors)
    plt.scatter(lf_bsl_w_ADP, lf_bsl_w_OP, marker="x", s=20, color="k", label="LF baseline")
    ax.scatter(lf_pareto[:, 0], lf_pareto[:, 1], marker="o", s=10, color="k", facecolors='none', label="LF Pareto")
    plt.grid(True, color="grey", linestyle="dashed")
    plt.ylim(y_lim_min, y_lim_max)
    plt.xlim(x_lim_min, x_lim_max)
    # ax.legend(loc="lower right")
    plt.ylabel(r'$w_{OP}$ [-]')
    ax.set_xticklabels([])
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'$Generation$')
    plt.tight_layout()
    
    # Plot 1: beta ADP pareto
    ax = plt.subplot(2,2,2)
    # vmax = np.max(np.abs(constraints[:,-3][idx]))
    # vmin = -vmax
    norm = plt.Normalize(constraints[:,-3][idx].min(), constraints[:,-3][idx].max())
    cmap = cm.get_cmap('hsv')
    colors = cmap(norm(constraints[:,-3][idx]))
    plt.scatter(lf_fit[:, 0], lf_fit[:, 1], marker="o", s=10, c=colors)
    plt.scatter(lf_bsl_w_ADP, lf_bsl_w_OP, marker="x", s=20, color="k", label="LF baseline")
    plt.grid(True, color="grey", linestyle="dashed")
    plt.ylim(y_lim_min, y_lim_max)
    plt.xlim(x_lim_min, x_lim_max)
    # ax.legend(loc="lower right")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'$\beta_{ADP} \ [^\circ]$')
    plt.tight_layout()
    
    # Plot 1: beta ADP pareto
    ax = plt.subplot(2,2,3)
    norm = plt.Normalize(constraints[:,-2][idx].min(), constraints[:,-2][idx].max())
    cmap = cm.get_cmap('hsv')
    colors = cmap(norm(constraints[:,-2][idx]))
    plt.scatter(lf_fit[:, 0], lf_fit[:, 1], marker="o", s=10, c=colors)
    plt.scatter(lf_bsl_w_ADP, lf_bsl_w_OP, marker="x", s=20, color="k", label="LF baseline")
    plt.grid(True, color="grey", linestyle="dashed")
    plt.ylim(y_lim_min, y_lim_max)
    plt.xlim(x_lim_min, x_lim_max)
    # ax.legend(loc="lower right")
    plt.xlabel(r'$w_{ADP}$ [-]')
    plt.ylabel(r'$w_{OP}$ [-]')
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'$|\beta_{OP1} - 4.0| \ [^\circ]$')
    plt.tight_layout()
    
    # Plot 1: beta ADP pareto
    ax = plt.subplot(2,2,4)
    norm = plt.Normalize(constraints[:,-1][idx].min(), constraints[:,-1][idx].max())
    cmap = cm.get_cmap('hsv')
    colors = cmap(norm(constraints[:,-1][idx]))
    plt.scatter(lf_fit[:, 0], lf_fit[:, 1], marker="o", s=10, c=colors)
    plt.grid(True, color="grey", linestyle="dashed")
    plt.ylim(y_lim_min, y_lim_max)
    plt.xlim(x_lim_min, x_lim_max)
    # ax.legend(loc="lower right")
    plt.xlabel(r'$w_{ADP}$ [-]')
    ax.set_yticklabels([])
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'$\beta_{OP2} \ [^\circ]$')
    plt.tight_layout()
    plt.show()

def plot_pareto_single_HF(lf_cand, lf_fit, constraints, df_chosen, hf_results=None, fontsize=14):

    df_c = df_chosen.reset_index(drop=True)

    lf_bsl_w_ADP= 0.07826
    lf_bsl_w_OP = 0.1489

    hf_bsl_w_ADP = 0.0349
    hf_bsl_w_OP = 0.0483

    lf_pareto = compute_pareto(lf_fit[:, 0], lf_fit[:, 1])

    x_lim_min = 0.025
    x_lim_max = 0.15
    y_lim_min = 0.075
    y_lim_max = 0.20
    
    # plt.subplots(figsize=(figsize[0], figsize[1]))
    fig = plt.figure(figsize=square_fig)

    # Plot 1: beta ADP pareto
    ax = plt.subplot(1,1,1)
    cmap='viridis'
    row_indices = np.arange(len(lf_fit))
    norm = plt.Normalize(row_indices.min() // pop_size, row_indices.max() // pop_size)
    cmap = cm.get_cmap('viridis')
    colors = cmap(norm(row_indices // pop_size))
    plt.scatter(lf_fit[:, 0], lf_fit[:, 1], marker="o", s=50, c=colors, label='LF Individuals')
    plt.scatter(lf_bsl_w_ADP, lf_bsl_w_OP, marker="x", s=90, color="k", label="LF baseline", linewidths=3)
    ax.scatter(lf_pareto[:, 0], lf_pareto[:, 1], marker="o", s=50, color="k", facecolors='none', label="LF Pareto", linewidths=2)
    plt.grid(True, color="grey", linestyle="dashed")
    plt.ylim(0.02, 0.20)
    plt.xlim(0.02, 0.10)
    ax.legend(loc="lower right")
    plt.xlabel(r'$w_{ADP}$ [-]')
    plt.ylabel(r'$w_{OP}$ [-]')
    # ax.set_xticklabels([])
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'$Generation$')

    pairs = list(zip(df_c["w_ADP"], df_c["w_OP"]))
    plt.scatter(df_c['w_ADP'], df_c['w_OP'], marker="o", facecolors='none', edgecolors='r', linewidth=2, s=100)
    for idx, row in df_c.iterrows():
        plt.text(row['w_ADP']+0.00015, row['w_OP']+0.00015, str(idx+1), fontsize=fontsize, fontweight='bold',
                color='r', ha='left', va='bottom')
        
    if not hf_results is None:
        linea_x = np.linspace(lf_bsl_w_ADP,hf_bsl_w_ADP)
        linea_y = np.linspace(lf_bsl_w_OP,hf_bsl_w_OP)
        plt.scatter(hf_bsl_w_ADP, hf_bsl_w_OP, marker="+", s=90, color="k", label="LF Baseline", linewidths=3)
        plt.plot(linea_x,linea_y,'k--')
        for idx in range(len(hf_results)):
            hf_res = hf_results[idx]
            if hf_res[2] == 1:
                linea_x = np.linspace(hf_res[0],pairs[idx][0])
                linea_y = np.linspace(hf_res[1],pairs[idx][1])
                plt.scatter(hf_res[0], hf_res[1], marker="*", s=200, color="gold", edgecolors='k', linewidths=2)
                plt.plot(linea_x,linea_y,'k--')

        plt.scatter(hf_res[0], hf_res[1], marker="*", s=200, facecolor="none", edgecolors='k', linewidths=2, label="HF recomputed")
        
        for idx in range(len(hf_results)):
            hf_res = hf_results[idx]
            if hf_res[2] == 0:
                linea_x = np.linspace(hf_res[0],pairs[idx][0])
                linea_y = np.linspace(hf_res[1],pairs[idx][1])
                plt.scatter(hf_res[0], hf_res[1], marker="*", s=200, color="red", edgecolors='k', linewidths=2)
                plt.plot(linea_x,linea_y,'k--')
    
    # plt.legend(frameon=False)
    plt.legend(frameon=True).get_frame().set_edgecolor("none")
    plt.tight_layout()
    plt.savefig("lf_pareto.pdf", bbox_inches="tight")
    plt.show()

def run_pod_analysis_DLR(df_chosen, baseline_file='./ogv1c.dat', pod_dataset='../POD_Dataset', nprofile=1000, nmode=5, figsize=[7.5, 6], fontsize=16):
    
    """
    Run POD analysis with the given design sensitivity parameters.
    All plots consolidated into a single figure with subplots.
    
    Args:
        design_sensitivity: list of parameter bounds dicts
        input_file: progen input template filename
        directory: output directory for results
        nprofile: number of profiles to generate
        nmode: number of POD modes
        n_processes: number of parallel processes (None = auto)
    """
    df_c = df_chosen.reset_index(drop=True)

    modes_color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    solution_style = ['solid','dashed','dotted']
    seed = 123
    random.seed(seed)
    np.random.seed(seed)

    design_sensitivity = [
        {'BetaLE':[118,129]},
        {'BetaTE':[60,66]},
        {'BetaST':[88,92]},
        {'x2SS':[0.013,0.033]},
        {'x3SS':[0.31,0.40]},
        {'y3SS':[0.192,0.21]},
        {'x4SS':[0.818,0.86]},
        {'m2DS':[0.127,0.18]},
        {'d2DS':[0.035,0.048]},
        {'d3DS':[0.022,0.05]},
        {'d4DS':[0.014,0.017]},
        {'Dmax_approx':[0.8,1.0]},
        {'rTE':[0.00495,0.0075]}]
    
    params_dict = {}

    for _, param in enumerate(design_sensitivity):
        for key, value in param.items():
            params_dict[key] = value

    dlr = DLR_POD_2D(baseline_file,  pod_dataset, '/home/mciarlatani/bin/BladeGenerator.exe', params_dict, nmode, nprofile, seed)

    x_coords = dlr.ffd.pts[:, 0]
    
    # Subplot 2: POD modes (top middle)

    # Normalize x coordinates by chord length (approximate)
    chord_approx = np.max(x_coords) - np.min(x_coords)

    plt.figure(figsize=(figsize[0], figsize[1]))
    for nn in range(1, nmode + 1):
        plt.plot(x_coords / chord_approx, dlr.phi_tilde[:, -nn], label=f"Mode {nn}", color = modes_color[nn-1], linewidth=1.5)
    plt.xlabel(r"$x/c$ [-]")
    plt.ylabel(r"POD / basis [m]")
    plt.ylim([-0.02,0.05])
    plt.legend()
    plt.tight_layout()
    plt.savefig("lf_modes.pdf", bbox_inches="tight")

    plt.figure(figsize=(figsize[0], figsize[1]))
    for nn in range(1, nmode + 1):

        for idx, row in df_c.iterrows():
            amplitude = np.array(row[['x1', 'x2', 'x3', 'x4', 'x5']].values).reshape(-1, 1)
            if nn == 1:
                plt.plot(x_coords / chord_approx, dlr.phi_tilde[:, -nn]*amplitude[-nn], color = modes_color[nn-1], linewidth=1.5, linestyle=solution_style[idx], label=f"Candidate {idx+1}")
            else:
                plt.plot(x_coords / chord_approx, dlr.phi_tilde[:, -nn]*amplitude[-nn], color = modes_color[nn-1], linewidth=1.5, linestyle=solution_style[idx])

    plt.xlabel(r"$x/c$ [-]")
    plt.ylabel(r"POD / basis [m]")
    plt.ylim([-0.0005,0.0015])
    plt.legend()
    plt.tight_layout()
    plt.savefig("lf_opt_modes.pdf", bbox_inches="tight")


    # plt.savefig("lf_scatter_modes.pdf", bbox_inches="tight")

    # for idx, row in df_c.iterrows():
    #     amplitude = np.array(row[['x1', 'x2', 'x3', 'x4', 'x5']].values).reshape(-1, 1)
    #     cont = 5
    #     for x in amplitude:
    #         plt.plot(x_coords / chord_approx, dlr.phi_tilde[:, -cont], label=f"mode {nn}", linewidth=1.5)
    
    # ax2.set(xlabel="$x / c$ [-]", ylabel="POD basis [-]", title="b) Geometric modes")
    # ax2.legend(loc="best")
    # ax2.grid(True, alpha=0.3)
    plt.show()

def plot_sf_scatter(df_opt, bsl_w_ADP, bsl_w_OP, selected=None):
    
    n_DOE = len(df_opt[df_opt['infill'] == 0])
    n_infill = len(df_opt) - n_DOE

    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=n_infill)
    colors = [0]*n_DOE + list(range(1, n_infill + 1))

    df_unfeasible = df_opt.loc[(df_opt["beta_ADP"] < beta_ADP_bounds[0]) | (df_opt["beta_ADP"] > beta_ADP_bounds[1]) |
                               (df_opt["beta_OP1"] < beta_OP1_bounds[0]) | (df_opt["beta_OP1"] > beta_OP1_bounds[1]) |
                               (df_opt["beta_OP2"] < beta_OP2_bounds[0]) | (df_opt["beta_OP2"] > beta_OP2_bounds[1])]   

    size = 20

    fig, ax = plt.subplots(figsize=(5.2, 3.64))
    
    # idx = 0
    # for point in selected:
    #     ax.scatter(hf_fit[point, 0], hf_fit[point, 1], marker="o", facecolors='none', edgecolors='red', linewidth=1.5, s=size*1.5)
    #     plt.text(hf_fit[point, 0]-0.001, hf_fit[point, 1]-0.001, str(idx+1), fontsize=6, fontweight='bold',
    #                 color='r', ha='left', va='bottom')
    #     idx+=1
                    
    # u_DOE = [u for u in unfeasible if u<n_DOE]
    # u_infill = [u for u in unfeasible if u>=n_DOE]
    
    ax.scatter(df_opt.loc[df_opt["infill"] == 0, ["w_ADP"]], df_opt.loc[df_opt["infill"] == 0, ["w_OP"]], marker='s', c=colors[:n_DOE], s=size, label="initial DOE")
    ax.scatter(df_opt.loc[df_opt["infill"] != 0, ["w_ADP"]], df_opt.loc[df_opt["infill"] != 0, ["w_OP"]], marker='o', c=colors[n_DOE:], s=size, label="initial DOE")
    # ax.scatter(hf_fit[u_DOE, 0], hf_fit[u_DOE, 1], marker='s', c='red', s=size, label="unfeasible design")
    # sc = ax.scatter(hf_fit[n_DOE:, 0], hf_fit[n_DOE:, 1], marker='o', c=colors[n_DOE:], s=size, label="HF infills")
    ax.scatter(df_unfeasible.loc[df_unfeasible["infill"] != 0, ["w_ADP"]], df_unfeasible.loc[df_unfeasible["infill"] != 0, ["w_OP"]], marker='o', c='red', s=size)
    ax.scatter(bsl_w_ADP, bsl_w_OP, marker="+", color="k", s=size, label="HF baseline")
    
    ax.set_ylim(0.030, 0.08)
    ax.set_xlim(0.025, 0.055)
    ax.set_xlabel('$w_{ADP}$ [-]')
    ax.set_ylabel('$w_{OP}$ [-]')

    # ax.set_axisbelow(True)
    # ax.grid(True, color="grey", linestyle="dashed")
    
    # ax.set_ylim(0.030, 0.08)
    # ax.set_xlim(0.025, 0.055)
    # cbar = plt.colorbar(sc, ax=ax, ticks=list(range(1, n_infill + 1)))
    # cbar.set_ticklabels([str(i) for i in range(1, n_infill + 1)])
    # cbar.set_label('Infill number')
    # # legend
    # ax.legend(loc="lower right")
    # ax.set_xlabel('$w_{ADP}$ [-]')
    # ax.set_ylabel('$w_{OP}$ [-]')
    # plt.savefig(os.path.join(os.getcwd(), "hf_pareto.pdf"), bbox_inches="tight")

    plt.show()

def plot_opt_comparison(mf_df, hf_df, bsl_w_ADP, bsl_w_OP, selected=None):

    assert len(mf_df[mf_df['infill'] == 0]) == len(hf_df[hf_df['infill'] == 0]), "MF and HF dataframes must have the number of DOE points"
    assert len(mf_df[mf_df['infill'] != 0]) == len(hf_df[hf_df['infill'] != 0]), "MF and HF dataframes must have the same length"

    n_DOE = len(mf_df[mf_df['infill'] == 0])
    n_infill = len(mf_df) - n_DOE

    cmap_hf = plt.get_cmap("magma")
    norm_hf = mcolors.Normalize(vmin=0, vmax=n_infill)
    colors_hf = [0]*n_DOE + list(range(1, n_infill + 1))

    cmap_mf = plt.get_cmap("viridis")
    norm_mf = mcolors.Normalize(vmin=0, vmax=n_infill)
    colors_mf = [0]*n_DOE + list(range(1, n_infill + 1))

    mf_unfeas = mf_df.loc[(mf_df["beta_ADP"] < beta_ADP_bounds[0]) | (mf_df["beta_ADP"] > beta_ADP_bounds[1]) |
                          (mf_df["beta_OP1"] < beta_OP1_bounds[0]) | (mf_df["beta_OP1"] > beta_OP1_bounds[1]) |
                          (mf_df["beta_OP2"] < beta_OP2_bounds[0]) | (mf_df["beta_OP2"] > beta_OP2_bounds[1])]   

    hf_unfeas = hf_df.loc[(hf_df["beta_ADP"] < beta_ADP_bounds[0]) | (hf_df["beta_ADP"] > beta_ADP_bounds[1]) |
                          (hf_df["beta_OP1"] < beta_OP1_bounds[0]) | (hf_df["beta_OP1"] > beta_OP1_bounds[1]) |
                          (hf_df["beta_OP2"] < beta_OP2_bounds[0]) | (hf_df["beta_OP2"] > beta_OP2_bounds[1])]   

    size = 35

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    ax.scatter(hf_df.loc[hf_df["infill"] == 0, ["w_ADP"]], hf_df.loc[hf_df["infill"] == 0, ["w_OP"]], marker='s', c=colors_mf[:n_DOE], s=size, label="Initial LES DOE")

    sc0 = ax.scatter(mf_df.loc[mf_df["infill"], ["w_ADP"]], mf_df.loc[mf_df["infill"], ["w_OP"]], marker='.', cmap=cmap_mf, c=colors_mf, s=0.0001)
    sc1 = ax.scatter(mf_df.loc[mf_df["infill"] != 0, ["w_ADP"]], mf_df.loc[mf_df["infill"] != 0, ["w_OP"]], marker='o', cmap=cmap_mf, c=colors_mf[n_DOE:], s=size, label="MF opt LES infill")
    sc2 = ax.scatter(hf_df.loc[hf_df["infill"] != 0, ["w_ADP"]], hf_df.loc[hf_df["infill"] != 0, ["w_OP"]], marker='v', cmap=cmap_hf, c=colors_hf[n_DOE:], s=size, label="HF opt infill")
    
    ax.scatter(bsl_w_ADP, bsl_w_OP, marker="+", color="k", s=size*2.0, label="LES baseline")
    
    ax.scatter(mf_unfeas.loc[mf_unfeas["infill"] == 0, ["w_ADP"]], mf_unfeas.loc[mf_unfeas["infill"] == 0, ["w_OP"]], marker='s', c='red', s=size, label='Infeasible design')
    ax.scatter(mf_unfeas.loc[mf_unfeas["infill"] != 0, ["w_ADP"]], mf_unfeas.loc[mf_unfeas["infill"] != 0, ["w_OP"]], marker='o', c='red', s=size)
    ax.scatter(hf_unfeas.loc[hf_unfeas["infill"] != 0, ["w_ADP"]], hf_unfeas.loc[hf_unfeas["infill"] != 0, ["w_OP"]], marker='v', c='red', s=size)
    
    ax.set_ylim(0.04, 0.08)
    ax.set_xlim(0.027, 0.057)
    ax.set_xlabel('$\mathcal{L}_{ADP}$ [-]')
    ax.set_ylabel('$\mathcal{L}_{OP}$ [-]')
    
    cbar1 = plt.colorbar(sc0, ax=ax, ticks=[0, 5, 10, 15], pad=0.005, fraction=0.050)
    cbar1.set_label('Infill number')
    
    cbar2 = plt.colorbar(sc2, ax=ax, ticks=[], pad=0.02, fraction=0.050)
    # # legend
    ax.legend(loc="lower right")
    ax.grid(True, color="grey", linestyle="dashed",alpha = 0.6)
    ax.set_xticks([0.03, 0.035, 0.04, 0.045, 0.05, 0.055])
    ax.set_yticks([0.04, 0.05, 0.06, 0.07, 0.08])
    plt.savefig(os.path.join(os.getcwd(), "compare_pareto.pdf"), bbox_inches="tight")

    plt.show()

def plot_paper_pareto(n_DOE, n_infill, nLFtonHF, lf_fit, hf_fit, unfeasible, bsl_w_ADP, bsl_w_OP, selected=None):

    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=n_infill)
    colors = [0]*n_DOE + list(range(1, n_infill + 1))
    import itertools
    lf_colors = list(itertools.chain.from_iterable(x if isinstance(x, list) else [x] for x in [0]*n_DOE*nLFtonHF + [[n]*nLFtonHF for n in range(1, n_infill + 1)]))

    lf_infill = lf_fit[n_DOE*nLFtonHF:, :]

    size = 10

    fig, ax = plt.subplots(figsize=(5.2, 3.64))
                    
    u_DOE = [u for u in unfeasible if u<n_DOE]
    u_infill = [u for u in unfeasible if u>=n_DOE]

    ax.scatter(lf_infill[:, 0], lf_infill[:, 1], marker='o', c=lf_colors[n_DOE*nLFtonHF:], s=size, label="Infill RANS")
    
    ax.scatter(hf_fit[:n_DOE, 0], hf_fit[:n_DOE, 1], marker='s', c=colors[:n_DOE], s=size, label="initial DOE")
    ax.scatter(hf_fit[u_DOE, 0], hf_fit[u_DOE, 1], marker='s', c='red', s=size, label="unfeasible design")

    sc = ax.scatter(hf_fit[n_DOE:, 0], hf_fit[n_DOE:, 1], marker='*', c=colors[n_DOE:], s=size, label="HF infills")
    ax.scatter(hf_fit[u_infill, 0], hf_fit[u_infill, 1], marker='o', c='red', s=size)
    ax.scatter(bsl_w_ADP, bsl_w_OP, marker="+", color="k", s=size, label="HF baseline")

    for idx in range(n_infill):
        hf_res = hf_fit[idx + n_DOE,:]
        pairs = lf_infill[idx*nLFtonHF,:]
        linea_x = np.linspace(hf_res[0],pairs[0])
        linea_y = np.linspace(hf_res[1],pairs[1])
        ax.plot(linea_x,linea_y,'--',color='k',alpha=0.7,linewidth=0.75)
    
    idx = 0
    for point in selected:
        ax.scatter(hf_fit[point, 0], hf_fit[point, 1], marker="o", facecolors='none', edgecolors='red', linewidth=1.0, s=size*2.0)
        plt.text(hf_fit[point, 0]-0.0025, hf_fit[point, 1]-0.006, str(idx+1), fontsize=6, fontweight='bold',
                    color='r', ha='left', va='bottom')
        idx+=1

    ax.set_axisbelow(True)
    ax.grid(True, color="grey", linestyle="dashed")
    
    ax.set_xlim(0.02, 0.10)
    ax.set_ylim(0.025, 0.20)
    
    # ax.set_ylim(0.030, 0.08)
    # ax.set_xlim(0.025, 0.055)
    cbar = plt.colorbar(sc, ax=ax, ticks=list(range(1, n_infill + 1)))
    cbar.set_ticklabels([str(i) for i in range(1, n_infill + 1)])
    cbar.set_label('Infill number')
    # legend
    ax.legend(loc="lower right")
    ax.set_xlabel('$w_{ADP}$ [-]')
    ax.set_ylabel('$w_{OP}$ [-]')
    plt.savefig(os.path.join(os.getcwd(), "mf_pareto.pdf"), bbox_inches="tight")

    plt.show()

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

def read_mf_lf_results(home_dir, master_dir, n_DOE, n_infill, nLFtonHF):

    directoryList = [os.path.join(home_dir, 'lf_doe')] + [os.path.join(home_dir, master_dir+f'_{n}/low_infill_{n}') for n in range(n_infill)]

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
            for i in range(n_DOE*nLFtonHF):
                loss_ADP = np.genfromtxt(os.path.join(dir,f'WOLF/wolf_g0_c{i}/ADP/MixedoutLossCoef.dat'),skip_header=1,delimiter=',')
                loss_OP1 = np.genfromtxt(os.path.join(dir,f'WOLF/wolf_g0_c{i}/OP1/MixedoutLossCoef.dat'),skip_header=1,delimiter=',')
                loss_OP2 = np.genfromtxt(os.path.join(dir,f'WOLF/wolf_g0_c{i}/OP2/MixedoutLossCoef.dat'),skip_header=1,delimiter=',')

                beta_ADP = np.genfromtxt(os.path.join(dir,f'WOLF/wolf_g0_c{i}/ADP/OutflowAngle.dat'),skip_header=1,delimiter=',')
                beta_OP1 = np.genfromtxt(os.path.join(dir,f'WOLF/wolf_g0_c{i}/OP1/OutflowAngle.dat'),skip_header=1,delimiter=',')
                beta_OP2 = np.genfromtxt(os.path.join(dir,f'WOLF/wolf_g0_c{i}/OP2/OutflowAngle.dat'),skip_header=1,delimiter=',')

                df_temp.loc[len(df_temp)] = [0, i+1, loss_ADP, (loss_OP1+loss_OP2)*0.5, loss_OP1, loss_OP2, beta_ADP, beta_OP1, beta_OP2,
                                             cand[i,0], cand[i,1], cand[i,2], cand[i,3], cand[i,4],
                                        os.path.join(dir, f"FFD/ogv1c_g0_c{i}.dat")]
        else:
            cont+=1
            for i in range(nLFtonHF):
                loss_ADP = np.genfromtxt(os.path.join(dir,f'WOLF/wolf_g0_c{i}/ADP/MixedoutLossCoef.dat'),skip_header=1,delimiter=',')
                loss_OP1 = np.genfromtxt(os.path.join(dir,f'WOLF/wolf_g0_c{i}/OP1/MixedoutLossCoef.dat'),skip_header=1,delimiter=',')
                loss_OP2 = np.genfromtxt(os.path.join(dir,f'WOLF/wolf_g0_c{i}/OP2/MixedoutLossCoef.dat'),skip_header=1,delimiter=',')

                beta_ADP = np.genfromtxt(os.path.join(dir,f'WOLF/wolf_g0_c{i}/ADP/OutflowAngle.dat'),skip_header=1,delimiter=',')
                beta_OP1 = np.genfromtxt(os.path.join(dir,f'WOLF/wolf_g0_c{i}/OP1/OutflowAngle.dat'),skip_header=1,delimiter=',')
                beta_OP2 = np.genfromtxt(os.path.join(dir,f'WOLF/wolf_g0_c{i}/OP2/OutflowAngle.dat'),skip_header=1,delimiter=',')

                df_temp.loc[len(df_temp)] = [cont, i+1, loss_ADP, (loss_OP1+loss_OP2)*0.5, loss_OP1, loss_OP2, beta_ADP, beta_OP1, beta_OP2,
                                                cand[i,0], cand[i,1], cand[i,2], cand[i,3], cand[i,4],
                                        os.path.join(dir, f"FFD/ogv1c_g0_c{i}.dat")]
        # print(df_temp)
        # input()

    return df_temp

def read_lf_results(dir, n_pop, n_gen, check=False):

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
    "unfeas_geom": pd.Series(dtype="float"),
    "unfeas_beta": pd.Series(dtype="float"),
    "fName": pd.Series(dtype="str")
    })

    cont = 0
    cand = np.genfromtxt(os.path.join(dir, "candidates.txt"))
    constraints = np.genfromtxt(os.path.join(dir, "constraints.txt"))

    for gen in range(n_gen):
        for ind in range(n_pop):
            
            loss_ADP = np.genfromtxt(os.path.join(dir,f'WOLF/wolf_g{gen}_c{ind}/ADP/MixedoutLossCoef.dat'),skip_header=1,delimiter=',')
            loss_OP1 = np.genfromtxt(os.path.join(dir,f'WOLF/wolf_g{gen}_c{ind}/OP1/MixedoutLossCoef.dat'),skip_header=1,delimiter=',')
            loss_OP2 = np.genfromtxt(os.path.join(dir,f'WOLF/wolf_g{gen}_c{ind}/OP2/MixedoutLossCoef.dat'),skip_header=1,delimiter=',')

            beta_ADP = np.genfromtxt(os.path.join(dir,f'WOLF/wolf_g{gen}_c{ind}/ADP/OutflowAngle.dat'),skip_header=1,delimiter=',')
            beta_OP1 = np.genfromtxt(os.path.join(dir,f'WOLF/wolf_g{gen}_c{ind}/OP1/OutflowAngle.dat'),skip_header=1,delimiter=',')
            beta_OP2 = np.genfromtxt(os.path.join(dir,f'WOLF/wolf_g{gen}_c{ind}/OP2/OutflowAngle.dat'),skip_header=1,delimiter=',')


            df_temp.loc[len(df_temp)] = [gen+1, ind+1, loss_ADP, (loss_OP1+loss_OP2)*0.5, loss_OP1, loss_OP2, beta_ADP, beta_OP1, beta_OP2,
                                        cand[cont,0], cand[cont,1], cand[cont,2], cand[cont,3], cand[cont,4],
                                        np.any(constraints[cont,:-3] >= 0, axis=0), np.any(constraints[cont,-3:] >= 0, axis=0),
                                        os.path.join(dir, f"FFD/ogv1c_g{gen}_c{ind}.dat")]
            
            cont+=1

    if check is True:
        lf_cand = np.loadtxt(os.path.join(dir, "candidates.txt"))
        lf_fit = np.loadtxt(os.path.join(dir, "fitnesses.txt"))

        for i in range(len(lf_cand)):
            delta_x = np.max(np.abs(df_temp[['x1','x2','x3','x4','x5']].iloc[i].to_numpy() - lf_cand[i,:]))
            delta_w = np.max(np.abs(df_temp[['w_ADP','w_OP']].iloc[i].to_numpy() - lf_fit[i,:]))
            assert np.max([delta_x, delta_w]) < 1e-8, f"Mismatch in LF optimization results for candidate {i}: delta_x = {delta_x}, delta_w = {delta_w}"

    return df_temp

def plot_opt_mf(lf_df, hf_df, nLFtoHF, bsl_w_ADP, bsl_w_OP, selected=None):

    n_DOE = len(hf_df[hf_df['infill'] == 0])
    n_infill = len(hf_df) - n_DOE

    cmap_hf = plt.get_cmap("viridis")
    norm_hf = mcolors.Normalize(vmin=0, vmax=n_infill)

    cmap_lf = plt.get_cmap("viridis")
    norm_lf = mcolors.Normalize(vmin=0, vmax=n_infill)

    hf_unfeas = hf_df.loc[(hf_df["beta_ADP"] < beta_ADP_bounds[0]) | (hf_df["beta_ADP"] > beta_ADP_bounds[1]) |
                          (hf_df["beta_OP1"] < beta_OP1_bounds[0]) | (hf_df["beta_OP1"] > beta_OP1_bounds[1]) |
                          (hf_df["beta_OP2"] < beta_OP2_bounds[0]) | (hf_df["beta_OP2"] > beta_OP2_bounds[1])]
    
    # selectedPlot = [sel + n_DOE for sel in selected]

    size = 15
    fig, ax = plt.subplots(figsize=square_fig)
    
    ax.scatter(hf_df.loc[hf_df["infill"] == 0, ["w_ADP"]], hf_df.loc[hf_df["infill"] == 0, ["w_OP"]], marker='s', c=hf_df.loc[hf_df["infill"] == 0, ["infill"]].to_numpy()[::-1], s=size, label="Initial LES DOE")
    ax.scatter(hf_df.loc[hf_df["infill"] != 0, ["w_ADP"]][::-1], hf_df.loc[hf_df["infill"] != 0, ["w_OP"]][::-1], marker='o', cmap=cmap_hf, c=hf_df.loc[hf_df["infill"] != 0, ["infill"]].to_numpy()[::-1], s=size, alpha=0.8)
    ax.scatter(hf_df.loc[hf_df["infill"] != 0, ["w_ADP"]].to_numpy()[0], hf_df.loc[hf_df["infill"] != 0, ["w_OP"]].to_numpy()[0], marker='o', cmap=cmap_hf, c=hf_df.loc[hf_df["infill"] != 0, ["infill"]].to_numpy()[0], s=size, label="MF opt LES infill")
    
    sc = ax.scatter(lf_df[["w_ADP"]], lf_df[["w_OP"]], marker='X', cmap=cmap_lf, c=lf_df[["infill"]].to_numpy(), s=size, linewidth=0.5, label="MF opt RANS infill")
    ax.scatter(hf_unfeas.loc[hf_unfeas["infill"] == 0, ["w_ADP"]], hf_unfeas.loc[hf_unfeas["infill"] == 0, ["w_OP"]], marker='s', c='red', s=size)
    ax.scatter(bsl_w_ADP, bsl_w_OP, marker="+", color="k", s=size*2.0, label="LES baseline")
    ax.scatter(hf_unfeas.loc[hf_unfeas["infill"] != 0, ["w_ADP"]], hf_unfeas.loc[hf_unfeas["infill"] != 0, ["w_OP"]], marker='o', c='red', s=size, label='Infeasible design')
    
    idx = 0
    for point in selected:
        x_HF = hf_df.iloc[point+n_DOE]['w_ADP']
        y_HF = hf_df.iloc[point+n_DOE]['w_OP']
        gen, ind = hf_df.iloc[point+n_DOE][['infill', 'cand']]

        print(gen)

        print(lf_df.loc[(lf_df["infill"] == gen) & (lf_df["cand"] == ind)])

        x_LF = lf_df.loc[(lf_df["infill"] == gen) & (lf_df["cand"] == ind)]['w_ADP']
        y_LF = lf_df.loc[(lf_df["infill"] == gen) & (lf_df["cand"] == ind)]['w_OP']

        # ax.scatter(x_HF, y_HF, marker='o', cmap=cmap_hf, c=gen, s=size)
        # ax.scatter(x_LF, y_LF, marker='X', cmap=cmap_lf, c=gen, s=size)

        x_line = np.linspace(x_HF, x_LF, 10)
        y_line = np.linspace(y_HF, y_LF, 10)
        ax.scatter(x_HF, y_HF, marker="o", facecolors='none', edgecolors='red', linewidth=1, s=size*1.5)
        ax.scatter(x_LF, y_LF, marker="X", facecolors='none', edgecolors='red', linewidth=1, s=size*1.5)
        
        txt = plt.text(x_HF-0.002, y_HF-0.0075, str(idx+1), fontsize=0.8*fs, fontweight='bold',
                    color='r', ha='left', va='bottom')
        idx+=1

        ax.plot(x_line, y_line, color='tab:grey', linestyle='dashed', linewidth=1)

    ax.set_xlim(0.02, 0.10)
    ax.set_ylim(0.02, 0.20)
    ax.set_xlabel('$\mathcal{L}_{ADP}$ [-]')
    ax.set_ylabel('$\mathcal{L}_{OP}$ [-]')
    ax.grid(True, color="grey", linestyle="dashed",alpha = 0.6)
    plt.legend(loc="lower right",markerscale=1.8)
    
    axins = ax.inset_axes([0.05,0.55,0.3,0.3])
    bbox = axins.get_position()

    # Ricrea gli scatter nell'inset (non copiare)
    axins.scatter(hf_df.loc[hf_df["infill"] == 0, ["w_ADP"]], hf_df.loc[hf_df["infill"] == 0, ["w_OP"]], marker='s', color='k', s=size)
    axins.scatter(hf_df.loc[hf_df["infill"] != 0, ["w_ADP"]][::-1], hf_df.loc[hf_df["infill"] != 0, ["w_OP"]][::-1], marker='o', cmap=cmap_hf, c=hf_df.loc[hf_df["infill"] != 0, ["infill"]].to_numpy()[::-1], s=size*2)
    axins.scatter(lf_df[["w_ADP"]], lf_df[["w_OP"]], marker='X', cmap=cmap_lf, c=lf_df[["infill"]].to_numpy(), s=size)
    axins.scatter(hf_unfeas.loc[hf_unfeas["infill"] != 0, ["w_ADP"]], hf_unfeas.loc[hf_unfeas["infill"] != 0, ["w_OP"]], marker='o', c='red', s=size*2)
    axins.scatter(hf_unfeas.loc[hf_unfeas["infill"] == 0, ["w_ADP"]], hf_unfeas.loc[hf_unfeas["infill"] == 0, ["w_OP"]], marker='s', c='red', s=size*2)
    axins.scatter(bsl_w_ADP, bsl_w_OP, marker="+", color="k", s=size*3.0)
    
    # Linee di connessione
    idx = 0
    for point in selected:
        x_HF = hf_df.iloc[point+n_DOE]['w_ADP']
        y_HF = hf_df.iloc[point+n_DOE]['w_OP']
        gen, ind = hf_df.iloc[point+n_DOE][['infill', 'cand']]
        x_LF = lf_df.loc[(lf_df["infill"] == gen) & (lf_df["cand"] == ind)]['w_ADP'].values[0]
        y_LF = lf_df.loc[(lf_df["infill"] == gen) & (lf_df["cand"] == ind)]['w_OP'].values[0]
        x_line = np.linspace(x_HF, x_LF, 10)
        y_line = np.linspace(y_HF, y_LF, 10)
        axins.plot(x_line, y_line, color='tab:grey', linestyle='dashed', linewidth=1)

        axins.scatter(x_HF, y_HF, marker="o", facecolors='none', edgecolors='red', linewidth=1.0, s=size*2.0)
        axins.text(x_HF-0.0010, y_HF-0.0025, str(idx+1), fontsize=fs*0.9, fontweight='bold', color='r', ha='left', va='bottom')
        idx+=1

    # Imposta i limiti dello zoom
    axins.set_xlim(0.027, 0.036)
    axins.set_ylim(0.041, 0.060)
    # axins.set_aspect('equal', adjustable='box')
    axins.grid(True, color="grey", linestyle="dashed", alpha=0.6)
    axins.set_xticks([0.03])
    axins.set_yticks([0.05])
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    
    # Aggiungi rettangolo di evidenza sul plot principale

    x0_ax, y0_ax = 0.024, 0.035
    wx_ax, wy_ax = 0.013, 0.026
    x0_axins, y0_axins = 0.0215, 0.1145
    wx_axins, wy_axins = 0.0290, 0.0635

    x1_line = np.linspace

    rect = Rectangle((x0_ax, y0_ax), wx_ax, wy_ax, linewidth=0.5, 
                     edgecolor='black', facecolor='none')
    
    rect_bg = Rectangle((x0_axins, y0_axins), wx_axins, wy_axins, linewidth=0.5, 
                     edgecolor='black', facecolor='white', zorder=axins.get_zorder() - 1)

    ax.add_patch(rect_bg)
    ax.add_patch(rect)

    ax.plot([x0_axins, x0_ax], [y0_axins, y0_ax+ wy_ax], 'k', linewidth=0.5)
    ax.plot([x0_axins+wx_axins, x0_ax + wx_ax], [y0_axins, y0_ax + wy_ax], 'k', linewidth=0.5)

    cbar = plt.colorbar(sc, ax=ax, ticks=[0, 5, 10, 15], pad=0.02, fraction=0.050)
    cbar.set_label('Infill number')

    plt.savefig(os.path.join(os.getcwd(), "mf_pareto.pdf"), bbox_inches="tight")
    
    plt.show()

def plot_opt_lf(lf_df, hf_df, selected=None):

    n_DOE = len(hf_df[hf_df['infill'] == 0])
    n_infill = len(hf_df) - n_DOE

    lf_fit = np.vstack([[lf_df['w_ADP'].astype(float).to_numpy().flatten()], [lf_df['w_OP'].astype(float).to_numpy().flatten()]]).T
    lf_pareto = compute_pareto(lf_fit[:, 0], lf_fit[:, 1])
    lf_pareto_idx = [np.where(lf_fit == pid)[0][0] for pid in lf_pareto]  
    df_pareto = lf_df.iloc[lf_pareto_idx].reset_index(drop=True)

    cmap_lf = plt.get_cmap("viridis")
    norm_lf = mcolors.Normalize(vmin=0, vmax=np.max(df_pareto['infill'].to_numpy()))

    size = 20
    fig, ax = plt.subplots(figsize=square_fig)

    sc = ax.scatter(lf_df["w_ADP"], lf_df["w_OP"], marker='d', cmap=cmap_lf, c=lf_df[["infill"]].to_numpy(), s=size, label="LF opt RANS")

    labels_added = set()
    for i, hf_row in hf_df[['x1', 'x2', 'x3', 'x4', 'x5']].iterrows():
        diff = np.abs(lf_df[['x1', 'x2', 'x3', 'x4', 'x5']].values - hf_row.values)
        mask = np.max(diff, axis=1) < 1e-7
        matched_rows = lf_df[mask]
        ADP_violate = (hf_df.loc[i, 'beta_ADP'] < beta_ADP_bounds[0]) or (hf_df.loc[i, 'beta_ADP'] > beta_ADP_bounds[1])
        OP1_violate = (hf_df.loc[i, 'beta_OP1'] < beta_OP1_bounds[0]) or (hf_df.loc[i, 'beta_OP1'] > beta_OP1_bounds[1])
        OP2_violate = (hf_df.loc[i, 'beta_OP2'] < beta_OP2_bounds[0]) or (hf_df.loc[i, 'beta_OP2'] > beta_OP2_bounds[1])

        x_lf, y_lf = matched_rows["w_ADP"].values[0], matched_rows["w_OP"].values[0]
        x_hf, y_hf = hf_df.loc[i, 'w_ADP'], hf_df.loc[i, "w_OP"]

        ax.scatter(x_lf, y_lf, marker='d', color='k', linewidths=lw, alpha = 0.8, s=size,facecolors='none', label="LF opt Pareto" if "pareto" not in labels_added else None)
        ax.scatter(x_hf, y_hf, color='peru', s=size*1.5, marker='*', linewidth=0.5, label="LF opt LES" if "les" not in labels_added else None)
        ax.plot([x_lf,x_hf],[y_lf,y_hf],'--',color='tab:grey',linewidth=0.5)
        labels_added.update(["pareto", "les"])
        
        if ADP_violate == True or OP1_violate == True or OP2_violate == True:
            ax.scatter(x_hf, y_hf, c='red', s=size*1.5, marker='*', linewidth=0.5)
            ax.plot([x_lf,x_hf],[y_lf,y_hf],'--',color='tab:red',linewidth=0.5)
            labels_added.add("infeasible")

    ax.scatter(lf_bsl_w_ADP, lf_bsl_w_OP, marker="x", color="k", s=size*2.0, label="RANS baseline")
    ax.scatter(hf_bsl_w_ADP, hf_bsl_w_OP, marker="+", color="k", s=size*2.0, label="LES baseline")
    if "infeasible" in labels_added:
        ax.scatter([], [], marker='*', c='red', s=size*1.5, linewidth=0.5, label='Infeasible design')

    # ax.scatter(mf_hf_opt.loc[mf_hf_opt["infill"] != 0, ["w_ADP"]][::-1], mf_hf_opt.loc[mf_hf_opt["infill"] != 0, ["w_OP"]][::-1], marker='o', cmap=cmap_mf, color='tab:green', s=size, label="HF infill", alpha=0.8)

    ax.set_xlim(0.02, 0.10)
    ax.set_ylim(0.02, 0.24)
    ax.set_xlabel('$\mathcal{L}_{ADP}$ [-]')
    ax.set_ylabel('$\mathcal{L}_{OP}$ [-]')
    ax.grid(True, color="grey", linestyle="dashed",alpha = 0.6)
    cbar = plt.colorbar(sc, ax=ax, ticks=[1, 25, 50], pad=0.02, fraction=0.050)
    cbar.set_label('Generation')
    plt.legend(ncols=2)
    
    axins = ax.inset_axes([0.58,0.05,0.4,0.17])
    bbox = axins.get_position()

    axins.scatter(hf_bsl_w_ADP, hf_bsl_w_OP, marker="+", color="k", s=size*2.0)

    cont = 0
    cont_unfeas = 0
    unfeas_ADP = 0
    unfeas_OP1 = 0
    unfeas_OP2 = 0
    for i, hf_row in hf_df[['x1', 'x2', 'x3', 'x4', 'x5']].iterrows():
        cont+=1
        diff = np.abs(lf_df[['x1', 'x2', 'x3', 'x4', 'x5']].values - hf_row.values)
        mask = np.max(diff, axis=1) < 1e-7
        matched_rows = lf_df[mask]
        ADP_violate = (hf_df.loc[i, 'beta_ADP'] < beta_ADP_bounds[0]) or (hf_df.loc[i, 'beta_ADP'] > beta_ADP_bounds[1])
        OP1_violate = (hf_df.loc[i, 'beta_OP1'] < beta_OP1_bounds[0]) or (hf_df.loc[i, 'beta_OP1'] > beta_OP1_bounds[1])
        OP2_violate = (hf_df.loc[i, 'beta_OP2'] < beta_OP2_bounds[0]) or (hf_df.loc[i, 'beta_OP2'] > beta_OP2_bounds[1])

        x_lf, y_lf = matched_rows["w_ADP"].values[0], matched_rows["w_OP"].values[0]
        x_hf, y_hf = hf_df.loc[i, 'w_ADP'], hf_df.loc[i, "w_OP"]

        print(matched_rows['infill'].values)
        axins.scatter(x_hf, y_hf, color='peru', s=size*1.5, marker='*', linewidth=0.5,alpha = 0.8)
        
        if ADP_violate == True or OP1_violate == True or OP2_violate == True:
            axins.scatter(x_hf, y_hf, c='red', s=size*1.5, marker='*', linewidth=0.5,alpha = 0.8)
            cont_unfeas += 1
            
        if ADP_violate == True:
            unfeas_ADP += 1
        if OP1_violate == True:
            unfeas_OP1 += 1
        if OP2_violate == True:
            unfeas_OP2 += 1

    print('Fraction of infeasible designs in LF Pareto:', cont_unfeas/cont)
    print('Fraction of ADP infeasible designs in LF Pareto:', unfeas_ADP/cont)
    print('Fraction of OP1 infeasible designs in LF Pareto:', unfeas_OP1/cont)
    print('Fraction of OP2 infeasible designs in LF Pareto:', unfeas_OP2/cont)

    # # Imposta i limiti dello zoom
    axins.set_xlim(0.025, 0.036)
    axins.set_ylim(0.041, 0.057)
    axins.grid(True, color="grey", linestyle="dashed", alpha=0.6)
    axins.set_xticks([0.03])
    axins.set_yticks([0.05])
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    
    # # Aggiungi rettangolo di evidenza sul plot principale

    x0_ax, y0_ax = 0.024, 0.039
    wx_ax, wy_ax = 0.013, 0.020
    x0_axins, y0_axins = 0.0645, 0.026
    wx_axins, wy_axins = 0.035, 0.0465

    # x1_line = np.linspace

    rect = Rectangle((x0_ax, y0_ax), wx_ax, wy_ax, linewidth=0.5, 
                     edgecolor='black', facecolor='none')
    
    rect_bg = Rectangle((x0_axins, y0_axins), wx_axins, wy_axins, linewidth=0.5, 
                     edgecolor='black', facecolor='white', zorder=axins.get_zorder() - 1)

    ax.add_patch(rect)
    ax.add_patch(rect_bg)

    ax.plot([x0_ax+wx_ax, x0_axins], [y0_ax, y0_axins], 'k', linewidth=0.5)
    ax.plot([x0_ax+wx_ax, x0_axins], [y0_ax + wy_ax, y0_axins + wy_axins], 'k', linewidth=0.5)

    plt.savefig(os.path.join(os.getcwd(), "lf_pareto.pdf"), bbox_inches="tight")
    
    # plt.show()

def plot_scatter_modes(df_opt, baseline, figsize=[8,4], fontsize=16, chosen=None):

    bsl_w_ADP, bsl_w_OP1, bsl_w_OP2 = baseline

    size = 20

    plt.figure(figsize=(figsize[0], figsize[1]))
    
    cmap='viridis'
    norm = plt.Normalize(0, 15)
    cmap = cm.get_cmap('viridis')

    # df_c = df_chosen.reset_index(drop=True)
    lista_idx = [5,4,3,2,1]

    xlim={'x5':[-0.05,0.05], 'x4':[-0.1,0.1], 'x3':[-0.1,0.1], 'x2':[-0.1,0.1], 'x1':[-0.1,0.1]}
    # xlim={'x5':[-0.1,0.1], 'x4':[-0.1,0.1], 'x3':[-0.1,0.1], 'x2':[-0.1,0.1], 'x1':[-0.1,0.1]}
    ylim={'w_ADP':[0.018,0.062], 'w_OP1':[0.026,0.134], 'w_OP2':[0.018,0.062]}

    fig = plt.gcf()
    
    for i in range(0,15):

        plt.subplot(3,5,i+1)

        argx = 'x'+str(lista_idx[i%5])
        if i < 5:
            argy = 'w_ADP'
            plt.plot(xlim[argx], [bsl_w_ADP, bsl_w_ADP], 'k--', linewidth=0.5, label ="LES baseline")
        elif i < 10:
            argy = 'w_OP1'
            plt.plot(xlim[argx], [bsl_w_OP1, bsl_w_OP1], 'k--', linewidth=0.5)
        else:
            argy = 'w_OP2'
            plt.plot(xlim[argx], [bsl_w_OP2, bsl_w_OP2], 'k--', linewidth=0.5)

        
        plt.scatter(df_opt.loc[df_opt["infill"] == 0, [argx]], df_opt.loc[df_opt["infill"] == 0, [argy]], marker="s", s=size, c=df_opt.loc[df_opt["infill"] == 0, ["infill"]].to_numpy(), label="Initial LES DOE")
        plt.scatter(df_opt.loc[df_opt["infill"] != 0, [argx]], df_opt.loc[df_opt["infill"] != 0, [argy]], marker="o", s=size, c=df_opt.loc[df_opt["infill"] != 0, ["infill"]].to_numpy(), label="MF opt LES infill")

        idx = 0
        n_DOE = len(df_opt[df_opt['infill'] == 0])
        if not(chosen is None):
            for point in chosen:
                x_HF = df_opt.iloc[point+n_DOE][argx]
                y_HF = df_opt.iloc[point+n_DOE][argy]

                plt.scatter(x_HF, y_HF, marker="o", facecolors='none', edgecolors='red', linewidth=1.5, s=size*1.5)
                
                if argy == 'w_ADP':
                    dx, dy = [-0.002, 0.002]
                elif argy == 'w_OP2':
                    dx, dy = [-0.002, 0.0012]
                else:
                    dx, dy = [0.006, -0.0055]
                
                txt = plt.text(x_HF+dx, y_HF+dy, str(idx+1), fontsize=8, fontweight='bold',
                            color='r', ha='left', va='bottom')
                idx+=1

        if i == 0:
            plt.ylabel(r'$w_{ADP}$')
        elif i == 5:
            plt.ylabel(r'$w_{OP1}$')
            plt.yticks([0.03, 0.08, 0.13])
        elif i == 10:
            plt.ylabel(r'$w_{OP2}$')
        else:
            plt.yticks([])

        if i in [10,11,12,13,14]:
            plt.xticks([np.min(xlim[argx]), 0, np.max(xlim[argx])])
            plt.xlabel(r'$d_'+str(i-9)+'$')
        else:   
            plt.xticks([])
        
        plt.xlim(xlim[argx])
        plt.ylim(ylim[argy])
    
    # Get handles from first subplot which has all labels including "LES baseline"
    plt.subplot(3,5,1)
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Add custom legend entries for the numbers
    custom_handles = handles[:3] + [
        Line2D([0], [0], marker='$1$', color='none', markerfacecolor='r', markeredgecolor='r', markersize=10, linestyle='', label='Best ADP'),
        Line2D([0], [0], marker='$2$', color='none', markerfacecolor='r', markeredgecolor='r', markersize=10, linestyle='', label='Best Trade-off'),
        Line2D([0], [0], marker='$3$', color='none', markerfacecolor='r', markeredgecolor='r', markersize=10, linestyle='', label='Best OP2')
    ]
    custom_labels = labels[:3] + ['Best ADP', 'Best Trade-off', 'Best OP2']
    
    fig.legend(custom_handles, custom_labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=6, frameon=True, columnspacing=1.0, handletextpad=0.5)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig("mfhf_scatter_modes.png", bbox_inches="tight")
    # plt.show()

def plot_beta(lf_df, hf_df):

    n_DOE = len(hf_df[hf_df['infill'] == 0])
    n_infill = len(hf_df) - n_DOE

    cmap_ADP = plt.get_cmap("Blues_r")
    norm_ADP = mcolors.Normalize(vmin=0, vmax=n_infill)

    cmap_OP1 = plt.get_cmap("Reds_r")
    norm_OP1 = mcolors.Normalize(vmin=0, vmax=n_infill)

    cmap_OP2 = plt.get_cmap("Greens_r")
    norm_OP2 = mcolors.Normalize(vmin=0, vmax=n_infill)

    size = 15
    plt.figure(figsize=(5, 3))

    beta_list = ['beta_ADP', 'beta_OP1', 'beta_OP2']
    marker = {'beta_ADP':'o', 'beta_OP1':'d', 'beta_OP2':'^'}
    cmap = {'beta_ADP':cmap_ADP, 'beta_OP1':cmap_OP1, 'beta_OP2':cmap_OP2}

    # select data for limits computation so all subplots share the same scale
    hf_sel = hf_df.loc[hf_df["infill"] != 0, beta_list]
    lf_sel = lf_df.loc[(lf_df["infill"] != 0) & (lf_df["cand"] == 1), beta_list]
    

    xlim = [-6,6]

    for beta in beta_list:
        HFPlot = hf_df[beta].to_numpy().flatten()
        LFPlot = np.hstack([lf_df[45:50][[beta]].to_numpy().flatten(), lf_df.loc[(lf_df["infill"] != 0) & (lf_df["cand"] == 1), [beta]].to_numpy().flatten()])

        xVerify = hf_df['x1'].to_numpy().flatten()
        for j in range(1,6):
            xVerify = hf_df[f'x{j}'].to_numpy().flatten()
            yVerify = np.hstack([lf_df[45:50][f'x{j}'].to_numpy().flatten(), lf_df.loc[(lf_df["infill"] != 0) & (lf_df["cand"] == 1), [f'x{j}']].to_numpy().flatten()])
            assert np.abs(np.min(xVerify-yVerify)) <1e-8

        sc = plt.scatter(HFPlot, LFPlot, marker=marker[beta], cmap=cmap[beta], label=beta,
               c=hf_df["infill"].to_numpy().flatten(), edgecolors='black', linewidths=0.8)
        
        plt.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], linestyle='--', linewidth=0.5, color='k')
        
        plt.xlabel('Beta HF')
        plt.ylabel('Beta LF')
        
        plt.gca().set_aspect('equal', adjustable='box')
        print('Mean error LF-HF (LF = HF + eps) for '+beta+': ', np.mean(LFPlot - HFPlot))
    

    cbar = plt.colorbar(sc, ticks=[0, 5, 10, 15], pad=0.02, fraction=0.050)
    cbar.set_label('Infill number')

    plt.tight_layout()
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(os.getcwd(), "beta_error.png"), bbox_inches="tight")
    # plt.show()
    plt.close('all')

def plot_chosen_profiles(mf_df, baseline, selected, figsize=[6.6, 2.5]):

    n_DOE = len(mf_df[mf_df['infill'] == 0])

    # Plot 5: Profile comparison
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1]))
    Delta = 0.0025
    
    axins1 = ax.inset_axes([0.37, 0.05, 0.20, 0.35])
    axins2 = ax.inset_axes([0.63, 0.2, 0.20, 0.45])

    cont = 1

    for point in selected:

        entry = mf_df.iloc[point+n_DOE]
        if cont == 1:
            lab = 'Best ADP'
            col = 'tab:blue'
        if cont == 2:
            lab = 'Best Trade-off'
            col = 'tab:orange'
        if cont == 3:
            lab = 'Best OP'
            col = 'tab:green'
        print(entry[['w_ADP','w_OP']].T)

        pro = np.loadtxt(os.path.join(entry['fName']), skiprows=2)*1e-3
        # ax.plot(pro[:, 0], pro[:, 1], linewidth=1, label=r"$L_{ADP}=$" + f"{row['w_ADP']:.4f};" + r"$L_{OP}=$" + f"{row['w_OP']:.4f}")
        ax.plot(pro[:, 0], pro[:, 1], linewidth=lw, label=lab, color=col)
        ax.set(xlabel='$x$ [m]', ylabel='$y$ [m]')
        cont += 1
        axins1.plot(pro[:, 0], pro[:, 1], linewidth=lw)
        axins2.plot(pro[:, 0], pro[:, 1], linewidth=lw)
    
    ax.plot(baseline[:, 0], baseline[:, 1], color="k", linestyle="dashed", linewidth=lw, label="Baseline")
    axins1.plot(baseline[:, 0], baseline[:, 1], color="k", linestyle="dashed", linewidth=lw)
    axins2.plot(baseline[:, 0], baseline[:, 1], color="k", linestyle="dashed", linewidth=lw)

    # ax.legend()
    axins1.set_xlim(min(baseline[:, 0]) - 0.0004, min(baseline[:, 0]) + 0.006)
    axins1.set_ylim(min(baseline[:, 1]) - 0.0006, min(baseline[:, 1]) +0.0035)
    axins1.set_xticks([])
    axins1.set_yticks([])
    axins2.set_xlim(max(baseline[:, 0]) - 0.0025, max(baseline[:, 0]) + 0.00025)
    axins2.set_ylim(max(baseline[:, 1]) - 0.0028, max(baseline[:, 1]) + 0.00050)
    axins2.set_xticks([])
    axins2.set_yticks([])

    mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="0.5")
    mark_inset(ax, axins2, loc1=2, loc2=4, fc="none", ec="0.5")
    
    plt.tight_layout()
    plt.axis('equal')
    ax.legend(frameon=True,framealpha=0,loc='upper left').get_frame().set_edgecolor("none")
    plt.savefig(os.path.join(os.getcwd(), "mf_profile_opt.pdf"), bbox_inches="tight")
    plt.xlim(-0.00005,0.07)
    # plt.xlim(-0.001,0.0025)
    plt.show()

def temp_read_hf_results(home_dir, master_dir, n_cand):

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

    cand = np.genfromtxt(os.path.join(home_dir, master_dir+".txt"))
    dir = os.path.join(home_dir, master_dir)
    for i in range(n_cand):
        temp_ADP = np.genfromtxt(os.path.join(dir,f'MUSICAA/musicaa_g0_c{i}/ADP/QoI_convergence.csv'),skip_header=1,delimiter=',')[-1,:]
        temp_OP1 = np.genfromtxt(os.path.join(dir,f'MUSICAA/musicaa_g0_c{i}/OP1/QoI_convergence.csv'),skip_header=1,delimiter=',')[-1,:]
        temp_OP2 = np.genfromtxt(os.path.join(dir,f'MUSICAA/musicaa_g0_c{i}/OP2/QoI_convergence.csv'),skip_header=1,delimiter=',')[-1,:]
        df_temp.loc[len(df_temp)] = [0, i+1, temp_ADP[0], (temp_OP1[0]+temp_OP2[0])*0.5, temp_OP1[0], temp_OP2[0], temp_ADP[1], temp_OP1[1], temp_OP2[1],
                                        cand[i,0], cand[i,1], cand[i,2], cand[i,3], cand[i,4],
                                os.path.join(dir, f"FFD/ogv1c_g0_c{i}.dat")]

    # cand = np.genfromtxt(os.path.join(home_dir, master_dir+".txt"))
    # dir = os.path.join(home_dir, master_dir)
    # for i in range(n_cand):

    #     temp_ADP = np.genfromtxt(os.path.join(home_dir,f'ADP_LES_{i}/MUSICAA/musicaa_g0_c0/ADP/QoI_convergence.csv'),skip_header=1,delimiter=',')[-1,:]
    #     temp_OP1 = np.genfromtxt(os.path.join(home_dir,f'OP1_LES_{i}/MUSICAA/musicaa_g0_c0/OP1/QoI_convergence.csv'),skip_header=1,delimiter=',')[-1,:]
    #     temp_OP2 = np.genfromtxt(os.path.join(home_dir,f'OP2_LES_{i}/MUSICAA/musicaa_g0_c0/OP2/QoI_convergence.csv'),skip_header=1,delimiter=',')[-1,:]
    #     df_temp.loc[len(df_temp)] = [0, i+1, temp_ADP[0], (temp_OP1[0]+temp_OP2[0])*0.5, temp_OP1[0], temp_OP2[0], temp_ADP[1], temp_OP1[1], temp_OP2[1],
    #                                     cand[i,0], cand[i,1], cand[i,2], cand[i,3], cand[i,4],
    #                             os.path.join(dir, f"FFD/ogv1c_g0_c{i}.dat")]

    # print(df_temp)

    return df_temp

def main():

    """Main analysis and plotting routine"""
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
    lf_dir = "/Data1/Mattia/RANS_OPT_baseDelta"
    # lf_dir = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/RANS_bruteForce/output_lf/"

    mf_hf_opt = read_hf_results(mf_dir, master_dir_mf, n_DOE, n_infill)
    hf_opt = read_hf_results(hf_dir, master_dir_hf, n_DOE, n_infill)
    mf_lf_opt = read_mf_lf_results(mf_dir, master_dir_mf, n_DOE, n_infill, nLFtoHF)
    pd.set_option('display.max_colwidth', None)
    print(mf_hf_opt[['infill','w_OP','w_ADP','fName']])

    # plot_opt_comparison(mf_hf_opt, hf_opt, hf_bsl_w_ADP, hf_bsl_w_OP, selected)
    # plot_opt_mf(mf_lf_opt, mf_hf_opt, nLFtoHF, hf_bsl_w_ADP, hf_bsl_w_OP, selected)
    # # plot_beta(mf_lf_opt, mf_hf_opt)
    # plot_scatter_modes(mf_hf_opt, [hf_bsl_w_ADP, hf_bsl_w_OP1, hf_bsl_w_OP2], chosen=selected)

    # lf_opt = read_lf_results(lf_dir, 30, 50)
    # # hf_RANS_rerun = temp_read_hf_results('/home/mciarlatani/Irene/irene_parallel_optimization/', 'RANS_OPT_baseDelta', 43)
    # # hf_RANS_rerun.to_csv('./RANS_OPT_baseDelta.df',index=False)
    # hf_RANS_rerun = pd.read_csv('./RANS_OPT_baseDelta.df')

    # print(lf_opt)
    # lf_opt = lf_opt.loc[lf_opt['unfeas_geom']==False].reset_index(drop=True)
    # lf_opt = lf_opt.loc[lf_opt['unfeas_beta']==False].reset_index(drop=True)
    # plot_opt_lf(lf_opt, hf_RANS_rerun, selected=None)
    # # plot_scatter_modes(df_pareto, [lf_bsl_w_ADP, lf_bsl_w_OP1, lf_bsl_w_OP2])



if __name__ == "__main__":
    main()
