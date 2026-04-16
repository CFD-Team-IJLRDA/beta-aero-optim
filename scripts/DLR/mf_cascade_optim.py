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

# Values corresponding to the mixed-out loss experiments (AIFLUID)
# LES mixed-out loss
hf_bsl_w_ADP = 0.0349
hf_bsl_w_OP = 0.0483

# non-adapted fine mesh mixed-out los
lf_bsl_w_ADP= 0.07826
lf_bsl_w_OP = 0.1489

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

def plot_pareto_single_HF(lf_cand, lf_fit, constraints, df_chosen, hf_results=None, figsize=[7.5, 6], fontsize=14):

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
    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    plt.rcParams.update({'font.size': fontsize})

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

def plot_chosen_profiles(df_choosen, baseline, figsize=[15.2, 5.64],fontsize=14):

    plt.rcParams.update({'font.size': fontsize})

    # Plot 5: Profile comparison
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1]))
    Delta = 0.0025
    ax.plot(baseline[:, 0], baseline[:, 1], color="k", linestyle="dashed", label="Baseline")
    axins1 = ax.inset_axes([0.27, 0.05, 0.35, 0.35])
    axins1.plot(baseline[:, 0], baseline[:, 1], color="k", linestyle="dashed")
    axins2 = ax.inset_axes([0.63, 0.3, 0.35, 0.35])
    axins2.plot(baseline[:, 0], baseline[:, 1], color="k", linestyle="dashed")

    cont = 1
    for index, row in df_choosen.iterrows():
    
        pro = np.loadtxt(os.path.join(row['fName']), skiprows=2)*1e-3
        # ax.plot(pro[:, 0], pro[:, 1], linewidth=1, label=r"$L_{ADP}=$" + f"{row['w_ADP']:.4f};" + r"$L_{OP}=$" + f"{row['w_OP']:.4f}")
        ax.plot(pro[:, 0], pro[:, 1], linewidth=1, label=r"Candidate " + f"{cont}")
        ax.set(xlabel='$x$ [m]', ylabel='$y$ [m]')
        cont += 1
        axins1.plot(pro[:, 0], pro[:, 1], linewidth=1)
        axins2.plot(pro[:, 0], pro[:, 1], linewidth=1)
        
    # ax.legend()
    axins1.set_xlim(min(baseline[:, 0]) - 0.15 * Delta, min(baseline[:, 1]) + 2 * Delta)
    axins1.set_ylim(min(baseline[:, 1]) - 0.25 * Delta, min(baseline[:, 1]) + Delta)
    axins1.set_xticks([])
    axins1.set_yticks([])
    axins2.set_xlim(max(baseline[:, 0]) - 3 * Delta, max(baseline[:, 0]) + 0.4 * Delta)
    axins2.set_ylim(max(baseline[:, 1]) - 1.25 * Delta, max(baseline[:, 1]) + 0.3 * Delta)
    axins2.set_xticks([])
    axins2.set_yticks([])

    mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="0.5")
    mark_inset(ax, axins2, loc1=2, loc2=4, fc="none", ec="0.5")
    
    plt.tight_layout()
    plt.axis('equal')
    ax.legend(frameon=True).get_frame().set_edgecolor("none")
    plt.savefig(os.path.join(os.getcwd(), "mf_profile_opt.pdf"), bbox_inches="tight")
    plt.show()

def plot_scatter_modes(df_opt, df_chosen, figsize=[15, 5], fontsize=16):

    plt.figure(figsize=(figsize[0], figsize[1]))
    plt.rcParams.update({'font.size': fontsize})
    
    cmap='viridis'
    norm = plt.Normalize(0, 49)
    cmap = cm.get_cmap('viridis')
    colors = cmap(norm(df_opt['gen'])) 

    df_c = df_chosen.reset_index(drop=True)
    lista_idx = [5,4,3,2,1]

    for i in range(0,10):
        arg1 = 'x'+str(lista_idx[i%5])
        if i < 5:
            arg2 = 'w_ADP'
        else:
            arg2 = 'w_OP'

        plt.subplot(2,5,i+1)
        plt.scatter(df_opt[arg1], df_opt[arg2], marker="o", s=10, c=colors)
        plt.scatter(df_c[arg1], df_c[arg2], marker="o", facecolors='none', edgecolors='r', linewidth=2, s=30)

        # Aggiungi l'indice accanto a ogni punto rosso
        for idx, row in df_c.iterrows():
            plt.text(row[arg1]+0.0015, row[arg2]+0.0015, str(idx+1), fontsize=fontsize-6, fontweight='bold',
                    color='r', ha='left', va='bottom')
            
        plt.xlim([-0.1,0.1])

        if i in [0,5]:
            plt.ylabel(r'$\mathcal{L}_{ADP}$' if i<5 else r'$\mathcal{L}_{OP}$')
        else:
            plt.yticks([])

        if i in [5,6,7,8,9]:
            plt.xlabel(r'$x_'+str(i-4)+'$')
            plt.xticks([-0.1,0,0.1])
        else:   
            plt.xticks([])
    
    plt.tight_layout()
    plt.savefig("lf_scatter_modes.pdf", bbox_inches="tight")
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

    plt.rcParams.update({'font.size': fontsize})

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

def plot_mf_scatter(n_DOE, n_infill, hf_fit, unfeasible, bsl_w_ADP, bsl_w_OP, selected=None):

    plt.rcParams.update({
        "figure.dpi": 300,
        "font.size": 8,
        'legend.fontsize': 8, 
        "axes.titlesize": 8,
        "axes.labelsize": 8
    })

    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=n_infill)
    colors = [0]*n_DOE + list(range(1, n_infill + 1))

    size = 20

    fig, ax = plt.subplots(figsize=(5.2, 3.64))
    
    idx = 0
    for point in selected:
        ax.scatter(hf_fit[point, 0], hf_fit[point, 1], marker="o", facecolors='none', edgecolors='red', linewidth=1.5, s=size*1.5)
        plt.text(hf_fit[point, 0]-0.001, hf_fit[point, 1]-0.001, str(idx+1), fontsize=6, fontweight='bold',
                    color='r', ha='left', va='bottom')
        idx+=1
                    
    u_DOE = [u for u in unfeasible if u<n_DOE]
    u_infill = [u for u in unfeasible if u>=n_DOE]
    
    ax.scatter(hf_fit[:n_DOE, 0], hf_fit[:n_DOE, 1], marker='s', c=colors[:n_DOE], s=size, label="initial DOE")
    ax.scatter(hf_fit[u_DOE, 0], hf_fit[u_DOE, 1], marker='s', c='red', s=size, label="unfeasible design")
    sc = ax.scatter(hf_fit[n_DOE:, 0], hf_fit[n_DOE:, 1], marker='o', c=colors[n_DOE:], s=size, label="HF infills")
    ax.scatter(hf_fit[u_infill, 0], hf_fit[u_infill, 1], marker='o', c='red', s=size)
    ax.scatter(bsl_w_ADP, bsl_w_OP, marker="+", color="k", s=size, label="HF baseline")

    ax.set_axisbelow(True)
    ax.grid(True, color="grey", linestyle="dashed")
    
    ax.set_ylim(0.030, 0.08)
    ax.set_xlim(0.025, 0.055)
    cbar = plt.colorbar(sc, ax=ax, ticks=list(range(1, n_infill + 1)))
    cbar.set_ticklabels([str(i) for i in range(1, n_infill + 1)])
    cbar.set_label('Infill number')
    # legend
    ax.legend(loc="lower right")
    ax.set_xlabel('$w_{ADP}$ [-]')
    ax.set_ylabel('$w_{OP}$ [-]')
    plt.savefig(os.path.join(os.getcwd(), "mf_pareto.pdf"), bbox_inches="tight")

    plt.show()

def plot_paper_pareto(n_DOE, n_infill, nLFtonHF, lf_fit, hf_fit, unfeasible, bsl_w_ADP, bsl_w_OP, selected=None):

    plt.rcParams.update({
        "figure.dpi": 300,
        "font.size": 8,
        'legend.fontsize': 8, 
        "axes.titlesize": 8,
        "axes.labelsize": 8
    })

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

def main():

    bsl_w_ADP = 0.0349
    bsl_w_OP = 0.0483

    """Main analysis and plotting routine"""

    nLFtonHF = 10

    n_DOE = 5
    n_infill = 15
    master_dir = 'output_paper'
    unfeasible = [0,1,2,3,9,10]
    # unfeasible = [0,1,2,3,9]
    selected = [n_DOE + 6, n_DOE, n_DOE + 3]  # indices of chosen optimal candidates

    plt.close('all')
    # Load mixed-out loss results
    mf_dir = "/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/" + master_dir

    mf_cand = np.loadtxt(os.path.join(mf_dir, "hf_candidates.txt"))
    hf_fit = np.loadtxt(os.path.join(mf_dir, "hf_fitnesses.txt"))

    lf_cand = np.loadtxt(os.path.join(mf_dir, "lf_candidates.txt"))
    lf_fit = np.loadtxt(os.path.join(mf_dir, "lf_fitnesses.txt"))
    candidates = np.genfromtxt(os.path.join(mf_dir, "candidates.txt"))

    df_opt = pd.DataFrame({
    "infill": pd.Series(dtype="int"),
    "cand": pd.Series(dtype="int"),
    "w_ADP": pd.Series(dtype="float"),
    "w_OP": pd.Series(dtype="float"),
    "x1": pd.Series(dtype="float"),
    "x2": pd.Series(dtype="float"),
    "x3": pd.Series(dtype="float"),
    "x4": pd.Series(dtype="float"),
    "x5": pd.Series(dtype="float"),
    "fName": pd.Series(dtype="str")
    })

    for idx in range(n_DOE):
        df_opt.loc[len(df_opt)] = [0, idx, hf_fit[idx, 0], hf_fit[idx, 1]
                                , candidates[idx,0], candidates[idx,1], candidates[idx,2], candidates[idx,3], candidates[idx,4],
                                os.path.join(mf_dir, "hf_doe/FFD/", f"ogv1c_g0_c{idx}.dat")]

    for idx in range(0,n_infill):
        df_opt.loc[len(df_opt)] = [idx+1, 0, hf_fit[idx+n_DOE, 0], hf_fit[idx+n_DOE, 1]
                                , candidates[idx+n_DOE,0], candidates[idx+n_DOE,1], candidates[idx+n_DOE,2], candidates[idx+n_DOE,3], candidates[idx+n_DOE,4]
                                , os.path.join(mf_dir, master_dir + f"_{idx}/high_infill_{idx}/FFD/ogv1c_g0_c0.dat")]
    print(df_opt)
    print(df_opt.loc[selected])
    # plot_chosen_profiles(df_opt.loc[selected], baseline)
    
    plot_mf_scatter(n_DOE, n_infill, hf_fit, unfeasible, bsl_w_ADP, bsl_w_OP, selected)


    # # initial DOE and baseline
    # ax.scatter(ar1_y_hf[:doe_size, 0], ar1_y_hf[:doe_size, 1], marker='s', color=initial_color, s=size, label="initial DOE")
    # ax.scatter(bsl_w_ADP, bsl_w_OP, marker="*", color="r", s=size, label="HF baseline")
    # # infills
    # sc = ax.scatter(ar1_y_hf[doe_size:, 0], ar1_y_hf[doe_size:, 1], marker='o', c=colors, cmap=cmap, s=size, label="HF infills")
    # # Axes settings
    # ax.set_axisbelow(True)
    # ax.grid(True, color="grey", linestyle="dashed")
    # ax.set_ylim(0.030, 0.075)
    # ax.set_xlim(0.025, 0.060)
    # # Create a dummy ScalarMappable for the colorbar
    # cbar = plt.colorbar(sc, ax=ax)
    # cbar.set_label('Infill number')
    # # legend
    # ax.legend(loc="lower right")
    # ax.set_xlabel('$w_{ADP}$ [-]')
    # ax.set_ylabel('$w_{OP}$ [-]')
    # plt.tight_layout()
    # # plt.savefig(os.path.join(os.getcwd(), "ar1_pareto_2.pdf"), bbox_inches="tight")

    # # Identify feasible candidates    
    # unfeas_cand = np.where(np.any(constraints >= 0, axis=1))[0]
    # unfeas_geom = np.where(np.any(constraints[:,:-3] >= 0, axis=1))[0]
    # unfeas_beta = np.where(np.any(constraints[:,-3:] >= 0, axis=1))[0]

    # feas_cand = list(set(nums) ^ set(unfeas_cand))
    # feas_geom = list(set(nums) ^ set(unfeas_geom))
    # feas_beta = list(set(nums) ^ set(unfeas_beta))

    # criteria = feas_cand.copy()

    # df_opt = df_opt.iloc[criteria].reset_index(drop=True)
    
    # # plot_pareto_with_constraints(lf_cand, lf_fit, criteria, constraints, [10.4,7.28])

    # lf_fit = lf_fit[criteria, :]
    # lf_cand = lf_cand[criteria, :]
    # lf_pareto = compute_pareto(lf_fit[:, 0], lf_fit[:, 1])
    
    # # Print Pareto front indices
    # lf_pareto_idx = [np.where(lf_fit == pid)[0][0] for pid in lf_pareto]
    
    # df_pareto = df_opt.iloc[lf_pareto_idx].reset_index(drop=True)

    # # Get best candidates
    # lf_ADP_cand_idx, lf_OP_cand_idx, lf_cand_idx = get_best_candidates_idx(lf_fit, lf_pareto, 1)
    # lf_cand_idx = [1192]
    
    # # Uncomment to save fitnesses
    # # np.savetxt("lf_fitnesses.txt", lf_opt)
    
    # fnames = ['']

    # plot_chosen_profiles(profiles, baseline)
    

    plt.show()


if __name__ == "__main__":
    main()
