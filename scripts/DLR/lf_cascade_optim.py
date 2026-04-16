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
    # plt.savefig("lf_pareto.pdf", bbox_inches="tight")
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
        ax.plot(pro[:, 0], pro[:, 1], linewidth=1, label=r"Design LF " + f"{cont}")
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
    plt.savefig(os.path.join(os.getcwd(), "lf_profile_opt.pdf"), bbox_inches="tight")

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

def main():
    """Main analysis and plotting routine"""
    plt.close('all')
    # Load mixed-out loss results
    lf_dir = "/Data1/Mattia/RANS_OPT_baseDelta"

    lf_cand = np.loadtxt(os.path.join(lf_dir, "candidates.txt"))
    lf_fit = np.loadtxt(os.path.join(lf_dir, "fitnesses.txt"))
    constraints = np.genfromtxt(os.path.join(lf_dir, "constraints.txt"))
    candidates = np.genfromtxt(os.path.join(lf_dir, "candidates.txt"))

    nums = list(range(0, constraints.shape[0]))

    df_opt = pd.DataFrame({
    "idx": pd.Series(dtype="int"),
    "gen": pd.Series(dtype="int"),
    "cand": pd.Series(dtype="int"),
    "w_ADP": pd.Series(dtype="float"),
    "w_OP": pd.Series(dtype="float"),
    "x1": pd.Series(dtype="float"),
    "x2": pd.Series(dtype="float"),
    "x3": pd.Series(dtype="float"),
    "x4": pd.Series(dtype="float"),
    "x5": pd.Series(dtype="float"),
    })

    for idx in nums:
        df_opt.loc[len(df_opt)] = [int(idx), idx // pop_size, idx % pop_size, lf_fit[idx, 0], lf_fit[idx, 1]
                                , candidates[idx,0], candidates[idx,1], candidates[idx,2], candidates[idx,3], candidates[idx,4]]
    df_opt['fName'] = df_opt.apply(lambda row: os.path.join(lf_dir, "FFD", f"ogv1c_g{int(row['gen'])}_c{int(row['cand'])}.dat"), axis=1)
    
    # Identify feasible candidates    
    unfeas_cand = np.where(np.any(constraints >= 0, axis=1))[0]
    unfeas_geom = np.where(np.any(constraints[:,:-3] >= 0, axis=1))[0]
    unfeas_beta = np.where(np.any(constraints[:,-3:] >= 0, axis=1))[0]

    feas_cand = list(set(nums) ^ set(unfeas_cand))
    feas_geom = list(set(nums) ^ set(unfeas_geom))
    feas_beta = list(set(nums) ^ set(unfeas_beta))

    criteria = feas_cand.copy()

    df_opt = df_opt.iloc[criteria].reset_index(drop=True)
    
    # plot_pareto_with_constraints(lf_cand, lf_fit, criteria, constraints, [10.4,7.28])

    lf_fit = lf_fit[criteria, :]
    lf_cand = lf_cand[criteria, :]
    lf_pareto = compute_pareto(lf_fit[:, 0], lf_fit[:, 1])
    
    # Print Pareto front indices
    lf_pareto_idx = [np.where(lf_fit == pid)[0][0] for pid in lf_pareto]
    
    df_pareto = df_opt.iloc[lf_pareto_idx].reset_index(drop=True)
    print("======================================================== LF Pareto candidates indices ========================================================")
    print(df_pareto[['w_ADP','w_OP','x1','x2','x3','x4','x5']])
    df_pareto[['x1','x2','x3','x4','x5']].to_csv('RANS_OPT_baseDelta.txt', sep=' ', header=False, index=False, float_format='%.12e')
    print(df_pareto)
    input()


    # Get best candidates
    lf_ADP_cand_idx, lf_OP_cand_idx, lf_cand_idx = get_best_candidates_idx(lf_fit, lf_pareto, 1)
    # lf_cand_idx = [1192]
    
    # Uncomment to save fitnesses
    # np.savetxt("lf_fitnesses.txt", lf_opt)
    
    df_chosen = df_opt.iloc[[lf_ADP_cand_idx] + lf_cand_idx + [lf_OP_cand_idx]]
    print("======================================================== Optimal candidate profiles ========================================================")
    print(df_chosen)
    df_chosen[["x1","x2","x3","x4","x5"]].to_csv("RANS_candidates.txt", sep=" ", header=False, index=False, float_format="%.12e")
    print("====================================================== Written in RANS_candidates.txt ======================================================")


    ## run_pod_analysis_DLR(df_chosen)
    # plot_pareto_single(lf_cand, lf_fit, constraints, df_chosen)
    plot_pareto_single_HF(lf_cand, lf_fit, constraints, df_chosen, [[0.02656, 0.0554, 0],[0.0287, 0.0512, 1],[0.0311, 0.0463, 1]])
    # plot_chosen_profiles(df_chosen, baseline)
    ## plot_scatter_modes(df_opt, df_chosen)
    
    # plot_pareto_single_HF(lf_cand, lf_fit, constraints, df_chosen, [[0.026417, 0.055268, 0],[0.028026, 0.051122, 1],[0.031148, 0.454955, 1]])

    # Plot 6: scatter pairplot
    data = lf_cand
    n_features = data.shape[1]
    n_samples = data.shape[0]

    # # Get optimal candidate profiles
    # lf_opt_cand = [np.loadtxt(os.path.join(df_opt.iloc[idx]['fName']), skiprows=2)*1e-3 for idx in [lf_ADP_cand_idx, lf_OP_cand_idx] + lf_cand_idx]
    
    # # Plot 5: Profile comparison
    # fig, ax = plt.subplots(figsize=(5.2, 3.64))
    # Delta = 0.0025
    
    # ax.plot(baseline[:, 0], baseline[:, 1], color="k", linestyle="dashed", label="baseline")
    # for num, pro in enumerate(lf_opt_cand):
    #     ax.plot(pro[:, 0], pro[:, 1], linewidth=1, label=f"Pareto candidate {num + 1}")
    # ax.set(xlabel='$x$ [m]', ylabel='$y$ [m]')
    # ax.legend()
    
    # # Leading Edge Inset
    # axins1 = ax.inset_axes([0.27, 0.05, 0.35, 0.35])
    # axins1.plot(baseline[:, 0], baseline[:, 1], color="k", linestyle="dashed")
    # for num, pro in enumerate(lf_opt_cand):
    #     axins1.plot(pro[:, 0], pro[:, 1], linewidth=1)
    # axins1.set_xlim(min(baseline[:, 0]) - 0.15 * Delta, min(baseline[:, 1]) + 2 * Delta)
    # axins1.set_ylim(min(baseline[:, 1]) - 0.25 * Delta, min(baseline[:, 1]) + Delta)
    # axins1.set_xticks([])
    # axins1.set_yticks([])
    # mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="0.5")
    
    # # Trailing Edge Inset
    # axins2 = ax.inset_axes([0.63, 0.3, 0.35, 0.35])
    # axins2.plot(baseline[:, 0], baseline[:, 1], color="k", linestyle="dashed")
    # for num, pro in enumerate(lf_opt_cand):
    #     axins2.plot(pro[:, 0], pro[:, 1], linewidth=1)
    # axins2.set_xlim(max(baseline[:, 0]) - 3 * Delta, max(baseline[:, 0]) + 0.4 * Delta)
    # axins2.set_ylim(max(baseline[:, 1]) - 1.25 * Delta, max(baseline[:, 1]) + 0.3 * Delta)
    # axins2.set_xticks([])
    # axins2.set_yticks([])
    # mark_inset(ax, axins2, loc1=2, loc2=4, fc="none", ec="0.5")
    
    # plt.tight_layout()
    # # plt.savefig(os.path.join(os.getcwd(), "lf_profile_opt.pdf"), bbox_inches="tight")
    # # plt.show()
    
    
    # # Plot 1: Beta Pareto plot
    # fig, ax = plt.subplots(figsize=(5.2, 3.64))
    # ax.scatter(lf_fit[:, 0], lf_fit[:, 1], marker="o", s=10, color="gray", edgecolors="none", label="LF candidates", zorder=-1)
    # ax.scatter(lf_pareto[:, 0], lf_pareto[:, 1], marker="o", s=10, color="k", label="LF Pareto")
    # ax.scatter(lf_bsl_w_ADP, lf_bsl_w_OP, marker="*", s=10, color="k", label="LF baseline")
    # ax.plot()
    # ax.set_axisbelow(True)
    # plt.grid(True, color="grey", linestyle="dashed")
    # ax.set_ylim(0.030, 0.275)
    # ax.set_xlim(0.020, 0.220)
    # ax.legend(loc="lower right")
    # ax.set_xlabel('$w_{ADP}$ [-]')
    # ax.set_ylabel('$w_{OP}$ [-]')
    # plt.tight_layout()
    # # plt.savefig(os.path.join(os.getcwd(), "lf_pareto_opt_1.pdf"), bbox_inches="tight")
    # plt.show()
    
    # # Plot 2: Pareto with HF baseline
    # fig, ax = plt.subplots(figsize=(5.2, 3.64))
    # ax.scatter(lf_fit[:, 0], lf_fit[:, 1], marker="o", s=10, color="gray", edgecolors="none", label="LF candidates", zorder=-1)
    # ax.scatter(lf_pareto[:, 0], lf_pareto[:, 1], marker="o", s=10, color="k", label="LF Pareto")
    # ax.scatter(lf_bsl_w_ADP, lf_bsl_w_OP, marker="*", s=10, color="k", label="LF baseline")
    # ax.scatter(bsl_w_ADP, bsl_w_OP, marker="*", s=10, color="r", label="HF baseline")
    # ax.annotate('', xy=(bsl_w_ADP, bsl_w_OP), xytext=(lf_bsl_w_ADP, lf_bsl_w_OP), arrowprops=dict(arrowstyle='-', linestyle="--", linewidth=0.5, color='black'))
    # ax.plot()
    # ax.set_axisbelow(True)
    # plt.grid(True, color="grey", linestyle="dashed")
    # ax.set_ylim(0.03, 0.275)
    # ax.set_xlim(0.020, 0.220)
    # ax.legend(loc="lower right")
    # ax.set_xlabel('$w_{ADP}$ [-]')
    # ax.set_ylabel('$w_{OP}$ [-]')
    # plt.tight_layout()
    # # plt.savefig(os.path.join(os.getcwd(), "lf_pareto_opt_2.pdf"), bbox_inches="tight")
    # plt.show()
    
    # # Plot 3: Pareto with recomputed values
    # X_recomp = np.loadtxt("../examples/LRN-CASCADE/cascade_musicaa_base/output_lf_opt/fitnesses.txt")
    
    # fig, ax = plt.subplots(figsize=(5.2, 3.64))
    # ax.scatter(lf_fit[:, 0], lf_fit[:, 1], marker="o", s=10, color="gray", edgecolors="none", label="LF candidates", zorder=-1)
    # ax.scatter(lf_pareto[:, 0], lf_pareto[:, 1], marker="o", s=10, color="k", label="LF Pareto")
    # ax.scatter(lf_bsl_w_ADP, lf_bsl_w_OP, marker="*", s=20, color="k", label="LF baseline")
    # ax.scatter(bsl_w_ADP, bsl_w_OP, marker="*", s=20, color="r", label="HF baseline")
    # plt.scatter(X_recomp[:, 0], X_recomp[:, 1], marker='o', c="r", s=10, label="HF recomputed")
    # for ii in range(len(X_recomp)):
    #     ax.annotate('', xy=(lf_opt[ii, 0], lf_opt[ii, 1]), xytext=(X_recomp[ii, 0], X_recomp[ii, 1]), arrowprops=dict(arrowstyle='-', linestyle="--", linewidth=0.5, color='black'))
    # plt.text(X_recomp[0, 0] - 0.003, X_recomp[0, 1] - 0.006, 1, fontsize=7)
    # plt.text(X_recomp[1, 0], X_recomp[1, 1] - 0.009, 2, fontsize=7)
    # plt.text(X_recomp[2, 0] -0.0023, X_recomp[2, 1] - 0.008, 3, fontsize=7)
    # plt.text(X_recomp[3, 0] -0.0015, X_recomp[3, 1] - 0.008, 4, fontsize=7)
    
    # # Zoomed inset
    # axins1 = ax.inset_axes([0.60, 0.60, 0.35, 0.35])
    # axins1.scatter(bsl_w_ADP, bsl_w_OP, marker="*", s=20, color="r", label="HF baseline")
    # axins1.scatter(X_recomp[:, 0], X_recomp[:, 1], marker='o', c="r", s=10, label="HF recomputed")
    # axins1.text(X_recomp[0, 0] - 0.001, X_recomp[0, 1] - 0.002, 1, fontsize=7)
    # axins1.text(X_recomp[1, 0], X_recomp[1, 1] - 0.003, 2, fontsize=7)
    # axins1.text(X_recomp[2, 0] -0.0006, X_recomp[2, 1] - 0.0023, 3, fontsize=7)
    # axins1.text(X_recomp[3, 0] -0.0005, X_recomp[3, 1] - 0.0023, 4, fontsize=7)
    # axins1.set_xlim(0.025, 0.040)
    # axins1.set_ylim(0.035, 0.055)
    # mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="k")
    
    # ax.set_axisbelow(True)
    # plt.grid(True, color="grey", alpha=0.5, linestyle="dashed")
    # ax.set_ylim(0.03, 0.275)
    # ax.set_xlim(0.020, 0.220)
    # ax.legend(loc="lower right")
    # ax.set_xlabel('$w_{ADP}$ [-]')
    # ax.set_ylabel('$w_{OP}$ [-]')
    # plt.tight_layout()
    # # plt.savefig(os.path.join(os.getcwd(), "lf_pareto_opt.pdf"), bbox_inches="tight")
    # plt.show()
    
    # # Plot 4: Effect of mesh fix
    # lf_opt_fixed = np.array([[0.06024, 0.5 * (0.1654 + 0.1419)],
    #                          [0.1369, 0.5 * (0.1276 + 0.1019)],
    #                          [0.1077, 0.5 * (0.1541 + 0.1544)],
    #                          [0.1242, 0.5 * (0.1244 + 0.1010)]])
    
    # fig, ax = plt.subplots(figsize=(5.2, 3.64))
    # ax.scatter(lf_fit[:, 0], lf_fit[:, 1], marker="o", s=10, color="gray", edgecolors="none", label="LF candidates", zorder=-1)
    # ax.scatter(lf_pareto[:, 0], lf_pareto[:, 1], marker="o", s=10, color="k", label="LF Pareto")
    # ax.scatter(lf_opt_fixed[:, 0], lf_opt_fixed[:, 1], marker="o", s=10, color="g", label="LF Pareto (fixed mesh)")
    # ax.scatter(lf_bsl_w_ADP, lf_bsl_w_OP, marker="*", s=20, color="k", label="LF baseline")
    # ax.scatter(bsl_w_ADP, bsl_w_OP, marker="*", s=20, color="r", label="HF baseline")
    # plt.scatter(X_recomp[:, 0], X_recomp[:, 1], marker='o', c="r", s=10, label="HF recomputed")
    # for ii in range(len(X_recomp)):
    #     ax.annotate('', xy=(lf_opt[ii, 0], lf_opt[ii, 1]), xytext=(X_recomp[ii, 0], X_recomp[ii, 1]), arrowprops=dict(arrowstyle='-', linestyle="--", linewidth=0.5, color='black'))
    #     ax.annotate('', xy=(lf_opt_fixed[ii, 0], lf_opt_fixed[ii, 1]), xytext=(X_recomp[ii, 0], X_recomp[ii, 1]), arrowprops=dict(arrowstyle='-', linestyle="--", linewidth=0.5, color='g'))
    # plt.text(X_recomp[0, 0] - 0.003, X_recomp[0, 1] - 0.006, 1, fontsize=7)
    # plt.text(X_recomp[1, 0], X_recomp[1, 1] - 0.009, 2, fontsize=7)
    # plt.text(X_recomp[2, 0] -0.0023, X_recomp[2, 1] - 0.008, 3, fontsize=7)
    # plt.text(X_recomp[3, 0] -0.0015, X_recomp[3, 1] - 0.008, 4, fontsize=7)
    
    # # Zoomed inset
    # axins1 = ax.inset_axes([0.60, 0.60, 0.35, 0.35])
    # axins1.scatter(bsl_w_ADP, bsl_w_OP, marker="*", s=20, color="r", label="HF baseline")
    # axins1.scatter(X_recomp[:, 0], X_recomp[:, 1], marker='o', c="r", s=10, label="HF recomputed")
    # axins1.text(X_recomp[0, 0] - 0.001, X_recomp[0, 1] - 0.002, 1, fontsize=7)
    # axins1.text(X_recomp[1, 0], X_recomp[1, 1] - 0.003, 2, fontsize=7)
    # axins1.text(X_recomp[2, 0] -0.0006, X_recomp[2, 1] - 0.0023, 3, fontsize=7)
    # axins1.text(X_recomp[3, 0] -0.0005, X_recomp[3, 1] - 0.0023, 4, fontsize=7)
    # axins1.set_xlim(0.025, 0.040)
    # axins1.set_ylim(0.035, 0.055)
    # mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="k")
    
    # ax.set_axisbelow(True)
    # plt.grid(True, color="grey", alpha=0.5, linestyle="dashed")
    # ax.set_ylim(0.03, 0.275)
    # ax.set_xlim(0.020, 0.220)
    # ax.legend(loc="lower right")
    # ax.set_xlabel('$w_{ADP}$ [-]')
    # ax.set_ylabel('$w_{OP}$ [-]')
    # plt.tight_layout()
    # # plt.savefig(os.path.join(os.getcwd(), "lf_fixed_pareto_opt.pdf"), bbox_inches="tight")
    # plt.show()
    
    # # Calculate percentage improvement
    # improvement = (np.array([bsl_w_ADP, bsl_w_OP]) - X_recomp) / np.array([bsl_w_ADP, bsl_w_OP]) * 100
    # print("\nPercentage improvement over baseline:")
    # print(improvement)
    
    # # Load NSGA-II optimal HF candidate and plot as LF
    # nsga_opt_cand = np.loadtxt("../examples/LRN-CASCADE/cascade_musicaa_base/output_nsga_opt/candidates.txt")
    # nsga_opt_fit = np.array([0.07463, 0.5 * (0.1787 + 0.07468)])
    
    # # Plot 7: Design space exploration
    # responses = lf_fit
    # optimal_candidates = [lf_cand[idx] for idx in [lf_ADP_cand_idx, lf_OP_cand_idx] + lf_cand_idx]
    
    # row_indices = np.arange(len(lf_fit))
    # norm = plt.Normalize(row_indices.min() // pop_size, row_indices.max() // pop_size)
    # cmap = cm.get_cmap('viridis')
    # colors = cmap(norm(row_indices // pop_size))
    
    # fig, axes = plt.subplots(2, n_features, figsize=(8.5, 2.8), constrained_layout=True)
    
    # for i in range(n_features):
    #     ax = axes[0, i]
    #     ax.set_yscale('log')
    #     if i == 0:
    #         ax.set_ylabel('$w_{ADP}$ [-]')
    #     ax.scatter(data[:, i], responses[:, 0], c=colors, s=15)
    #     for idx in lf_pareto_idx:
    #         ax.scatter(lf_cand[idx, i], lf_fit[idx, 0], color="k", s=15)
    #     for cand, fit in zip(optimal_candidates, lf_opt):
    #         ax.scatter(cand[i], fit[0], facecolors="none", edgecolors="r", s=15)
    #     ax.scatter(nsga_opt_cand[i], nsga_opt_fit[0], marker="D", color="gray", s=15, zorder=100)
    #     ax.set_xlabel(labels[i])
    #     if i != 0:
    #         ax.yaxis.minorticks_off()
    #         ax.set_yticklabels([])
    #         ax.tick_params(left=False)
    #         ax.yaxis.set_major_locator(plt.NullLocator())
    
    # for i in range(n_features):
    #     ax = axes[1, i]
    #     ax.set_yscale('log')
    #     if i == 0:
    #         ax.set_ylabel('$w_{OP}$ [-]')
    #     ax.scatter(data[:, i], responses[:, 1], c=colors, s=15)
    #     for idx in lf_pareto_idx:
    #         ax.scatter(lf_cand[idx, i], lf_fit[idx, 1], color="k", s=15)
    #     for cand, fit in zip(optimal_candidates, lf_opt):
    #         ax.scatter(cand[i], fit[1], facecolors="none", edgecolors="r", s=15)
    #     ax.scatter(nsga_opt_cand[i], nsga_opt_fit[0], marker="D", color="gray", s=15, zorder=100)
    #     ax.set_xlabel(labels[i])
    #     if i != 0:
    #         ax.yaxis.minorticks_off()
    #         ax.set_yticklabels([])
    #         ax.tick_params(left=False)
    #         ax.yaxis.set_major_locator(plt.NullLocator())
    
    # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.05)
    # cbar.set_label('Generation number')
    # # plt.savefig(os.path.join(os.getcwd(), "lf_space.pdf"), bbox_inches="tight")
    # plt.show()
    
    # # Display recomputed values
    # # print("\nRecomputed values (X_recomp):")
    # # print(X_recomp)

    plt.show()


if __name__ == "__main__":
    main()
