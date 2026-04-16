# Converted from wall_resolution.ipynb
# --- code cells below ---


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

# import Louenas/Johnson post-processing class
sys.path.append(os.path.join(os.getcwd(), 'jscripts'))
from mod_postProcessMusicaa import postProcess

# plot params
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "Times"
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8
figsize = (5.2, 3.64)

input_dir = "/home/mciarlatani/Hilbert/beta-aero-optim/Optimization/musicaa_RANSrerun/output/MUSICAA/musicaa_g0_c0/ADP"
c_ax = 67

wall_blocks = [3, 4, 7, 6]

pp_OGV_RANS = postProcess(rep=input_dir, bloc=wall_blocks, bloc_fst=[1, 2, 3, 4, 5, 6, 7, 8, 9], is_2D=True, ngh=5, is_RANS=False, is_extended=True, is_stats=True, is_little_endian=True)
pp_OGV_RANS.init_postProcess()
## Calculate utau
filepath_wall_vector = os.path.join(input_dir, 'norm_surf.dat')
df = pd.read_csv(filepath_wall_vector, sep=r'\s+', header=None)

# Step 2: Extract x and y components of the vectors
# Define normal vectors
nx = df[2]
nx = np.array(nx)
ny = df[3]
ny = np.array(ny)
normal = np.array([nx, ny])

#Define tangent vectors
tx = -ny 
ty = nx 
tangent = np.array([tx, ty])

#Extract the wall values 
tau_w, utau, Cf, Re_tau, y_plus_x, x_plus_x, tu_mean, tu_x, tu_y, tu_z, y_delta, y_1, y_0, y_plus_x_real= pp_OGV_RANS.calc_bl(normal_w=normal, tangent_w=tangent, is_turb=True)

x_list = []
for i in wall_blocks:
    new_x_value = pp_OGV_RANS.x[i][:,0]
    x_list.append(new_x_value)
x_wall = np.concatenate(x_list)

fig, ax = plt.subplots(figsize=figsize)
ax.plot(x_wall / c_ax, y_plus_x, '-')
plt.ylabel('y+')
plt.xlabel('x-position')
# plt.legend()
plt.savefig(r'wall_resolution_fig1.png', dpi=150, bbox_inches='tight')
plt.show()

x0_list = []
x1_list = []
y0_list = []
y1_list = []

# these are given in info.ini
Roref = 0.219261536536
Muref = 1.81068915267e-05

for i in wall_blocks:
    
    new_x0_value = pp_OGV_RANS.x[i][:,0]
    x0_list.append(new_x0_value)
    
    new_x1_value = pp_OGV_RANS.x[i][:,1]
    x1_list.append(new_x1_value)
    
    new_y0_value = pp_OGV_RANS.y[i][:,0]
    y0_list.append(new_y0_value)
    
    new_y1_value = pp_OGV_RANS.y[i][:,1]
    y1_list.append(new_y1_value)

x0_concat = np.concatenate(x0_list)
x1_concat = np.concatenate(x1_list)
y0_concat = np.concatenate(y0_list)
y1_concat = np.concatenate(y1_list)

yplus = abs(utau) * np.sqrt((y1_concat - y0_concat)**2 + (x1_concat - x0_concat)**2) / 1000 * Roref / Muref 
yplus_deltay_only = abs(utau) * abs((y1_concat - y0_concat)) / 1000 * Roref / Muref
print('The mean y+ is', np.mean(yplus), np.mean(yplus_deltay_only))

le_idx = np.argmin(x0_concat)
te_idx = np.argmax(x0_concat)

# le_idx > te_idx
fig, ax = plt.subplots(figsize=figsize)
plt.plot(x0_concat[le_idx:], y0_concat[le_idx:], c="r", label="suction side")
plt.plot(x0_concat[:te_idx], y0_concat[:te_idx], c="r")
plt.plot(x0_concat[te_idx:le_idx], y0_concat[te_idx:le_idx], c="b", label="pressure side")
ax.set_ylabel("$y$ [mm]")
ax.set_xlabel("$x / c_{ax}$ [-]")
ax.legend()

fig, ax = plt.subplots(figsize=figsize)
ax.plot(-x0_concat[te_idx:le_idx] / c_ax, yplus[te_idx:le_idx], "-.", c="b", label="pressure side")
ax.plot(x0_concat[le_idx:] / c_ax, yplus[le_idx:], "--", c="r", label="suction side")
ax.plot(x0_concat[:te_idx] / c_ax, yplus[:te_idx], "--", c="r")
ax.set_ylabel("$y+$ [-]")
ax.set_xlabel("$x / c_{ax}$ [-]")
ax.legend()

fig, ax = plt.subplots(figsize=figsize)
ax.plot(-x0_concat[te_idx:le_idx] / c_ax, yplus_deltay_only[te_idx:le_idx], "-.", c="b", label="pressure side")
ax.plot(x0_concat[le_idx:] / c_ax, yplus_deltay_only[le_idx:], "--", c="r", label="suction side")
ax.plot(x0_concat[:te_idx] / c_ax, yplus_deltay_only[:te_idx], "--", c="r")
ax.set_ylabel(r"$\Delta y+$ [-]")
ax.set_xlabel("$x / c_{ax}$ [-]")
ax.legend()

Delta_z = 0.1
scale_z = 0.001

zplus_deltaz_only = abs(utau) * Delta_z * scale_z * Roref / Muref
print('The mean z+ is', np.mean(zplus_deltaz_only))

fig, ax = plt.subplots(figsize=figsize)
ax.plot(-x0_concat[te_idx:le_idx] / c_ax, zplus_deltaz_only[te_idx:le_idx], "-.", c="b", label="pressure side")
ax.plot(x0_concat[le_idx:] / c_ax, zplus_deltaz_only[le_idx:], "--", c="r", label="suction side")
ax.plot(x0_concat[:te_idx] / c_ax, zplus_deltaz_only[:te_idx], "--", c="r")
ax.set_ylabel(r"$\Delta z+$ [-]")
ax.set_xlabel("$x / c_{ax}$ [-]")
ax.legend()

delta_x0 = np.append(abs(x0_concat[1:] - x0_concat[:-1]), abs(x0_concat[-1] - x0_concat[1]))
xplus_deltax_only = abs(utau) * delta_x0 / 1000 * Roref / Muref
print('The mean x+ is', np.mean(xplus_deltax_only))

fig, ax = plt.subplots(figsize=figsize)
ax.plot(-x0_concat[te_idx:le_idx] / c_ax, xplus_deltax_only[te_idx:le_idx], "-.", c="b", label="pressure side")
ax.plot(x0_concat[le_idx:] / c_ax, xplus_deltax_only[le_idx:], "--", c="r", label="suction side")
ax.plot(x0_concat[:te_idx] / c_ax, xplus_deltax_only[:te_idx], "--", c="r")
ax.set_ylabel(r"$\Delta x+$ [-]")
ax.set_xlabel("$x / c_{ax}$ [-]")
ax.legend()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=figsize)

# x+
ax1.plot(-x0_concat[te_idx:le_idx] / c_ax, xplus_deltax_only[te_idx:le_idx], "-.", c="b", label="pressure side")
ax1.plot(x0_concat[le_idx:] / c_ax, xplus_deltax_only[le_idx:], "--", c="r", label="suction side")
ax1.plot(x0_concat[:te_idx] / c_ax, xplus_deltax_only[:te_idx], "--", c="r")
ax1.set_ylabel(r"$\Delta x+$ [-]")
# ax1.set_xlabel("$x / c_{ax}$ [-]")
# ax1.legend()

# y+
ax2.plot(-x0_concat[te_idx:le_idx] / c_ax, yplus_deltay_only[te_idx:le_idx], "-.", c="b", label="pressure side")
ax2.plot(x0_concat[le_idx:] / c_ax, yplus_deltay_only[le_idx:], "--", c="r", label="suction side")
ax2.plot(x0_concat[:te_idx] / c_ax, yplus_deltay_only[:te_idx], "--", c="r")
ax2.set_ylabel(r"$\Delta y+$ [-]")
# ax2.set_xlabel("$x / c_{ax}$ [-]")
ax2.legend()

# z+
ax3.plot(-x0_concat[te_idx:le_idx] / c_ax, zplus_deltaz_only[te_idx:le_idx], "-.", c="b", label="pressure side")
ax3.plot(x0_concat[le_idx:] / c_ax, zplus_deltaz_only[le_idx:], "--", c="r", label="suction side")
ax3.plot(x0_concat[:te_idx] / c_ax, zplus_deltaz_only[:te_idx], "--", c="r")
ax3.set_ylabel(r"$\Delta z+$ [-]")
ax3.set_xlabel("$x / c_{ax}$ [-]")
# ax3.legend()

# Optional: Add spacing
plt.tight_layout()

# Show the plot
plt.savefig(r'wall_resolution_fig2.png', dpi=150, bbox_inches='tight')
plt.show()
fig.savefig(os.path.join("wall_resolution.pdf"), bbox_inches='tight')

