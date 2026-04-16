import matplotlib.pyplot as plt
import numpy as np
import os
import re
import scipy.interpolate as si

from pyevtk.hl import gridToVTK
from typing import Any


def get_block_info(dat_dir: str) -> dict:
    """
    Returns a dictionnary containing relevant information on the blocks
    used by MUSICAA: number of blocks, block size, snapshots.
    """
    block_info: dict[str, Any] = {}

    # get number of blocks
    filename = os.path.join(dat_dir, "param_blocks.ini")
    nbl_ = read_next_line_in_file(filename, "Number of Blocks")
    nbl = int(re.findall(r"\d+", nbl_)[0])
    block_info["nbl"] = nbl

    # iterate for each block
    total_nb_points = 0
    total_nb_lines = 0
    total_nb_planes = 0
    with open(filename, "r") as f:
        filedata = f.readlines()
    for bl in range(nbl):
        bl += 1
        pattern = f"! Block #{bl}"
        block_info[f"block_{bl}"] = {}
        for i, line in enumerate(filedata):
            if pattern in line:
                block_info[f"block_{bl}"]["nx"] = int(re.findall(r"\d+", filedata[i + 3])[0])
                block_info[f"block_{bl}"]["ny"] = int(re.findall(r"\d+", filedata[i + 4])[0])
                block_info[f"block_{bl}"]["nz"] = int(re.findall(r"\d+", filedata[i + 5])[0])
                # get eventual sensors if any
                nb_snaps = int(re.findall(r"\d+", filedata[i + 16])[0])
                block_info[f"block_{bl}"]["nb_snaps"] = nb_snaps
                block_info[f"block_{bl}"]["nb_points"] = 0
                block_info[f"block_{bl}"]["nb_lines"] = 0
                block_info[f"block_{bl}"]["nb_planes"] = 0
                point_nb = 0
                line_nb = 0
                plane_nb = 0
                if nb_snaps > 0:
                    for snap in range(nb_snaps):
                        snap += 1
                        info_snap = [
                            int(dim) for dim in re.findall(r"\d+", filedata[i + 20 + snap])]
                        nx1, nx2 = info_snap[0], info_snap[1]
                        ny1, ny2 = info_snap[2], info_snap[3]
                        nz1, nz2 = info_snap[4], info_snap[5]
                        freq = info_snap[6]
                        nvars = info_snap[7]
                        # line
                        if (
                            (nx1 == nx2 and ny1 == ny2 and nz1 != nz2)
                            or (nx1 == nx2 and nz1 == nz2 and ny1 != ny2)
                            or (ny1 == ny2 and nz1 == nz2 and nx1 != nx2)
                        ):
                            snap_type = "line"
                            param_freq = 1
                            line_nb += 1
                            total_nb_lines += 1
                            snap_id = line_nb
                        # point
                        elif (nx1 == nx2 and ny1 == ny2 and nz1 == nz2):
                            snap_type = "point"
                            param_freq = 0
                            point_nb += 1
                            total_nb_points += 1
                            snap_id = point_nb
                        # plane
                        else:
                            snap_type = "plane"
                            snap = plane_nb + 1
                            param_freq = 2
                            plane_nb += 1
                            total_nb_planes += 1
                            snap_id = plane_nb
                        if freq == 0:
                            # frequency is prescribed in param.ini
                            freq = int(read_next_line_in_file(
                                os.path.join(dat_dir, "param.ini"),
                                "Snapshot frequencies").split()[param_freq])
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"] = {}
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"]["nx1"] = nx1
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"]["nx2"] = nx2
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"]["ny1"] = ny1
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"]["ny2"] = ny2
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"]["nz1"] = nz1
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"]["nz2"] = nz2
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"]["freq"] = freq
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"]["nvars"] = nvars
                        var_list = filedata[i + 20 + snap].split()[-nvars:]
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"]["var_list"] = var_list
                        position = point_nb + line_nb + plane_nb
                        block_info[f"block_{bl}"][f"{snap_type}_{snap_id}"]["position"] = position
                    block_info[f"block_{bl}"]["nb_points"] = point_nb
                    block_info[f"block_{bl}"]["nb_lines"] = line_nb
                    block_info[f"block_{bl}"]["nb_planes"] = plane_nb
    block_info["total_nb_points"] = total_nb_points
    block_info["total_nb_lines"] = total_nb_lines
    block_info["total_nb_planes"] = total_nb_planes
    return block_info


def get_min_max(var: str, data: dict, nblock: int, nckpt: int) -> tuple[float, float]:
    """
    Computes the global min/max values of the specified variable across all blocks and checkpoints.
    """
    vmins = []
    vmaxs = []
    for bl in range(1, nblock + 1):
        for ckpt in range(1, nckpt + 1):
            vmins.append(data[bl]["planes"][ckpt][var].min())
            vmaxs.append(data[bl]["planes"][ckpt][var].max())

    vmin = min(vmins)
    vmax = max(vmaxs)
    return vmin, vmax


def line_interp(data: dict, var: str, lims: list[float], bl_list: list[int]) -> np.ndarray:
    """
    Interpolates data along a line defined by lims.
    """
    # flatten data
    var_flat_ = []
    x_flat_ = []
    y_flat_ = []
    for bl in bl_list:
        var_flat_.append(data[bl][f"{var}"].flatten())
        x_flat_.append(data[bl]["x"].flatten())
        y_flat_.append(data[bl]["y"].flatten())
    var_flat = np.hstack(var_flat_)
    x_flat = np.hstack(x_flat_)
    y_flat = np.hstack(y_flat_)

    # create line
    x1, y1 = lims[0], lims[1]
    x2, y2 = lims[2], lims[3]
    x_interp = np.linspace(x1, x2, 1000)
    y_interp = np.linspace(y1, y2, 1000)

    # interpolate
    var_interp = si.griddata((x_flat, y_flat), var_flat,
                             (x_interp, y_interp), method="linear")

    return var_interp


def mixed_out(data: dict) -> dict:
    """
    Computes mixed-out quantities:
    see A. Prasad (2004): https://doi.org/10.1115/1.1928289
    """
    # conservation of mass
    m_bar = np.nanmean(data["rhou_interp"])
    v_bar = np.nanmean(data["rho*uv_interp"]) / m_bar
    w_bar = np.nanmean(data["rho*uw_interp"]) / m_bar
    vv_bar = v_bar**2
    ww_bar = w_bar**2

    # conservation of momentum
    x_mom = np.nanmean(data["rho*uu_interp"] + data["p_interp"])
    y_mom = np.nanmean(data["rho*uv_interp"])
    z_mom = np.nanmean(data["rho*uw_interp"])

    # conservation of energy
    gam = data["gam"]
    R = data["R"]
    e = data["R"] * data["gam"] / (data["gam"] - 1) *\
        np.nanmean(data["rhou_interp"] * data["T_interp"]) +\
        0.5 * np.nanmean(data["rhou_interp"] * (data["uu_interp"]
                                                + data["vv_interp"]
                                                + data["ww_interp"]))

    # quadratic equation
    Q = 1 / m_bar**2 * (1 - 2 * gam / (gam - 1))
    L = 2 / m_bar**2 * (gam / (gam - 1) * x_mom - x_mom)
    C = 1 / m_bar**2 * (x_mom**2 + y_mom**2 + z_mom**2) - 2 * e / m_bar

    # select subsonic root
    p_bar = (-L - np.sqrt(L**2 - 4 * Q * C)) / 2 / Q
    u_bar = (x_mom - p_bar) / m_bar
    V2_bar = u_bar**2 + vv_bar + ww_bar
    rho_bar = m_bar / u_bar
    T_bar = p_bar / rho_bar / R
    c_bar = np.sqrt(gam * R * T_bar)
    M_bar = np.sqrt(V2_bar) / c_bar
    p0_bar = p_bar * (1 + (gam - 1) / 2 * M_bar**2)**(gam / (gam - 1))

    # store
    mixed_out_state = {"p_bar": p_bar,
                       "rho_bar": rho_bar,
                       "T_bar": T_bar,
                       "V2_bar": V2_bar,
                       "M_bar": M_bar,
                       "p0_bar": p0_bar}

    return mixed_out_state


def read_grid(
        dir_data: str, file_input: str
) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    print("Reading grid...")
    bl = file_input.split('bl')[-1][0]
    dict_info = read_info(dir_data)
    nx = dict_info[f'nx_bl{bl}']
    ny = dict_info[f'ny_bl{bl}']
    nz = dict_info[f'nz_bl{bl}']
    ngh = dict_info['ngh']
    ngh = 5
    nx_ngh = nx + 2 * ngh
    ny_ngh = ny + 2 * ngh

    f = open(file_input, 'r')
    x = np.fromfile(f, dtype=('<f8'), count=nx_ngh * ny_ngh).reshape((nx_ngh, ny_ngh, 1), order='F')
    y = np.fromfile(f, dtype=('<f8'), count=nx_ngh * ny_ngh).reshape((nx_ngh, ny_ngh, 1), order='F')
    z = np.zeros(1)
    x = x[ngh:-ngh, ngh:-ngh, 0]
    y = y[ngh:-ngh, ngh:-ngh, 0]
    return nx, ny, nz, x, y, z


def read_info(dir_data: str) -> dict:
    dict_info: dict[Any, Any] = {}
    lines = open(os.path.join(dir_data, 'info.ini'), 'r').readlines()
    dict_info["nbloc"] = int(lines[0].split()[4])
    dict_info["is_curv"] = lines[0].split()[5]
    # block info
    for ind in range(dict_info["nbloc"]):
        dict_info["nx_bl" + str(ind + 1)] = int(lines[1 + ind].split()[5])
        dict_info["ny_bl" + str(ind + 1)] = int(lines[1 + ind].split()[6])
        dict_info["nz_bl" + str(ind + 1)] = int(lines[1 + ind].split()[7])
    # sim info
    pattern = r"([A-Za-z0-9 ]+)\s*=\s*(.*)"
    for line in lines[ind + 2:]:
        match = re.match(pattern, line.strip())
        if match:
            keywords_list = match.group(1).strip().split()
            values = match.group(2).split()
            for val_idx in range(len(values)):
                dict_info[keywords_list[val_idx]] = float(values[val_idx])
    return dict_info


def read_next_line_in_file(fname: str, pattern: str) -> str:
    """
    Returns the next line of fname containing pattern.
    """
    filedata = open(fname, 'r').readlines()
    index = [idx for idx, s in enumerate(filedata) if pattern in s][0] + 1
    if index:
        return filedata[index].strip()
    else:
        raise Exception(f"{pattern} not found in {fname}")


def read_planes(file_input: str, dict_info: dict, var_names: list[str]) -> tuple[dict, int]:
    """
    Reads the plane data of the block contained in file_input.
    Returns the data dictionary: {checkpoint_id: {variable: data array of shape (nx, ny)}}

    Note: the data shape is retrieved from dict_info and the list of variables var_names.
    """
    print("Reading plane...")
    bl = file_input.split('_')[-1].split('.')[0]
    nx, ny, _ = dict_info[f'nx_{bl}'], dict_info[f'ny_{bl}'], dict_info[f'nz_{bl}']
    mu_ref = dict_info['Muref']

    # Find nb of planes written in file
    f = open(file_input, 'rb')
    f8_dtype = np.dtype('f8')
    data = np.fromfile(f, dtype=f8_dtype, count=-1)
    planes: dict[Any, Any] = {}

    nb_ckpt = data.size // (nx * ny) // len(var_names)
    for nvar, var in enumerate(var_names):
        for i in range(nb_ckpt):
            i += 1
            planes[i] = {}
            ckpt_id = len(var_names) * (i - 1)
            for nvar, var in enumerate(var_names):
                planes[i][var] = data[
                    nx * ny * (nvar + ckpt_id): nx * ny * (nvar + 1 + ckpt_id)
                ].reshape((nx, ny), order='F')
                if var == 'mut':
                    planes[i][var] = planes[i][var] / mu_ref
                planes[i][f'vmax_{var}'] = planes[i][var].max()
                planes[i][f'vmin_{var}'] = planes[i][var].min()

    return planes, nb_ckpt


def read_restart(
        file_input: str, nx: int, ny: int, nz: int, is_2D: bool, is_RANS: bool
) -> tuple[np.ndarray, ...]:
    print("Reading restart...")
    f = open(file_input, 'r')
    ro = np.fromfile(f, dtype=('<f8'), count=nx * ny * nz).reshape((nx, ny, nz), order='F')
    rou = np.fromfile(f, dtype=('<f8'), count=nx * ny * nz).reshape((nx, ny, nz), order='F')
    rov = np.fromfile(f, dtype=('<f8'), count=nx * ny * nz).reshape((nx, ny, nz), order='F')
    row = np.fromfile(f, dtype=('<f8'), count=nx * ny * nz).reshape((nx, ny, nz), order='F')
    roe = np.fromfile(f, dtype=('<f8'), count=nx * ny * nz).reshape((nx, ny, nz), order='F')
    if is_RANS:
        nutil = np.fromfile(f, dtype=('<f8'), count=nx * ny * nz).reshape((nx, ny, nz), order='F')
    f.close()

    if is_2D:
        ro_, rou_, rov_, row_, roe_ = ro, rou, rov, row, roe
        ro, rou, rov, row, roe = (
            np.ndarray((nx, ny)), np.ndarray((nx, ny)),
            np.ndarray((nx, ny)), np.ndarray((nx, ny)),
            np.ndarray((nx, ny))
        )
        for j in range(ny):
            for i in range(nx):
                ro[i][j] = ro_[i][j][0]
                rou[i][j] = rou_[i][j][0]
                rov[i][j] = rov_[i][j][0]
                row[i][j] = row_[i][j][0]
                roe[i][j] = roe_[i][j][0]

        if is_RANS:
            nutil_ = nutil
            nutil = np.ndarray((nx, ny))
            for j in range(ny):
                for i in range(nx):
                    nutil[i][j] = nutil_[i][j][0]
    if is_RANS:
        return ro, rou, rov, row, roe, nutil
    else:
        return ro, rou, rov, row, roe


def read_stats(file_input: str, nx: int, ny: int) -> dict:
    print("Reading stats...")
    stats = {}
    if file_input[-14:].split('_')[0][-1] == '1':
        var_list = ['rho', 'u', 'v', 'w', 'p', 'T', 'rhou', 'rhov', 'rhow', 'rhoe',
                    'rho**2', 'uu', 'vv', 'ww', 'uv', 'uw', 'vw', 'vT', 'p**2', 'T**2',
                    'mu', 'divloc', 'divloc**2']

    else:
        var_list = ['e', 'h', 'c', 's', 'M', '0.5*q', 'g', 'la', 'cp', 'cv',
                    'prr', 'eck', 'rho*dux', 'rho*duy', 'rho*duz', 'rho*dvx', 'rho*dvy',
                    'rho*dvz', 'rho*dwx', 'rho*dwy', 'rho*dwz', 'p*div', 'rho*div', 'b1',
                    'b2', 'b3', 'rhoT', 'uT', 'vT', 'e**2', 'h**2', 'c**2', 's**2',
                    'qq/cc2', 'g**2', 'mu**2', 'la**2', 'cv**2', 'cp**2', 'prr**2', 'eck**2',
                    'p*u', 'p*v', 's*u', 's*v', 'p*rho', 'h*rho', 'T*p', 'p*s', 'T*s', 'rho*s',
                    'g*rho', 'g*p', 'g*s', 'g*T', 'g*u', 'g*v', 'p*dux', 'p*dvy', 'p*dwz',
                    'p*duy', 'p*dvx', 'rho*div**2', 'dux**2', 'duy**2', 'duz**2', 'dvx**2',
                    'dvy**2', 'dvz**2', 'dwx**2', 'dwy**2', 'dwz**2', 'b1**2', 'b2**2', 'b3**2',
                    'rho*b1', 'rho*b2', 'rho*b3', 'rho*uu', 'rho*vv', 'rho*ww',
                    'rho*T**2', 'rho*b1**2', 'rho*b2**2', 'rho*b3**2', 'rho*uv', 'rho*uw',
                    'rho*vw', 'rho*vT', 'rho*u**2*v', 'rho*v**3', 'rho*w**2*v', 'rho*v**2*u',
                    'rho*dux**2', 'rho*dvy**2', 'rho*dwz**2', 'rho*duy*dvx', 'rho*duz*dwx',
                    'rho*dvz*dwy', 'u**3', 'p**3', 'u**4', 'p**4', 'Frhou', 'Frhov', 'Frhow',
                    'Grhov', 'Grhow', 'Hrhow', 'Frhovu', 'Frhouu', 'Frhovv', 'Frhoww',
                    'Grhovu', 'Grhovv', 'Grhoww', 'Frhou_dux', 'Frhou_dvx', 'Frhov_dux',
                    'Frhov_duy', 'Frhov_dvx', 'Frhov_dvy', 'Frhow_duz', 'Frhow_dvz',
                    'Frhow_dwx', 'Grhov_duy', 'Grhov_dvy', 'Grhow_duz', 'Grhow_dvz',
                    'Grhow_dwy', 'Hrhow_dwz', 'la*dTx', 'la*dTy', 'la*dTz',
                    'h*u', 'h*v', 'h*w', 'rho*h*u', 'rho*h*v', 'rho*h*w', 'rho*h*u', 'rho*s*v',
                    'rho*s*w', 'rho*u**3', 'rho*v**3', 'rho*w**3', 'rho*w**2*u', 'h0', 'e0', 's0',
                    'T0', 'p0', 'rh0']

    f = open(file_input, 'rb')
    dtype = np.dtype('f8')
    for var in var_list:
        stats[var] = np.fromfile(f, dtype=dtype, count=nx * ny).reshape((nx, ny), order='F')
    f.close()

    return stats


# Quadrilateral elements
def create_elm(nb: int, line: int) -> list[list[int]]:
    quad = []
    for i in range(nb):
        bg = i          # bas-gauche
        bd = bg + 1     # bas-droite
        hd = bd + line  # haut-droite
        hg = bg + line  # haut-gauche

        if bd % line != 0:
            quad.append([bg, bd, hd, hg])  # liste des coins des elements
    return quad


# Plotting function for mesh
def plot_elm(nod_x: np.ndarray, nod_y: np.ndarray, qd: list, edgecolor: str):
    for cell in qd:
        vertices_x = [nod_x[cell[i]] for i in range(len(cell))]
        vertices_y = [nod_y[cell[i]] for i in range(len(cell))]
        plt.fill(vertices_x, vertices_y, edgecolor=edgecolor, linewidth=0.3, fill=False)


# Block to be plotted
def plot_grid(
        dir_data: str, is_curv: bool,
        bl_list: list, every: int = 5,
        ngh: int = 5, figsize: tuple[float, float] = (12, 10),
        xlabel: str = "$x$ [mm]", ylabel: str = "$y$ [mm]",
        # Zoom box parameters (easily modifiable)
        box1_xlim: tuple[float, float] = (-3, 5),
        box1_ylim: tuple[float, float] = (-0.7, 3.2),
        box2_xlim: tuple[float, float] = (63, 68),
        box2_ylim: tuple[float, float] = (18.5, 21.5)
):
    """
    Plots mesh with 3 subplots:
    - Top: full view with zoom boxes highlighted
    - Bottom left: zoom on box1
    - Bottom right: zoom on box2
    """
    # General information
    dict_info = read_info(dir_data)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])  # Top subplot (full width)
    ax2 = fig.add_subplot(gs[1, 0])  # Bottom left
    ax3 = fig.add_subplot(gs[1, 1])  # Bottom right
    
    axes = [ax1, ax2, ax3]
    
    # Plot on all three subplots
    for ax_idx, ax in enumerate(axes):
        plt.sca(ax)  # Set current axis
        
        for bl_num in bl_list:
            bl_file = os.path.join(dir_data, f'grid_bl{bl_num}_ngh{ngh}.bin')
            nx, ny, nz, x, y, z = read_grid(dir_data, bl_file)
            x, y, z = x[::every], y[::every], z
            node_x, node_y = x.flat[::every], y.flat[::every]

            nb_cells = (nx // every - 1) * ny // every
            if is_curv:
                quad = create_elm(nb_cells, ny // every)
                if bl_num == 6:
                    plot_elm(node_x, node_y-40.4, quad, 'tab:blue')
                else:
                    plot_elm(node_x, node_y, quad, 'tab:blue')
            else:
                ax.vlines(x[0], ymin=y[0, 0], ymax=y[-1, 0], colors='tab:blue', linewidth=1)
                ax.hlines(y[:, 0], xmin=x[0, 0], xmax=x[0, -1], colors='tab:blue', linewidth=1)
            
            # Add block number label in green at the center of the block
            # x_center = (x.min() + x.max()) / 2
            # y_center = (y.min() + y.max()) / 2
            # ax.text(x_center, y_center, str(bl_num), color='green', fontsize=10, 
            #        ha='center', va='center', fontweight='bold')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal', adjustable='datalim')
        
        # Set limits and titles for each subplot
        if ax_idx == 0:  # Top subplot - full view
            ax.set_title('Full Mesh View', fontweight='bold')
            # Draw rectangles for zoom boxes
            from matplotlib.patches import Rectangle
            rect1 = Rectangle((box1_xlim[0], box1_ylim[0]), 
                            box1_xlim[1] - box1_xlim[0], 
                            box1_ylim[1] - box1_ylim[0],
                            linewidth=1.5, edgecolor='tab:orange', facecolor='none', linestyle='-')
            rect2 = Rectangle((box2_xlim[0], box2_ylim[0]), 
                            box2_xlim[1] - box2_xlim[0], 
                            box2_ylim[1] - box2_ylim[0],
                            linewidth=1.5, edgecolor='tab:green', facecolor='none', linestyle='-')
            ax.add_patch(rect1)
            ax.add_patch(rect2)
        elif ax_idx == 1:  # Bottom left - zoom box 1
            ax.set_xlim(box1_xlim)
            ax.set_ylim(box1_ylim)
            ax.set_title('Zoom Box 1 (Orange)', fontweight='bold', color='tab:orange')
        elif ax_idx == 2:  # Bottom right - zoom box 2
            ax.set_xlim(box2_xlim)
            ax.set_ylim(box2_ylim)
            ax.set_title('Zoom Box 2 (Green)', fontweight='bold', color='tab:green')
    
    plt.show()


def write_pvd(dir_data, plane_nb, bl, nb_chknpts):
    """
    Generates the .pvd file to read the .vtk files as an animation with paraview.
    """
    f = open(os.path.join(dir_data, f"plane_{plane_nb}_bl{bl}.pvd"), "w")
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="Collection" version="0.1"\n')
    f.write('         byte_order="LittleEndian"\n')
    f.write('         compressor="vtkZLibDataCompressor">\n')
    f.write('  <Collection>\n')
    for i in range(nb_chknpts):
        i += 1
        f.write(f'    <DataSet timestep="{i}" group="" part="0"\n')
        fname = f"BL{bl}/par_plane_{plane_nb}_{i}_bl{bl}.vts"
        f.write(f'             file=\"{fname}\"/>\n')
    f.write('  </Collection>\n')
    f.write('</VTKFile>')
    f.close()


def write_para(input_dir, output_dir, plane_nb: int, var_names: list[str],
               plane: bool = False, stats: bool = False, rest: bool = False, tstar: str = ""):
    """
    Converts MUSICAA data to .vtk format.
    """
    plane_nb_str = str(plane_nb).rjust(3, '0')
    if tstar:
        unit = str(tstar).split('.')[0].rjust(4, '0')
        deci = str(tstar).split('.')[1].ljust(4, '0')

    # General information
    dict_info = read_info(input_dir)
    n_bl = dict_info['nbloc']
    nz = dict_info['nz_bl1']
    ngh = int(dict_info["ngh"])

    data: dict[str, Any] = {}
    for bl in range(1, n_bl + 1):
        # Retrieve data from readvars
        if not tstar:
            rest_file = os.path.join(input_dir, f'restart_bl{bl}.bin')
        else:
            rest_file = os.path.join(input_dir, f'restart{unit}_{deci}_bl{bl}.bin')
        nx, ny, nz, x, y, z = read_grid(
            input_dir, os.path.join(input_dir, f'grid_bl{bl}_ngh{ngh}.bin'))
        nx, ny, nz = dict_info[f'nx_bl{bl}'], dict_info[f'ny_bl{bl}'], dict_info[f'nz_bl{bl}']
        if rest:
            ro, rou, rov, row, roe = read_restart(rest_file, nx, ny, nz, False, False)
            # Store in dict
            data[f'block_{bl}'] = {}
            data[f'block_{bl}']['ro'] = ro
            data[f'block_{bl}']['rou'] = rou
            data[f'block_{bl}']['rov'] = rov
            data[f'block_{bl}']['row'] = row
            data[f'block_{bl}']['roe'] = roe
            data[f'block_{bl}']['x'], data[f'block_{bl}']['y'] = x, y

            # Make coordinates 3D for Paraview
            data[f'block_{bl}']['x'] = np.repeat(x[:, :, np.newaxis], nz, axis=2)
            data[f'block_{bl}']['y'] = np.repeat(y[:, :, np.newaxis], nz, axis=2)
            z_3d = np.repeat(z[np.newaxis, ...], ny, axis=0)
            z_3d = np.repeat(z_3d[np.newaxis, ...], nx, axis=0)

            # Save to file
            out_restart = os.path.join(output_dir, f'par_restart_bl{bl}')
            print(f"INFO -- converted restart of block {bl} saved to: {out_restart}")
            os.makedirs(out_restart, exist_ok=True)
            gridToVTK(
               out_restart, data[f'block_{bl}']['x'],
               data[f'block_{bl}']['y'], z_3d,
               pointData=data[f'block_{bl}']
            )

        if plane:
            # Make coordinates 3D for Paraview
            x = np.repeat(x[:, :, np.newaxis], 1, axis=2)
            y = np.repeat(y[:, :, np.newaxis], 1, axis=2)
            z = np.zeros((1))
            z_3d = np.repeat(z[np.newaxis, ...], ny, axis=0)
            z_3d = np.repeat(z_3d[np.newaxis, ...], nx, axis=0)

            plane_file = os.path.join(input_dir, f'plane_{plane_nb_str}_bl{bl}.bin')
            planes, nb_chkpnts = read_planes(plane_file, dict_info, var_names)
            planes_: dict[int, Any] = {}
            for i in range(nb_chkpnts):
                i += 1
                planes_[i] = {}
                for var in var_names:
                    planes_[i][var] = np.repeat(planes[i][var][:, :, np.newaxis], 1, axis=2)
                # Save to file
                os.makedirs(os.path.join(output_dir, f'par_planes/BL{bl}'), exist_ok=True)
                gridToVTK(
                    os.path.join(
                        output_dir, f'par_planes/BL{bl}/par_plane_{plane_nb_str}_{i}_bl{bl}'),
                    x, y, z_3d, pointData=planes_[i]
                )

            out_planes = os.path.join(output_dir, 'par_planes')
            print(f"INFO -- converted planes of block {bl} saved to: {out_planes}")
            write_pvd(out_planes, plane_nb_str, bl, nb_chkpnts)

        if stats:
            stats1 = read_stats(os.path.join(input_dir, f"stats1_bl{bl}.bin"), nx, ny)
            stats2 = read_stats(os.path.join(input_dir, f"stats2_bl{bl}.bin"), nx, ny)

            # Make coordinates 3D for Paraview
            nx, ny, nz, x, y, z = read_grid(
                input_dir, os.path.join(input_dir, f'grid_bl{bl}_ngh{ngh}.bin'))
            x = np.repeat(x[:, :, np.newaxis], 1, axis=2)
            y = np.repeat(y[:, :, np.newaxis], 1, axis=2)
            z = np.zeros((1))
            z_3d = np.repeat(z[np.newaxis, ...], ny, axis=0)
            z_3d = np.repeat(z_3d[np.newaxis, ...], nx, axis=0)

            stats_data: dict = {}
            for var in stats1.keys():
                stats_data[var] = np.repeat(stats1[var][:, :, np.newaxis], 1, axis=2)
            for var in stats2.keys():
                stats_data[var] = np.repeat(stats2[var][:, :, np.newaxis], 1, axis=2)

            # Save to file
            out_stats = os.path.join(output_dir, f'par_planes/par_stats_bl{bl}')
            print(f"INFO -- converted stats of block {bl} saved to: {out_stats}")
            gridToVTK(out_stats, x, y, z_3d, pointData=stats_data)
