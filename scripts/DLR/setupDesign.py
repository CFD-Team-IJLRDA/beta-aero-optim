"""
DLRprofiles.py - Unified Blade Profile Analysis Tool

This script provides a comprehensive suite for blade profile analysis including:
- Sensitivity analysis with linear parameter sweeps
- POD (Proper Orthogonal Decomposition) analysis with multifidelity sampling
- Geometric constraint verification

ARCHITECTURE:
-------------
The code has been refactored to eliminate redundancies through function consolidation:

1. UNIFIED GENERATION & SIMULATION:
   - generate_design_variants(): Handles all sampling modes (linear, random, multifidelity)
   - generate_and_simulate(): Unified pipeline for variant generation, BladeGenerator execution, 
     and data extraction with optional interpolation

2. ANALYSIS MODES:
   - Sensitivity (-s): Linear parameter sweeps with visualization
   - POD (-p): Multifidelity sampling, POD decomposition, mode analysis
   - Constraints (-c): Geometric constraint verification on generated profiles

3. CORE FUNCTIONS:
   - read_progen_input/write_progen_input: Template-based progen.input file I/O
   - extract_data_from_tec: Tecplot file parsing
   - run_blade_generator_parallel: Parallel BladeGenerator execution
   - compute_pod: POD basis computation
   - compute_QoIs: Geometric quantities of interest

USAGE:
------
python3 DLRprofiles.py -s  # Sensitivity analysis
python3 DLRprofiles.py -p  # POD analysis
python3 DLRprofiles.py -c  # Constraint verification

REFACTORING HISTORY:
--------------------
- Consolidated generate_design_variants (linear/random/mf modes)
- Unified generate_and_simulate + generate_and_simulate_pod
- Removed redundant _simulate_single_variant_pod worker function
- All modes now use same parallel processing infrastructure
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import subprocess
import numpy.random as rng
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

from aero_optim.ffd.ffd import FFD_2D, DLR_POD_2D
from aero_optim.geom import split_profile, get_chords, get_camber_th, get_area, get_cog, get_circle, plot_profile, plot_sides
from interpolation import interpolate_profile

OD = OrderedDict  # Type alias

# Define function to extract m and theta from tecplot file
def extract_data_from_tec(file_path, zone_name="NURB_Profil_gestaf0"):
    m_values = []
    theta_values = []
    in_zone = False

    with open(file_path, 'r') as file:
        for line in file:
            if f'ZONE T="{zone_name}"' in line:
                in_zone = True  # Start reading the data for the desired zone
            elif in_zone and line.strip().startswith("ZONE"):
                in_zone = False  # Stop reading after the zone ends
            elif in_zone:
                # Extract m and theta from the current line (first two columns)
                try:
                    values = line.split()
                    m_values.append(float(values[0]))
                    theta_values.append(float(values[1]))
                except ValueError:
                    continue  # Skip any lines that don't contain valid float values
    
    # Remove duplicate last point (first and last points are identical in tecplot files)
    m_array = np.array(m_values)
    theta_array = np.array(theta_values)
    if len(m_array) > 1 and np.allclose(m_array[0], m_array[-1]) and np.allclose(theta_array[0], theta_array[-1]):
        m_array = m_array[:-1]
        theta_array = theta_array[:-1]
    
    return m_array, theta_array

def run_blade_generator_parallel(variant_dir_and_exe):
    """Wrapper function for running BladeGenerator in parallel"""
    variant_dir, blade_generator_path = variant_dir_and_exe
    try:
        result = subprocess.run([blade_generator_path], cwd=variant_dir, 
                              capture_output=True, text=True, check=False)
        return (variant_dir, result.returncode, result.stdout, result.stderr)
    except Exception as e:
        return (variant_dir, -1, "", str(e))

def extract_data_parallel(variant_dir):
    """Wrapper function for extracting data in parallel"""
    output_file = os.path.join(variant_dir, 'Output', 'Profile_mergedGestaf.tec')
    
    if os.path.exists(output_file):
        try:
            m, theta = extract_data_from_tec(output_file)
            data_2d = np.column_stack((m, theta))
            return (variant_dir, data_2d, None)
        except Exception as e:
            return (variant_dir, np.array([]), str(e))
    else:
        return (variant_dir, np.array([]), "Output file not found")

def process_blade_generator_results(results, context="execution"):
    """Helper function to process BladeGenerator results consistently"""
    successful_runs = 0
    for profile, returncode, stdout, stderr in results:
        if returncode == 0 or returncode == 100:
            successful_runs += 1
            # if returncode == 100:
                # print(f'  Completed with warnings: {profile}')
            # else:
                # print(f'  Completed: {profile}')
        else:
            # print(f'  Warning: BladeGenerator failed for {profile} (exit code: {returncode})')
            if stderr:
                print(f'    Error: {stderr}')
    return successful_runs

def read_progen_input(path: str) -> Tuple[List[str], OD]:
    """
    Legge il file `progen.input`, salva le prime 3 righe in `header`
    e il resto in un OrderedDict `entries`.
    - Commenti (linee che iniziano con '#' o '!') e linee vuote sono conservate
      come chiavi speciali '__comment_N' o '__blank_N' con valore = riga originale.
    - Linee con '=' vengono parse come key = value (split sulla prima '=').
    - Altre linee vengono parse come "key [value...]" (primo token = key, resto = value).
    - Se una chiave appare più volte, il valore viene salvato come lista di (sep, value) tuples per preservare il separatore.
    Ritorna (header_lines, entries).
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header = [lines[i].rstrip('\n') for i in range(min(3, len(lines)))]
    rest = lines[3:] if len(lines) > 3 else []

    # store entries as OrderedDict[key] = (sep, value) or list of (sep, value)
    entries: OD[str, Union[Tuple[Union[str, None], str], List[Tuple[Union[str, None], str]]]] = OrderedDict()
    comment_idx = 0
    blank_idx = 0

    for raw in rest:
        line = raw.rstrip('\n')
        stripped = line.lstrip()
        if stripped == '':
            blank_idx += 1
            entries[f'__blank_{blank_idx}'] = ''  # rappresenta una linea vuota
            continue
        if stripped.startswith('#') or stripped.startswith('!'):
            comment_idx += 1
            entries[f'__comment_{comment_idx}'] = line
            continue

        # detect separator to preserve formatting: '=' or whitespace
        if '=' in line:
            key, val = line.split('=', 1)
            sep = '='
            key = key.strip()
            val = val.strip()
        else:
            parts = line.split(None, 1)
            sep = None
            key = parts[0].strip()
            val = parts[1].strip() if len(parts) > 1 else ''

        item = (sep, val)
        if key in entries:
            existing = entries[key]
            if isinstance(existing, list):
                existing.append(item)
            else:
                entries[key] = [existing, item]
        else:
            entries[key] = item

    return header, entries

def get_parameter_value(entries: OD, key: str) -> str:
    """Extract parameter value from entries, handling both single values and lists."""
    if key not in entries:
        return 'N/A'
    
    entry = entries[key]
    if isinstance(entry, list):
        return entry[0][1] if entry else 'N/A'
    else:
        return entry[1] if entry else 'N/A'

def write_progen_input(out_dir: str, header: List[str], entries: OD, input_dict: dict = None) -> None:
    """
    Scrive su file con lo stesso formato di progen.input:
    - Scrive le prime 3 righe (header) così come sono.
    - Poi scrive le entries nell'ordine dato:
      * '__blank_N' -> riga vuota
      * '__comment_N' -> riga commento così com'è
      * key -> 'key = value' o 'key value' a seconda del separatore originale
    out_dir must be a directory path; the function will create it if needed and write 'progen.input' inside.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'progen.input')    
    # Check if file already exists
    if os.path.exists(out_file):
        # print(f"File {out_file} already exists, skipping write")
        return

    with open(out_file, 'w', encoding='utf-8') as f:
        # header
        for i in range(3):
            line = header[i] if i < len(header) else ''
            f.write(line.rstrip('\n') + '\n')

        # body
        for key, val in entries.items():
            if key.startswith('__blank_'):
                f.write('\n')
            elif key.startswith('__comment_'):
                # val contiene la riga così com'era
                f.write(val.rstrip('\n') + '\n')
            else:
                # entries store either (sep, value) or list of (sep, value)
                items = val if isinstance(val, list) else [val]

                # If input_dict provided and key present in input_dict, replace values
                if input_dict is not None and key in input_dict:
                    ov = input_dict[key]
                    # choose separator preference from existing items (prefer first item's sep)
                    first_sep = items[0][0] if items else None
                    if isinstance(ov, list):
                        # build items list from input_dict list
                        items = [(first_sep, str(x)) for x in ov]
                    else:
                        items = [(first_sep, str(ov))]

                for sep, v in items:
                    if sep == '=':
                        f.write(f"{key} = {v}\n")
                    else:
                        # write with single space separator to mimic original whitespace format
                        if v == '':
                            f.write(f"{key}\n")
                        else:
                            f.write(f"{key} {v}\n")

def generate_design_variants(bounds: Union[dict, List[dict]], header: List[str], entries: OD, N: int, 
                           out_base_dir: str, seed: int | None = None, sampling_mode: str = 'linear') -> Tuple[List[str], List[dict]]:
    """Generate N design parameter dicts and write progen.input files.

    Args:
        bounds: dict {key: [low, high]} or list of dicts [{key: [low, high]}]
        header, entries: parsed progen input used as template
        N: number of variants to generate
        out_base_dir: parent directory where tempi folders will be created
        seed: optional RNG seed
        sampling_mode: 'linear', 'random', or 'mf' (multifidelity)

    Returns:
        Tuple of (list of directories, list of parameter dicts)
    """
    os.makedirs(out_base_dir, exist_ok=True)
    written_paths = []
    parameter_values = []
    
    print(f'Generating {N} {sampling_mode} design variants in {out_base_dir}...')
    
    if seed is not None:
        rng.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    
    # Convert bounds to uniform format
    if isinstance(bounds, dict):
        bounds_dict = bounds
    else:  # List of single-key dicts
        bounds_dict = {}
        for b in bounds:
            bounds_dict.update(b)
    
    if sampling_mode == 'mf':
        # Multifidelity sampling
        from aero_optim.mf_sm.mf_models import get_sampler
        mf_sampler = get_sampler(len(bounds_dict), [0, 1], seed, True)
        x_lf, _ = mf_sampler.sample_mf(N, 1)
        keyList = list(bounds_dict.keys())
        
        for i in range(N):
            variant = {}
            for j, key in enumerate(keyList):
                lo, hi = bounds_dict[key]
                variant[key] = x_lf[i, j] * (hi - lo) + lo
            parameter_values.append(variant)
    else:
        # Linear or random sampling
        for i in range(1, N + 1):
            variant = {}
            for k, (lo, hi) in bounds_dict.items():
                if sampling_mode == 'random':
                    variant[k] = rng.uniform(float(lo), float(hi))
                else:  # linear
                    variant[k] = float(lo) + (float(hi) - float(lo)) * (i - 1) / (N - 1)
            parameter_values.append(variant)
    
    # Write progen.input files
    for i, variant in enumerate(parameter_values, 0):
        # print(f'{sampling_mode.capitalize()} variant {i}: {variant}')
        out_dir = os.path.join(out_base_dir, f'temp{i}')
        write_progen_input(out_dir, header, entries, input_dict=variant)
        written_paths.append(out_dir)

    return written_paths, parameter_values

def generate_and_simulate(bounds: Union[dict, List[dict]], header: List[str], entries: OD, N: int, 
                         out_base_dir: str, interpolate: dict, blade_generator_path: str = '/home/mciarlatani/bin/BladeGenerator.exe',
                         seed: int | None = None, max_workers: int = None, sampling_mode: str = 'mf') -> List[np.ndarray]:
    """
    Unified function to generate N progen.input files with parameter variations, run BladeGenerator.exe,
    and extract tecplot data. Supports multiple sampling modes and optional interpolation.
    
    Args:
        bounds: dict {key: [low, high]} or list of dicts [{key: [low, high]}] for parameter bounds
        header, entries: parsed progen input used as template
        N: number of variants to generate
        out_base_dir: parent directory where tempi folders will be created
        interpolate: dict with 'interpolate' flag and 'original_file'
        blade_generator_path: path to BladeGenerator.exe
        seed: optional RNG seed
        max_workers: maximum number of parallel workers (None = use all CPU cores)
        sampling_mode: 'linear', 'random', or 'mf' (multifidelity)
        
    Returns:
        List of 2D numpy arrays, each containing [m, theta] data from extract_data_from_tec
    """
    if max_workers is None:
        max_workers = cpu_count()
    
    # Step 1: Generate N progen.input files
    print(f'Step 1: Generating {N} {sampling_mode} parameter variants...')
    variant_dirs, _ = generate_design_variants(bounds, header, entries, N, out_base_dir, seed, sampling_mode)
    
    # Step 2: Run BladeGenerator.exe in parallel
    print(f'Step 2: Running BladeGenerator.exe in {len(variant_dirs)} directories using {max_workers} workers...')
    
    blade_args = [(variant_dir, blade_generator_path) for variant_dir in variant_dirs]
    
    with Pool(processes=max_workers) as pool:
        results = pool.map(run_blade_generator_parallel, blade_args)
    
    # Process results using helper function
    successful_runs = process_blade_generator_results(results, "simulation")
    print(f'  Completed: {successful_runs}/{len(variant_dirs)} simulations successful')
    
    # Step 3: Extract tecplot data in parallel
    print(f'Step 3: Extracting tecplot data from generated outputs using {max_workers} workers...')
    
    with Pool(processes=max_workers) as pool:
        extraction_results = pool.map(extract_data_parallel, variant_dirs)
    
    # Process extraction results and apply interpolation if requested
    extracted_data = []
    successful_extractions = 0
    
    # Load original profile once if interpolation is requested
    original_profile = None
    if interpolate and interpolate.get('interpolate', False):
        original_file = interpolate.get('original_file')
        if original_file:
            original_profile = np.genfromtxt(original_file, skip_header=2, delimiter=' ')
            if original_profile.shape[1] == 3:
                original_profile = original_profile[:, :2]
    
    for variant_dir, data_2d, error in extraction_results:
        if data_2d.size > 0:
            # Apply interpolation if configured
            if original_profile is not None:
                data_2d = interpolate_profile(data_2d, original_profile, kind='linear')
            
            extracted_data.append(data_2d)
            successful_extractions += 1
            # print(f'  Extracted data from {variant_dir}: shape {data_2d.shape}')
        else:
            extracted_data.append(np.array([]))
            if error:
                print(f'  Error extracting data from {variant_dir}: {error}')
            successful_extractions += 1
    print(f'  Extracted data shape: {data_2d.shape}')

    return extracted_data

def get_valid_center(
        x: np.ndarray, y: np.ndarray, dmin: float, dmax: float,
        le: bool = True, percent: float = 10, resolution: int = 50
) -> np.ndarray | None:
    """
    **Computes** and **returns** the center of a valid circle in regards of the
    leading/trailing edge constraints.

    In particular, it checks if circles of radius dmin can fit in both the leading
    and trailing edges, and if such circles have their centers located at a distance
    to the leading/trailing edge that is smaller than dmax.
    **Returns** None if the constraint is not respected.
    """
    # sort coordinates
    count = int(len(x) * percent / 100)
    indices = np.argsort(x) if le else np.argsort(x)[::-1]
    x_sorted = x[indices][:count]
    y_sorted = y[indices][:count]

    # find bounding box for circle center location search
    profile = np.column_stack([x_sorted, y_sorted])
    min_x, min_y = profile.min(axis=0)
    max_x, max_y = profile.max(axis=0)

    # generate grid of candidate points, starting near leading edge
    x_vals = np.linspace(min_x, max_x, resolution)
    y_vals = np.linspace(min_y, max_y, resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    candidate_points = np.vstack([X.ravel(), Y.ravel()]).T

    # sort candidates by smallest x and on the right side of the le/te edge
    candidate_points = candidate_points[np.argsort(candidate_points[:, 0])]
    candidate_points = (
        candidate_points[candidate_points[:, 0] > profile[0, 0]] if le
        else candidate_points[candidate_points[:, 0] < profile[0, 0]]
    )

    # keep only the candidates at a distance from the le/te edge
    # comprised between dmin and dmax
    dists = cdist(candidate_points, np.array([[x_sorted[0], y_sorted[0]]]))
    idx, _ = np.where((dists > dmin) & (dists < dmax))

    # check if there is at least one pt that gives a valid circle center
    for pt in candidate_points[idx]:
        if np.min(cdist([pt], profile)) > dmin:
            return pt
    return None

def plot_failed_profiles(failed_profiles_file: str, directory: str, n_samples: int = 15):
    """
    Legge gli indici dei profili falliti, sceglie n_samples casuali e li plotta.
    
    Args:
        failed_profiles_file: path al file con gli indici dei profili falliti
        directory: directory base dove si trovano le cartelle tempX
        n_samples: numero di profili da plottare (default 15)
    """
    # Leggi gli indici dal file
    with open(os.path.join(directory, failed_profiles_file), 'r') as f:
        failed_indices = [int(line.strip()) for line in f if line.strip()]
    
    print(f"Found {len(failed_indices)} failed profiles")
    
    # Scegli n_samples indici casuali
    if len(failed_indices) < n_samples:
        print(f"Warning: Only {len(failed_indices)} profiles available, plotting all")
        selected_indices = failed_indices
    else:
        selected_indices = random.sample(failed_indices, n_samples)
    
    print(f"Selected {len(selected_indices)} profiles to plot: {selected_indices}")
    
    # Prepara tutti i plot ma non mostrarli ancora
    figures = []
    for idx in selected_indices:
        profile_path = os.path.join(directory, f'temp{idx}', 'Output', 'Profile_mergedGestaf.tec')
        
        if not os.path.exists(profile_path):
            print(f"Warning: Profile file not found: {profile_path}")
            continue
        
        # Estrai i dati
        m, theta = extract_data_from_tec(profile_path)
        profile = np.column_stack((m, theta))
        
        # Calcola le componenti necessarie per plot_sides
        upper, lower = split_profile(profile)
        c = get_chords(profile)[0]
        camber_line, thmax, Xthmax, th_vec = get_camber_th(upper, lower, interpolate=True)
        
        # Calcola i cerchi LE e TE
        O_le = get_valid_center(profile[:, 0], profile[:, 1], dmin=0.005 * c, dmax=1.4 * 0.005 * c, le=True)
        O_te = get_valid_center(profile[:, 0], profile[:, 1], dmin=0.005 * c, dmax=1.4 * 0.005 * c, le=False)
        le_circle = get_circle(O_le, 0.005 * c) if O_le is not None else np.array([])
        te_circle = get_circle(O_te, 0.005 * c) if O_te is not None else np.array([])
        
        # Crea il plot (nuova figura)
        plot_sides(upper, lower, camber_line, le_circle, te_circle, th_vec, False)

# ============================================================================
# POD-SPECIFIC FUNCTIONS
# ============================================================================

def compute_pod(profiles, nprofile: int, nmode: int):
    """Compute POD basis and related matrices from profile stack."""
    S = np.stack([p[:, -1] for p in profiles], axis=1)
    print(f"S shape: {S.shape}")
    
    S_mean = 1 / nprofile * np.sum(S, axis=1)
    print(f"S_mean shape: {S_mean.shape}")
    F = S[:, :] - S_mean[:, None]
    print(f"shape of F: {F.shape}")

    C = np.matmul(np.transpose(F), F)
    print(f"shape of C: {C.shape}")
    eigenvalues, eigenvectors = LA.eigh(C)
    print(f"shape of eigenvectors: {eigenvectors.shape}")
    phi = np.matmul(F, eigenvectors)
    print(f"shape of phi: {phi.shape}")

    phi_tilde = phi[:, -nmode:]
    print(f"shape of phi_tilde: {phi_tilde.shape}")
    V_tilde_inv = np.linalg.inv(eigenvectors)[-nmode:, :]
    # print(eigenvalues[-nmode:])
    # # print(V_tilde_inv[-nmode,:])
    # input()
    print(f"shape of V_tilde_inv: {V_tilde_inv.shape}")
    D_tilde = S_mean[:, None] + np.matmul(phi_tilde, V_tilde_inv)
    return S, S_mean, F, C, eigenvalues, eigenvectors, phi, phi_tilde, V_tilde_inv, D_tilde

def sample_ellipsoid(V_tilde_inv, confidence_level=0.90, nSamples=1000):

    nModes = V_tilde_inv.shape[0]
    mean = V_tilde_inv.T.mean(axis=0)
    cov = np.cov(V_tilde_inv.T, rowvar=False)

    # 95% confidence ellipsoid
    radius2 = chi2.ppf(confidence_level, df=nModes)  # chi-square for 3D
    eigvals, eigvecs = np.linalg.eigh(cov)
    radii = np.sqrt(eigvals * radius2)

    u = np.random.normal(size=(nSamples, nModes))

    u /= np.linalg.norm(u, axis=1)[:, None]
    r = np.random.rand(nSamples) ** (1/nModes)
    Delta = (mean + (u * r[:, None]) @ np.diag(radii) @ eigvecs.T).T

    print('Deltas shape:', Delta.shape)

    return Delta

def run_pod_analysis(design_sensitivity, input_file, baseline_file, directory, nprofile=1000, nmode=5, n_processes=None):
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
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    
    if n_processes is None:
        n_processes = min(mp.cpu_count(), 90)
    
    ncontrol = len(design_sensitivity)

    header, params = read_progen_input(input_file)
    
    print(f"Using {n_processes} processes for parallel simulation")

    profiles = generate_and_simulate(design_sensitivity, header, params, nprofile, directory, 
                                     interpolate, '/home/mciarlatani/bin/BladeGenerator.exe', 10, n_processes, 
                                     sampling_mode='mf')
    print(f"First profile shape: {profiles[0].shape}")

    # for i in range(len(profiles)):
    #     profiles[i] = profiles[i][:-1,:]
    
    # Compute POD
    S, S_mean, F, C, eigenvalues, eigenvectors, phi, phi_tilde, V_tilde_inv, D_tilde = compute_pod(profiles, nprofile, nmode)
    
    # POD boundaries
    l_bound = np.array([min(v) for v in V_tilde_inv])
    u_bound = np.array([max(v) for v in V_tilde_inv])

    print('Lower bound_POD:', l_bound)
    print('Upper bound_POD:', u_bound)
    
    y_min = S_mean + np.sum(phi_tilde * np.array(l_bound), axis=1)
    y_max = S_mean + np.sum(phi_tilde * np.array(u_bound), axis=1)

    plt.figure()
    for i in range(nmode):
        j = 0
        plt.subplot(nmode, nmode, i * nmode + j + 1)
        plt.ylabel(f'V_{i+1}')
        for j in range(nmode):
            plt.subplot(nmode, nmode, i * nmode + j + 1)
            if nmode-i == 1:
                plt.xlabel(f'V_{j+1}')
            if i == j:
                plt.hist(dlr.V_tilde_inv[i, :], bins=50, alpha=0.7)
            else:
                plt.scatter(dlr.V_tilde_inv[i,:], dlr.V_tilde_inv[j,:], alpha=0.15)
                plt.axis('equal')
            plt.xticks([])
            plt.yticks([])
    plt.show()

    # print('V_tilde_inv shape:', V_tilde_inv.shape)

    # nModes = V_tilde_inv.shape[0]
    # mean = V_tilde_inv.T.mean(axis=0)
    # cov = np.cov(V_tilde_inv.T, rowvar=False)

    # # 90% confidence ellipsoid
    # radius2 = chi2.ppf(0.90, df=nModes)  # chi-square for 3D
    # eigvals, eigvecs = eigh(cov)

    # # Radii along principal axes
    # radii = np.sqrt(eigvals * radius2)

    # # Parametric angles
    # u = np.linspace(0, 2*np.pi, 50)
    # v = np.linspace(0, np.pi, 50)
    # u, v = np.meshgrid(u, v)

    # # Parametric equation of a unit sphere
    # x = np.cos(u) * np.sin(v)
    # y = np.sin(u) * np.sin(v)
    # z = np.cos(v)

    # # Flatten the arrays for matrix multiplication
    # xyz = np.stack([x.flatten(), y.flatten(), z.flatten()])

    # # Transform to ellipsoid
    # ellipsoid = eigvecs @ np.diag(radii) @ xyz
    # ellipsoid = ellipsoid + mean.reshape(3,1)

    # # Reshape back for plotting
    # x_ell = ellipsoid[0].reshape(x.shape)
    # y_ell = ellipsoid[1].reshape(y.shape)
    # z_ell = ellipsoid[2].reshape(z.shape)

    # Delta = sample_ellipsoid(V_tilde_inv, confidence_level=0.90, nSamples=1000)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # sc = ax.scatter(V_tilde_inv[0, :], V_tilde_inv[1, :], V_tilde_inv[2, :], c=V_tilde_inv[2, :], cmap='viridis', s=50)
    # sc = ax.scatter(Delta[0, :], Delta[1, :], Delta[2, :], c='tab:red', marker='x', s=30)
    # ax.plot_surface(x_ell, y_ell, z_ell, color='r', alpha=0.2)
    # ax.set_xlabel(r'$V_1$')
    # ax.set_ylabel(r'$V_2$')
    # ax.set_zlabel(r'$V_3$')
    # # plt.colorbar(sc, ax=ax, label='Z value')
    # plt.show()
    
    # Use x coordinates from profiles (same length as y_min/y_max)
    x_coords = profiles[0][:, 0]
    upper_min, lower_min = split_profile(np.column_stack((x_coords, y_min)))
    upper_max, lower_max = split_profile(np.column_stack((x_coords, y_max)))
    
    # ========================================================================
    # CONSOLIDATED PLOT - All POD analysis in one figure
    # ========================================================================
    
    # Use 16:9 aspect ratio for saved figure
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Random reconstructions (top left)
    ax1 = plt.subplot(gs[0, 0])
    nplots = 5
    for ii in range(nplots):
        idx = random.randint(0, len(profiles) - 1)
        ax1.plot(profiles[idx][:, 0], S[:, idx], label=f"sample {idx}", linewidth=1)
        ax1.plot(profiles[idx][:, 0], D_tilde[:, idx], linestyle="dashed", color="k", linewidth=1)
    ax1.set(xlabel="$x$ [m]", ylabel="$y$ [m]", title=f"a) Reconstructed profiles ({nmode} modes)")
    ax1.legend(loc="best", fontsize=6)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: POD modes (top middle)
    ax2 = plt.subplot(gs[0, 1])
    # Normalize x coordinates by chord length (approximate)
    chord_approx = np.max(x_coords) - np.min(x_coords)
    for nn in range(1, nmode + 1):
        ax2.plot(x_coords / chord_approx, phi_tilde[:, -nn], label=f"mode {nn}", marker = 'x', linewidth=1.5)
    ax2.set(xlabel="$x / c$ [-]", ylabel="POD basis [-]", title="b) Geometric modes")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Energy and error (top right)
    ax3 = plt.subplot(gs[1, 0])
    eigen_nrj = []
    error = []
    for nn in range(1, ncontrol + 1):
        phi_tilde_tmp = phi[:, -nn:]
        V_tilde_inv_tmp = np.linalg.inv(eigenvectors)[-nn:, :]
        D_tilde_tmp = S_mean[:, None] + np.matmul(phi_tilde_tmp, V_tilde_inv_tmp)
        eigen_nrj.append(eigenvalues[-nn] / np.sum(eigenvalues) * 100)
        err = np.sqrt(np.sum([(y_true - y_pred) ** 2 for y_true, y_pred in zip(S.T, D_tilde_tmp.T)]) / nprofile)
        error.append(err)
    ax3.axvline(nmode, color="k", linestyle="dashed", linewidth=1)
    ax3.plot(range(1, len(eigen_nrj) + 1), eigen_nrj, color="blue", marker="s", ms=4, label="energy", linewidth=1.5)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(range(1, len(error) + 1), error, color="red", marker="s", ms=4, label="RMSE", linewidth=1.5)
    ax3_twin.set_yscale("log")
    ax3.set(xlabel="$N_i$ [-]", ylabel="$\\lambda_i / \\sum_{n=1}^{N_m} \\lambda_n$ [pct]", title="c) Energy and error")
    ax3_twin.set(ylabel="RMSE [m]")
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc="center left", bbox_to_anchor=(0.5, 0.5))
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Design space (bottom, spanning all columns)
    ax4 = plt.subplot(gs[1, 1])
    # Use baseline profile from first sample for reference
    baseline_y = profiles[0][:, 1]  # or S_mean for mean profile
    ax4.plot(x_coords, baseline_y, label="baseline profile", color="k", linewidth=2)
    ax4.plot(x_coords, y_min, label="min profile", color="b", linewidth=1.5)
    ax4.plot(x_coords, y_max, label="max profile", color="r", linewidth=1.5)
    ax4.fill_between(lower_min[:, 0], lower_min[:, 1], lower_max[:, 1], color="b", alpha=0.15)
    ax4.fill_between(upper_max[:, 0], upper_min[:, 1], upper_max[:, 1], color="r", alpha=0.15)
    ax4.set(xlabel="$x$ [m]", ylabel="$y$ [m]", title="d) POD design space")
    ax4.legend(loc="best")
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal', adjustable='datalim')
    
    plt.show()
    plt.close()

def run_pod_analysis_DLR(design_sensitivity, input_file, baseline_file, directory, nprofile=1000, nmode=5, n_processes=None):
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
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    
    ncontrol = len(design_sensitivity)

    Path(directory).mkdir(exist_ok=True)
    os.system('cp ' + input_file + ' ' + directory + '/progen.input '+directory)
    
    params_dict = {}
    for i, param in enumerate(design_sensitivity):
        for key, value in param.items():
            params_dict[key] = value

    if n_processes is None:
        n_processes = min(mp.cpu_count(), 90)

    dlr = DLR_POD_2D(baseline_file,  directory, '/home/mciarlatani/bin/BladeGenerator.exe', params_dict, nmode, nprofile, seed, perturb_POD=perturb)

    from sklearn.decomposition import PCA

    profiles = np.array([p[:, -1]-dlr.pts[:,-1] for p in dlr.profiles])
    pca = PCA(n_components=5, svd_solver="full")
    pca.fit(profiles)
    print(pca.explained_variance_ratio_)

    plt.figure()
    for p in pca.components_:
        plt.plot(dlr.pts[:,0], p)
    plt.show()


def orig_pod_analysis_DLR(design_sensitivity, input_file, baseline_file, directory, nprofile=1000, nmode=5, n_processes=None):
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
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    
    ncontrol = len(design_sensitivity)

    Path(directory).mkdir(exist_ok=True)
    os.system('cp ' + input_file + ' ' + directory + '/progen.input '+directory)
    
    params_dict = {}
    for i, param in enumerate(design_sensitivity):
        for key, value in param.items():
            params_dict[key] = value

    if n_processes is None:
        n_processes = min(mp.cpu_count(), 90)

    dlr = DLR_POD_2D(baseline_file,  directory, '/home/mciarlatani/bin/BladeGenerator.exe', params_dict, nmode, nprofile, seed, perturb_POD=perturb)

    # POD boundaries
    l_bound, u_bound = dlr.get_bound()

    print('Lower bound_DLR:', l_bound)
    print('Upper bound_DLR:', u_bound)

    y_min = dlr.apply_ffd(np.array(l_bound))[:,1]
    y_max = dlr.apply_ffd(np.array(u_bound))[:,1]
    # y_max = dlr.S_mean + np.sum(dlr.phi_tilde * np.array(u_bound), axis=1)

    # Use x coordinates from profiles (same length as y_min/y_max)
    x_coords = dlr.ffd.pts[:, 0]
    upper_min, lower_min = split_profile(np.column_stack((x_coords, y_min)))
    upper_max, lower_max = split_profile(np.column_stack((x_coords, y_max)))

    # from scipy.stats import norm
    # dlr.V_tilde_inv[nmode-1,:] = norm.ppf(dlr.V_tilde_inv[nmode-1,:]-np.min(dlr.V_tilde_inv[nmode-1,:])/(np.max(dlr.V_tilde_inv[nmode-1,:])-np.min(dlr.V_tilde_inv[nmode-1,:])))

    # plt.figure()
    # for i in range(nmode):
    #     j = 0
    #     plt.subplot(nmode, nmode, i * nmode + j + 1)
    #     plt.ylabel(f'V_{i+1}')
    #     for j in range(nmode):
    #         plt.subplot(nmode, nmode, i * nmode + j + 1)
    #         if nmode-i == 1:
    #             plt.xlabel(f'V_{j+1}')
    #         if i == j:
    #             plt.hist(dlr.V_tilde_inv[i,:], bins=50, alpha=0.7)
    #         else:
    #             plt.scatter(dlr.V_tilde_inv[i,:], dlr.V_tilde_inv[j,:], alpha=0.15)
    #             plt.axis('equal')
    #         plt.xticks([])
    #         plt.yticks([])
    # plt.show()
    
    # ========================================================================
    # CONSOLIDATED PLOT - All POD analysis in one figure
    # ========================================================================
    
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    ax1 = plt.subplot(gs[0, 0])
    nplots = 5

    if perturb == 'Baseline':
        add = dlr.ffd.pts[:, 1]
    else:
        add = 0.0

    for ii in range(nplots):
        idx = random.randint(0, len(dlr.profiles) - 1)
        ax1.plot(dlr.profiles[idx][:, 0], dlr.S[:, idx]+add, label=f"sample {idx}", linewidth=1)
        ax1.plot(dlr.profiles[idx][:, 0], dlr.D_tilde[:, idx]+add, linestyle="dashed", color="k", linewidth=1)
    ax1.set(xlabel="$x$ [m]", ylabel="$y$ [m]", title=f"a) Reconstructed profiles ({nmode} modes)")
    ax1.legend(loc="best", fontsize=6)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: POD modes (top middle)
    ax2 = plt.subplot(gs[0, 1])
    # Normalize x coordinates by chord length (approximate)
    chord_approx = np.max(x_coords) - np.min(x_coords)
    for nn in range(1, nmode + 1):
        ax2.plot(x_coords / chord_approx, dlr.phi_tilde[:, -nn], label=f"mode {nn}", linewidth=1.5)
    ax2.set(xlabel="$x / c$ [-]", ylabel="POD basis [-]", title="b) Geometric modes")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Energy and error (top right)
    ax3 = plt.subplot(gs[1, 0])
    eigen_nrj = []
    error = []
    for nn in range(1, ncontrol + 1):
        phi_tilde_tmp = dlr.phi[:, -nn:]
        V_tilde_inv_tmp = np.linalg.inv(dlr.eigenvectors)[-nn:, :]
        D_tilde_tmp = dlr.S_mean[:, None] + np.matmul(phi_tilde_tmp, V_tilde_inv_tmp)
        eigen_nrj.append(dlr.eigenvalues[-nn] / np.sum(dlr.eigenvalues) * 100)
        err = np.sqrt(np.sum([(y_true - y_pred) ** 2 for y_true, y_pred in zip(dlr.S.T, D_tilde_tmp.T)]) / nprofile)
        # err = np.sqrt(np.sum([(y_true - y_pred) ** 2 for y_true, y_pred in zip(dlr.S.T, D_tilde_tmp.T)]) / nprofile)
        error.append(err)
    ax3.axvline(nmode, color="k", linestyle="dashed", linewidth=1)
    ax3.plot(range(1, len(eigen_nrj) + 1), eigen_nrj, color="blue", marker="s", ms=4, label="energy", linewidth=1.5)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(range(1, len(error) + 1), error, color="red", marker="s", ms=4, label="RMSE", linewidth=1.5)
    ax3_twin.set_yscale("log")
    ax3.set(xlabel="$N_i$ [-]", ylabel="$\\lambda_i / \\sum_{n=1}^{N_m} \\lambda_n$ [pct]", title="c) Energy and error")
    ax3_twin.set(ylabel="RMSE [m]")
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc="center left", bbox_to_anchor=(0.5, 0.5))
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Design space (bottom, spanning all columns)
    ax4 = plt.subplot(gs[1, 1])
    # Use baseline profile from first sample for reference
    baseline_y = dlr.ffd.pts[:, 1]  # or S_mean for mean profile
    ax4.plot(x_coords, baseline_y, label="baseline profile", color="k", linewidth=2)
    ax4.plot(x_coords, y_min, label="min profile", color="b", linewidth=1.5)
    ax4.plot(x_coords, y_max, label="max profile", color="r", linewidth=1.5)
    ax4.fill_between(lower_min[:, 0], lower_min[:, 1], lower_max[:, 1], color="b", alpha=0.15)
    ax4.fill_between(upper_max[:, 0], upper_min[:, 1], upper_max[:, 1], color="r", alpha=0.15)
    ax4.set(xlabel="$x$ [m]", ylabel="$y$ [m]", title="d) POD design space")
    ax4.legend(loc="best")
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal', adjustable='datalim')


    plt.savefig('POD_Info.png', dpi=300)
    
    plt.show()
    plt.close()

def compare_POD(design_sensitivity, input_file, baseline_file, directory, nprofile=1000, nmode=5, n_processes=None):
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
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    
    ncontrol = len(design_sensitivity)

    Path(directory).mkdir(exist_ok=True)
    os.system('cp ' + input_file + ' ' + directory + '/progen.input '+directory)
    
    params_dict = {}
    for i, param in enumerate(design_sensitivity):
        for key, value in param.items():
            params_dict[key] = value

    if n_processes is None:
        n_processes = min(mp.cpu_count(), 90)

    dlr = DLR_POD_2D(baseline_file,  directory, '/home/mciarlatani/bin/BladeGenerator.exe', params_dict, nmode, nprofile, seed, perturb_POD=perturb)

    header, params = read_progen_input(input_file)

    print(f"Using {n_processes} processes for parallel simulation")

    profiles = generate_and_simulate(design_sensitivity, header, params, nprofile, directory, 
                                    interpolate, '/home/mciarlatani/bin/BladeGenerator.exe', 10, n_processes, 
                                     sampling_mode='mf')

    # with Pool(processes=n_processes) as pool:
    #     profiles = pool.starmap(dlr.interpolate_profile, [(p, dlr.pts, 'linear') for p in profiles])

    # plt.scatter(dlr.profiles[0][:,0], dlr.profiles[0][:,-1], marker = 'o', facecolor='none', label='DLR profile')
    # plt.scatter(profiles[0][:,0], profiles[0][:,-1], marker = 'X', facecolor='none', label='DLR profile')
    
    # plt.scatter(dlr.profiles[0][:,0], dlr.profiles[0][:,1], marker = 'o', facecolors='none', edgecolors='tab:blue', label='DLR profile 0')
    # plt.scatter(profiles[0][:,0], profiles[0][:,1], marker = 'x', label='POD profile 0')
    # for idx, (x, y) in enumerate(profiles[0][:, :2]):
    #     plt.text(x, y, str(idx+1), fontsize=8, ha='right', va='bottom', color='red')
    # plt.legend()
    # plt.show()
    # plt.close()

    # Compute POD
    S, S_mean, F, C, eigenvalues, eigenvectors, phi, phi_tilde, V_tilde_inv, D_tilde = compute_pod(profiles, nprofile, nmode)

    # print('Differenza S_mean pct:', (S_mean - dlr.S_mean)/np.mean(dlr.S_mean)*100)
    # print('Differenza S:', np.mean(S - dlr.S, axis=1))

    print('profile POD:\n', profiles[0][:,0])
    print('profile DLR:\n', dlr.profiles[0][:,0])

    print('S_mean POD: ', S_mean.shape)
    print('S_mean DLR: ', dlr.S_mean.shape)

    l_bound, u_bound = dlr.get_bound()
    print('Lower bound_DLR:', l_bound)
    l_bound = np.array([min(v) for v in V_tilde_inv])
    print('Lower bound_POD:', l_bound)

    print('Upper bound_DLR:', u_bound)
    u_bound = np.array([max(v) for v in V_tilde_inv])
    print('Upper bound_DLR:', u_bound)

# ============================================================================
# END POD-SPECIFIC FUNCTIONS
# ============================================================================

def plot_sensitivity(design_sensitivity: List[dict], input_file: str, directory: str, baseline_file: str) -> None:
    axis_limits = {
    'LE': {'xlim': (-0.001, 0.01), 'ylim': (-0.001, 0.007)},  # Leading Edge parameters
    'TE': {'xlim': (0.05, 0.07), 'ylim': (0.015, 0.025)}     # Trailing Edge parameters
    }

    input_dir = os.path.dirname(input_file) + '/' if os.path.dirname(input_file) else ''

    os.system(f'cd {input_dir} && /home/mciarlatani/bin/BladeGenerator.exe')
    header, params = read_progen_input(input_file)

    for dicts in design_sensitivity:
        folder_name = '+'.join(dicts.keys())
        print(f'Generating designs for: {folder_name}')
        generate_design_variants(dicts, header, params, 5, os.path.join(directory, folder_name), seed=123)

    # Find and process all test directories
    directories = []
    if os.path.exists(directory):
        for folder in os.listdir(directory):
            folder_path = os.path.join(directory, folder)
            if os.path.isdir(folder_path) and folder != 'Baseline':
                for subdir in os.listdir(folder_path):
                    subdir_path = os.path.join(folder_path, subdir)
                    if os.path.isdir(subdir_path) and subdir.startswith('temp'):
                        directories.append(subdir_path)
    
    print(f'Found {len(directories)} test directories to process')
    
    # Run BladeGenerator for each test directory in parallel
    if directories:
        print(f'Running BladeGenerator in parallel using {NCPU} workers...')
        
        blade_args = [(profile, '/home/mciarlatani/bin/BladeGenerator.exe') for profile in directories]
        
        with Pool(processes=NCPU) as pool:
            results = pool.map(run_blade_generator_parallel, blade_args)
            
    else:
        print('No test directories found to process')

    # Load baseline data
    baseline_data = None
    if os.path.exists(baseline_file):
        baseline_data = (np.genfromtxt(baseline_file, skip_header=2, delimiter=' ').T.tolist())[:-1]
        print('Baseline data loaded')

    # Group test data by parameter
    parameter_groups = {}
    for profile in directories:
        param_name = profile.split(os.sep)[2]  # Extract parameter name (e.g., 'x2SS' from 'Test_Sensitivity/sensitivity/x2SS/temp3')
        test_name = os.path.basename(profile)
        output_file = os.path.join(profile, 'Output', 'Profile_mergedGestaf.tec')
        
        if os.path.exists(output_file):
            try:
                m, theta = extract_data_from_tec(output_file)
                if param_name not in parameter_groups:
                    parameter_groups[param_name] = []
                parameter_groups[param_name].append((test_name, m, theta))
            except Exception as e:
                print(f'Warning: Could not extract data from {output_file}: {e}')

    # Create plots with additional subplot for random profiles
    if not parameter_groups or baseline_data is None:
        print("No data available for plotting")
    else:
        n_groups = len(parameter_groups)
        # Add one more slot for random profiles subplot
        total_plots = n_groups + 1
        n_cols = min(4, total_plots)
        n_rows = (total_plots + n_cols - 1) // n_cols
        
        # Calculate figure size with 16:9 aspect ratio
        fig_width = 16
        fig_height = 9
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        if total_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else list(axes)

        baseline_m, baseline_theta = baseline_data
        _, baseline_entries = read_progen_input(input_file)

        for idx, (param_name, test_data) in enumerate(parameter_groups.items()):
            ax = axes[idx]
            param_keys = param_name.split('+')
            
            # Create baseline label with parameter values
            baseline_values = []
            for key in param_keys:
                value = get_parameter_value(baseline_entries, key)
                try:
                    formatted_value = str(np.round(float(value), 5))
                except (ValueError, TypeError):
                    formatted_value = str(value)
                baseline_values.append(f"{key}={formatted_value}")
            
            baseline_label = f"Baseline ({', '.join(baseline_values)})"
            ax.plot(baseline_m, baseline_theta, 'k-', linewidth=3, label=baseline_label, alpha=0.9)
            
            # Collect parameter values for colormap
            param_values = []
            plot_data = []
            
            for test_name, m, theta in test_data:
                # Find corresponding directory
                test_dir = next((d for d in directories if test_name in d and param_name in d), None)
                if test_dir:
                    progen_path = os.path.join(test_dir, 'progen.input')
                    try:
                        _, entries = read_progen_input(progen_path)
                        # Use first parameter for coloring
                        value = get_parameter_value(entries, param_keys[0])
                        param_values.append(float(value))
                        plot_data.append((test_name, m, theta))
                    except (ValueError, Exception) as e:
                        print(f'Warning: Could not read parameters from {progen_path}: {e}')
            
            # Plot test variants with colormap
            if param_values and plot_data:
                vmin, vmax = min(param_values), max(param_values)
                if vmax > vmin:
                    norm = plt.Normalize(vmin=vmin, vmax=vmax)
                    cmap = plt.cm.viridis
                    
                    for (test_name, m, theta), param_val in zip(plot_data, param_values):
                        color = cmap(norm(param_val))
                        ax.plot(m, theta, color=color, linestyle='--', linewidth=2, alpha=0.8)
                    
                    # Add colorbar with custom ticks
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
                    cbar.set_label(param_keys[0])
                    
                    mean_val = (vmin + vmax) / 2
                    cbar.set_ticks([np.round(vmin,5), np.round(mean_val,5), np.round(vmax,5)])
                else:
                    # All values are the same
                    for test_name, m, theta in plot_data:
                        ax.plot(m, theta, 'b--', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('m')
            ax.set_ylabel('theta')
            ax.set_title(f'Parameter: {param_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Remove tick labels
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            # Set axis limits based on parameter key content
            for key_pattern, limits in axis_limits.items():
                if key_pattern in param_name:
                    ax.set_xlim(limits['xlim'])
                    ax.set_ylim(limits['ylim'])
                    break

        # Generate and plot random profiles in the last subplot
        ax_random = axes[len(parameter_groups)]
        
        # Generate N=10 random profiles using uniform sampling
        N_random = 5
        print(f'Generating {N_random} random profiles...')
        
        # Create combined bounds dictionary from design_sensitivity
        combined_bounds = {}
        for param_dict in design_sensitivity:
            combined_bounds.update(param_dict)
        
        # Generate random profiles using unified function
        # Note: no interpolation for sensitivity random profiles
        random_profiles = generate_and_simulate(combined_bounds, header, params, N_random, 
                                            os.path.join(directory, 'random_profiles'), 
                                            interpolate, 
                                            '/home/mciarlatani/bin/BladeGenerator.exe', 
                                            seed=42, max_workers=min(50, cpu_count()),
                                            sampling_mode='random')
        
        # Plot baseline
        ax_random.plot(baseline_m, baseline_theta, 'k-', linewidth=3, label='Baseline', alpha=0.9)
        
        # Plot random profiles
        for i, profile_data in enumerate(random_profiles):
            if profile_data.size > 0:
                ax_random.plot(profile_data[:, 0], profile_data[:, 1], linestyle='--', 
                                linewidth=1.5, alpha=0.7)
        
        ax_random.set_xlabel('m')
        ax_random.set_ylabel('theta')
        ax_random.set_title('Random Profiles (N=10)')
        ax_random.legend()
        ax_random.grid(True, alpha=0.3)
        
        # Remove tick labels
        ax_random.set_xticklabels([])
        ax_random.set_yticklabels([])
        
        # Hide remaining empty subplots
        for idx in range(len(parameter_groups) + 1, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig('Sensitivity_Profiles.png', dpi=300)
        plt.show()

# ============================================================================
# CONSTRAINT-SPECIFIC FUNCTIONS
# ============================================================================

def compute_QoIs(profile, verbose = False):
    # LE and TE defined erroneously as min and max x-coord
    c, c_ax = get_chords(profile)
    upper, lower = split_profile(profile)

    camber_line, thmax, Xthmax, th_vec = get_camber_th(upper, lower, interpolate=True)
    th_over_c = thmax/c
    Xth_over_cax = Xthmax/c_ax

    area = abs(get_area(profile))
    area_over_c2 = area/c**2
    cog = get_cog(profile)
    Xcg_over_cax = cog[0] / c_ax
    O_le = get_valid_center(profile[:, 0], profile[:, 1], dmin=0.005 * c, dmax=1.4 * 0.005 * c, le=True)
    O_te = get_valid_center(profile[:, 0], profile[:, 1], dmin=0.005 * c, dmax=1.4 * 0.005 * c, le=False)
    le_cond = 1 if O_le is None else 0
    te_cond = 1 if O_te is None else 0

    if verbose:
        print(f"baseline chord = {c} m, baseline axial chord = {c_ax} m")
        print(f"baseline th_max = {thmax} m, "
            f"Xth_max {Xthmax} m, "
            f"th_max/c = {th_over_c}, "
            f"Xth_max/c_ax = {Xth_over_cax}")
        print(f"baseline area = {area} m2, baseline area/c^2 = {area_over_c2}")
        print(f"baseline area = {area} m2, baseline X_cg/c_ax = {Xcg_over_cax}")
    le_circle = get_circle(O_le, 0.005 * c) if O_le is not None else np.array([])
    te_circle = get_circle(O_te, 0.005 * c) if O_te is not None else np.array([])
    # plot_sides(upper, lower, camber_line, le_circle, te_circle, th_vec, False)
    return(th_over_c, Xth_over_cax, area_over_c2, Xcg_over_cax, le_cond, te_cond)

def process_profile_parallel(args):
    """Worker function for parallel processing of profiles"""
    i, profile, bsl_th_over_c, bsl_Xth_over_cax, bsl_area_over_c2, bsl_Xcg_over_cax, bsl_le_cond, bsl_te_cond = args
        
    th_over_c, Xth_over_cax, area_over_c2, Xcg_over_cax, le_cond, te_cond = compute_QoIs(profile, False)
    
    th_cond = abs(th_over_c - bsl_th_over_c) / bsl_th_over_c - 0.3
    Xth_cond = abs(Xth_over_cax - bsl_Xth_over_cax) / bsl_Xth_over_cax - 0.2
    area_cond = abs(area_over_c2 - bsl_area_over_c2) / bsl_area_over_c2 - 0.2
    cog_cond = abs(Xcg_over_cax - bsl_Xcg_over_cax) / bsl_Xcg_over_cax - 0.2

    # Check violations
    th_violation = th_cond > 0
    Xth_violation = Xth_cond > 0    
    area_violation = area_cond > 0
    cog_violation = cog_cond > 0
    le_violation = le_cond > 0
    te_violation = te_cond > 0

    # Check if profile violates any constraint
    profile_violates = th_cond >= 0 or Xth_cond >= 0 or area_cond >= 0 or cog_cond >= 0

    return (i, th_violation, Xth_violation, area_violation, cog_violation, le_violation, te_violation, profile_violates)

def run_constraint_verification(design_sensitivity, input_file, baseline_file, output_dir='sensitivity', 
                                nprofile=1000, nmode=5, n_processes=None,interp=True, mode='STD', seed=123):
    """
    Run constraint verification analysis on generated profiles.
    
    Args:
        design_sensitivity: list of parameter bounds dicts
        input_file: progen input template filename
        baseline_file: path to baseline profile file
        output_dir: directory for output files
        nprofile: number of profiles to generate
        nmode: number of POD modes (if mode='POD')
        n_processes: number of parallel processes (None = auto)
        mode: 'POD' or 'STD' (standard)
        seed: random seed
    """
    
    random.seed(seed)
    np.random.seed(seed)
    header, params = read_progen_input(input_file)
    
    baseline = np.genfromtxt(baseline_file, skip_header=2, skip_footer=1, delimiter=' ')
    bsl_th_over_c, bsl_Xth_over_cax, bsl_area_over_c2, bsl_Xcg_over_cax, le_cond, te_cond = compute_QoIs(baseline, False)
    
    print(f"Using {NCPU} processes for parallel simulation")
    
    if mode == 'STD':
    
        # Generate and simulate profiles
        profiles = generate_and_simulate(design_sensitivity, header, params, nprofile, output_dir, 
                                    interpolate, '/home/mciarlatani/bin/BladeGenerator.exe', seed, NCPU,
                                    sampling_mode='mf')
    
    elif mode == 'POD':
        
        # Process based on mode
        Delta = np.zeros((nmode, nprofile))

        profiles = []
        params_dict = {}
        for i, param in enumerate(design_sensitivity):
            for key, value in param.items():
                params_dict[key] = value
                
        Path(output_dir).mkdir(exist_ok=True)
        os.system('cp ' + input_file + ' ' + output_dir + '/progen.input '+output_dir)
        dlr = DLR_POD_2D(baseline_file,  output_dir, '/home/mciarlatani/bin/BladeGenerator.exe', params_dict, nmode, nprofile, seed, perturb_POD=perturb)
        
        print(f"phi_tilde shape: {dlr.phi_tilde.shape}")
        l_bound = np.array([min(v) for v in dlr.V_tilde_inv])
        u_bound = np.array([max(v) for v in dlr.V_tilde_inv])
        print(f"Latent space bounds: Lower={l_bound}, Upper={u_bound}")
        Delta = sample_ellipsoid(dlr.V_tilde_inv, 0.90, nprofile)
        # for i in range(len(u_bound)):
        #     Delta[i, :] = rng.uniform(float(l_bound[i]), float(u_bound[i]), nprofile)

        modes = np.matmul(dlr.phi_tilde, Delta)

        for perturbation in modes.T:
            profiles.append(np.column_stack((dlr.pts[:,0],perturbation+dlr.S_mean)))

        for i in range(10):
            plt.plot(profiles[i][:,0], profiles[i][:,1], label=f'Profile {i+1}')
        plt.legend()
        plt.show()
    
    # Parallel processing of profiles
    print(f"Processing {len(profiles)} profiles using {NCPU} workers...")
    
    # Prepare arguments for parallel processing
    profile_args = [(i, profiles[i], bsl_th_over_c, bsl_Xth_over_cax, bsl_area_over_c2, bsl_Xcg_over_cax, le_cond, te_cond)
                    for i in range(len(profiles))]
    
    # Process profiles in parallel
    with Pool(processes=NCPU) as pool:
        results = pool.map(process_profile_parallel, profile_args)
    
    # Initialize counters
    contTh   = 0
    contXth  = 0
    contArea = 0
    contCog  = 0
    contrLE  = 0 
    contrTE  = 0
    cont     = 0
    profile_idx = []
    # Process results
    for i, th_violation, Xth_violation, area_violation, cog_violation, le_violation, te_violation, profile_violates in results:
        
        if th_violation:
            contTh += 1
        if Xth_violation:
            contXth += 1
        if area_violation:
            contArea += 1
        if cog_violation:
            contCog += 1
        # if le_violation:
        #     contrLE += 1
        # if te_violation:
        #     contrTE += 1
        
        if profile_violates:
            profile_idx.append(i)
            cont += 1
    
    # Save failed profiles to file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "failed_profiles.txt"), "w") as out:
        out.write("\n".join([str(idx) for idx in profile_idx]))
    
    # Print summary
    print("\n" + "=" * 60)
    print("CONSTRAINT VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total profiles violating th_max/c:     {contTh}/{nprofile} ({100*contTh/nprofile:.1f}%)")
    print(f"Total profiles violating Xth_max/c_ax: {contXth}/{nprofile} ({100*contXth/nprofile:.1f}%)")
    print(f"Total profiles violating area/c^2:     {contArea}/{nprofile} ({100*contArea/nprofile:.1f}%)")
    print(f"Total profiles violating X_cg/c_ax:    {contCog}/{nprofile} ({100*contCog/nprofile:.1f}%)")
    print(f"Total profiles violating r_LE/c_ax:    {contrLE}/{nprofile} ({100*contrLE/nprofile:.1f}%)")
    print(f"Total profiles violating r_TE/c_ax:    {contrTE}/{nprofile} ({100*contrTE/nprofile:.1f}%)")
    print(f"Total profiles violating constraints:  {cont}/{nprofile} ({100*cont/nprofile:.1f}%)")
    print(f"Failed profile indices saved to: {os.path.join(output_dir, 'failed_profiles.txt')}")
    print("=" * 60)

    return profile_idx

def plot_failed_profiles_histogram(failed_profiles_file: str, directory: str, design_sensitivity: List[dict], nprofiles: int):
    """
    Analizza i profili falliti e crea istogrammi per ogni parametro di design.
    
    Args:
        failed_profiles_file: nome del file con gli indici dei profili falliti
        directory: directory base dove si trovano le cartelle tempX
        design_sensitivity: lista di dizionari con i parametri di design
        nprofiles: numero totale di profili generati
    """
    # Estrai le chiavi dei parametri da design_sensitivity
    param_keys = []
    for param_dict in design_sensitivity:
        param_keys.extend(param_dict.keys())
    
    # Leggi gli indici dei profili falliti
    failed_file_path = os.path.join(directory, failed_profiles_file)
    with open(failed_file_path, 'r') as f:
        failed_indices = set(int(line.strip()) for line in f if line.strip())
    
    print(f"Found {len(failed_indices)} failed profiles")
    
    # Raccogli i valori dei parametri per tutti i profili
    all_params = {key: [] for key in param_keys}
    failed_params = {key: [] for key in param_keys}
    
    print(f"Reading parameter values from {nprofiles} directories...")
    for i in range(nprofiles):
        temp_dir = os.path.join(directory, f'temp{i}')
        progen_path = os.path.join(temp_dir, 'progen.input')

        _, entries = read_progen_input(progen_path)
        
        # Estrai valori per ogni parametro
        for key in param_keys:
            value_str = get_parameter_value(entries, key)
            if value_str != 'N/A':
                value = float(value_str)
                all_params[key].append(value)
                
                # Se questo profilo è fallito, aggiungi anche a failed_params
                if i in failed_indices:
                    failed_params[key].append(value)
    
    # Crea i subplot
    n_params = len(param_keys)
    n_cols = min(4, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    # Calculate figure size with 16:9 aspect ratio
    fig_width = 16
    fig_height = 9
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else list(axes)
    
    # Crea un istogramma per ogni parametro
    for idx, key in enumerate(param_keys):
        ax = axes[idx]
        
        if not all_params[key]:
            ax.text(0.5, 0.5, f'No data for {key}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(key)
            continue
        
        # Determina i bins per l'istogramma
        all_values = np.array(all_params[key])
        failed_values = np.array(failed_params[key]) if failed_params[key] else np.array([])
        
        # Usa 20 bins o meno se ci sono pochi dati
        n_bins = min(20, len(set(all_values)))
        
        # Crea i bins uniformi basati su tutti i valori
        bins = np.linspace(all_values.min(), all_values.max(), n_bins + 1)
        
        # Calcola gli istogrammi
        all_counts, bin_edges = np.histogram(all_values, bins=bins)
        failed_counts, _ = np.histogram(failed_values, bins=bins) if len(failed_values) > 0 else (np.zeros(len(bins)-1), bins)
        
        # Calcola le posizioni delle barre
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Offset per le barre rosse e blu
        offset = bin_width * 0.2
        
        # Plotta le barre blu (tutti i profili)
        ax.bar(bin_centers - offset, all_counts, width=bin_width*0.4, 
              color='blue', alpha=0.7, label='All profiles', edgecolor='black')
        
        # Plotta le barre rosse (profili falliti)
        ax.bar(bin_centers + offset, failed_counts, width=bin_width*0.4, 
              color='red', alpha=0.7, label='Failed profiles', edgecolor='black')
        
        # Configurazione del plot
        ax.set_title(f'{key}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # Nascondi subplot vuoti
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Salva il plot
    output_path = os.path.join('Failed_Profiles_Histogram.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Histogram saved to: {output_path}")
    
    plt.show()
    
    # # Stampa statistiche riassuntive
    # print("\n" + "=" * 60)
    # print("PARAMETER DISTRIBUTION SUMMARY")
    # print("=" * 60)
    # for key in param_keys:
    #     all_vals = np.array(all_params[key])
    #     failed_vals = np.array(failed_params[key]) if failed_params[key] else np.array([])
        
    #     if len(all_vals) > 0:
    #         print(f"\n{key}:")
    #         print(f"  Total samples: {len(all_vals)}")
    #         print(f"  Range: [{all_vals.min():.6f}, {all_vals.max():.6f}]")
    #         print(f"  Mean: {all_vals.mean():.6f} ± {all_vals.std():.6f}")
            
    #         if len(failed_vals) > 0:
    #             print(f"  Failed samples: {len(failed_vals)}")
    #             print(f"  Failed range: [{failed_vals.min():.6f}, {failed_vals.max():.6f}]")
    #             print(f"  Failed mean: {failed_vals.mean():.6f} ± {failed_vals.std():.6f}")
    #             print(f"  Failure rate: {len(failed_vals)/len(all_vals)*100:.1f}%")
    # print("=" * 60)

# ============================================================================
# END CONSTRAINT-SPECIFIC FUNCTIONS
# ============================================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--sensitivity", action='store_true', help="Sensitivity analysis plots")
    parser.add_argument("-p", "--pod", action='store_true', help="POD analysis plots")
    parser.add_argument("-c", "--constraints", action='store_true', help="Verify constraints")
    parser.add_argument("-v", "--verbose", type=int, help="logger verbosity level", default=3)
    
    args = parser.parse_args()
    
    directory = 'POD_Dataset_RANS'
    baseline_file = 'Baseline/Output/Profile_mergedGestaf.tec'
    baseline_profile = './LRN-Bladegen.dat'
    perturb = 'Baseline'  # 'TrueMean' or 'Baseline'
    plt.close('all')

    # input_file = 'Baseline_DLR/progen.input'
    # baseline_file = 'Baseline_DLR/Output/Profile_mergedGestaf.tec'
    # design_sensitivity = [
    #     {'BetaLE':[118,129]},
    #     {'BetaTE':[60,66]},
    #     {'BetaST':[88,92]},
    #     {'x2SS':[0.013,0.033]},
    #     {'x3SS':[0.31,0.40]},
    #     {'y3SS':[0.192,0.21]},
    #     {'x4SS':[0.818,0.86]},
    #     {'m2DS':[0.127,0.18]},
    #     {'d2DS':[0.035,0.048]},
    #     {'d3DS':[0.022,0.05]},
    #     {'d4DS':[0.014,0.017]},
    #     {'Dmax_approx':[0.8,1.0]},
    #     {'rTE':[0.00495,0.0075]}]
    
    # NCPU = 90

    input_file = 'Baseline_DLR/progen.input'
    baseline_file = 'Baseline_DLR/Output/Profile_mergedGestaf.tec'
    design_sensitivity = [{'BetaLE':[110.6727,132.6727]},
        {'BetaTE':[58.0394,70.0394]},
        {'BetaST':[88,92]},
        {'x2SS':[0.0122990,0.0522990]},
        {'x3SS':[0.31,0.40]},
        {'y3SS':[0.177428,0.217428]},
        {'x4SS':[0.81321,0.89321]},
        {'m2DS':[0.11946,0.239465]},
        {'d2DS':[0.035454,0.055454]},
        {'d3DS':[0.020002,0.080002]},
        {'d4DS':[0.0013360,0.0017360]},
        {'Dmax_approx':[0.8,1.2]},
        {'rTE':[0.0040,0.01136]}]
    
    nProfiles = 1000
    nModes = 5
    NCPU = 90
    interpolate = {'interpolate': True,'original_file': baseline_profile}

    constraint_mode = 'POD'  # 'POD' or 'STD'
    
    # Constraint-specific parameters
    Path(directory).mkdir(exist_ok=True)
    
    if args.sensitivity:
        print("Running sensitivity analysis...")
        plot_sensitivity(design_sensitivity, input_file, os.path.join(directory,'sensitivity'), baseline_profile)

    if args.pod:
        print("Running POD analysis...")
        # Set multiprocessing start method for compatibility
        mp.set_start_method('spawn', force=True) if mp.get_start_method() != 'spawn' else None
        run_pod_analysis_DLR(design_sensitivity, input_file, baseline_profile, os.path.join(directory), nprofile=nProfiles, nmode=nModes, n_processes=None)
    
    if args.constraints:
        print("Running constraint verification...")
        # # Set multiprocessing start method for compatibility
        mp.set_start_method('spawn', force=True) if mp.get_start_method() != 'spawn' else None
        run_constraint_verification(
            design_sensitivity=design_sensitivity,
            input_file=input_file,
            baseline_file=baseline_profile,
            output_dir=os.path.join(directory),
            nprofile=nProfiles,
            nmode=nModes,
            n_processes=None,
            interp=False,
            mode=constraint_mode,
            seed=123
        )

        plot_failed_profiles_histogram('failed_profiles.txt',os.path.join(directory),design_sensitivity,nProfiles)
        # plot_failed_profiles(failed_profiles_file='failed_profiles.txt', directory=os.path.join(directory,'constraints'), n_samples=15)

    if not args.sensitivity and not args.pod and not args.constraints:
        print("No analysis specified. Use -s for sensitivity, -p for POD, or -c for constraints.")
        parser.print_help()