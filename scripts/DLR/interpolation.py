import os
import random
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy.stats import qmc
from scipy.interpolate import interp1d
import multiprocessing as mp

from aero_optim.ffd.ffd import FFD_2D
from aero_optim.geom    import (get_area, get_camber_th, get_chords, get_circle, get_circle_centers,
                             get_cog, get_radius_violation, split_profile, plot_profile, plot_sides)
from aero_optim.mf_sm.mf_models import get_sampler

import matplotlib.pyplot as plt
import numpy as np
import os
from collections import OrderedDict
from typing import Tuple, List, Union
import numpy.random as rng

OD = OrderedDict  # Type alias

def interp_surface(surface, nodes, kind='linear'):
    """
    Interpola una superficie (dorso o ventre) sui nodi specificati.
    
    Args:
        surface: array 2D con coordinate (x, y) della superficie da interpolare
        nodes: array 1D con le coordinate x dei nodi target per l'interpolazione
        kind: tipo di interpolazione ('linear', 'cubic', etc.)
    
    Returns:
        array 2D con coordinate (x, y) interpolate sui nodi specificati
    """
    # Ensure array shape
    surface = np.asarray(surface)

    x = surface[:, 0]
    y = surface[:, 1]

    # Sort by x to ensure monotonic x for interpolation
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    f = interp1d(x_sorted, y_sorted, kind=kind, bounds_error=False, fill_value='extrapolate')
    return np.column_stack((nodes, f(nodes)))

def split_airfoil(prof):
        """Separa un profilo in dorso e ventre identificando i punti di cambio direzione x."""
        x_coords = prof[:, 0]
        x_diff = np.diff(x_coords)
        signs = np.sign(x_diff)
        sign_changes = np.diff(signs)
        change_indices = np.where(sign_changes != 0)[0]
        
        # Clean up change_indices per gestire cambi multipli vicini
        if len(change_indices) > 2:
            cleaned_indices = []
            used = np.zeros(len(change_indices), dtype=bool)
            
            for i, idx in enumerate(change_indices):
                if used[i]:
                    continue
                group = [j for j, other_idx in enumerate(change_indices) 
                        if abs(other_idx - idx) <= 3 and not used[j]]
                for j in group:
                    used[j] = True
                group_indices = [change_indices[j] for j in group]
                group_x_diffs = [abs(x_diff[gi]) for gi in group_indices]
                best_in_group = group_indices[np.argmin(group_x_diffs)]
                cleaned_indices.append(best_in_group)
            change_indices = np.array(cleaned_indices)
        
        # Assicurati di avere esattamente 2 punti di cambio (LE e TE)
        if len(change_indices) != 2:
            le_idx = np.argmin(x_coords)
            te_idx = np.argmax(x_coords)
            change_indices = np.array([le_idx-1, te_idx-1]) if le_idx > 0 and te_idx > 0 else np.array([0, len(x_diff)-2])
            change_indices = np.clip(change_indices, 0, len(x_diff)-1)
        
        assert len(change_indices) == 2, "Impossibile identificare LE e TE del profilo"
        
        split_indices = np.sort(change_indices + 1)
        
        # Separa in due zone
        zone_1 = prof[split_indices[0]+1:split_indices[1]+1, :].copy()
        zone_1 = zone_1[np.argsort(zone_1[:, 0], kind='stable')][:-1]
        
        zone_2 = np.vstack((prof[:split_indices[0]+1, :], prof[split_indices[1]+1:, :])).copy()
        zone_2 = zone_2[np.argsort(-zone_2[:, 0], kind='stable')][:-1]
        
        # Identifica dorso (upper) e ventre (lower) dalla media y
        if np.mean(zone_2[:,1]) > np.mean(zone_1[:,1]):
            lower, upper = zone_1, zone_2
        else:
            upper, lower = zone_1, zone_2
        
        return upper, lower, split_indices

def interpolate_profile(profile, original_profile, kind='linear'):
    """
    Interpola un profilo generico usando i nodi definiti da un profilo originale.
    Separa entrambi i profili in dorso e ventre, poi interpola ciascuna superficie
    sui nodi corrispondenti del profilo originale.
    
    Args:
        profile: 2D array (x, y) del profilo da interpolare (nuovo profilo)
        original_profile: 2D array (x, y) del profilo che definisce i nodi di interpolazione
        kind: tipo di interpolazione ('linear', 'cubic', etc.)
        
    Returns:
        2D array con profilo interpolato che mantiene esattamente le coordinate x del profilo originale
    """
    
    # CASO SPECIALE: Se profile e original_profile sono lo stesso (auto-interpolazione)
    # restituisci direttamente una copia per evitare errori numerici
    
    # Calcola min e max del profilo originale
    x_min_orig = np.min(original_profile[:, 0])
    x_max_orig = np.max(original_profile[:, 0])
    
    # Scala il profilo nuovo per matchare il range del profilo originale
    x_min_new = np.min(profile[:, 0])
    x_max_new = np.max(profile[:, 0])
    
    # Crea una copia del profilo da scalare
    profile_scaled = profile.copy()
    
    # Scala le coordinate x del nuovo profilo per matchare il range dell'originale
    if x_max_new != x_min_new:  # Evita divisione per zero
        profile_scaled[:, 0] = (profile[:, 0] - x_min_new) / (x_max_new - x_min_new) * (x_max_orig - x_min_orig) + x_min_orig
    
    # Separa il profilo nuovo (da interpolare) dopo lo scaling
    new_upper, new_lower, _ = split_airfoil(profile_scaled)
    
    # Separa il profilo originale (che fornisce i nodi di interpolazione)
    orig_upper, orig_lower, orig_split_indices = split_airfoil(original_profile)
    
    # OTTIMIZZAZIONE: Crea le funzioni di interpolazione una sola volta invece che ad ogni iterazione
    # Prepara le superfici ordinate per x
    new_upper_sorted = new_upper[np.argsort(new_upper[:, 0])]
    new_lower_sorted = new_lower[np.argsort(new_lower[:, 0])]
    orig_upper_sorted = orig_upper[np.argsort(orig_upper[:, 0])]
    orig_lower_sorted = orig_lower[np.argsort(orig_lower[:, 0])]
    
    # Crea funzioni di interpolazione (una volta sola)
    f_new_upper = interp1d(new_upper_sorted[:, 0], new_upper_sorted[:, 1], kind=kind, bounds_error=False, fill_value='extrapolate')
    f_new_lower = interp1d(new_lower_sorted[:, 0], new_lower_sorted[:, 1], kind=kind, bounds_error=False, fill_value='extrapolate')
    f_orig_upper = interp1d(orig_upper_sorted[:, 0], orig_upper_sorted[:, 1], kind=kind, bounds_error=False, fill_value='extrapolate')
    f_orig_lower = interp1d(orig_lower_sorted[:, 0], orig_lower_sorted[:, 1], kind=kind, bounds_error=False, fill_value='extrapolate')
    
    # Crea un profilo interpolato con le stesse coordinate x del profilo originale
    interpolated = np.zeros_like(original_profile)
    interpolated[:, 0] = original_profile[:, 0]  # Copia esattamente le coordinate x
    
    # OTTIMIZZAZIONE: Interpola tutti i punti in una volta (vectorizzato)
    x_coords = original_profile[:, 0]
    y_coords_orig = original_profile[:, 1]
    
    # Calcola y su entrambe le superfici per tutti i punti
    y_upper_orig = f_orig_upper(x_coords)
    y_lower_orig = f_orig_lower(x_coords)
    y_upper_new = f_new_upper(x_coords)
    y_lower_new = f_new_lower(x_coords)
    
    # Determina per ogni punto se appartiene a dorso o ventre (operazione vectorizzata)
    dist_to_upper = np.abs(y_coords_orig - y_upper_orig)
    dist_to_lower = np.abs(y_coords_orig - y_lower_orig)
    is_upper = dist_to_upper < dist_to_lower
    
    # Assegna i valori interpolati (operazione vectorizzata)
    interpolated[:, 1] = np.where(is_upper, y_upper_new, y_lower_new)
    
    # Post-processing: correggi punti vicini al trailing edge che sono stati classificati male
    # OTTIMIZZAZIONE: Calcola distanze in modo vectorizzato
    distances = np.sqrt(np.diff(interpolated[:, 0])**2 + np.diff(interpolated[:, 1])**2)
    mean_distance = np.mean(distances)
    threshold = mean_distance * 0.05
    
    # Identifica coppie di punti consecutivi troppo vicini
    close_pairs = np.where(distances < threshold)[0]
    # print(len(close_pairs), " coppie di punti vicini trovate per correzione")

    # Se non ci sono coppie problematiche, restituisci subito
    if len(close_pairs) == 0:
        return interpolated
    
    # Filtra solo le coppie vicine agli split indices
    near_split_mask = np.array([
        (abs(idx - orig_split_indices[0]) < 3 or 
         abs(idx - orig_split_indices[1]) < 3 or
         abs(idx + 1 - orig_split_indices[0]) < 3 or 
         abs(idx + 1 - orig_split_indices[1]) < 3)
        for idx in close_pairs
    ])
    close_pairs = close_pairs[near_split_mask]
    
    # Se non ci sono coppie problematiche vicino agli split, restituisci subito
    if len(close_pairs) == 0:
        return interpolated
    
    # Per le coppie rimanenti, applica la correzione (necessario loop per has_intersection)
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    def has_intersection(prof, idx1, idx2):
        """Verifica se modificare i,j crea intersezioni con altri segmenti"""
        segs = [(idx1-1, idx1), (idx1, idx2), (idx2, idx2+1)]
        segs = [(a, b) for a, b in segs if 0 <= a < len(prof) and 0 <= b < len(prof)]
        for a, b in segs:
            p1, p2 = prof[a], prof[b]
            for k in range(len(prof) - 1):
                if abs(k - a) <= 1 or abs(k - b) <= 1:
                    continue
                p3, p4 = prof[k], prof[k + 1]
                if ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4):
                    return True
        return False
    
    # Precalcola tutti i valori y necessari per le correzioni (vectorizzato)
    x_vals = interpolated[close_pairs, 0]
    x_vals_next = interpolated[close_pairs + 1, 0]
    
    y_upper_vals = f_new_upper(x_vals)
    y_lower_vals = f_new_lower(x_vals)
    y_upper_vals_next = f_new_upper(x_vals_next)
    y_lower_vals_next = f_new_lower(x_vals_next)
    
    for pair_idx, idx in enumerate(close_pairs):
        i = idx
        j = idx + 1
        
        y_i_upper = y_upper_vals[pair_idx]
        y_i_lower = y_lower_vals[pair_idx]
        y_j_upper = y_upper_vals_next[pair_idx]
        y_j_lower = y_lower_vals_next[pair_idx]
        
        # Testa configurazioni alternative
        configs = [
            (y_i_upper, y_j_lower, abs(y_i_upper - y_j_lower)),
            (y_i_lower, y_j_upper, abs(y_i_lower - y_j_upper))
        ]
        
        current_dist = abs(interpolated[i, 1] - interpolated[j, 1])
        best_config = None
        best_dist = current_dist
        
        for yi, yj, dist in configs:
            test_prof = interpolated.copy()
            test_prof[i, 1], test_prof[j, 1] = yi, yj
            if not has_intersection(test_prof, i, j) and dist > best_dist * 1.5:
                best_config = (yi, yj)
                best_dist = dist
        
        if best_config:
            interpolated[i, 1], interpolated[j, 1] = best_config

    return interpolated

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

def main():
    """
    Funzione di test per l'interpolazione di profili sui nodi di un profilo originale.
    """
    original_file = '/home/mciarlatani/Hilbert/beta-aero-optim/Optimization/cascade_wolf_base/ogv1c.dat'
    original_profile = np.genfromtxt(original_file, skip_header=2, delimiter=' ')
    x, y = extract_data_from_tec('/home/mciarlatani/Hilbert/beta-aero-optim/LFWorkingOpt/test_musicaa_mf_DLR/POD_Dataset/temp201/Output/Profile_mergedGestaf.tec', zone_name="NURB_Profil_gestaf0")
    
    # Test 1: Auto-interpolazione (deve produrre risultato identico all'originale)
    print("=" * 60)
    print("TEST 1: Auto-interpolazione (ref = interpolated)")
    print("=" * 60)
    
    # ref = original_profile.copy()
    ref = (np.row_stack((x, y)).T) * 1000
    interpolated = interpolate_profile(ref, original_profile, kind='linear')

    plt.scatter(ref[:,0], ref[:,1], label='Original Profile', marker='o',facecolors='none', edgecolors='tab:green')
    plt.plot(interpolated[:,0], interpolated[:,1], label='Interpolated Profile', marker='x')
    for idx, (x, y) in enumerate(interpolated[:, :2]):
        plt.text(x, y, str(idx+1), fontsize=8, ha='right', va='bottom', color='blue')
    plt.legend()
    plt.show()
    plt.close('all')
    
    # # Verifica che siano identici
    # max_diff = np.max(np.abs(original_profile - interpolated))
    # print(f"Differenza massima tra originale e auto-interpolato: {max_diff:.10f}")
    
    # if max_diff < 1e-10:
    #     print("✓ AUTO-INTERPOLAZIONE CORRETTA: profili identici!")
    # else:
    #     print("✗ ERRORE: auto-interpolazione non produce risultato identico")
    #     print(f"Primo punto orig: {original_profile[0,:]} | interp: {interpolated[0,:]}")
    #     print(f"Ultimo punto orig: {original_profile[-1,:]} | interp: {interpolated[-1,:]}")
    
    # # Plot per visualizzazione
    # plt.figure(figsize=(14, 6))
    
    # plt.subplot(1, 2, 1)
    # plt.scatter(interpolated[:,0], interpolated[:,1], color='tab:green', marker='x', s=100, label='Interpolated', zorder=2)
    # plt.scatter(original_profile[:,0], original_profile[:,1], color='tab:blue', marker='o', facecolors='none', s=80, label='Original', zorder=1)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Auto-interpolazione: Original vs Interpolated')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.axis('equal')
    
    # plt.subplot(1, 2, 2)
    # diff = original_profile - interpolated
    # plt.scatter(original_profile[:,0], diff[:,1], color='red', marker='o', s=20)
    # plt.xlabel('x')
    # plt.ylabel('Δy (original - interpolated)')
    # plt.title(f'Differenza (max: {max_diff:.2e})')
    # plt.grid(True, alpha=0.3)
    # plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
    
    # plt.tight_layout()
    # plt.show()
    
    # print("\n" + "=" * 60)
    # print("TEST 2: Interpolazione di un nuovo profilo")
    # print("=" * 60)
    
    # Test 2: Interpolazione di un nuovo profilo (se necessario, decommenta)
    # x, y = extract_data_from_tec('path/to/file.tec')
    # new_profile = (np.row_stack((x, y)).T) * 1000
    # interpolated_new = interpolate_profile(new_profile, original_profile, kind='linear')
    # print(f"Profilo nuovo interpolato: {interpolated_new.shape}")




if __name__ == '__main__':
    # This guard is necessary for multiprocessing on some systems
    mp.set_start_method('spawn', force=True) if mp.get_start_method() != 'spawn' else None
    main()
