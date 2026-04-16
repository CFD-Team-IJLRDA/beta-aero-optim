import numpy as np
import matplotlib.pyplot as plt


def calculate_curve_length(pts, indices):
    """
    Calcola la lunghezza della curva passante per i punti tra due indici.
    
    Args:
        pts: Array con coordinate dei punti (Nx2 o Nx3)
        indices: Lista con [indice_inizio, indice_fine], estremo destro escluso
                 Supporta anche indici che attraversano il punto 0 (es. [270, 8])
    
    Returns:
        float: Lunghezza totale della curva
    
    Example:
        >>> length = calculate_curve_length(pts1, [8, 15])  # Da punto 8 a 14
        >>> length = calculate_curve_length(pts1, [270, 8])  # Da punto 270 a 7 (passando per 0)
    """
    idx_start, idx_end = indices
    n_points = len(pts)
    
    # Gestisce il caso in cui gli indici attraversano il punto 0
    if idx_start < idx_end:
        # Caso normale: da idx_start a idx_end-1
        point_indices = list(range(idx_start, idx_end))
    else:
        # Caso che attraversa lo zero: da idx_start alla fine, poi da 0 a idx_end-1
        point_indices = list(range(idx_start, n_points)) + list(range(0, idx_end))
    
    # Calcola la lunghezza totale sommando le distanze euclidee tra punti consecutivi
    total_length = 0.0
    for i in range(len(point_indices) - 1):
        print(point_indices[i], point_indices[i+1])
        p1 = pts[point_indices[i], :2]  # Prime 2 coordinate (x, y)
        p2 = pts[point_indices[i + 1], :2]
        distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        total_length += distance
    
    print(f"Lunghezza curva da indice {idx_start} a {idx_end-1}: {total_length:.6f}")
    print(f"Numero di segmenti: {len(point_indices) - 1}")
    
    return total_length


def plot_points_with_indices(pts1, pts2):
    """
    Plotta i punti di pts1 e pts2 (prime due colonne) con gli indici accanto ad ogni punto.
    
    Args:
        pts1: Array con coordinate dei punti del primo profilo
        pts2: Array con coordinate dei punti del secondo profilo
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot pts1
    ax1.plot(pts1[:, 0], pts1[:, 1], 'o-', markersize=4, linewidth=1, label='Profile 1')
    ax1.plot(pts2[:, 0], pts2[:, 1]-delta*np.max(pts1[:, 1]), 's-', markersize=4, linewidth=1, label='Profile 2', color='tab:orange')

    for idx, (x, y) in enumerate(pts1[:, :2]):
        ax1.text(x, y, str(idx+1), fontsize=8, ha='right', va='bottom', color='red')
    for idx, (x, y) in enumerate(pts2[:, :2]):
        ax1.text(x, y-delta*np.max(pts1[:, 1]), str(idx+1), fontsize=8, ha='right', va='bottom', color='blue')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Profile 1 ({len(pts1)} points)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend()
    
    # Plot pts2
    # ax2.plot(pts2[:, 0], pts2[:, 1], 's-', markersize=4, linewidth=1, label='Profile 2', color='tab:orange')
    # for idx, (x, y) in enumerate(pts2[:, :2]):
    #     ax2.text(x, y, str(idx+1), fontsize=8, ha='right', va='bottom', color='blue')
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # ax2.set_title(f'Profile 2 ({len(pts2)} points)')
    # ax2.grid(True, alpha=0.3)
    # ax2.axis('equal')
    # ax2.legend()
    
    plt.tight_layout()
    plt.show()

def reorder_blade(pts) -> list[list[float]]:
    """
    **Returns** the blade profile after reordering in clockwise direction.
    """
    d = np.sqrt([x**2 + y**2 for x, y, _ in pts])
    start = np.argmin(d)

    try:
        [y0, y1] = [pts[start][1], pts[start+1][1]]
    except:
        [y0, y1] = [pts[start-1][1], pts[start][1]]

    # Ordine orario: y diminuisce (y1 < y0), altrimenti ribalta la lista
    if y1 > y0:
        return [[p[0], p[1], p[2]] for p in pts[:start] + pts[start:]]
    else:
        return [[p[0], p[1], p[2]] for p in (pts[:start] + pts[start:])[::-1]]
    
# file1 = '/home/mciarlatani/Hilbert/beta-aero-optim/LFWorkingOpt/test_musicaa_mf_DLR/output_mf/lf_doe/FFD/ogv1c_g0_c0.dat'
# file2 = '../../examples/LRN-CASCADE/data/lrn_cascade.dat'
    
file1 = '/home/mciarlatani/Hilbert/beta-aero-optim/Optimization/RANS_baseline/output/FFD/ogv1c_g0_c0.dat'
file2 = '/home/mciarlatani/Hilbert/beta-aero-optim/Optimization/RANS_baseline/ogv1c.dat'
    
# file1 = '/home/mciarlatani/Hilbert/beta-aero-optim/Optimization/test_musicaa_mf_DLR/output_mf/lf_doe/FFD/' \
# 'ogv1c_g0_c3.dat'
# file2 = '/home/mciarlatani/Hilbert/beta-aero-optim/Optimization/test_musicaa_mf_DLR/something_FFD/lf_doe/FFD/' \
# 'ogv1c_g0_c3.dat'

delta = 0.0

pts1 = (np.genfromtxt(file1, skip_header=2)).tolist()

if len(pts1[0]) == 2:
    pts1 = [p + [0.0] for p in pts1]
pts2 = np.genfromtxt(file2, skip_header=2).tolist()
if len(pts2[0]) == 2:
    pts2 = [p + [0.0] for p in pts2]

# print(pts2)
pts1 = np.asarray(reorder_blade(pts1))
pts2 = np.asarray(reorder_blade(pts2))

print(pts2-pts1)
input()
# print(pts1)

# Chiama la funzione per visualizzare i profili
plot_points_with_indices(pts1, pts2)

calculate_curve_length(pts1, [0, 12])  # Esempio di utilizzo
calculate_curve_length(pts1, [267, 1])  # Esempio di utilizzo