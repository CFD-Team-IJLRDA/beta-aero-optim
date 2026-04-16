import numpy as np
import matplotlib.pyplot as plt
import re

def parse_geo_file(filepath):
    """
    Parse un file .geo_unrolled e estrae tutti i punti definiti come Point(I) = {x, y, z, lc}.
    
    Args:
        filepath: percorso del file .geo_unrolled
    
    Returns:
        dict: dizionario con indice -> (x, y, z) come chiave-valore
    """
    points = {}
    
    # Pattern regex per matchare: Point(I) = {x, y, z, lc};
    # Cattura: indice I, coordinate x, y, z
    pattern = r'Point\((\d+)\)\s*=\s*\{\s*([+-]?[\d.eE+-]+)\s*,\s*([+-]?[\d.eE+-]+)\s*,\s*([+-]?[\d.eE+-]+)\s*,'
    
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                idx = int(match.group(1))
                x = float(match.group(2))
                y = float(match.group(3))
                z = float(match.group(4))
                points[idx] = (x, y, z)
    
    return points

def plot_geo_points(file1, file2):
    """
    Legge due file .geo_unrolled e plotta i punti con i loro indici.
    
    Args:
        file1: percorso del primo file .geo_unrolled
        file2: percorso del secondo file .geo_unrolled
    """
    # Parse i due file
    points1 = parse_geo_file(file1)
    points2 = parse_geo_file(file2)
    
    print(f"File 1: {len(points1)} punti trovati")
    print(f"File 2: {len(points2)} punti trovati")
    
    # Crea figura con due subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot file 1
    if points1:
        indices1 = list(points1.keys())
        x1 = [points1[i][0] for i in indices1]
        y1 = [points1[i][1] for i in indices1]
        
        ax1.scatter(x1, y1, c='tab:blue', s=30, alpha=0.7, edgecolors='black', linewidths=0.5)
        
        # Aggiungi indici accanto ai punti
        for idx in indices1:
            x, y, z = points1[idx]
            ax1.text(x, y, str(idx), fontsize=7, ha='right', va='bottom', color='red', alpha=0.8)
        
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('y', fontsize=12)
        ax1.set_title(f'File 1: {file1.split("/")[-1]}\n({len(points1)} points)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
    else:
        ax1.text(0.5, 0.5, 'Nessun punto trovato', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title('File 1 Points')
    
    # Plot file 2
    if points2:
        indices2 = list(points2.keys())
        x2 = [points2[i][0] for i in indices2]
        y2 = [points2[i][1] for i in indices2]
        
        ax2.scatter(x2, y2, c='tab:orange', s=30, alpha=0.7, edgecolors='black', linewidths=0.5)
        
        # Aggiungi indici accanto ai punti
        for idx in indices2:
            x, y, z = points2[idx]
            ax2.text(x, y, str(idx), fontsize=7, ha='right', va='bottom', color='blue', alpha=0.8)
        
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('y', fontsize=12)
        ax2.set_title(f'File 2: {file2.split("/")[-1]}\n({len(points2)} points)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
    else:
        ax2.text(0.5, 0.5, 'Nessun punto trovato', ha='center', va='center', transform=ax2.transAxes)
    
    ax2.set_title('File 2 Points')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # # Percorsi dei file
    # file1 = '/home/mciarlatani/GPROptimization/beta-aero-optim/examples/test-Cascade/beta_geom_constrained/FFD_mf/lf_doe/MESH/lrn_cascade_g0_c6.geo_unrolled'
    # file2 = '/home/mciarlatani/GPROptimization/beta-aero-optim/examples/test-Cascade/beta_geom_constrained/DLR_mf/lf_doe/MESH/lrn_cascade_g0_c6.geo_unrolled'
    
    # Percorsi dei file
    file1 = '/home/mciarlatani/Hilbert/beta-aero-optim/Optimization/FailedSims/FailedADP/ogv1c_g1_c14.geo_unrolled'
    file2 = '/home/mciarlatani/Hilbert/beta-aero-optim/Optimization/FailedSims/Sim2/ogv1c_g3_c0.geo_unrolled'

    # Plot i punti
    plot_geo_points(file1, file2)
