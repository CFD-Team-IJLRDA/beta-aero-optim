import dill as pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from pprint import pprint

# --- Valutazione e plot superficie modello finale ---
from mpl_toolkits.mplot3d import Axes3D

nModels = 13  # numero totale di modelli salvati (0..12)
directory = "output_mf_normal"

model_path = os.path.join(directory, f"model_{nModels-1}.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# # Mostra chiavi / attributi del modello appena caricato per diagnostica
# print("Loaded model:", model_path)
# print(model.models[0].y_hf_DOE)
# if isinstance(model, dict):
#     print("Model is a dict. Keys:")
#     pprint(list(model.keys()))
# else:
#     # mostra attributi pubblici
#     public_attrs = [a for a in dir(model) if not a.startswith('_')]
#     print("Model public attributes and methods:")
#     pprint(public_attrs)
#     # mostra __dict__ se presente
#     if hasattr(model, '__dict__'):
#         print("model.__dict__ keys:")
#         pprint(list(model.__dict__.keys()))
#     # verifica presenza di metodi utili
#     for method in ('get_DOE', 'evaluate', 'evaluate_std', 'predict'):
#         print(f"Has {method}:", hasattr(model, method))

doe = model.get_DOE()
x_min, x_max = doe[:,0].min(), doe[:,0].max()
y_min, y_max = doe[:,1].min(), doe[:,1].max()
# Crea griglia uniforme
n_grid = 30
xg = np.linspace(x_min, x_max, n_grid)
yg = np.linspace(y_min, y_max, n_grid)
X, Y = np.meshgrid(xg, yg)
grid_points = np.column_stack([X.ravel(), Y.ravel()])

# --- Animazione evoluzione modelli ---
from matplotlib.animation import FuncAnimation

# Prepara i dati per l'animazione
models_data = []
for i in range(nModels):
    model_path = os.path.join(directory, f"model_{i}.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        Z_i = model.evaluate(grid_points)
        std_i = model.evaluate_std(grid_points)
        Z0_i = np.array(Z_i)[0].reshape(X.shape)
        Z1_i = np.array(Z_i)[1].reshape(X.shape)
        std0_i = np.array(std_i)[0].reshape(X.shape)
        std1_i = np.array(std_i)[1].reshape(X.shape)
        doe_i = model.get_DOE()
        doe_z0_i = model.evaluate(doe_i)[0][:,0]
        doe_z1_i = model.evaluate(doe_i)[1][:,0]
        models_data.append((i, Z0_i, Z1_i, std0_i, std1_i, doe_i, doe_z0_i, doe_z1_i))

if models_data:
    # Trova i valori min/max globali per normalizzare
    all_z0 = [data[1] for data in models_data]
    all_z1 = [data[2] for data in models_data]
    all_std0 = [data[3] for data in models_data]
    all_std1 = [data[4] for data in models_data]
    z0_min, z0_max = min(z.min() for z in all_z0), max(z.max() for z in all_z0)
    z1_min, z1_max = min(z.min() for z in all_z1), max(z.max() for z in all_z1)
    
    fig = plt.figure(figsize=(22, 8))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133)
    
    # Calcola range globali per std (manteniamo anche i min/max reali) ma usiamo
    # un range fisso per le colorbar come richiesto dall'utente.
    std0_min, std0_max = min(s.min() for s in all_std0), max(s.max() for s in all_std0)
    std1_min, std1_max = min(s.min() for s in all_std1), max(s.max() for s in all_std1)
    # Range fisso desiderato
    FIX_VMIN, FIX_VMAX = 0.0, 0.003
    mappable0 = mpl.cm.ScalarMappable(cmap='viridis', norm=mpl.colors.Normalize(vmin=FIX_VMIN, vmax=FIX_VMAX))
    mappable0.set_array([])
    cbar0 = fig.colorbar(mappable0, ax=ax1, shrink=0.6, pad=0.08, label='Std risposta 1')
    mappable1 = mpl.cm.ScalarMappable(cmap='plasma', norm=mpl.colors.Normalize(vmin=FIX_VMIN, vmax=FIX_VMAX))
    mappable1.set_array([])
    cbar1 = fig.colorbar(mappable1, ax=ax2, shrink=0.6, pad=0.08, label='Std risposta 2')

    # Variabile per tracciare il frame corrente
    current_frame = [0]
    
    def update_plot(frame_idx):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        i, Z0_i, Z1_i, std0_i, std1_i, doe_i, doe_z0_i, doe_z1_i = models_data[frame_idx]

        # Plot prima risposta - normalizza la std usando il range fisso
        norm_std0 = np.clip(std0_i, FIX_VMIN, FIX_VMAX)
        denom0 = (FIX_VMAX - FIX_VMIN) if (FIX_VMAX - FIX_VMIN) > 0 else 1.0
        norm_std0 = (norm_std0 - FIX_VMIN) / denom0
        surf1 = ax1.plot_surface(X, Y, Z0_i, facecolors=plt.cm.viridis(norm_std0), alpha=0.7)

        # Trova i punti nuovi aggiunti rispetto al modello precedente
        new_points_mask = np.zeros(len(doe_i), dtype=bool)
        if frame_idx > 0:
            prev_doe = models_data[frame_idx-1][5]  # DOE del modello precedente
            for idx, row in enumerate(doe_i):
                if not any(np.allclose(row, prev_row) for prev_row in prev_doe):
                    new_points_mask[idx] = True

        # Plot prima i punti vecchi (rossi), poi i nuovi (triangoli neri) per sovrapporli
        old_points_mask = ~new_points_mask
        if np.any(old_points_mask):
            ax1.scatter(doe_i[old_points_mask,0], doe_i[old_points_mask,1], doe_z0_i[old_points_mask], 
                       color='r', label='DOE precedenti', s=40)
        if np.any(new_points_mask):
            ax1.scatter(doe_i[new_points_mask,0], doe_i[new_points_mask,1], doe_z0_i[new_points_mask], 
                       color='black', label='Nuovi punti', s=60)

        ax1.set_title(f'Risposta 1 - Model {i} (colore=std)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Risposta 1')
        ax1.set_zlim(z0_min, z0_max)
        ax1.legend()

        # Plot seconda risposta con la stessa logica (range fisso)
        norm_std1 = np.clip(std1_i, FIX_VMIN, FIX_VMAX)
        denom1 = (FIX_VMAX - FIX_VMIN) if (FIX_VMAX - FIX_VMIN) > 0 else 1.0
        norm_std1 = (norm_std1 - FIX_VMIN) / denom1
        surf2 = ax2.plot_surface(X, Y, Z1_i, facecolors=plt.cm.plasma(norm_std1), alpha=0.7)

        if np.any(old_points_mask):
            ax2.scatter(doe_i[old_points_mask,0], doe_i[old_points_mask,1], doe_z1_i[old_points_mask], 
                       color='r', label='DOE precedenti', s=40)
        if np.any(new_points_mask):
            ax2.scatter(doe_i[new_points_mask,0], doe_i[new_points_mask,1], doe_z1_i[new_points_mask], 
                       color='black', label='Nuovi punti', s=80)

        ax2.set_title(f'Risposta 2 - Model {i} (colore=std)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Risposta 2')
        ax2.set_zlim(z1_min, z1_max)
        ax2.legend()

        # Scatter 2D: risposta1 vs risposta2 sui punti DOE (vecchi vs nuovi)
        # x = risposta1, y = risposta2
        if np.any(old_points_mask):
            ax3.scatter(doe_z0_i[old_points_mask], doe_z1_i[old_points_mask], color='r', label='DOE precedenti', s=40)
        if np.any(new_points_mask):
            ax3.scatter(doe_z0_i[new_points_mask], doe_z1_i[new_points_mask], color='black', label='Nuovi punti', s=60)
        ax3.set_xlabel('Risposta 1')
        ax3.set_ylabel('Risposta 2')
        # imposta limiti coerenti con i min/max globali delle risposte
        ax3.set_xlim(0, z0_max)
        ax3.set_ylim(0, z1_max)
        ax3.set_title(f'Risposta1 vs Risposta2 - Model {i}')
        ax3.legend()

        plt.draw()
    
    def next_frame(event):
        current_frame[0] = (current_frame[0] + 1) % len(models_data)
        update_plot(current_frame[0])
    
    def prev_frame(event):
        current_frame[0] = (current_frame[0] - 1) % len(models_data)
        update_plot(current_frame[0])
    
    # Crea i bottoni
    from matplotlib.widgets import Button
    ax_prev = plt.axes([0.42, 0.02, 0.08, 0.04])
    ax_next = plt.axes([0.52, 0.02, 0.08, 0.04])
    btn_prev = Button(ax_prev, 'Precedente')
    btn_next = Button(ax_next, 'Successivo')
    btn_prev.on_clicked(prev_frame)
    btn_next.on_clicked(next_frame)
    
    # Mostra il primo frame
    update_plot(0)
    plt.tight_layout()
    plt.show()
