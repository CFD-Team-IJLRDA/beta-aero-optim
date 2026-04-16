import argparse
import dill as pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['animation.embed_limit'] = 50  # 50MB invece di 20MB
from matplotlib.widgets   import Button
from matplotlib.animation import FuncAnimation, HTMLWriter, PillowWriter
from pprint import pprint

# --- Valutazione e plot superficie modello finale ---
from mpl_toolkits.mplot3d import Axes3D

# Configurazione feasibility thresholds (modificabili manualmente)
FEASIBILITY_BOUNDS = {
    'ADP': [-0.8, 0.8],  # theta_ADP bounds
    'OP1': [-1.5, 1.5],      # theta_OP1 bounds  
    'OP2': [-1.5, 1.5]       # theta_OP2 bounds
}

def save_as_html(filename='model_animation.html'):
    """Salva come HTML interattivo"""
    
    def animate_func(frame):
        update_plot(frame)
        return []
    
    anim = FuncAnimation(fig, animate_func, frames=len(models_data), 
                        interval=2000, repeat=True, blit=False)
    
    anim.save(filename, writer=HTMLWriter(fps=1, embed_frames=True))
    print(f"HTML interattivo salvato come {filename}")

def save_animation_as_gif(filename='model_animation.gif', fps=1):

    def animate_func(frame):
        update_plot(frame)
        return []

    anim = FuncAnimation(fig, animate_func, frames=len(models_data), 
                        interval=1000/fps, repeat=True, blit=False)

    writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer)
    print(f"Animazione salvata come {filename}")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot multi-fidelity surrogate model results')
parser.add_argument('-d', '--directory', type=str, required=True, 
                    help='Directory containing model files (e.g., output_mf)')
parser.add_argument('-n', '--nmodels', type=int, required=True,
                    help='Number of model files to process (e.g., 20)')
parser.add_argument('-html', '--html_output', type=str, 
                    help='Save HTML animation to specified directory')
args = parser.parse_args()

nModels = args.nmodels
directory = args.directory


# colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']  # cicla tra colormaps
colormaps = ['viridis', 'viridis', 'plasma', 'plasma', 'plasma']  # cicla tra colormaps
titles = [r'$Loss_{ADP}$', r'$Loss_{OP}$', r'$\theta_{ADP}$', r'$\theta_{OP1}$', r'$\theta_{OP2}$', r'$Loss_{ADP}$ vs $Loss_{OP}$']  # cicla tra colormaps

model_path = os.path.join(directory, f"model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

doe = model.get_DOE()
x_min, x_max = doe[:,0].min(), doe[:,0].max()
y_min, y_max = doe[:,1].min(), doe[:,1].max()
# Crea griglia uniforme
n_grid = 30
xg = np.linspace(x_min, x_max, n_grid)
yg = np.linspace(y_min, y_max, n_grid)
X, Y = np.meshgrid(xg, yg)
grid_points = np.column_stack([X.ravel(), Y.ravel()])

# Rileva numero di risposte del modello
sample_eval = model.evaluate(grid_points[:1])  # valuta solo un punto per rilevare la struttura
if isinstance(sample_eval, list):
    n_responses = len(sample_eval)
else:
    # se non è una lista, assumiamo che sia un singolo array
    sample_eval = np.asarray(sample_eval)
    if sample_eval.ndim > 1 and sample_eval.shape[0] > 1:
        n_responses = sample_eval.shape[0]
    else:
        n_responses = 1

print(f"Rilevate {n_responses} risposte del modello")

# Prepara i dati per l'animazione
models_data = []
for i in range(nModels+1):
    if i < nModels:
        model_path = os.path.join(directory, f"model_{i}.pkl")
    else:
        model_path = os.path.join(directory, f"model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        Z_i = model.evaluate(grid_points)
        std_i = model.evaluate_std(grid_points)
        
        # Estrai dinamicamente tutte le risposte
        Z_responses = []
        std_responses = []
        doe_responses = []
        
        for resp_idx in range(n_responses):
            if isinstance(Z_i, list):
                Z_resp = np.array(Z_i[resp_idx]).reshape(X.shape)
                std_resp = np.array(std_i[resp_idx]).reshape(X.shape)
            else:
                Z_array = np.asarray(Z_i)
                std_array = np.asarray(std_i)
                if Z_array.ndim > 1 and Z_array.shape[0] > resp_idx:
                    Z_resp = Z_array[resp_idx].reshape(X.shape)
                    std_resp = std_array[resp_idx].reshape(X.shape)
                else:
                    # fallback per modelli con una sola risposta
                    Z_resp = Z_array.reshape(X.shape)
                    std_resp = std_array.reshape(X.shape)
            
            Z_responses.append(Z_resp)
            std_responses.append(std_resp)
            
            # Estrai valori DOE per questa risposta
            doe_i = model.get_DOE()
            doe_eval = model.evaluate(doe_i)
            if isinstance(doe_eval, list):
                doe_z = np.asarray(doe_eval[resp_idx])
            else:
                doe_array = np.asarray(doe_eval)
                if doe_array.ndim > 1 and doe_array.shape[0] > resp_idx:
                    doe_z = doe_array[resp_idx]
                else:
                    doe_z = doe_array
            
            # Appiattisci se necessario
            if doe_z.ndim > 1:
                doe_z = doe_z[:, 0] if doe_z.shape[1] > 0 else doe_z.ravel()
            doe_responses.append(doe_z)
        
        models_data.append((i, Z_responses, std_responses, doe_i, doe_responses))

if models_data:
    # Trova i valori min/max globali per normalizzare per ogni risposta
    all_z_responses = [[] for _ in range(n_responses)]
    all_std_responses = [[] for _ in range(n_responses)]
    
    for data in models_data:
        _, Z_responses, std_responses, _, _ = data
        for resp_idx in range(n_responses):
            all_z_responses[resp_idx].append(Z_responses[resp_idx])
            all_std_responses[resp_idx].append(std_responses[resp_idx])
    
    # Calcola min/max per ogni risposta
    z_mins = []
    z_maxs = []
    std_mins = []
    std_maxs = []
    
    for resp_idx in range(n_responses):
        z_min = min(z.min() for z in all_z_responses[resp_idx])
        z_max = max(z.max() for z in all_z_responses[resp_idx])
        std_min = min(s.min() for s in all_std_responses[resp_idx])
        std_max = max(s.max() for s in all_std_responses[resp_idx])
        z_mins.append(z_min)
        z_maxs.append(z_max)
        std_mins.append(std_min)
        std_maxs.append(std_max)
    
    # Crea layout dinamico: n_responses subplot 3D + 1 subplot 2D per scatter
    total_subplots = n_responses + 1
    cols = min(3, total_subplots)  # max 3 colonne
    rows = (total_subplots + cols - 1) // cols  # calcola righe necessarie
    
    fig = plt.figure(figsize=(8 * cols, 6 * rows))
    
    # Crea subplot 3D per ogni risposta
    axes_3d = []
    
    for resp_idx in range(n_responses):
        ax = fig.add_subplot(rows, cols, resp_idx + 1, projection='3d')
        axes_3d.append(ax)
    
    # Crea subplot 2D per scatter (ultimo)
    ax_scatter = fig.add_subplot(rows, cols, n_responses + 1)
    
    # Calcola range individuali per std e crea colorbar fisse per ogni risposta
    mappables = []
    cbars = []
    
    for resp_idx in range(n_responses):
        cmap = colormaps[resp_idx % len(colormaps)]
        # Usa il range specifico della deviazione standard per questa risposta
        vmin = std_mins[resp_idx]
        vmax = std_maxs[resp_idx]
        mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=axes_3d[resp_idx], shrink=0.6, pad=0.08, 
                           label=f'Standard deviation')
        mappables.append(mappable)
        cbars.append(cbar)

    # Variabile per tracciare il frame corrente
    current_frame = [0]
    
    def update_plot(frame_idx):
        # Clear tutti gli assi
        for ax in axes_3d:
            ax.clear()
        ax_scatter.clear()
        
        i, Z_responses, std_responses, doe_i, doe_responses = models_data[frame_idx]

        # Trova i punti nuovi aggiunti rispetto al modello precedente
        new_points_mask = np.zeros(len(doe_i), dtype=bool)
        if frame_idx > 0:
            prev_doe = models_data[frame_idx-1][3]  # DOE del modello precedente
            for idx, row in enumerate(doe_i):
                if not any(np.allclose(row, prev_row) for prev_row in prev_doe):
                    new_points_mask[idx] = True

        old_points_mask = ~new_points_mask

        # Plot per ogni risposta
        for resp_idx in range(n_responses):
            ax = axes_3d[resp_idx]
            Z_resp = Z_responses[resp_idx]
            std_resp = std_responses[resp_idx]
            doe_z_resp = doe_responses[resp_idx]
            cmap = colormaps[resp_idx % len(colormaps)]
            
            # Normalizza la std usando il range specifico di questa risposta
            vmin = std_mins[resp_idx]
            vmax = std_maxs[resp_idx]
            norm_std = np.clip(std_resp, vmin, vmax)
            denom = (vmax - vmin) if (vmax - vmin) > 0 else 1.0
            norm_std = (norm_std - vmin) / denom
            
            # Plot superficie
            surf = ax.plot_surface(X, Y, Z_resp, facecolors=plt.cm.get_cmap(cmap)(norm_std), alpha=0.4)
            
            # Aggiungi isolinee per subplot 3, 4, 5 (risposta 3, 4, 5)
            if resp_idx == 2:  # subplot 3 (risposta 3)
                ax.contour(X, Y, Z_resp, levels=[FEASIBILITY_BOUNDS['ADP'][0], FEASIBILITY_BOUNDS['ADP'][1]], colors='lime', linewidths=5)
            elif resp_idx == 3:  # subplot 4 (risposta 4) 
                ax.contour(X, Y, Z_resp, levels=[FEASIBILITY_BOUNDS['OP1'][0], FEASIBILITY_BOUNDS['OP1'][1]], colors='lime', linewidths=5)
            elif resp_idx == 4:  # subplot 5 (risposta 5)
                ax.contour(X, Y, Z_resp, levels=[FEASIBILITY_BOUNDS['OP2'][0], FEASIBILITY_BOUNDS['OP2'][1]], colors='lime', linewidths=5)

            # Plot punti DOE
            # Linee 229-236, sostituisci con:
            offset = (z_maxs[resp_idx] - z_mins[resp_idx]) * 0.03
            
            if np.any(old_points_mask):
                ax.scatter(doe_i[old_points_mask,0], doe_i[old_points_mask,1], doe_z_resp[old_points_mask], 
                          color='r', label='Previous DOE', s=20)
            if np.any(new_points_mask):
                ax.scatter(doe_i[new_points_mask,0][0], doe_i[new_points_mask,1][0], doe_z_resp[new_points_mask][0], 
                          color='green', label='HF infill', s=60)
                
                if resp_idx in [0,1]:  # Primo subplot
                    ax.plot([doe_i[new_points_mask,0][0], doe_i[new_points_mask,0][0]], 
                            [doe_i[new_points_mask,1][0], doe_i[new_points_mask,1][0]], 
                            [z_maxs[resp_idx], doe_z_resp[new_points_mask][0]], 'green', linewidth=4)
                else:
                    ax.plot([doe_i[new_points_mask,0][0], doe_i[new_points_mask,0][0]], 
                            [doe_i[new_points_mask,1][0], doe_i[new_points_mask,1][0]], 
                            [z_mins[resp_idx], doe_z_resp[new_points_mask][0]], 'green', linewidth=4)
                ax.scatter(doe_i[new_points_mask,0][1:], doe_i[new_points_mask,1][1:], doe_z_resp[new_points_mask][1:], 
                          color='black', label='LF infill', s=60)

            if resp_idx == 0:  # Primo subplot
                ax.view_init(elev=30, azim=53)  # Sostituisci con i tuoi valori preferiti
            elif resp_idx == 1:  # Secondo subplot  
                ax.view_init(elev=30, azim=118)  # Sostituisci con i tuoi valori preferiti

            ax.set_title(f'{titles[resp_idx]} - Infill {i}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel(f'{titles[resp_idx]}')
            ax.set_zlim(z_mins[resp_idx], z_maxs[resp_idx])
            ax.legend()

        # Scatter 2D: risposta1 vs risposta2 sui punti DOE (solo se ci sono almeno 2 risposte)
        doe_z0 = doe_responses[0]
        doe_z1 = doe_responses[1]
        
        if np.any(old_points_mask):
            ax_scatter.scatter(doe_z0[old_points_mask], doe_z1[old_points_mask]
                            , color='r', label='Previous DOE', s=40)
        if np.any(new_points_mask):
            ax_scatter.scatter(doe_z0[new_points_mask][1:], doe_z1[new_points_mask][1:]
                            , alpha=1.0, color='black', label='LF infill', s=60)
            ax_scatter.scatter(doe_z0[new_points_mask][0], doe_z1[new_points_mask][0]
                            , alpha=1.0, color='green', label='HF infill', s=60)

        ax_scatter.set_xlabel(f'{titles[0]}')
        ax_scatter.set_ylabel(f'{titles[1]}')
        ax_scatter.set_xlim(0, z_maxs[0])
        ax_scatter.set_ylim(0, z_maxs[1])
        ax_scatter.set_title(f'{titles[-1]} - Infill {i}')
        ax_scatter.legend()

        # for i, ax in enumerate(axes_3d[:2]):  # solo primi due subplot
        #     elev = ax.elev
        #     azim = ax.azim
        #     print(f"Subplot {i+1}: elevation={elev:.1f}, azimuth={azim:.1f}")

        plt.draw()
    
    def next_frame(event):
        current_frame[0] = (current_frame[0] + 1) % len(models_data)
        update_plot(current_frame[0])
    
    def prev_frame(event):
        current_frame[0] = (current_frame[0] - 1) % len(models_data)
        update_plot(current_frame[0])
    
    # Crea i bottoni
    ax_prev = plt.axes([0.42, 0.02, 0.08, 0.04])
    ax_next = plt.axes([0.52, 0.02, 0.08, 0.04])
    btn_prev = Button(ax_prev, 'Precedente')
    btn_next = Button(ax_next, 'Successivo')
    btn_prev.on_clicked(prev_frame)
    btn_next.on_clicked(next_frame)
    
    # Mostra il primo frame
    update_plot(0)
    plt.tight_layout()
    
    # Salva HTML se richiesto
    if args.html_output:
        save_as_html(args.html_output)
    
    plt.show()
