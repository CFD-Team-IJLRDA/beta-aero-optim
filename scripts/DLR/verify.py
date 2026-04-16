import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def plot_ffd_profiles(xxx_folder_name):
    """
    Carica e plotta tutti i file .dat nella cartella FFD specificata.
    
    Args:
        xxx_folder_name (str): Nome della cartella XXX sotto output_mf
    """
    # Costruisce il percorso completo
    base_path = os.path.expanduser("~/GPROptimization/beta-aero-optim/examples/test-Cascade/beta_opt_feasible/output_mf")
    ffd_folder = os.path.join(base_path, xxx_folder_name, "FFD")
    
    # Verifica che la cartella esista
    if not os.path.exists(ffd_folder):
        print(f"Errore: cartella non trovata {ffd_folder}")
        return
    
    # Trova tutti i file .dat
    dat_files = glob.glob(os.path.join(ffd_folder, "*.dat"))
    
    if not dat_files:
        print(f"Nessun file .dat trovato in {ffd_folder}")
        return
    
    print(f"Trovati {len(dat_files)} file .dat in {ffd_folder}")
    
    # Crea il plot
    plt.figure(figsize=(10, 6))
    
    for dat_file in dat_files:
        try:
            # Carica il file saltando le prime 2 righe di header
            data = np.loadtxt(dat_file, skiprows=2)
            
            # Estrae x e y (prime due colonne)
            x = data[:, 0]
            y = data[:, 1]
            
            # Plotta con il nome del file come label
            filename = os.path.basename(dat_file)
            plt.plot(x, y, label=filename, linewidth=1.5)
            
        except Exception as e:
            print(f"Errore nel caricare {dat_file}: {e}")
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Profili FFD da {xxx_folder_name}/FFD')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Esempio di utilizzo
    xxx = input("Inserisci il nome della cartella XXX: ")
    plot_ffd_profiles(xxx)
