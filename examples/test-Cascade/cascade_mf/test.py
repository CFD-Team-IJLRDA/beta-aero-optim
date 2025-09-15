
import pickle
import json
import pandas as pd

# Percorsi dei file
PKL_PATH = "output_mf/lf_doe/df_dict.pkl"
JSON_PATH = "df_dict_reduced.json"

with open(PKL_PATH, "rb") as f:
    df_dict = pickle.load(f)

# Costruisci un nuovo dizionario:
# - per ADP, OP1, OP2: solo l'ultimo LossCoef
# - per le altre chiavi: tutte le colonne dell'ultima riga
result = {}
for gid in df_dict:
    result[gid] = {}
    for cid in df_dict[gid]:
        result[gid][cid] = {}
        for op in df_dict[gid][cid]:
            df = df_dict[gid][cid][op]
            if op in ["ADP", "OP1", "OP2"]:
                key = f"{op}Loss"
                if hasattr(df, 'columns') and "LossCoef" in df.columns:
                    last_loss = df["LossCoef"].iloc[-1]
                    result[gid][cid][key] = float(last_loss)
                else:
                    result[gid][cid][key] = None
            else:
                # Per le altre chiavi, prendi tutte le colonne dell'ultima riga se df è un DataFrame
                if hasattr(df, 'iloc') and hasattr(df, 'to_dict'):
                    if len(df) > 0:
                        last_row = df.iloc[-1].to_dict()
                        # Converti eventuali valori numpy a float
                        for k, v in last_row.items():
                            if hasattr(v, 'item'):
                                last_row[k] = v.item()
                        result[gid][cid][op] = last_row
                    else:
                        result[gid][cid][op] = {}
                else:
                    # Se df non è un DataFrame, salva il valore direttamente
                    result[gid][cid][op] = df

with open(JSON_PATH, "w") as f:
    json.dump(result, f, indent=2)

print(f"Dizionario ridotto salvato in {JSON_PATH}")

