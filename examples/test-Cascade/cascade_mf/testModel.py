import dill as pickle
import numpy as np
import os

outdir = "output_mf"  # Cambia con la tua directory reale
model_path = "output_mf/model.pkl"

with open(model_path, "rb") as handle:
    model = pickle.load(handle)

print(model.get_DOE())

print("\n=== Attributi e metodi pubblici di MultiobjectiveModel ===")
print([attr for attr in dir(model) if not attr.startswith('_')])

print("\n=== Attributi di istanza del modello ===")
print(model.__dict__)

print("\n=== Attributi e metodi pubblici di MFSMT ADP ===")
print([attr for attr in dir(model.models[0]) if not attr.startswith('_')])

print("\n=== Attributi e metodi pubblici di MFSMT OP ===")
print([attr for attr in dir(model.models[1]) if not attr.startswith('_')])

# # Punto di esempio (modifica con il tuo punto reale)
# x = np.array([[0.1, 0.2, 0.3, 0.4]])

# # Predizione
# if hasattr(model, "predict"):
#     y_pred = model.predict(x)
#     print("Predizione QoI:", y_pred)
# elif hasattr(model, "predict_lf"):
#     y_pred = model.predict_lf(x)
#     print("Predizione QoI:", y_pred)
# else:
#     raise Exception("Il modello caricato non supporta la predizione diretta.")