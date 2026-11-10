"""Point-wise dataset for the INR: inputs (params, coords) and targets per cell, for a list of blades.

params = [c1..c5, inlet angle]   (z-scored with training statistics)
coords = [x_hat, y_hat]          (T_bal output in [0, 1], z-scored with training statistics)
targets = [p, rho, Ux, Uy, nut]  (z-scored per channel with training statistics); Ma is derived, not learned.
Each blade contributes its cells once per operating point (ADP 43 deg, OP1 48 deg, OP2 38 deg).
"""
import json
import os

import numpy as np
import torch

from .paths import WORK as HERE                    # datasets live in the surrogate work folder (paths.py)
DATA = os.path.join(HERE, "data")       # set_data("data_mean") switches to the time-averaged fields


def set_data(name):
    global DATA
    DATA = os.path.join(HERE, name)


INLET_ANGLE = np.array([43.0, 48.0, 38.0])       # ADP, OP1, OP2
TARGETS = ["p", "rho", "Ux", "Uy", "nut"]        # columns 0..4 of the stored fields (Ma, column 5, is derived)


def load_balance():
    return np.load(os.path.join(DATA, "balance.npz"))


def balance(X, bal):
    return np.c_[np.interp(X[:, 0], bal["grid_x"], bal["F_x"]), np.interp(X[:, 1], bal["grid_y"], bal["F_y"])]


def splits():
    return json.load(open(os.path.join(DATA, "splits.json")))


def load_blade(name):
    return np.load(os.path.join(DATA, "blades", f"{name}.npz"))


def periodic_interpolator(X, F, pitch=0.04039):
    """
    Linear interpolator of cell-centre values F (N, k) at physical points, periodic in y with the pitch: the cells are
    copied one pitch up and down, so that points between the periodic boundaries and the first row of cell centres
    (the first/last ~14 samples of the measurement lines) are inside the interpolation region instead of NaN.
    The cascade is periodic with the separation (0, pitch) (cyclicAMI), so the copies are exact.
    """
    from scipy.interpolate import LinearNDInterpolator
    off = np.array([[0.0, 0.0], [0.0, pitch], [0.0, -pitch]])
    Xa = np.vstack([X + o for o in off])
    return LinearNDInterpolator(Xa, np.vstack([F, F, F]))


COORDS = {"plain": 2, "periodic": 4}      # coordinate embedding -> number of network input features
RAMP = 0.02                                 # periodic embedding: width (in x_hat) of the ramp of b(x_hat) outside the blade


def blade_extent(bal):
    """x_hat of the reference LE and TE (all blades are aligned on them by T_geom)."""
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    return tuple(float(np.interp(x, bal["grid_x"], bal["F_x"])) for x in (meta["x_le_ref"], meta["x_te_ref"]))


def embed(xh, mode, bal):
    """
    Coordinate features of the network. 'plain': (x_hat, y_hat).
    'periodic' (blade-aware): (x_hat, sin 2 pi y_hat, cos 2 pi y_hat, b(x_hat) y_hat). The two horizontal edges of the
    reference square are the same physical line upstream of the LE and downstream of the TE (the periodic cut lines),
    but the two different blade surfaces in between. With b = 0 away from the blade (smoothstep ramp of width RAMP
    just outside [x_LE, x_TE]) and b = 1 along it, the features, hence the predictions, are exactly periodic in y_hat on the
    cut lines and free on the blade (suction and pressure sides stay distinct).
    """
    if mode == "plain":
        return xh
    x_le, x_te = blade_extent(bal)
    x, y = xh[:, 0], xh[:, 1]
    t = np.clip(np.minimum((x - (x_le - RAMP)) / RAMP, ((x_te + RAMP) - x) / RAMP), 0, 1)
    b = t * t * (3 - 2 * t)
    return np.c_[x, np.sin(2 * np.pi * y), np.cos(2 * np.pi * y), b * y]


def mp_aug(name):
    """CFD fields (3 OPs, M, 5) linearly interpolated at the MP1 + MP2 line points of a blade, and those points in the
    reference (T_geom) domain (M, 2); cached in DATA/mp_aug/ (see build_mp_aug.py). NaN end points are dropped."""
    z = np.load(os.path.join(DATA, "mp_aug", f"{name}.npz"))
    return z["vals"], z["ref"]


def raw_arrays(names, bal, coords="plain", mp_points=0):
    """
    Unnormalized (params, coords, targets, blade_id, op_id) arrays for all cells of the blades x 3 OPs.
    mp_points > 0 also appends the MP1/MP2 line samples of each blade (mp_aug), repeated mp_points times
    (training-set augmentation; they are mapped through the same frozen T_bal, which is not refitted).
    """
    P, X, Y, B, O = [], [], [], [], []  # noqa: E741  (O: operating-point index)
    for b, name in enumerate(names):
        d = load_blade(name)
        xh = embed(balance(d["X_geom"], bal), coords, bal)
        if mp_points:
            vals, ref = mp_aug(name)
            xa = np.tile(embed(balance(ref, bal), coords, bal), (mp_points, 1))
        n = len(xh)
        for k in range(3):
            P.append(np.tile(np.r_[d["pod"], INLET_ANGLE[k]], (n, 1)))
            X.append(xh)
            Y.append(d["fields"][k][:, :5])
            B.append(np.full(n, b))
            O.append(np.full(n, k))
            if mp_points:
                P.append(np.tile(np.r_[d["pod"], INLET_ANGLE[k]], (len(xa), 1)))
                X.append(xa)
                Y.append(np.tile(vals[k], (mp_points, 1)))
                B.append(np.full(len(xa), b))
                O.append(np.full(len(xa), k))
    return (np.vstack(P).astype(np.float32), np.vstack(X).astype(np.float32), np.vstack(Y).astype(np.float32),
            np.concatenate(B), np.concatenate(O))


class Normalizer:
    def __init__(self, P, X, Y):
        self.stats = {k: (v.mean(0), v.std(0) + 1e-12) for k, v in (("P", P), ("X", X), ("Y", Y))}

    def __call__(self, key, a):
        m, s = self.stats[key]
        return (a - m) / s

    def inverse_Y(self, y):
        m, s = self.stats["Y"]
        return y * s + m

    def save(self, path):
        np.savez(path, **{f"{k}_{i}": v for k, (m, s) in self.stats.items() for i, v in (("mean", m), ("std", s))})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        obj = cls.__new__(cls)
        obj.stats = {k: (z[f"{k}_mean"], z[f"{k}_std"]) for k in ("P", "X", "Y")}
        return obj


def to_device(norm, P, X, Y, device):
    def t(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
    return t(norm("P", P)), t(norm("X", X)), t(norm("Y", Y))
