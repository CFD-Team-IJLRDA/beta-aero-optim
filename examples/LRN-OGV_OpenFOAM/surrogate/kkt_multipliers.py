"""
Lagrange multipliers of the gradient-optimised blades (grad_optim.py runs): which constraints limit each local optimum.

At each local optimum c* of the run (history.csv, source == surrogate) the KKT stationarity condition
    -grad f = sum_{i active} lambda_i grad g_i,   lambda_i >= 0
is solved for lambda by non-negative least squares (same INR ensemble, same constraints, same autodiff gradients as the
optimisation). Constraints are written in normalised form g~ <= 0 so that lambda is comparable between them:
    cap        (L_OP - eps) / eps                 (relax by 1 % of eps)
    angles     (|phi - phi_ref| - tol) / tol       (widen the window by 10 %)
    geometry   g as in GeometryConstraints        (already relative; widen the tolerance by 10 % of its value)
    r_LE, BladeGen family                         (relax by 0.1)
    POD box    z - 1, -z  (z in [0, 1])           (widen the box by 10 % of its span)
Output: the predicted relative decrease of the objective if one constraint is relaxed as above, gain_% = 100 lambda delta / f
(first-order, the meaning of lambda in constrained_optimisation.md Sec. 3), and the KKT residual ||grad f + sum lambda grad g|| / ||grad f||
(small: a genuine constrained optimum). Written to <run>/kkt.csv.
Usage: ~/miniconda3/envs/ogv_torch/bin/python kkt_multipliers.py sopt/grad2_N200_s0_o1 [...]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.optimize import nnls

import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
import surrogate_optim as SO
from aero_optim.optim.cascade_foam import ANGLE_TOL, GeometryConstraints
from aero_optim.shape.bladegen_pod import BladeGenPOD
from aero_optim.utils import from_dat
from grad_optim import Model
from aero_optim.surrogate.torch_geometry import TorchGeometry
from aero_optim.surrogate.predict import OPS, DiffPredictor

ACTIVE = 5e-3                                       # |g~| below this: active (the optimiser stops at g~ = -1e-3, its margin)
GEOM = ["t_max/c", "x_tmax/c_ax", "A/c2", "x_cg/c_ax", "r_LE", "BladeGen family"]


def main():
    pod = BladeGenPOD.load(os.path.join(SO.POD_DIR, "basis.npz"))
    ds = np.load(os.path.join(SO.POD_DIR, "dataset.npz"))
    geometry = GeometryConstraints(np.array(from_dat(SO.BASELINE_DAT, 2, 1))[:, :2], np.array([pod.project(d) for d in ds["D"][~ds["failed"]]]))
    TG = TorchGeometry(geometry.ref, pod, geometry)
    lo, hi = pod.bounds[:, 0], pod.bounds[:, 1]
    span = hi - lo
    ref, tol = np.array([SO.ANGLE_REF[o] for o in OPS]), np.array([ANGLE_TOL[o] for o in OPS])
    for run in sys.argv[1:]:
        cfg = json.load(open(os.path.join(run, "run.json")))
        P = DiffPredictor(cfg["models"], pod, data=cfg["data"], device="cuda:0")
        M = Model(P, cfg["kappa"])
        h = pd.read_csv(os.path.join(run, "history.csv"))
        rows = []
        for _, r in h[h.source == "surrogate"].iterrows():
            c = r[[f"c{k + 1}" for k in range(5)]].to_numpy(float)
            v, J = M.at(c)
            obj = 1 if r.start == "anchor1" else 0
            f, df = v[obj], J[obj] * span                                   # gradients in z (c = lo + z span)
            ct = torch.tensor(c, dtype=torch.float64)
            gg, gJ = TG(ct).numpy(), torch.autograd.functional.jacobian(TG, ct).numpy() * span
            s = np.sign(v[2:] - ref)
            names = [f"angle {o}" for o in OPS] + GEOM + [f"box c{k + 1} {e}" for k in range(5) for e in ("max", "min")]
            g = np.r_[(np.abs(v[2:] - ref) - tol) / tol, gg, [x for k in range(5) for x in ((c[k] - hi[k]) / span[k], (lo[k] - c[k]) / span[k])]]
            G = np.r_[s[:, None] * J[2:] * span / tol[:, None], gJ, [x for k in range(5) for x in (np.eye(5)[k], -np.eye(5)[k])]]
            delta = np.r_[[0.1] * 3, [0.1 * geometry.tol[k] for k in GEOM[:4]], 0.1, 0.1, [0.1] * 10]
            if r.start.startswith("eps"):
                eps = float(r.start[3:])
                names, delta = ["cap L_OP"] + names, np.r_[0.01, delta]
                g, G = np.r_[(v[1] - eps) / eps, g], np.r_[[J[1] * span / eps], G]
            act = np.flatnonzero(g > -ACTIVE)
            lam = np.zeros(len(g))
            if len(act):
                lam[act] = nnls(G[act].T, -df)[0]
            res = np.linalg.norm(df + G.T @ lam) / np.linalg.norm(df)
            row = {"start": r.start, "objective": ["L_ADP", "L_OP"][obj], "L_ADP": v[0], "L_OP": v[1], "feasible": r.feasible,
                   "kkt_residual": res, "active": "; ".join(names[i] for i in act)}
            row.update({f"gain_% {n}": 100 * lam[i] * delta[i] / f for i, n in enumerate(names) if lam[i] > 0})
            rows.append(row)
        out = pd.DataFrame(rows)
        out.to_csv(os.path.join(run, "kkt.csv"), index=False)
        print(run, "\n", out.round(4).to_string(), "\n")


if __name__ == "__main__":
    main()
