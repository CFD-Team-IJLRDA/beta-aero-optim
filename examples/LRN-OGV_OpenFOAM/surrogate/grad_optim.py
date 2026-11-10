"""
Gradient-based optimisation on the INR surrogate (autodiff through aero_optim.surrogate.predict.DiffPredictor), same problem as surrogate_optim.py
(5 POD coefficients, objectives L_ADP and L_OP, all constraints active).

Front by the epsilon-constraint method: minimise L_ADP subject to L_OP <= eps and all constraints, with SLSQP, for a sweep of eps
(--swap: minimise L_OP subject to L_ADP <= eps);
plus the two anchors (min L_ADP, min L_OP). Each problem is started from the --starts best feasible CFD-known budget blades
(closest in L_OP to eps), so it uses the same information as the NSGA-II run. Objective, outflow-angle and geometric
constraints: exact gradients by autodiff (geometry: aero_optim.surrogate.torch_geometry).
--kappa > 0 optimises mean + kappa * std of the ensemble (penalises regions where the 3 models disagree).

Outputs in --out (same format as surrogate_optim.py, so verify_cfd.py and analyse_data_efficiency.py apply): history.csv (every
local optimum found), pareto.csv (feasible non-dominated, predicted), run.json (timings, evaluations).
Usage (from the work folder): ~/miniconda3/envs/ogv_torch/bin/python <this folder>/grad_optim.py --models runs/eff_N200_s0_m{0,1,2} --data data_subsets/N200_s0
         --out sopt/grad_N200_s0_o1 [--n-eps 12] [--starts 3] [--kappa 0] [--device cuda:0]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
import surrogate_optim as SO
from aero_optim.optim.cascade_foam import ANGLE_TOL, CONSTRAINTS, GeometryConstraints
from aero_optim.shape.bladegen_pod import BladeGenPOD
from aero_optim.utils import from_dat
from aero_optim.surrogate.paths import DOE
from aero_optim.surrogate.predict import OPS, DiffPredictor
from aero_optim.surrogate.torch_geometry import TorchGeometry

GEOM = CONSTRAINTS[:SO.N_GEOM]


class Model:
    """Cached value + gradient of the surrogate QoIs at c (one autodiff pass per new point)."""
    def __init__(self, P, kappa):
        self.P, self.kappa, self.c, self.n_eval = P, kappa, None, 0

    def at(self, c):
        if self.c is None or not np.array_equal(c, self.c):
            ct = torch.tensor(c, dtype=torch.float64, device=self.P.device, requires_grad=True)
            q = self.P.qois_torch(ct)                                     # (models, 3 OPs, [loss, angle])
            w = q[:, :, 0].mean(0) + self.kappa * q[:, :, 0].std(0)
            out = torch.cat([torch.stack([w[0], 0.5 * (w[1] + w[2])]), q[:, :, 1].mean(0)])   # L_ADP, L_OP, 3 angles
            J = torch.stack([torch.autograd.grad(o, ct, retain_graph=True)[0] for o in out])
            self.c, self.v, self.J = c.copy(), out.detach().cpu().numpy(), J.cpu().numpy()
            self.q = q.detach().cpu().numpy()
            self.n_eval += 1
        return self.v, self.J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-eps", type=int, default=8)
    ap.add_argument("--starts", type=int, default=1)
    ap.add_argument("--caps", default="useful", choices=["useful", "full"],
                    help="useful: caps between the lowest predicted L_OP and the baseline L_OP; full: up to the L_OP of the min-L_ADP blade")
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--swap", action="store_true",
                    help="swap the roles: minimise L_OP subject to L_ADP <= eps (caps between the lowest predicted L_ADP and the baseline L_ADP)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pod = BladeGenPOD.load(os.path.join(SO.POD_DIR, "basis.npz"))
    ds = np.load(os.path.join(SO.POD_DIR, "dataset.npz"))
    geometry = GeometryConstraints(np.array(from_dat(SO.BASELINE_DAT, 2, 1))[:, :2], np.array([pod.project(d) for d in ds["D"][~ds["failed"]]]))
    P = DiffPredictor(a.models, pod, data=a.data, device=a.device)
    for m, *_ in P.models:
        for p in m.parameters():
            p.requires_grad_(False)
    prob = SO.SurrogateProblem(P, pod, geometry, SO.ANGLE_REF, ANGLE_TOL)
    M = Model(P, a.kappa)
    lo, hi = pod.bounds[:, 0], pod.bounds[:, 1]
    span = hi - lo
    t0 = time.time()
    prob.initial_population(a.data, 32)
    known = pd.DataFrame(prob.history)
    known = known[known.feasible]
    ref, tol = np.array([SO.ANGLE_REF[o] for o in OPS]), np.array([ANGLE_TOL[o] for o in OPS])

    TG = TorchGeometry(geometry.ref, pod, geometry)                         # geometric constraints with autodiff gradients
    gcache = {}

    def geom_all(c):
        key = c.tobytes()
        if key not in gcache:
            gcache.clear()
            ct = torch.tensor(c, dtype=torch.float64)
            gcache[key] = (TG(ct).numpy(), torch.autograd.functional.jacobian(TG, ct).numpy())
        return gcache[key]
    geom = lambda c: geom_all(c)[0]                                          # noqa: E731
    geom_jac = lambda c: geom_all(c)[1]                                      # noqa: E731
    # SLSQP works on z in [0, 1]^5 (c = lo + z span); inequality constraints are fun >= 0, with a small margin MARGIN so that
    # solutions on a constraint stay feasible; objectives scaled by 1/L0 (O(1)).
    C = lambda z: lo + z * span                                              # noqa: E731
    MARGIN, L0 = 1e-3, 0.05
    cons = [{"type": "ineq", "fun": lambda z: -geom(C(z)) - MARGIN, "jac": lambda z: -geom_jac(C(z)) * span},
            {"type": "ineq", "fun": lambda z: tol * (1 - MARGIN) - np.abs(M.at(C(z))[0][2:] - ref),
             "jac": lambda z: -np.sign(M.at(C(z))[0][2:] - ref)[:, None] * M.at(C(z))[1][2:] * span}]

    def solve(obj, c0, eps=None):                                           # cap eps on the other loss (index 1 - obj)
        cs = cons + ([{"type": "ineq", "fun": lambda z: (eps - M.at(C(z))[0][1 - obj]) / L0,
                       "jac": lambda z: -M.at(C(z))[1][1 - obj] * span / L0}] if eps is not None else [])
        r = minimize(lambda z: M.at(C(z))[0][obj] / L0, (c0 - lo) / span, jac=lambda z: M.at(C(z))[1][obj] * span / L0,
                     method="SLSQP", bounds=[(0, 1)] * 5, constraints=cs, options={"maxiter": 100, "ftol": 1e-6})
        n_iter.append(r.nit)
        return C(r.x)

    def record(c, tag):
        M.at(c)
        q = M.q
        row = prob.row(1, len(prob.history), "surrogate", c, geometry(pod.reconstruct(c), c), q[:, :, 0].mean(0), q[:, :, 1].mean(0), q[:, :, 0].std(0))
        row["start"] = tag
        prob.history.append(row)
        return row

    rng, n_iter = np.random.default_rng(a.seed), []
    main, capped = (1, 0) if a.swap else (0, 1)                             # objective minimised, objective capped by eps
    QC = ["L_ADP", "L_OP"][capped]
    starts = lambda target: known.iloc[np.argsort(np.abs(known[QC].to_numpy() - target) + 1e-9 * rng.random(len(known)))[:a.starts]]  # noqa: E731
    for obj in (0, 1):                                                       # anchors: min L_ADP, min L_OP
        tgt = known[QC].min() if obj == capped else known[QC].max()
        for _, s in starts(tgt).iterrows():
            record(solve(obj, s[[f"c{k + 1}" for k in range(5)]].to_numpy(float)), f"anchor{obj}")
    h = pd.DataFrame(prob.history)
    sf = h[(h.source == "surrogate") & h.feasible]
    e_lo, e_hi = sf[QC].min(), sf.loc[sf[["L_ADP", "L_OP"][main]].idxmin(), QC]
    if a.caps == "useful":
        doe = pd.read_csv(os.path.join(DOE, "doe_results.csv"), dtype={"blade": str})
        e_hi = min(e_hi, float(doe.loc[doe.blade == "baseline", QC].iloc[0]))
    prev = None
    for eps in np.linspace(e_hi, e_lo, a.n_eps + 2)[1:-1]:                   # from the loosest cap down: warm start from the previous one
        c0s = [s[[f"c{k + 1}" for k in range(5)]].to_numpy(float) for _, s in starts(eps).iterrows()]
        if prev is not None and prev["feasible"]:
            c0s = [np.array([prev[f"c{k + 1}"] for k in range(5)])] + c0s[:a.starts - 1]
        for c0 in c0s:
            prev = record(solve(main, c0, eps), f"eps{eps:.5f}")
    wall = time.time() - t0
    front = prob.save(a.out)
    h = pd.DataFrame(prob.history)
    json.dump({"optimiser": "SLSQP, epsilon-constraint", "surrogate": "inr", "models": a.models, "data": a.data, "kappa": a.kappa, "swap": a.swap,
               "n_eps": a.n_eps, "starts": a.starts, "caps": a.caps, "geometric_constraints": "torch autodiff", "seed": a.seed, "local_optima": int((h.source == "surrogate").sum()),
               "feasible_candidates": int(h[h.source == "surrogate"].feasible.sum()), "pareto_size": len(front),
               "surrogate_evaluations": M.n_eval, "slsqp_iterations_median": int(np.median(n_iter)), "optimisation_wall_s": round(wall, 1), "s_per_evaluation": round(wall / M.n_eval, 4),
               "surrogate_fit_s": sum(json.load(open(os.path.join(m, "summary.json")))["train_wall_s"] for m in a.models),
               "predictor_device": a.device, "ensemble_size": len(a.models)}, open(os.path.join(a.out, "run.json"), "w"), indent=1)
    print(f"{len(front)} Pareto blades from {int((h.source == 'surrogate').sum())} local optima, {M.n_eval} evaluations (value + gradient), {wall:.0f} s")


if __name__ == "__main__":
    main()
