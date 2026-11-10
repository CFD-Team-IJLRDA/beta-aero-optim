"""
NSGA-II optimisation of the LRN-OGV cascade with the INR surrogate instead of CFD; the problem is the brute-force one
(aero_optim/optim/cascade_foam.py): design variables = the 5 POD coefficients, objectives L_ADP = w_ADP and
L_OP = (w_OP1 + w_OP2)/2, constraints ACTIVE and identical: geometry (t_max/c, x_tmax/c_ax, A/c^2, x_cg/c_ax, r_LE),
"in the BladeGen family", outflow-angle windows around the baseline at the three operating points. Candidates that
violate a geometric constraint are not evaluated (F = [1, 1], as in the brute-force run).
The only constraint that cannot be checked is the CFD run quality (set to satisfied).

Initial population: like the brute-force run, the pop_size best blades (non-dominated sorting + crowding, infeasible ranked
by constraint violation) of a set of blades whose CFD values are known: the training+validation blades of the data folder
(--data), i.e. exactly the CFD budget the surrogate was trained on.

Outputs in --out: history.csv (every evaluated candidate: coefficients, geometry, predicted losses/angles and the spread
of the ensemble, constraints, feasibility), pareto.csv (feasible non-dominated candidates, predicted), loss_plane.png.

Surrogates: the INR ensemble (--models run dirs) or, with --gp, the scalar GP baseline (gp_baseline.GPModel: one GP per OP and QoI on
the 5 POD coefficients) fitted on the same CFD budget (train + val blades of --data).

Usage: ~/miniconda3/envs/ogv_torch/bin/python surrogate_optim.py (--models runs/m0 runs/m1 runs/m2 | --gp) --out sopt/x
         [--data data_mean] [--pop 32] [--gen 30] [--seed 1] [--angle-margin 0.0] [--device cuda:1]
"""
import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
HERE = _repo.WORK                 # data, runs, sopt: the surrogate work folder (aero_optim/surrogate/paths.py)
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowding  # noqa: E402
from pymoo.core.population import Population  # noqa: E402
from pymoo.core.problem import Problem  # noqa: E402
from pymoo.optimize import minimize  # noqa: E402
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting  # noqa: E402

from aero_optim.optim.cascade_foam import (ANGLE_TOL, CONSTRAINTS, GEOM_TOL, NOT_SIMULATED_F,  # noqa: E402
                                           GeometryConstraints)
from aero_optim.shape.bladegen_pod import BladeGenPOD  # noqa: E402
from aero_optim.utils import from_dat  # noqa: E402
from aero_optim.surrogate import data as D  # noqa: E402
from aero_optim.surrogate.predict import OPS, Predictor  # noqa: E402
from gp_baseline import GPModel, table  # noqa: E402

from aero_optim.surrogate.paths import BASELINE_DAT, POD_DIR  # noqa: E402
ANGLE_REF = {"ADP": 0.8572, "OP1": 1.8329, "OP2": 0.0227}     # OpenFOAM baseline angles (medium mesh DoE, doe_results.csv)
N_GEOM = len(GEOM_TOL) + 2


class GPPredictor:
    """Same interface as aero_optim.surrogate.predict.Predictor: qois(profiles, pods) -> (n, 1, 3 OPs, [loss, angle])."""
    def __init__(self, data):
        D.set_data(data)
        sp = D.splits()
        X, Q = table(sp["train"] + sp["val"])
        t0 = time.time()
        self.gp = GPModel(X, Q)
        self.fit_s = time.time() - t0

    def qois(self, profiles, pods):
        return self.gp.predict(np.asarray(pods))[:, None]


class SurrogateProblem(Problem):
    def __init__(self, predictor, pod, geometry, angle_ref, angle_tol, angle_margin=0.0):
        self.P, self.pod, self.geometry = predictor, pod, geometry
        self.angle_ref, self.angle_tol, self.margin = angle_ref, angle_tol, angle_margin
        self.history, self.gen = [], 1
        super().__init__(n_var=pod.n_modes, n_obj=2, n_ieq_constr=len(CONSTRAINTS), xl=pod.bounds[:, 0], xu=pod.bounds[:, 1])

    def row(self, gen, cid, source, x, geom, loss, angle, loss_std=None):
        r = {"gen": gen, "cid": cid, "source": source}
        r.update({f"c{k + 1}": v for k, v in enumerate(x)})
        r.update(geom)
        simulated = loss is not None
        for k, op in enumerate(OPS):
            r[f"{op}_w"] = loss[k] if simulated else np.nan
            r[f"{op}_angle"] = angle[k] if simulated else np.nan
            r[f"g_angle_{op}"] = abs(angle[k] - self.angle_ref[op]) - self.angle_tol[op] + self.margin if simulated else 1.
            if loss_std is not None:
                r[f"{op}_w_std"] = loss_std[k]
        r["g_run_quality"] = -1. if simulated else 1.
        r["L_ADP"] = loss[0] if simulated else NOT_SIMULATED_F[0]
        r["L_OP"] = 0.5 * (loss[1] + loss[2]) if simulated else NOT_SIMULATED_F[1]
        g = np.array([r[k] for k in CONSTRAINTS])
        r["CV"] = float(np.sum(np.maximum(g, 0.)))
        r["feasible"] = r["CV"] == 0.
        return r

    def _evaluate(self, X, out, *args, **kwargs):
        geoms = [self.geometry(self.pod.reconstruct(x), x) for x in X]
        ok = [i for i, g in enumerate(geoms) if all(g[k] <= 0 for k in CONSTRAINTS[:N_GEOM])]
        q = self.P.qois([self.pod.reconstruct(X[i]) for i in ok], X[ok]) if ok else np.zeros((0, 1, 3, 2))
        rows = []
        for cid, (x, g) in enumerate(zip(X, geoms)):
            if cid in ok:
                e = q[ok.index(cid)]                                      # (n_models, 3, 2)
                rows.append(self.row(self.gen, cid, "surrogate", x, g, e[:, :, 0].mean(0), e[:, :, 1].mean(0), e[:, :, 0].std(0)))
            else:
                rows.append(self.row(self.gen, cid, "surrogate", x, g, None, None))
        self.history.extend(rows)
        out["F"] = np.array([[r["L_ADP"], r["L_OP"]] for r in rows])
        out["G"] = np.array([[r[k] for k in CONSTRAINTS] for r in rows])
        self.gen += 1

    def initial_population(self, data, pop_size):
        """The pop_size best blades among the CFD-known blades of the data folder (train + val), true values."""
        D.set_data(data)
        sp = D.splits()
        rows = []
        for name in sp["train"] + sp["val"]:
            d = D.load_blade(name)
            x = np.asarray(d["pod"], float)
            qoi = d["qoi"]                                               # (3 OPs, [loss, angle]) time-averaged CFD
            rows.append(self.row(0, int(name.split("_")[-1]) if name != "baseline" else -1, "cfd", x,
                                 self.geometry(self.pod.reconstruct(x), x), qoi[:, 0], qoi[:, 1]))
        self.history.extend(rows)
        pop = Population.new(X=np.array([[r[f"c{k + 1}"] for k in range(self.n_var)] for r in rows]),
                             F=np.array([[r["L_ADP"], r["L_OP"]] for r in rows]),
                             G=np.array([[r[k] for k in CONSTRAINTS] for r in rows]), H=np.zeros((len(rows), 0)))
        for ind in pop:
            ind.evaluated.update({"F", "G", "H"})
        return RankAndCrowding().do(self, pop, n_survive=pop_size), len(rows), sum(r["feasible"] for r in rows)

    def save(self, out):
        h = pd.DataFrame(self.history)
        h.to_csv(os.path.join(out, "history.csv"), index=False)
        feas = h[h.feasible]
        front = feas.iloc[NonDominatedSorting().do(feas[["L_ADP", "L_OP"]].to_numpy(), only_non_dominated_front=True)] if len(feas) else feas
        front = front.sort_values("L_ADP")
        front.to_csv(os.path.join(out, "pareto.csv"), index=False)
        fig, ax = plt.subplots(figsize=(6, 4.6))
        for src, col, lab in (("cfd", "#b5b3ab", "CFD-known blades"), ("surrogate", "#2a78d6", "surrogate candidates")):
            s = feas[feas.source == src]
            ax.scatter(s.L_ADP, s.L_OP, s=8, color=col, lw=0, label=f"{lab} (feasible, {len(s)})")
        ax.plot(front.L_ADP, front.L_OP, color="black", lw=1, marker="o", ms=3, label=f"predicted Pareto front ({len(front)})")
        ax.set_xlabel("L_ADP")
        ax.set_ylabel("L_OP")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "loss_plane.png"), dpi=300)
        plt.close(fig)
        return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--gp", action="store_true", help="GP baseline instead of the INR ensemble")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data_mean")
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--gen", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--angle-margin", type=float, default=0.0, help="deg: tighten the angle windows (conservative)")
    ap.add_argument("--checkpoint", default="best_qoi_model.pt")
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pod = BladeGenPOD.load(os.path.join(POD_DIR, "basis.npz"))
    ds = np.load(os.path.join(POD_DIR, "dataset.npz"))
    train = np.array([pod.project(d) for d in ds["D"][~ds["failed"]]])
    baseline = np.array(from_dat(BASELINE_DAT, 2, 1))[:, :2]
    geometry = GeometryConstraints(baseline, train)
    if args.gp:
        P = GPPredictor(args.data)
        fit_s = P.fit_s
    else:
        P = Predictor(args.models, data=args.data, checkpoint=args.checkpoint, device=args.device)
        fit_s = sum(json.load(open(os.path.join(m, "summary.json")))["train_wall_s"] for m in args.models)
    problem = SurrogateProblem(P, pod, geometry, ANGLE_REF, ANGLE_TOL, args.angle_margin)
    t0 = time.time()
    pop, n_known, n_feas = problem.initial_population(args.data, args.pop)
    print(f"initial population: {args.pop} of {n_known} CFD-known blades ({n_feas} feasible)", flush=True)
    algorithm = NSGA2(pop_size=args.pop, sampling=pop, eliminate_duplicates=True)
    minimize(problem, algorithm, ("n_gen", args.gen + 1), seed=args.seed, verbose=False)
    wall = time.time() - t0
    front = problem.save(args.out)
    h = pd.DataFrame(problem.history)
    json.dump({"surrogate": "gp" if args.gp else "inr", "models": args.models, "surrogate_fit_s": round(fit_s, 1), "data": args.data, "pop": args.pop, "gen": args.gen, "seed": args.seed,
               "cfd_known_blades": n_known, "surrogate_evaluations": int((h.source == "surrogate").sum()),
               "feasible_candidates": int(h[h.source == "surrogate"].feasible.sum()), "pareto_size": len(front),
               "angle_margin": args.angle_margin, "optimisation_wall_s": round(wall, 1),
               "s_per_evaluation": round(wall / max(1, int((h.source == "surrogate").sum())), 4),
               "predictor_device": "cpu" if args.gp else args.device, "ensemble_size": 1 if args.gp else len(args.models)}, open(os.path.join(args.out, "run.json"), "w"), indent=1)
    print(f"done: {len(front)} Pareto blades, {int((h.source == 'surrogate').sum())} surrogate evaluations", flush=True)


if __name__ == "__main__":
    main()
