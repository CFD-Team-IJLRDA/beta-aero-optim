"""
Brute-force RANS optimisation of the LRN-OGV cascade: NSGA-II on the POD shape coefficients,
each candidate simulated with OpenFOAM at the three operating points.

- Design variables: the POD coefficients c1..cN of the BladeGen shape basis, within the POD box.
- Objectives: L_ADP = w_ADP and L_OP = (w_OP1 + w_OP2) / 2 (mixed-out loss coefficients).
- Constraints (feasible when <= 0), after Ciarlatani et al., ASME GT2026, Eqs. (3)-(4):
  - geometry, checked before any CFD (a violating blade is not simulated):
    t_max/c, x_tmax/c_ax, A/c^2 and x_cg/c_ax within +-30/20/20/20 % of the baseline,
    r_LE > 0.5 % c, and "in_bladegen": the blade must lie in the BladeGen family, i.e. its
    distance to the nearest POD training blade (standardised coefficients) must not exceed the
    95th percentile of the training blades' own nearest-neighbour distances;
  - flow, after CFD: |outflow angle - reference| <= tolerance at each operating point
    (reference = the OpenFOAM baseline by default), and every run must be usable
    (stationary time-average, see cascade_qoi.run_quality).
  Blades that were not simulated or whose runs failed get objectives [1, 1] (total loss);
  NSGA-II ranks infeasible blades by constraint violation only.
- The initial population is selected (non-dominated sorting + crowding) from a finished DoE,
  so it costs no CFD.

Outputs in config["optim"]["outdir"]: history.csv (every candidate, including the DoE blades
it started from), pareto.csv, Figs/loss_plane_g<gen>.png, and the usual profiles/MESH/OPENFOAM.
Re-running the same config resumes: finished blades are reloaded instead of rerun.
"""
import copy
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pymoo.algorithms.moo.nsga2 import RankAndCrowding  # noqa: E402
from pymoo.core.population import Population  # noqa: E402
from pymoo.core.problem import Problem  # noqa: E402
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from aero_optim.geom import (get_area, get_camber_th, get_chords, get_cog,  # noqa: E402
                             le_radius_violation, self_intersects, split_profile)
from aero_optim.shape.bladegen_pod import BladeGenPOD  # noqa: E402
from aero_optim.simulator.cascade_batch import run_blades  # noqa: E402
from aero_optim.simulator.cascade_qoi import run_quality  # noqa: E402
from aero_optim.simulator.openfoam import OpenFOAMSimulator  # noqa: E402
from aero_optim.utils import from_dat  # noqa: E402

logger = logging.getLogger(__name__)

OPS = ["ADP", "OP1", "OP2"]
GEOM_TOL = {"t_max/c": 0.3, "x_tmax/c_ax": 0.2, "A/c2": 0.2, "x_cg/c_ax": 0.2}  # relative to baseline
ANGLE_TOL = {"ADP": 0.8, "OP1": 1.5, "OP2": 1.5}  # deg
CONSTRAINTS = [f"g_{k}" for k in GEOM_TOL] + ["g_r_le", "g_in_bladegen"] \
    + [f"g_angle_{op}" for op in OPS] + ["g_run_quality"]
NOT_SIMULATED_F = [1., 1.]
DPI = 600


def blade_metrics(profile: np.ndarray) -> dict[str, float]:
    """**Returns** the geometric quantities constrained in the paper (Eq. 3)."""
    upper, lower = split_profile(profile)
    c, c_ax = get_chords(profile)
    _, t_max, x_tmax, _ = get_camber_th(upper, lower, interpolate=True)
    return {"t_max/c": t_max / c, "x_tmax/c_ax": x_tmax / c_ax, "A/c2": abs(get_area(profile)) / c**2,
            "x_cg/c_ax": get_cog(profile)[0] / c_ax, "c": c}


class GeometryConstraints:
    """Geometric constraints of a candidate blade, relative to the baseline blade."""
    def __init__(self, baseline: np.ndarray, train_coeffs: np.ndarray, tol: dict | None = None,
                 r_le_min: float = 0.005, cloud_quantile: float = 95.):
        self.ref = blade_metrics(baseline)
        self.tol = {**GEOM_TOL, **(tol or {})}
        self.r_le_min = r_le_min
        self.scale = train_coeffs.std(axis=0)
        self.tree = cKDTree(train_coeffs / self.scale)
        nn = self.tree.query(train_coeffs / self.scale, k=2)[0][:, 1]
        self.max_dist = float(np.percentile(nn, cloud_quantile))

    def __call__(self, profile: np.ndarray, coeffs: np.ndarray) -> dict[str, float]:
        """**Returns** the metrics and the constraint values g_* (<= 0: satisfied)."""
        out = {"dist_bladegen": float(self.tree.query(coeffs / self.scale)[0])}
        if self_intersects(profile):
            return {**out, **{g: 1. for g in CONSTRAINTS[:len(GEOM_TOL) + 2]}}
        m = blade_metrics(profile)
        out.update({k: m[k] for k in GEOM_TOL})
        out.update({f"g_{k}": abs(m[k] - self.ref[k]) / self.ref[k] - self.tol[k] for k in GEOM_TOL})
        out["g_r_le"] = le_radius_violation(profile, self.r_le_min * m["c"]) / (self.r_le_min * m["c"])
        out["g_in_bladegen"] = out["dist_bladegen"] / self.max_dist - 1.
        return out


def flow_outcome(case_dir: str, angle_ref: dict[str, float], angle_tol: dict[str, float],
                 max_rel_std: float = 5., max_drift: float = 1.) -> dict:
    """
    **Returns** the objectives and flow constraints of a simulated blade from its OpenFOAM
    folder (qois.csv and each operating point's qoi_history.csv).
    """
    out = {}
    summary_file = os.path.join(case_dir, "qois.csv")
    summary = pd.read_csv(summary_file, index_col=0) if os.path.isfile(summary_file) else None
    reasons, quality = [], []
    for op in OPS:
        ok = summary is not None and op in summary.index and summary.at[op, "status"] == "ok"
        w = summary.at[op, "MixedoutLossCoef"] if ok else np.nan
        angle = summary.at[op, "OutflowAngle"] if ok else np.nan
        out[f"{op}_w"], out[f"{op}_angle"] = w, angle
        out[f"g_angle_{op}"] = abs(angle - angle_ref[op]) - angle_tol[op] if ok else 1.
        if not ok:
            reasons.append(f"{op}: run failed")
            quality.append(1.)
            continue
        q = run_quality(pd.read_csv(os.path.join(case_dir, op, "qoi_history.csv"), index_col=0),
                        max_rel_std, max_drift)
        out[f"{op}_rel_std_%"], out[f"{op}_drift_%"] = q["rel_std_%"], q["drift_%"]
        if q["reason"]:
            reasons.append(f"{op}: {q['reason']}")
        quality.append(1. if not w > 0 else max(q["rel_std_%"] / max_rel_std, q["drift_%"] / max_drift) - 1.)
    out["g_run_quality"] = max(quality)
    out["run_quality"] = "; ".join(reasons)
    simulated = all(np.isfinite(out[f"{op}_w"]) for op in OPS)
    out["L_ADP"] = out["ADP_w"] if simulated else NOT_SIMULATED_F[0]
    out["L_OP"] = 0.5 * (out["OP1_w"] + out["OP2_w"]) if simulated else NOT_SIMULATED_F[1]
    return out


class CascadeFoamProblem(Problem):
    """
    The optimisation problem; `_evaluate` meshes and simulates a whole generation in parallel.

    Config `"optim"` entries:
    - outdir (str): output folder of the optimisation.
    - pod_dir (str): folder with basis.npz and dataset.npz (the BladeGen training blades).
    - init_doe (str): finished DoE folder (doe_results.csv, OPENFOAM/) to start from.
    - pop_size (int), n_gen (int), seed (int), budget (int): NSGA-II and parallel settings.
    - angle_tol (dict), angle_ref (dict, optional): outflow-angle windows [deg];
      the reference defaults to the baseline angles of init_doe.
    - geom_tol (dict), r_le_min (float), cloud_quantile (float): geometric constraints.
    - max_rel_std, max_drift (float): run quality thresholds [%].
    """
    def __init__(self, config: dict):
        opt = config["optim"]
        self.config = copy.deepcopy(config)
        self.outdir = opt["outdir"]
        self.config["study"]["outdir"] = self.outdir
        os.makedirs(os.path.join(self.outdir, "Figs"), exist_ok=True)
        self.budget = opt.get("budget", 96)
        self.pod = BladeGenPOD.load(os.path.join(opt["pod_dir"], "basis.npz"))
        ds = np.load(os.path.join(opt["pod_dir"], "dataset.npz"))
        train = np.array([self.pod.project(d) for d in ds["D"][~ds["failed"]]])
        self.baseline = np.array(from_dat(config["study"]["file"], 2, 1))[:, :2]
        self.geometry = GeometryConstraints(self.baseline, train, opt.get("geom_tol"),
                                            opt.get("r_le_min", 0.005), opt.get("cloud_quantile", 95.))
        self.init_doe = opt["init_doe"]
        doe = pd.read_csv(os.path.join(self.init_doe, "doe_results.csv"), dtype={"blade": str})
        base = doe[doe.blade == "baseline"].iloc[0]
        self.baseline_L = (base["L_ADP"], base["L_OP"])
        self.angle_ref = opt.get("angle_ref") or {op: float(base[f"{op}_angle"]) for op in OPS}
        self.angle_tol = {**ANGLE_TOL, **opt.get("angle_tol", {})}
        self.quality = (opt.get("max_rel_std", 5.), opt.get("max_drift", 1.))
        self.sim = OpenFOAMSimulator(self.config)
        self.gen = 1
        self.history: list[dict] = []
        logger.info(f"outflow angle windows: { {op: (round(self.angle_ref[op] - self.angle_tol[op], 3), round(self.angle_ref[op] + self.angle_tol[op], 3)) for op in OPS} }")
        logger.info(f"BladeGen family: distance <= {self.geometry.max_dist:.3f} (standardised POD units)")
        super().__init__(n_var=self.pod.n_modes, n_obj=2, n_ieq_constr=len(CONSTRAINTS),
                         xl=self.pod.bounds[:, 0], xu=self.pod.bounds[:, 1])

    def _row(self, gen, cid, name, source, coeffs, geom, flow) -> dict:
        row = {"gen": gen, "cid": cid, "blade": name, "source": source}
        row.update({f"c{k + 1}": v for k, v in enumerate(coeffs)})
        row.update(geom)
        row.update(flow)
        g = np.array([row[k] for k in CONSTRAINTS])
        row["CV"] = float(np.sum(np.maximum(g, 0.)))
        row["feasible"] = row["CV"] == 0.
        return row

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs):
        gid = self.gen
        geoms = [self.geometry(self.pod.reconstruct(x), x) for x in X]
        feasible = [i for i, g in enumerate(geoms) if all(g[k] <= 0 for k in CONSTRAINTS[:len(GEOM_TOL) + 2])]
        names = [f"g{gid:03d}_c{cid:03d}" for cid in range(len(X))]
        logger.info(f"generation {gid}: {len(feasible)} / {len(X)} candidates pass the geometric constraints")
        run_blades(self.config, self.sim, [(names[i], gid, i, self.pod.reconstruct(X[i])) for i in feasible],
                   self.outdir, self.budget)
        rows = []
        for cid, (x, geom) in enumerate(zip(X, geoms)):
            if cid in feasible:
                flow = flow_outcome(self.sim.get_sim_outdir(gid, cid), self.angle_ref, self.angle_tol, *self.quality)
            else:
                flow = {**{f"g_angle_{op}": 1. for op in OPS}, "g_run_quality": 1., "run_quality": "not simulated",
                        "L_ADP": NOT_SIMULATED_F[0], "L_OP": NOT_SIMULATED_F[1]}
            rows.append(self._row(gid, cid, names[cid], "optim", x, geom, flow))
        self.history.extend(rows)
        out["F"] = np.array([[r["L_ADP"], r["L_OP"]] for r in rows])
        out["G"] = np.array([[r[k] for k in CONSTRAINTS] for r in rows])
        self.observe()
        self.gen += 1

    def initial_population(self, pop_size: int) -> Population:
        """
        **Evaluates** the finished DoE blades (no CFD) and **returns** the pop_size best ones
        (non-dominated sorting, then crowding; infeasible ones ranked by constraint violation).
        """
        doe = pd.read_csv(os.path.join(self.init_doe, "doe_results.csv"), dtype={"blade": str})
        doe = doe[doe.blade != "baseline"]
        cols = [f"c{k + 1}" for k in range(self.pod.n_modes)]
        rows = []
        for _, r in doe.iterrows():
            x = r[cols].to_numpy(float)
            case = os.path.join(self.init_doe, "OPENFOAM", f"openfoam_g0_c{int(r.blade)}")
            rows.append(self._row(0, int(r.blade), r.blade, "doe", x, self.geometry(self.pod.reconstruct(x), x),
                                  flow_outcome(case, self.angle_ref, self.angle_tol, *self.quality)))
        self.history.extend(rows)
        pop = Population.new(X=np.array([[r[c] for c in cols] for r in rows]),
                             F=np.array([[r["L_ADP"], r["L_OP"]] for r in rows]),
                             G=np.array([[r[k] for k in CONSTRAINTS] for r in rows]),
                             H=np.zeros((len(rows), 0)))
        for ind in pop:
            ind.evaluated.update({"F", "G", "H"})
        pop = RankAndCrowding().do(self, pop, n_survive=pop_size)
        logger.info(f"initial population: {pop_size} of {len(rows)} DoE blades "
                    f"({sum(r['feasible'] for r in rows)} feasible)")
        self.observe(gen=0)
        return pop

    def observe(self, gen: int | None = None):
        """**Writes** history.csv and pareto.csv and **plots** the loss plane."""
        gen = self.gen if gen is None else gen
        h = pd.DataFrame(self.history)
        h.to_csv(os.path.join(self.outdir, "history.csv"), index=False)
        feas = h[h.feasible]
        front = feas.iloc[NonDominatedSorting().do(feas[["L_ADP", "L_OP"]].to_numpy(), only_non_dominated_front=True)] \
            if len(feas) else feas
        front = front.sort_values("L_ADP")
        front.to_csv(os.path.join(self.outdir, "pareto.csv"), index=False)
        logger.info(f"after generation {gen}: {len(feas)} feasible blades, {len(front)} on the Pareto front")

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        sim_ok = h[h.L_ADP < NOT_SIMULATED_F[0]]
        ax.scatter(sim_ok[~sim_ok.feasible].L_ADP, sim_ok[~sim_ok.feasible].L_OP, s=8, color="#d9d8d3", lw=0,
                   label=f"infeasible ({(~sim_ok.feasible).sum()})")
        ax.scatter(feas[feas.source == "doe"].L_ADP, feas[feas.source == "doe"].L_OP, s=8, color="#b5b3ab", lw=0,
                   label=f"DoE, feasible ({(feas.source == 'doe').sum()})")
        opt = feas[feas.source == "optim"]
        if len(opt):
            sc = ax.scatter(opt.L_ADP, opt.L_OP, s=12, c=opt.gen, cmap="Blues", vmin=0, vmax=max(gen, 1), lw=0,
                            label=f"optimisation, feasible ({len(opt)})")
            fig.colorbar(sc, ax=ax, label="generation")
        ax.plot(front.L_ADP, front.L_OP, color="#0b0b0b", lw=1, marker="o", ms=3, label=f"Pareto front ({len(front)})")
        ax.scatter(*self.baseline_L, marker="*", s=160, color="#eb6834", zorder=5, label="baseline")
        ax.set_xlabel("L_ADP = w_ADP")
        ax.set_ylabel("L_OP = (w_OP1 + w_OP2) / 2")
        ax.set_title(f"generation {gen}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(self.outdir, "Figs", f"loss_plane_g{gen:03d}.png"), dpi=DPI)
        plt.close(fig)
