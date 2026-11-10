"""
CFD verification of a surrogate-optimised front on the SAME solver set-up as the brute-force optimisation and the DoE
used for the surrogate (medium mesh, 6000 iterations, QoIs averaged over 4000-6000), so that losses are comparable.

Takes <run>/pareto.csv (written by surrogate_optim.py), keeps the blades proposed by the surrogate (source == surrogate),
picks at most --k of them evenly along the front (both ends always included), meshes and simulates them (3 OPs) and writes
<run>/verification.csv: predicted vs CFD loss/angle per OP, CFD objectives, feasibility with CFD values (angle windows and
run quality; geometry was already checked) and the wall time of the verification.

The CFD cases (MESH/, OPENFOAM/, profiles/) go to <AERO_RUNS>/cfd_sopt/<run name>/, outside this folder.

Usage: PATH=~/miniconda3/envs/aero_venv/bin:$PATH ~/miniconda3/envs/aero_venv/bin/python verify_cfd.py --run sopt/inr_N100_s0_o1 [--k 25] [--budget 48]
"""
import argparse
import copy
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
from aero_optim.optim.cascade_foam import ANGLE_TOL, CONSTRAINTS, OPS, flow_outcome  # noqa: E402
from aero_optim.shape.bladegen_pod import BladeGenPOD  # noqa: E402
from aero_optim.simulator.cascade_batch import run_blades  # noqa: E402
from aero_optim.simulator.openfoam import OpenFOAMSimulator  # noqa: E402

HERE = _repo.WORK                 # data, runs, sopt: the surrogate work folder (aero_optim/surrogate/paths.py)
from aero_optim.surrogate.paths import CFD_SOPT as CFD_ROOT, EXAMPLE as EX, POD_DIR, RUNS  # noqa: E402,F401  (AERO_RUNS on murphy)
ANGLE_REF = {"ADP": 0.8572, "OP1": 1.8329, "OP2": 0.0227}


def medium_config(outdir):
    c = json.load(open(os.path.join(EX, "openfoam_config.json")))
    c["study"]["outdir"] = outdir
    c["study"]["file"] = os.path.join(EX, "baseline_322.dat")
    c["mesh"].pop("template", None)                     # default (medium) cascade template
    c["simulator"].update(end_time=6000, sample_start=4000)
    c["simulator"]["ref_input"] = os.path.join(EX, c["simulator"]["ref_input"])
    return c


def pick(front, k):
    """k blades evenly spread along the front (ordered by L_ADP), ends included."""
    front = front.sort_values("L_ADP").reset_index(drop=True)
    if len(front) <= k:
        return front
    return front.iloc[np.unique(np.round(np.linspace(0, len(front) - 1, k)).astype(int))].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--budget", type=int, default=48, help="solver processes at once (3 per blade)")
    ap.add_argument("--tag", default="cfd")
    ap.add_argument("--select", default="spread", choices=["spread", "better"],
                    help="spread: --k blades evenly along the front; better: every predicted blade better than the baseline in both "
                         "losses (blades already simulated by an earlier check of this run are reused, not re-run)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run = os.path.abspath(a.run) if os.path.isabs(a.run) else os.path.join(HERE, a.run)
    outdir = os.path.join(CFD_ROOT, os.path.basename(run.rstrip("/")))
    os.makedirs(outdir, exist_ok=True)
    os.chdir(EX)
    config = medium_config(outdir)
    pod = BladeGenPOD.load(os.path.join(POD_DIR, "basis.npz"))
    front = pd.read_csv(os.path.join(run, "pareto.csv"))
    surr = front[front.source == "surrogate"]
    sim = OpenFOAMSimulator(config)
    cols = [f"c{j + 1}" for j in range(pod.n_modes)]
    if a.select == "spread":
        sel = pick(surr, a.k)
        ids = [(1, i) for i in range(len(sel))]
    else:
        doe = pd.read_csv(os.path.join(RUNS, "doe1000_bladegen", "doe_results.csv"), dtype={"blade": str})
        b0 = doe.loc[doe.blade == "baseline", ["L_ADP", "L_OP"]].to_numpy()[0]
        sel = surr[(surr.L_ADP < b0[0]) & (surr.L_OP < b0[1])].sort_values("L_ADP").reset_index(drop=True)
        old = os.path.join(run, "verification_cfd.csv")
        old = pd.read_csv(old)[cols].to_numpy() if os.path.isfile(old) else np.zeros((0, len(cols)))
        ids = []
        for i, r in sel.iterrows():
            hit = np.flatnonzero(np.isclose(old, r[cols].to_numpy(float), rtol=0, atol=1e-12).all(1))
            ids.append((1, int(hit[0])) if len(hit) else (2, i))          # gid 1 = earlier check (reused), gid 2 = new
    blades = [(f"s_g{g}_c{c:03d}", g, c, pod.reconstruct(r[cols].to_numpy(float))) for (g, c), (_, r) in zip(ids, sel.iterrows())]
    t0 = time.time()
    status = run_blades(config, sim, blades, outdir, a.budget)
    wall = time.time() - t0
    rows = []
    for (g_, c_), (i, r) in zip(ids, sel.iterrows()):
        f = flow_outcome(sim.get_sim_outdir(g_, c_), ANGLE_REF, ANGLE_TOL)
        row = {"blade": f"s_g{g_}_c{c_:03d}", "status": status.get((g_, c_), "missing"), "reused": g_ == 1 and a.select == "better"}
        row.update({f"c{j + 1}": r[f"c{j + 1}"] for j in range(pod.n_modes)})
        for op in OPS:
            row.update({f"pred_{op}_w": r[f"{op}_w"], f"pred_{op}_angle": r[f"{op}_angle"],
                        f"cfd_{op}_w": f[f"{op}_w"], f"cfd_{op}_angle": f[f"{op}_angle"], f"g_angle_{op}": f[f"g_angle_{op}"]})
        row.update(pred_L_ADP=r["L_ADP"], pred_L_OP=r["L_OP"], cfd_L_ADP=f["L_ADP"], cfd_L_OP=f["L_OP"],
                   g_run_quality=f["g_run_quality"], run_quality=f["run_quality"])
        g = [row[k] for k in CONSTRAINTS if k.startswith("g_angle") or k == "g_run_quality"]
        geom_ok = all(r[k] <= 0 for k in CONSTRAINTS if k not in row)
        row["cfd_feasible"] = bool(geom_ok and all(v <= 0 for v in g))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(run, f"verification_{a.tag}.csv"), index=False)
    json.dump({"select": a.select, "blades": len(blades), "cfd_runs": 3 * len(blades),
               "new_blades": int(sum(g == 2 for g, _ in ids)) if a.select == "better" else len(blades), "wall_s": round(wall), "budget_procs": a.budget,
               "solver": "rhoSimpleFoam, medium mesh, 6000 it, avg 4000-6000"},
              open(os.path.join(run, f"verification_{a.tag}.json"), "w"), indent=1)
    ok = out[out.status == "ok"]
    print(f"{len(ok)}/{len(out)} simulated in {wall / 60:.1f} min; CFD-feasible {int(out.cfd_feasible.sum())}; "
          f"mean |loss error| ADP {100 * ((ok.pred_ADP_w - ok.cfd_ADP_w).abs() / ok.cfd_ADP_w).mean():.2f} %")


if __name__ == "__main__":
    main()
