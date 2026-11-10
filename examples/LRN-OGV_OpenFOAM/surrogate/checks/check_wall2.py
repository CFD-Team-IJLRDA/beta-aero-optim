"""
Sanity checks of a dataset folder built from scratch (default data_wall2/, the wall x2-mesh DoE):
  1. transformation quality over all blades (quality.csv): det J min, RBF boundary residual, cells outside the re-cut;
  2. on 5 random blades + the baseline, T_geom rebuilt from the mesh: boundary residual |T_geom(c_i) - target_i|,
     rows of X_geom reproduced, outside-re-cut fraction;
  3. QoIs recomputed from the stored mean fields, linearly interpolated on MP1/MP2 (cascade_qoi.plane_qois),
     vs the stored qoi (the solver's line samples averaged over the same window). Two variants: plain (the line ends
     outside the cell-centre hull are NaN and dropped, as in eval_inr.py's 'snapshot' QoIs) and periodic (cell centres
     copied +- one pitch, so the whole line is interpolated);
  4. aero_optim.surrogate.data.raw_arrays on 3 blades: shapes / dtypes vs the reference dataset (data_mean/).
Writes <data>/checks/qoi_consistency.csv and prints a summary.

Usage: ~/miniconda3/envs/ipynb/bin/python checks/check_wall2.py [--data data_wall2] [--ref data_mean] [--jobs 32]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts folder: _repo, train_inr, ...
import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
HERE = _repo.WORK                 # data, runs, results: the surrogate work folder
sys.path.insert(0, _repo.SRC)
import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
OPS = ["ADP", "OP1", "OP2"]


def plane_df(f):
    return pd.DataFrame({"p": f[:, 0], "rho": f[:, 1], "Ux": f[:, 2], "Uy": f[:, 3]})


def qoi_one(args):
    data, name = args
    from scipy.interpolate import LinearNDInterpolator
    from aero_optim.simulator.cascade_qoi import plane_qois
    d = np.load(os.path.join(HERE, data, "blades", f"{name}.npz"))
    F = np.hstack([d["fields"][k][:, :4] for k in range(3)])
    it = LinearNDInterpolator(d["X_phys"], F)
    m1, m2 = it(d["mp1_phys"]), it(d["mp2_phys"])
    pitch = json.load(open(os.path.join(HERE, data, "meta.json")))["pitch"]
    per = LinearNDInterpolator(np.vstack([d["X_phys"] + [0, s * pitch] for s in (0, 1, -1)]), np.vstack([F, F, F]))
    p1, p2 = per(d["mp1_phys"]), per(d["mp2_phys"])
    rows = []
    for k, op in enumerate(OPS):
        c = slice(4 * k, 4 * k + 4)
        q = plane_qois(plane_df(m1[:, c]), plane_df(m2[:, c]))
        qp = plane_qois(plane_df(p1[:, c]), plane_df(p2[:, c]))
        rows.append(dict(blade=name, op=op, w_fields=q["MixedoutLossCoef"], w_fields_periodic=qp["MixedoutLossCoef"],
                         w_qoi=d["qoi"][k, 0], phi_fields=q["OutflowAngle"], phi_fields_periodic=qp["OutflowAngle"],
                         phi_qoi=d["qoi"][k, 1],
                         nan_mp1=int(np.isnan(m1[:, 4 * k]).sum()), nan_mp2=int(np.isnan(m2[:, 4 * k]).sum())))
    return rows


def qoi_table(data, jobs):
    names = sorted(f[:-4] for f in os.listdir(os.path.join(HERE, data, "blades")))
    with ProcessPoolExecutor(jobs) as ex:
        df = pd.DataFrame([r for rows in ex.map(qoi_one, [(data, n) for n in names]) for r in rows])
    df["w_rel_err_%"] = 100 * (df.w_fields / df.w_qoi - 1)
    df["phi_err_deg"] = df.phi_fields - df.phi_qoi
    df["w_rel_err_periodic_%"] = 100 * (df.w_fields_periodic / df.w_qoi - 1)
    df["phi_err_periodic_deg"] = df.phi_fields_periodic - df.phi_qoi
    return df


def qoi_summary(df, label):
    print(f"\nQoI from interpolated mean fields vs stored qoi, {label} ({df.blade.nunique()} blades):")
    for tag, sfx in (("plain (NaN line ends dropped)", ""), ("periodic (whole line)", "_periodic")):
        print(f" {tag}:")
        for op in OPS + ["all"]:
            s = df if op == "all" else df[df.op == op]
            e, a = s[f"w_rel_err{sfx}_%"], s[f"phi_err{sfx}_deg"]
            print(f"  {op:4s} loss: mean |rel err| {e.abs().mean():.2f} %, mean (bias) {e.mean():+.2f} %, "
                  f"median |.| {e.abs().median():.2f} %, max |.| {e.abs().max():.2f} % | "
                  f"angle: mean |err| {a.abs().mean():.4f} deg, max {a.abs().max():.4f} deg")


def morph_check(data, name):
    import torch
    from aero_optim.surrogate.foam import CascadeMesh
    from aero_optim.surrogate.morph import RBFMorph, geometric_targets
    from aero_optim.surrogate.recut import PITCH, Recut
    import build_dataset as B
    meta = json.load(open(os.path.join(HERE, data, "meta.json")))
    case = next(b["case"] for b in meta["blades"] if b["name"] == name)
    d = np.load(os.path.join(HERE, data, "blades", f"{name}.npz"))
    mesh = CascadeMesh(os.path.join(case, "mesh", "constant", "polyMesh"))
    rc = Recut(mesh)
    pts, lab = rc.boundary()
    tgt = geometric_targets(rc, pts, lab, B.X_LE, B.X_TE)
    morph = RBFMorph(pts, tgt, **B.RBF)
    res = float(np.abs(morph(pts).numpy() - tgt).max())
    X_cut = rc.apply(mesh.centroids)
    Xg = morph(X_cut).numpy()
    Xg[:, 1] = np.clip(Xg[:, 1], 0.0, PITCH)
    return dict(blade=name, cells=len(Xg), n_centres=len(pts), residual_recomputed=res,
                residual_stored=float(d["rbf_residual"]), X_geom_maxdiff=float(np.abs(Xg - d["X_geom"]).max()),
                X_phys_maxdiff=float(np.abs(mesh.centroids - d["X_phys"]).max()),
                outside_recut=rc.check(X_cut), detJ_min=float(morph.jacobian_det(X_cut).min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_wall2")
    ap.add_argument("--ref", default="data_mean", help="reference dataset for the layout check and the QoI comparison")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    pd.set_option("display.width", 200)

    q = pd.read_csv(os.path.join(HERE, args.data, "quality.csv"))
    print(f"1. quality over {len(q)} blades (min / median / max):")
    print(q[["cells", "detJ_min", "detJ_q01", "rbf_residual", "outside_recut"]].describe().T[["min", "50%", "max"]].to_string())
    print(f"   blades with detJ_min <= 0: {(q.detJ_min <= 0).sum()}, with outside_recut > 0: {(q.outside_recut > 0).sum()}, "
          f"with rbf_residual >= 1e-11: {(q.rbf_residual >= 1e-11).sum()}")

    names = sorted(n for n in q.blade if n != "baseline")
    pick = list(np.random.default_rng(args.seed).choice(names, 5, replace=False)) + ["baseline"]
    with ProcessPoolExecutor(6) as ex:
        m = pd.DataFrame(list(ex.map(morph_check, [args.data] * 6, pick)))
    print("\n2. T_geom rebuilt from the mesh on 5 random blades + baseline:")
    print(m.to_string(index=False))

    out = os.path.join(HERE, args.data, "checks")
    os.makedirs(out, exist_ok=True)
    df = qoi_table(args.data, args.jobs)
    df.to_csv(os.path.join(out, "qoi_consistency.csv"), index=False)
    qoi_summary(df, args.data)
    print(f"  interpolation NaN points per line (line ends outside the cell-centre hull): "
          f"MP1 max {df.nan_mp1.max()}, MP2 max {df.nan_mp2.max()}")
    if args.ref:
        qoi_summary(qoi_table(args.ref, args.jobs), f"{args.ref} (same check, for reference)")

    from aero_optim.surrogate import data as D
    print("\n4. aero_optim.surrogate.data.raw_arrays on 3 blades:")
    sp = {}
    for data in (args.data, args.ref):
        D.set_data(data)
        bal = D.load_balance()
        names3 = D.splits()["train"][:2] + ["baseline"]
        arrs = D.raw_arrays(names3, bal)
        arrs_p = D.raw_arrays(names3[:1], bal, coords="periodic", mp_points=1) if os.path.isdir(os.path.join(D.DATA, "mp_aug")) else None
        sp[data] = [(a.shape, a.dtype) for a in arrs]
        print(f"  {data} {names3}: " + ", ".join(f"{s}/{t}" for s, t in sp[data])
              + (f" | periodic + mp_aug: " + ", ".join(f"{a.shape}/{a.dtype}" for a in arrs_p) if arrs_p else ""))
        x = arrs[1]
        print(f"    x_hat range [{x[:, 0].min():.4f}, {x[:, 0].max():.4f}], y_hat [{x[:, 1].min():.4f}, {x[:, 1].max():.4f}]")
    ok = all(a[0][1:] == b[0][1:] and a[1] == b[1] for a, b in zip(sp[args.data], sp[args.ref]))
    print(f"  trailing shapes and dtypes identical to {args.ref}: {ok}")


if __name__ == "__main__":
    main()
