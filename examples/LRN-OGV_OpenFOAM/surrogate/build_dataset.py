"""
Step 1: OpenFOAM cascade cases -> per-blade arrays in the physical and in the reference (rectangle) domain.

For every blade (DoE kept blades, the optimisation Pareto blades, the baseline):
  X_phys (N, 2)     cell centroids [m]
  X_geom (N, 2)     T_geom(X_phys): re-cut + RBF morph onto [x_in, x_out] x [0, pitch] (T_bal is applied later,
                    after the train split, see fit_balance.py)
  fields (3, N, 6)  ADP/OP1/OP2 x cells x (p, rho, Ux, Uy, nut, Ma) at the last iteration
                    (--mean: fieldAverage means over the MP sampling window; Ma derived from the means)
  pod (5,)          POD coefficients c1..c5
  qoi (3, 2)        CFD mixed-out loss and outflow angle per OP (time-averaged, from qois.csv;
                    --mean: from qois_fieldavg.csv, the same window as the mean fields)
  mp1_*/mp2_*       measurement-plane points (1001 each) in the physical and reference domains
  detJ_min, rbf_residual, le, te
Rows are aligned: X_phys[i] <-> X_geom[i] <-> fields[:, i] (no inverse map needed).

Usage: ~/miniconda3/envs/ipynb/bin/python build_dataset.py [--jobs 32]
       ~/miniconda3/envs/ipynb/bin/python build_dataset.py --mean     # -> data_mean/, geometry reused from data/
       # any DoE straight from its mean fields, without a snapshot dataset (geometry recomputed, same re-cut/morph):
       ~/miniconda3/envs/ipynb/bin/python build_dataset.py --doe <runs>/doe1000_wall2 --out data_wall2 \
           --fields mean --qoi-file qois.csv --no-pareto --mesh "wall x2 template (cascade_wall2.geo)"
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
HERE = _repo.WORK                 # data, runs, sopt: the surrogate work folder (aero_optim/surrogate/paths.py)
RUNS = _repo.RUNS
OPS = ["ADP", "OP1", "OP2"]
FIELDS = ["p", "rho", "Ux", "Uy", "nut", "Ma"]
X_LE, X_TE = -0.00011, 0.06725          # reference LE/TE abscissae (baseline blade) [m]
RBF = dict(radius=0.1, kernel="wendland_c2")
MP1 = (-0.02, -0.0285704, 0.0118176)     # x, y_start, y_end: same lines as the solver's sampling
MP2 = (0.087, 0.000001, 0.040389)


def blade_list(doe=f"{RUNS}/doe1000_bladegen", pareto=True):
    """Kept blades of the DoE (dataset/dataset.csv) and, if pareto, the NSGA-II Pareto blades (medium mesh only)."""
    rows = []
    d = pd.read_csv(f"{doe}/dataset/dataset.csv", dtype={"blade": str})
    for _, r in d.iterrows():
        case = f"{doe}/{r.case}"
        rows.append(dict(name="baseline" if r.blade == "baseline" else f"doe_{r.blade}",
                         set="baseline" if r.blade == "baseline" else "doe", case=case,
                         pod=[r[f"c{k}"] for k in range(1, 6)]))
    if not pareto:
        return rows
    p = pd.read_csv(f"{RUNS}/optim_nsga2/pareto.csv", dtype={"blade": str})
    for _, r in p.iterrows():
        g, c = r.blade[1:].split("_c")
        rows.append(dict(name=f"pareto_{r.blade}", set="pareto", case=f"{RUNS}/optim_nsga2/OPENFOAM/openfoam_g{int(g)}_c{int(c)}",
                         pod=[r[f"c{k}"] for k in range(1, 6)]))
    return rows


def process(item, mean=False, data=None, fields_mean=False, qoi_file="qois.csv"):
    """
    mean=True: legacy data_mean/ mode (geometry reused from data/, fields swapped for the means).
    data (output folder, relative to HERE) given: build from scratch into it; fields_mean reads the fieldAverage
    means of the last time folder, qoi_file the QoIs over the same window (its status column must be all 'ok').
    """
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, _repo.SRC)
    from aero_optim.surrogate.foam import CascadeMesh, read_fields
    from aero_optim.surrogate.morph import RBFMorph, geometric_targets
    from aero_optim.surrogate.recut import PITCH, Recut
    out = os.path.join(HERE, data or ("data_mean" if mean else "data"), "blades", f"{item['name']}.npz")
    if os.path.isfile(out):
        return item["name"], "exists"
    if mean:
        try:
            g = dict(np.load(os.path.join(HERE, "data", "blades", f"{item['name']}.npz")))
            g["fields"] = np.stack([np.stack([read_fields(os.path.join(item["case"], op), mean=True)[f] for f in FIELDS], axis=1)
                                    for op in OPS]).astype(np.float32)
            q = pd.read_csv(os.path.join(item["case"], "qois_fieldavg.csv"), index_col=0)
            if not (q.status == "ok").all():
                return item["name"], f"FAILED fieldavg status {list(q.status)}"
            g["qoi"] = np.array([[q.at[op, "MixedoutLossCoef"], q.at[op, "OutflowAngle"]] for op in OPS])
            np.savez_compressed(out, **g)
            return item["name"], "ok"
        except Exception as e:
            return item["name"], f"FAILED {type(e).__name__}: {e}"
    try:
        mesh = CascadeMesh(os.path.join(item["case"], "mesh", "constant", "polyMesh"))
        rc = Recut(mesh)
        X_cut = rc.apply(mesh.centroids)
        outside = rc.check(X_cut)
        pts, lab = rc.boundary()
        morph = RBFMorph(pts, geometric_targets(rc, pts, lab, X_LE, X_TE), **RBF)
        X_geom = morph(X_cut).numpy()
        X_geom[:, 1] = np.clip(X_geom[:, 1], 0.0, PITCH)          # sub-micron overshoots at the walls
        detJ = morph.jacobian_det(X_cut).numpy()
        mps = {}
        for name, (x, y0, y1) in (("mp1", MP1), ("mp2", MP2)):
            ph = np.c_[np.full(1001, x), np.linspace(y0, y1, 1001)]
            mps[f"{name}_phys"] = ph
            mps[f"{name}_ref"] = morph(rc.apply(ph)).numpy()
        fields = np.stack([np.stack([read_fields(os.path.join(item["case"], op), mean=fields_mean)[f] for f in FIELDS], axis=1)
                           for op in OPS]).astype(np.float32)
        q = pd.read_csv(os.path.join(item["case"], qoi_file), index_col=0)
        if data is not None and "status" in q and not (q.status == "ok").all():
            return item["name"], f"FAILED {qoi_file} status {list(q.status)}"
        qoi = np.array([[q.at[op, "MixedoutLossCoef"], q.at[op, "OutflowAngle"]] for op in OPS])
        np.savez_compressed(out, X_phys=mesh.centroids, X_geom=X_geom, fields=fields, pod=np.array(item["pod"]),
                            qoi=qoi, detJ_min=detJ.min(), detJ_q01=np.quantile(detJ, 0.01), rbf_residual=morph.residual,
                            outside_recut=outside, le=rc.le, te=rc.te, **mps)
        return item["name"], f"ok cells {len(X_geom)} detJmin {detJ.min():.4f}"
    except Exception as e:  # keep going, report at the end
        return item["name"], f"FAILED {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mean", action="store_true", help="time-averaged fields -> data_mean/ (needs data/ built first)")
    ap.add_argument("--doe", default=None, help="DoE folder (default: doe1000_bladegen); needs --out")
    ap.add_argument("--out", default=None, help="output folder (relative to this script); builds the geometry from scratch")
    ap.add_argument("--fields", choices=["last", "mean"], default="last", help="last-iteration fields or fieldAverage means (--out only)")
    ap.add_argument("--qoi-file", default="qois.csv", help="per-case QoI table (--out only)")
    ap.add_argument("--no-pareto", action="store_true", help="skip the NSGA-II Pareto blades (they exist on the medium mesh only)")
    ap.add_argument("--mesh", default=None, help="free-text mesh description recorded in meta.json (--out only)")
    args = ap.parse_args()
    if args.out is None and (args.doe or args.fields != "last" or args.qoi_file != "qois.csv" or args.no_pareto or args.mesh):
        ap.error("--doe/--fields/--qoi-file/--no-pareto/--mesh need --out (the default folders are data/ and data_mean/)")
    if args.out is not None and args.mean:
        ap.error("--mean is the legacy data_mean/ mode; with --out use --fields mean")
    doe = args.doe or f"{RUNS}/doe1000_bladegen"
    data = os.path.join(HERE, args.out or ("data_mean" if args.mean else "data"))
    os.makedirs(os.path.join(data, "blades"), exist_ok=True)
    items = blade_list(doe, pareto=not args.no_pareto)[: args.limit]
    t0 = time.time()
    with ProcessPoolExecutor(args.jobs) as ex:
        if args.out is None:
            res = list(ex.map(process, items, [args.mean] * len(items)))
        else:
            n = len(items)
            res = list(ex.map(process, items, [False] * n, [args.out] * n, [args.fields == "mean"] * n, [args.qoi_file] * n))
    bad = [r for r in res if r[1].startswith("FAILED")]
    print(f"{len(res)} blades in {time.time() - t0:.0f} s, {len(bad)} failed")
    for r in bad[:20]:
        print(r)
    meta = dict(ops=OPS, fields=FIELDS, x_le_ref=X_LE, x_te_ref=X_TE, rbf=RBF, pitch=0.04039,
                blades=[{k: v for k, v in it.items() if k != "case"} | {"case": it["case"]} for it in items])
    if args.mean:    # same geometry, hence the same splits, T_bal and quality table
        meta["fields_source"] = "fieldAverage means over the MP sampling window (qois_fieldavg.csv)"
        for f in ("splits.json", "balance.npz", "quality.csv"):
            shutil.copy(os.path.join(HERE, "data", f), os.path.join(data, f))
    if args.out is not None:
        cells = [len(np.load(f)["X_phys"]) for f in (os.path.join(data, "blades", f"{it['name']}.npz") for it in items)
                 if os.path.isfile(f)]
        info = {}
        if os.path.isfile(os.path.join(doe, "dataset", "dataset.json")):
            info = json.load(open(os.path.join(doe, "dataset", "dataset.json"))).get("cfd", {})
        meta["doe"] = doe
        meta["mesh"] = dict(description=args.mesh, cells_min=min(cells, default=None),
                            cells_median=int(np.median(cells)) if cells else None, cells_max=max(cells, default=None))
        meta["iterations"], meta["averaging_window"] = info.get("iterations"), info.get("averaging_window")
        meta["fields_source"] = ("fieldAverage means (last time folder) over the QoI averaging window" if args.fields == "mean"
                                 else "last iteration") + f"; QoIs from {args.qoi_file}"
        meta["pareto"] = not args.no_pareto
    json.dump(meta, open(os.path.join(data, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
