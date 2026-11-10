"""
Step 1b: splits + T_bal (paper Sec. 4.1): per-axis empirical CDF of the geometry-aligned cell centres,
averaged over the TRAINING blades only and then frozen, so every blade (train or unseen) uses the same map.

    x_hat = F_x(x_geom),  y_hat = F_y(y_geom)   in [0, 1]

F is the training-averaged eCDF on a fine grid, linearly interpolated; a small linear blend (EPS) makes it
strictly increasing, hence exactly invertible.
Writes data/splits.json, data/balance.npz and data/quality.csv (per-blade transformation quality).

Usage: ~/miniconda3/envs/ipynb/bin/python fit_balance.py
       # another dataset folder, with the blade-to-split assignment of an existing splits.json (matched by name):
       ~/miniconda3/envs/ipynb/bin/python fit_balance.py --data data_wall2 --keep-splits data_mean/splits.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
HERE = _repo.WORK                 # data, runs, sopt: the surrogate work folder (aero_optim/surrogate/paths.py)
SEED, F_VAL, F_TEST = 0, 0.1, 0.1
N_GRID, EPS = 4001, 1e-3


def ecdf_on_grid(v, grid):
    v = np.sort(v)
    return np.searchsorted(v, grid, side="right") / len(v)


def keep_splits(ref_path, names):
    """
    Splits of ref_path restricted to the blades present (names); blades absent from every reference split go to train.
    **Returns** (splits, report) with report = {split: dropped count, 'added_to_train': [names]}.
    """
    ref = json.load(open(ref_path))
    have, keys = set(names), ("test", "val", "train", "pareto", "baseline")
    splits = {k: sorted(n for n in ref[k] if n in have) for k in keys}
    report = {k: len(ref[k]) - len(splits[k]) for k in keys}
    known = set().union(*(ref[k] for k in keys))
    added = sorted(n for n in names if n not in known)
    splits["train"] = sorted(splits["train"] + added)
    report["added_to_train"] = added
    return splits, report, ref.get("seed", SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="dataset folder (relative to this script)")
    ap.add_argument("--keep-splits", default=None, metavar="SPLITS_JSON",
                    help="reuse this splits.json (matched by blade name) instead of a new random split")
    args = ap.parse_args()
    data = os.path.join(HERE, args.data)
    files = sorted(glob.glob(os.path.join(data, "blades", "*.npz")))
    names = [os.path.basename(f)[:-4] for f in files]
    q = []
    for f, n in zip(files, names):
        d = np.load(f)
        q.append(dict(blade=n, cells=len(d["X_geom"]), detJ_min=float(d["detJ_min"]), detJ_q01=float(d["detJ_q01"]),
                      rbf_residual=float(d["rbf_residual"]), outside_recut=float(d["outside_recut"]),
                      x_te=float(d["te"][0]), y_te=float(d["te"][1])))
    q = pd.DataFrame(q)
    q.to_csv(os.path.join(data, "quality.csv"), index=False)

    if args.keep_splits:
        splits, report, seed = keep_splits(os.path.join(HERE, args.keep_splits), names)
        print(f"splits from {args.keep_splits}: dropped (not in {args.data}) "
              f"{ {k: v for k, v in report.items() if k != 'added_to_train'} }, "
              f"added to train {len(report['added_to_train'])}: {report['added_to_train']}")
    else:
        doe = sorted(n for n in names if n.startswith("doe_"))
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(doe)
        n_test, n_val = round(F_TEST * len(doe)), round(F_VAL * len(doe))
        splits = {"test": sorted(perm[:n_test]), "val": sorted(perm[n_test:n_test + n_val]),
                  "train": sorted(perm[n_test + n_val:]), "pareto": sorted(n for n in names if n.startswith("pareto_")),
                  "baseline": ["baseline"]}
        seed = SEED
    json.dump({k: list(v) for k, v in splits.items()} | {"seed": seed}, open(os.path.join(data, "splits.json"), "w"), indent=1)

    X = [np.load(os.path.join(data, "blades", f"{n}.npz"))["X_geom"] for n in splits["train"]]
    lo = np.min([x.min(axis=0) for x in X], axis=0)
    hi = np.max([x.max(axis=0) for x in X], axis=0)
    grids = [np.linspace(lo[k], hi[k], N_GRID) for k in range(2)]
    F = [np.mean([ecdf_on_grid(x[:, k], grids[k]) for x in X], axis=0) for k in range(2)]
    F = [(1 - EPS) * f + EPS * np.linspace(0, 1, N_GRID) for f in F]      # strictly increasing
    for f in F:
        f[0], f[-1] = 0.0, 1.0
        assert np.all(np.diff(f) > 0)
    np.savez(os.path.join(data, "balance.npz"), grid_x=grids[0], F_x=F[0], grid_y=grids[1], F_y=F[1])

    print({k: len(v) for k, v in splits.items()})
    print(q[["cells", "detJ_min", "detJ_q01", "rbf_residual", "outside_recut"]].describe().T[["min", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
