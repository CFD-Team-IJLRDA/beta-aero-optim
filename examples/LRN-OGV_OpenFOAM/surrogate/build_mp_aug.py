"""
Caches, per blade, the CFD fields linearly interpolated (periodically: aero_optim.surrogate.data.periodic_interpolator, so no point
is lost at the periodic boundaries) at the MP1 and MP2 line points (1001 each) and those points
in the reference domain -> <data>/mp_aug/<blade>.npz (vals (3 OPs, M, 5) = p, rho, Ux, Uy, nut; ref (M, 2)).
Used by train_inr.py --mp-aug (training-set augmentation on the measurement lines).

Usage: ~/miniconda3/envs/ipynb/bin/python build_mp_aug.py [--data data_mean] [--jobs 4]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
from aero_optim.surrogate import data as D


def one(args):
    name, data = args
    D.set_data(data)
    out = os.path.join(D.DATA, "mp_aug", f"{name}.npz")
    if os.path.isfile(out):
        return
    d = D.load_blade(name)
    pts = np.vstack([d["mp1_phys"], d["mp2_phys"]])
    ref = np.vstack([d["mp1_ref"], d["mp2_ref"]])
    vals = np.stack([D.periodic_interpolator(d["X_phys"], d["fields"][k][:, :5])(pts) for k in range(3)])
    ok = ~np.isnan(vals).any(axis=(0, 2))
    np.savez(out, vals=vals[:, ok].astype(np.float32), ref=ref[ok])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_mean")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    D.set_data(args.data)
    os.makedirs(os.path.join(D.DATA, "mp_aug"), exist_ok=True)
    sp = D.splits()
    names = sp["train"] + sp["val"] + sp["test"] + sp["pareto"] + sp["baseline"]
    with ProcessPoolExecutor(args.jobs) as ex:
        list(ex.map(one, [(n, args.data) for n in names]))
    print(len(os.listdir(os.path.join(D.DATA, "mp_aug"))), "blades cached")


if __name__ == "__main__":
    main()
