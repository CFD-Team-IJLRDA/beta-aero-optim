"""
Scalar baseline: Gaussian-process regression (Kriging) straight from the 5 POD coefficients to the CFD
time-averaged mixed-out loss and outflow angle, one GP per operating point and per QoI.
Kernel: constant x anisotropic Matern 5/2 + white noise, hyperparameters by marginal likelihood (10 restarts).
Trained on the same training blades as the INR (optionally a subset), evaluated on test / pareto / baseline.

Usage: ~/miniconda3/envs/ipynb/bin/python gp_baseline.py [--train-subset N] [--name gp_full] [--data data_mean] [--fit-on train+val]
Outputs in runs/<name>/: gp_eval_<set>.csv, gp_summary.md (+ fit_s in summary.json)
GPModel is also the GP surrogate of surrogate_optim.py (--gp).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
from aero_optim.surrogate.data import load_blade, set_data, splits

HERE = _repo.WORK                 # data, runs, sopt: the surrogate work folder (aero_optim/surrogate/paths.py)
OPS = ["ADP", "OP1", "OP2"]


def table(names):
    X, Q = [], []
    for n in names:
        d = load_blade(n)
        X.append(d["pod"])
        Q.append(d["qoi"])
    return np.array(X), np.array(Q)        # (n, 5), (n, 3 OPs, 2 QoIs)


class GPModel:
    """One GP per OP and per QoI on the standardised POD coefficients; predict(X) -> (n, 3 OPs, [loss, angle])."""
    def __init__(self, X, Q, verbose=False):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.models = {}
        for k, op in enumerate(OPS):
            for j, q in enumerate(["w", "phi"]):
                kern = ConstantKernel(1.0) * Matern(length_scale=np.ones(X.shape[1]), nu=2.5) + WhiteKernel(1e-3)
                gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=10, random_state=0)
                gp.fit((X - self.mu) / self.sd, Q[:, k, j])
                self.models[(op, q)] = gp
                if verbose:
                    print(op, q, gp.kernel_, flush=True)

    def predict(self, X):
        Z = (np.atleast_2d(X) - self.mu) / self.sd
        return np.stack([np.stack([self.models[(op, q)].predict(Z) for q in ("w", "phi")], -1) for op in OPS], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-subset", type=int, default=None)
    ap.add_argument("--name", default="gp_full")
    ap.add_argument("--data", default="data", help="data or data_mean (QoIs over the same window as the mean fields)")
    ap.add_argument("--fit-on", default="train", choices=["train", "train+val"],
                    help="train+val: all the CFD budget of the data folder (the GP needs no validation set)")
    args = ap.parse_args()
    set_data(args.data)
    out = os.path.join(HERE, "runs", args.name)
    os.makedirs(out, exist_ok=True)
    sp = splits()
    tr = sp["train"] + (sp["val"] if args.fit_on == "train+val" else [])
    tr = tr[: args.train_subset] if args.train_subset else tr
    Xtr, Qtr = table(tr)
    t0 = time.time()
    gp = GPModel(Xtr, Qtr, verbose=True)
    json.dump({"fit_s": round(time.time() - t0, 1), "n_fit": len(tr), "data": args.data, "fit_on": args.fit_on},
              open(os.path.join(out, "summary.json"), "w"), indent=1)
    text = [f"# GP baseline ({len(tr)} training blades, {args.data}, fit on {args.fit_on})", ""]
    for s in ("val", "test", "pareto", "baseline"):
        if args.fit_on == "train+val" and s == "val":
            continue
        X, Q = table(sp[s])
        P = gp.predict(X)
        rows = []
        for i, n in enumerate(sp[s]):
            for k, op in enumerate(OPS):
                rows.append({"blade": n, "op": op, "w_pred": P[i, k, 0], "w_avg": Q[i, k, 0], "phi_pred": P[i, k, 1], "phi_avg": Q[i, k, 1]})
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out, f"gp_eval_{s}.csv"), index=False)
        text += [f"### {s}", "", "| OP | loss rel. MAE | angle MAE |", "|---|---|---|"]
        for op in OPS + ["all"]:
            d = df if op == "all" else df[df.op == op]
            text.append(f"| {op} | {100 * np.mean(np.abs(d.w_pred - d.w_avg) / d.w_avg):.2f} % | {np.mean(np.abs(d.phi_pred - d.phi_avg)):.3f}° |")
        text.append("")
    open(os.path.join(out, "gp_summary.md"), "w").write("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
