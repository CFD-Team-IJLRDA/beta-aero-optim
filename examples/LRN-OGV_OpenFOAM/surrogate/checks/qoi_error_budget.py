"""
Which predicted field drives the INR's mixed-out-loss error? Per test blade and OP:
loss from the INR on MP1/MP2 vs loss from the CFD fields (linearly interpolated on the same lines, the 'snap' reference);
then the INR value of ONE field on ONE line is replaced by the CFD value and the loss recomputed.
The drop of the mean |relative error| when a field is corrected = how much of the error that field carries.
Also reports the INR error of each field on MP2 as % of that field's range over the domain (cf. the field NRMSE).

Usage: ~/miniconda3/envs/ogv_torch/bin/python checks/qoi_error_budget.py --name mean_ddp [--gpu 1] [--n 88]
Output: runs/<name>/qoi_error_budget.md
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts folder: _repo, train_inr, ...
import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
HERE = _repo.WORK                 # data, runs, results: the surrogate work folder
sys.path.insert(0, _repo.SRC)
from eval_inr import Surrogate, plane_df  # noqa: E402
from aero_optim.simulator.cascade_qoi import plane_qois  # noqa: E402
from aero_optim.surrogate.data import load_blade, periodic_interpolator, splits  # noqa: E402

OPS = ["ADP", "OP1", "OP2"]
FIELDS = ["p", "rho", "Ux", "Uy"]


def loss(m1, m2):
    return plane_qois(plane_df(m1), plane_df(m2))["MixedoutLossCoef"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--n", type=int, default=88)
    args = ap.parse_args()
    run = os.path.join(HERE, "runs", args.name)
    sur = Surrogate(run, torch.device(f"cuda:{args.gpu}"))
    rows = []
    for name in splits()["test"][: args.n]:
        d = load_blade(name)
        for k, op in enumerate(OPS):
            true = d["fields"][k][:, :5]
            it = periodic_interpolator(d["X_phys"], true)
            c1, c2 = it(d["mp1_phys"]), it(d["mp2_phys"])
            ok = ~(np.isnan(c1).any(1) | np.isnan(c2).any(1))       # line ends outside the cell-centre hull
            p1, p2 = sur(d["pod"], k, d["mp1_ref"]), sur(d["pod"], k, d["mp2_ref"])
            c1, c2, p1, p2 = c1[ok], c2[ok], p1[ok], p2[ok]
            w_cfd = loss(c1, c2)
            r = {"blade": name, "op": op, "err_inr": loss(p1, p2) / w_cfd - 1}
            rng = true.max(0) - true.min(0)
            for j, f in enumerate(FIELDS):
                for line in ("MP1", "MP2"):
                    q1, q2 = p1.copy(), p2.copy()
                    (q1 if line == "MP1" else q2)[:, j] = (c1 if line == "MP1" else c2)[:, j]
                    r[f"err_fix_{f}_{line}"] = loss(q1, q2) / w_cfd - 1
                r[f"mp2_err_{f}_pct_range"] = 100 * np.sqrt(np.mean((p2[:, j] - c2[:, j]) ** 2)) / rng[j]
                r[f"mp1_err_{f}_pct_range"] = 100 * np.sqrt(np.mean((p1[:, j] - c1[:, j]) ** 2)) / rng[j]
            r["err_fix_all_MP2"] = loss(p1, c2) / w_cfd - 1
            r["err_fix_all_MP1"] = loss(c1, p2) / w_cfd - 1
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run, "qoi_error_budget.csv"), index=False)
    lines = [f"# Loss-error budget of `{args.name}` ({df.blade.nunique()} test blades)", "",
             "Mean |relative loss error| (%) with the INR on both lines, and after replacing one INR field on one line by the CFD value.",
             "Reference: loss of the CFD fields interpolated on the same lines.", "",
             "| OP | INR | " + " | ".join(f"fix {f} {l}" for f in FIELDS for l in ("MP1", "MP2")) + " | fix all MP1 | fix all MP2 |",
             "|---|---|" + "---|" * (2 * len(FIELDS) + 2)]
    for op in OPS + ["all"]:
        s = df if op == "all" else df[df.op == op]
        m = lambda c: f"{100 * s[c].abs().mean():.1f}"
        lines.append(f"| {op} | {m('err_inr')} | " + " | ".join(m(f"err_fix_{f}_{l}") for f in FIELDS for l in ("MP1", "MP2"))
                     + f" | {m('err_fix_all_MP1')} | {m('err_fix_all_MP2')} |")
    lines += ["", "INR RMS error on MP1, % of the field range over the domain:", "",
              "| OP | " + " | ".join(FIELDS) + " |", "|---|" + "---|" * len(FIELDS)]
    for op in OPS:
        s = df[df.op == op]
        lines.append(f"| {op} | " + " | ".join(f"{s[f'mp1_err_{f}_pct_range'].mean():.2f}" for f in FIELDS) + " |")
    lines += ["", "INR RMS error on MP2, % of the field range over the domain:", "",
              "| OP | " + " | ".join(FIELDS) + " |", "|---|" + "---|" * len(FIELDS)]
    for op in OPS:
        s = df[df.op == op]
        lines.append(f"| {op} | " + " | ".join(f"{s[f'mp2_err_{f}_pct_range'].mean():.2f}" for f in FIELDS) + " |")
    open(os.path.join(run, "qoi_error_budget.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
