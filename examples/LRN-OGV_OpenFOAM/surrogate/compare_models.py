"""
QoI-level comparison of surrogates on the same blades: INR runs (QoIs computed from the predicted fields on MP1/MP2)
and GP baselines (direct scalar regression). Reference: the CFD time-averaged QoIs ('avg' columns).
Field metrics (NRMSE per channel) exist only for the INR and are reported separately.

Metrics per set, OP and QoI (w = mixed-out loss, phi = outflow angle), over the blades of the set:
  bias (mean signed error), MAE, rel. MAE (mean |err| / true, w only), RMSE, NRMSE (RMSE / (max - min) of the CFD values over the set),
  R2, max |err|, Spearman rank correlation (ranking quality, what an optimiser needs).

Usage: ~/miniconda3/envs/ipynb/bin/python compare_models.py --inr paper_cfg --gp gp_full [--out compare_paper_cfg.md]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
HERE = _repo.WORK                 # data, runs, sopt: the surrogate work folder (aero_optim/surrogate/paths.py)
OPS = ["ADP", "OP1", "OP2"]
TARGETS = ["p", "rho", "Ux", "Uy", "nut"]


def metrics(true, pred, rel=True):
    e = pred - true
    rmse = np.sqrt(np.mean(e ** 2))
    m = {"bias": np.mean(e), "MAE": np.mean(np.abs(e)), "rel. MAE": 100 * np.mean(np.abs(e) / np.abs(true)) if rel else np.nan,
         "RMSE": rmse, "NRMSE": 100 * rmse / (true.max() - true.min()),
         "R2": 1 - np.sum(e ** 2) / np.sum((true - true.mean()) ** 2), "max |err|": np.max(np.abs(e)),
         "Spearman": spearmanr(true, pred).statistic if len(true) > 2 else np.nan}
    return m


def fmt(k, v, q):
    if np.isnan(v):
        return "–"
    if k in ("rel. MAE", "NRMSE"):
        return f"{v:.2f} %"
    if k in ("R2", "Spearman"):
        return f"{v:.3f}"
    return f"{v:+.2e}" if k == "bias" and q == "w" else f"{v:+.3f}°" if k == "bias" else f"{v:.2e}" if q == "w" else f"{v:.3f}°"


def main():
    np.seterr(all="ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--inr", required=True)
    ap.add_argument("--gp", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    inr_dir, gp_dir = os.path.join(HERE, "runs", args.inr), os.path.join(HERE, "runs", args.gp)
    cfg = json.load(open(os.path.join(inr_dir, "config.json")))
    hist = pd.read_csv(os.path.join(inr_dir, "history.csv"))
    best = hist.loc[hist.val_mse.idxmin()]
    lines = [f"# Surrogate comparison: INR `{args.inr}` vs GP `{args.gp}`", "",
             f"INR data: `{cfg.get('data', 'data')}`, {cfg['train_blades']} training blades, best validation epoch "
             f"{int(best.epoch)} / {len(hist)} (train MSE {best.train_mse:.3e}, val MSE {best.val_mse:.3e}"
             + (f", test MSE {best.test_mse:.3e}" if "test_mse" in hist else "") + ", normalized fields).",
             "Reference for all QoI metrics: CFD time-averaged QoIs. NRMSE = RMSE / (max - min of the CFD values over the set).", ""]
    for s in ("test", "pareto", "baseline"):
        inr = pd.read_csv(os.path.join(inr_dir, f"eval_{s}.csv"))
        gp = pd.read_csv(os.path.join(gp_dir, f"gp_eval_{s}.csv"))
        n = inr.blade.nunique()
        lines += [f"## {s} ({n} blade{'s' if n > 1 else ''} x 3 OPs)", ""]
        for q, lab in (("w", "mixed-out loss w"), ("phi", "outflow angle")):
            keys = list(metrics(np.arange(3.0), np.arange(3.0)).keys())
            if n == 1:
                keys = ["bias", "rel. MAE"] if q == "w" else ["bias"]
            lines += [f"**{lab}**", "", "| OP | model | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
            for op in OPS + ["all"]:
                for name, df in (("INR", inr), ("GP", gp)):
                    d = df if op == "all" else df[df.op == op]
                    m = metrics(d[f"{q}_avg"].to_numpy(), d[f"{q}_pred"].to_numpy(), rel=q == "w")
                    lines.append(f"| {op} | {name} | " + " | ".join(fmt(k, m[k], q) for k in keys) + " |")
            lines.append("")
        lines += ["**INR fields** (NRMSE per blade and OP = RMSE / range of the CFD field, then averaged; physical units)", "",
                  "| OP | " + " | ".join(TARGETS) + " |", "|---|" + "---|" * len(TARGETS)]
        for op in OPS + ["all"]:
            d = inr if op == "all" else inr[inr.op == op]
            lines.append(f"| {op} | " + " | ".join(f"{100 * d[f'nrmse_{t}'].mean():.2f} %" for t in TARGETS) + " |")
        lines.append("")
    out = os.path.join(HERE, "runs", args.out or f"compare_{args.inr}_vs_{args.gp}.md")
    open(out, "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
