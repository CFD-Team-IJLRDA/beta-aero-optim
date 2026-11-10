"""
Compact comparison of experiment arms (groups of runs that differ only by seed) on the 88 test blades:
mixed-out loss relative error per OP (each seed + mean, and the ensemble of the arm's seeds), against the GP.
Writes figures/batches/<batch>.png (600 DPI) and figures/batches/<batch>.csv; the table goes into RESULTS.md by hand.

Usage: ~/miniconda3/envs/ipynb/bin/python summarize_batch.py ideas_9oct   (arms defined in BATCHES below)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
HERE = _repo.WORK                 # data, runs, sopt: the surrogate work folder (aero_optim/surrogate/paths.py)
OPS = ["ADP", "OP1", "OP2"]
BATCHES = {
    "ideas_9oct": {
        "baseline": ["base_s0", "mean_s1", "mean_s2"],
        "periodic": ["per_s0", "per_s1", "per_s2"],
        "MP points": ["aug_s0", "aug_s1", "aug_s2"],
        "periodic + MP points": ["peraug_s0", "peraug_s1", "peraug_s2"],
    },
}


def metrics(d):
    out = {op: 100 * np.mean(np.abs(d[d.op == op].w_pred / d[d.op == op].w_avg - 1)) for op in OPS}
    out["angle"] = np.mean(np.abs(d.phi_pred - d.phi_avg))
    s = d[d.op == "ADP"]
    out["rho_ADP"] = spearmanr(s.w_avg, s.w_pred).statistic
    return out


def main():
    batch = sys.argv[1]
    arms = BATCHES[batch]
    out = os.path.join(HERE, "figures", "batches")
    os.makedirs(out, exist_ok=True)
    rows, per_seed = [], {}
    for arm, runs in arms.items():
        for ck, tag in (("", "best val MSE"), ("_qoi", "best val QoI")):
            files = [os.path.join(HERE, "runs", r, f"eval_test{ck}.csv") for r in runs]
            if not all(os.path.isfile(f) for f in files):
                continue
            ds = [pd.read_csv(f) for f in files]
            m = pd.DataFrame([metrics(d) for d in ds])
            ens = ds[0].copy()
            ens["w_pred"] = np.mean([d.w_pred for d in ds], 0)
            ens["phi_pred"] = np.mean([d.phi_pred for d in ds], 0)
            e = metrics(ens)
            rows.append({"arm": arm, "checkpoint": tag, "seeds": len(ds),
                         **{f"{k} mean": m[k].mean() for k in OPS}, **{f"{k} std": m[k].std() for k in OPS},
                         "angle mean": m["angle"].mean(), "rho_ADP mean": m["rho_ADP"].mean(),
                         **{f"{k} ens": e[k] for k in OPS}, "angle ens": e["angle"], "rho_ADP ens": e["rho_ADP"]})
            if ck == "":
                per_seed[arm] = (m, e)
    g = metrics(pd.read_csv(os.path.join(HERE, "runs", "gp_mean", "gp_eval_test.csv")))
    rows.append({"arm": "GP", "checkpoint": "-", "seeds": 1, **{f"{k} mean": g[k] for k in OPS},
                 "angle mean": g["angle"], "rho_ADP mean": g["rho_ADP"]})
    df = pd.DataFrame(rows)
    df.round(3).to_csv(os.path.join(out, f"{batch}.csv"), index=False)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    names = list(per_seed)
    for ax, op in zip(axes, OPS):
        for i, arm in enumerate(names):
            m, e = per_seed[arm]
            ax.plot(np.full(len(m), i), m[op], "o", color="#9fc3ee", ms=5, label="single seed" if i == 0 else None)
            ax.plot([i - 0.25, i + 0.25], [m[op].mean()] * 2, color="#2a78d6", lw=2, label="mean over seeds" if i == 0 else None)
            ax.plot(i, e[op], "*", color="#eb6834", ms=11, label="ensemble of the seeds" if i == 0 else None)
        ax.axhline(g[op], color="#52514e", ls="--", lw=1, label="GP")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" + ", "\n+ ") for n in names], fontsize=8)
        ax.set_title(op)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("test loss relative error [%]")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(f"{batch}: mixed-out loss error on 88 test blades (checkpoint: best val MSE)", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(out, f"{batch}.png"), dpi=600)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
