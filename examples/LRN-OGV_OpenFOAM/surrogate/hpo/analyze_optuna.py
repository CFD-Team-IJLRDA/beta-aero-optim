"""
Complete analysis of the Optuna campaign (hpo_results/optuna_1): what was run, how pruning behaved, how reliable it is,
what the completed trials say about each hyperparameter.

Outputs in hpo_results/optuna_1/: optuna_analysis.png (6 panels), optuna_marginals.csv, optuna_summary.txt (printed numbers).
Metric: validation mixed-out-loss error, mean of the 3 OPs, best epoch (%).

Usage: ~/miniconda3/envs/ipynb/bin/python hpo/analyze_optuna.py [campaign_dir]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts folder: _repo, train_inr, ...
import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
HERE = _repo.WORK                 # data, runs, results: the surrogate work folder
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "hpo_results", "optuna_1")
OUT = []


def say(s=""):
    OUT.append(s)
    print(s)


def main():
    d = pd.read_csv(os.path.join(R, "results.csv"))
    d["n"] = d.trial.str[1:].astype(int)
    d = d.sort_values("n").reset_index(drop=True)
    H = {t: pd.read_csv(os.path.join(R, "trials", t, "history.csv")).set_index("epoch").val_w_relmae.dropna() for t in d.trial}
    d["value_at_stop"] = [H[t].iloc[-1] if len(H[t]) else np.nan for t in d.trial]
    done, pru = d[d.status == "done"], d[d.status == "pruned"]

    say(f"trials {len(d)}: done {len(done)}, pruned {len(pru)} (at epoch 25: {(pru.pruned_at == 25).sum()}, at epoch 75: {(pru.pruned_at == 75).sum()}); "
        f"all trials use {int(d.epochs.iloc[0])} epochs, OneCycle (5 % warm-up), ReLU modulation, periodic coordinates (fixed in the space)")
    say(f"GPU time: done trials {done.train_min.sum() / 60:.0f} trial-h, pruned {pru.train_min.sum() / 60:.0f} trial-h "
        f"(4 trials share a GPU, so ~1/4 of that in GPU-h); median done trial {done.train_min.median():.0f} min")

    # --- is pruning reliable? rank of the early value vs the final value, among the completed trials
    say("\nPruning reliability (completed trials only): Spearman rank correlation of the value at epoch E with the final best value")
    rel = {}
    for ep in (25, 50, 75, 100, 150, 200):
        v = np.array([H[t].get(ep, np.nan) for t in done.trial])
        rel[ep] = spearmanr(v, done.val_w_relmae).statistic
        say(f"  epoch {ep:3d}: rho = {rel[ep]:.2f}, median value {np.nanmedian(v):.2f} %")
    # what rank would the final top 10 have had at the pruning rungs (among all trials that reached the rung)?
    top = done.sort_values("val_w_relmae").head(10)
    for ep in (25, 75):
        reach = d[(d.pruned_at.isna() | (d.pruned_at >= ep))]
        vals = pd.Series({t: H[t].get(ep, np.nan) for t in reach.trial}).dropna()
        thr = vals.quantile(1 / 3)
        ranks = [int((vals < vals[t]).sum() + 1) for t in top.trial]
        say(f"  at the epoch-{ep} rung ({len(vals)} trials reached it, the best third survives, threshold {thr:.2f} %): "
            f"the final top-10 had ranks {ranks}")

    # --- prune rate by hyperparameter value
    say("\nPrune rate and best completed value by setting (all trials / completed):")
    rows = []
    for k, bins in (("arch", None), ("hidden", None), ("layers", [11, 16, 21, 29]), ("batch", None), ("lr", [0.0009, 0.002, 0.004, 0.011]),
                    ("weight_decay", [0.0002, 0.001, 0.003, 0.011]), ("no_resnet", None), ("mp_aug", None), ("max_grad_norm", None),
                    ("activation", None), ("fourier_features", None)):
        col = d[k] if bins is None else pd.cut(d[k], bins)
        for v, g in d.groupby(col, observed=True):
            gd = g[g.status == "done"]
            rows.append(dict(param=k, value=str(v), trials=len(g), done=len(gd), prune_rate=round(1 - len(gd) / len(g), 2),
                             best=round(gd.val_w_relmae.min(), 2) if len(gd) else np.nan,
                             median_done=round(gd.val_w_relmae.median(), 2) if len(gd) else np.nan))
    m = pd.DataFrame(rows)
    m.to_csv(os.path.join(R, "optuna_marginals.csv"), index=False)
    say(m.to_string(index=False))

    # --- did TPE beat its random start-up phase?
    best_so_far = done.sort_values("n").val_w_relmae.cummin()
    first30 = done[done.n < 30]
    say(f"\nStart-up phase (first 30 trials = random): {len(first30)} completed, best {first30.val_w_relmae.min():.2f} %, median {first30.val_w_relmae.median():.2f} %")
    later = done[done.n >= 30]
    say(f"TPE phase (trials 30-299): {len(later)} completed ({100 * len(later) / (len(d) - 30):.0f} % of its trials vs "
        f"{100 * len(first30) / 30:.0f} % in the random phase), best {later.val_w_relmae.min():.2f} %, median {later.val_w_relmae.median():.2f} %")
    say(f"Completed trials: min {done.val_w_relmae.min():.2f}, median {done.val_w_relmae.median():.2f}, max {done.val_w_relmae.max():.2f} %; "
        f"{(done.val_w_relmae < 2.2).sum()} below 2.2 %")

    # --- plots
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.2))
    a = ax[0, 0]
    a.scatter(pru.n, pru.value_at_stop.clip(upper=15), s=8, color="#b9b3d6", label=f"pruned ({len(pru)}), value when stopped")
    a.scatter(done.n, done.val_w_relmae, s=14, color="#2a78d6", label=f"completed ({len(done)}), final")
    a.step(done.sort_values("n").n, best_so_far, color="#eb6834", where="post", label="best so far")
    a.axvline(30, color="#52514e", ls=":", lw=0.8)
    a.text(31, 14, "TPE starts", fontsize=7, color="#52514e")
    a.set_ylim(0, 15)
    a.set_xlabel("trial number")
    a.set_ylabel("validation loss error [%]")
    a.legend(fontsize=6.5, loc="upper right", bbox_to_anchor=(1, 0.93))
    a.set_title("(a) all 300 trials", fontsize=8)
    a = ax[0, 1]
    for t in done.trial:
        a.plot(H[t].index, H[t].values, color="#2a78d6", lw=0.5, alpha=0.5)
    for t in top.head(3).trial:
        a.plot(H[t].index, H[t].values, color="#eb6834", lw=1.1)
    a.axvline(25, color="#52514e", ls=":", lw=0.8)
    a.axvline(75, color="#52514e", ls=":", lw=0.8)
    a.set_ylim(1, 12)
    a.set_yscale("log")
    a.set_xlabel("epoch")
    a.set_ylabel("validation loss error [%]")
    a.set_title("(b) learning curves of the 51 completed trials\n(orange: best 3; dotted: pruning rungs 25 and 75)", fontsize=8)
    a = ax[0, 2]
    a.bar([str(e) for e in rel], list(rel.values()), color="#2a78d6")
    a.set_ylim(0, 1)
    a.set_xlabel("epoch of the early value")
    a.set_ylabel("rank correlation with the final value")
    a.set_title("(c) how well does an early value predict the final one?\n(completed trials only)", fontsize=8)
    for i, k in enumerate(("hidden", "layers", "lr")):
        a = ax[1, i]
        a.scatter(d[k], d.value_at_stop.clip(upper=15), s=7, color="#b9b3d6")
        a.scatter(done[k], done.val_w_relmae, s=14, color="#2a78d6")
        a.set_xscale("log" if k in ("lr", "hidden") else "linear")
        a.set_ylim(0, 15)
        a.set_xlabel(k)
        a.set_ylabel("validation loss error [%]")
        a.set_title(f"({'def'[i]}) {k}: light = pruned (value when stopped), blue = completed", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "optuna_analysis.png"), dpi=600)
    plt.close(fig)

    # second figure: completed trials only, log y axis, one panel per parameter, with how many trials were sampled
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.4))
    rng = np.random.default_rng(0)
    for a, (k, lab, logx) in zip(ax.ravel(), (("hidden", "width (hidden units)", True), ("layers", "layers", False),
                                               ("lr", "peak learning rate", True), ("n_params", "number of parameters", True),
                                               ("batch", "batch (points)", True), ("weight_decay", "weight decay", True))):
        x = done[k].astype(float)
        a.scatter(x * (1 + 0.03 * rng.standard_normal(len(x))) if k in ("hidden", "batch") else x, done.val_w_relmae, s=16,
                  color="#2a78d6", alpha=0.8)
        if k in ("hidden", "batch"):
            for v in sorted(d[k].unique()):
                a.text(v, 0.985, f"{int(((done[k] == v)).sum())}/{int((d[k] == v).sum())}", transform=a.get_xaxis_transform(),
                       ha="center", va="top", fontsize=6.5, color="#52514e")
        a.set_yscale("log")
        a.set_xscale("log" if logx else "linear")
        a.set_ylim(1.5, 4.0)
        a.set_yticks([1.5, 2, 3, 4])
        a.set_yticklabels(["1.5", "2", "3", "4"])
        a.set_xlabel(lab)
        a.set_ylabel("validation loss error [%] (log)")
        a.grid(alpha=0.3, which="both")
        if k == "hidden":
            a.set_title("completed trials only; labels = completed / sampled", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "optuna_params.png"), dpi=600)
    open(os.path.join(R, "optuna_summary.txt"), "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
