"""
Compact summary of the phase-2 screening (hpo_results/screening/results.csv, one factor at a time, 2 seeds per arm).
Metric: validation mixed-out-loss error (mean over the 3 OPs, best epoch), in %.

Outputs in hpo_results/screening/:
  screening_arms.csv    one row per arm: factor, setting, mean / per-seed metric, per-OP means, delta vs reference
  screening.png         (a) every arm's seeds and mean, grouped by factor, reference band = range of the reference seeds;
                        (b) the 6 x 6 omega x omega0 grid (mean of 2 seeds)
Usage: ~/miniconda3/envs/ipynb/bin/python hpo/summarize_screening.py
"""
import os
import sys
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts folder: _repo, train_inr, ...
import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
HERE = _repo.WORK                 # data, runs, results: the surrogate work folder
CAMP = os.path.join(HERE, "hpo_results", "screening")
OMEGAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
YMAX = 20.0                     # axis clipped here; diverged arms are marked with an arrow and their value


def main():
    d = pd.read_csv(os.path.join(CAMP, "results.csv"))
    d["arm"] = d.trial.str.replace(r"_s\d+$", "", regex=True)
    g = d.groupby(["factor", "arm"]).agg(
        val=("val_w_relmae", "mean"), seeds=("val_w_relmae", lambda x: " / ".join(f"{v:.2f}" for v in x)),
        ADP=("val_w_relmae_ADP", "mean"), OP1=("val_w_relmae_OP1", "mean"), OP2=("val_w_relmae_OP2", "mean"),
        angle=("val_phi_mae", "mean"), val_mse=("val_mse", "mean"), train_min=("train_min", "mean")).reset_index()
    ref = g[g.arm == "ref"].iloc[0]
    g["delta_vs_ref"] = g.val - ref.val
    g = g.sort_values(["factor", "val"])
    g.round(3).to_csv(os.path.join(CAMP, "screening_arms.csv"), index=False)
    ref_seeds = d[d.arm == "ref"].val_w_relmae

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13, 5.2))
    ax = fig.add_axes([0.05, 0.3, 0.62, 0.62])
    order = [f for f in g.factor.unique() if f not in ("omega x omega0", "reference")]
    x, ticks, labels, sep = 0, [], [], []
    for f in order:
        s = g[g.factor == f].sort_values("val")
        for _, r in s.iterrows():
            vals = d[d.arm == r.arm].val_w_relmae
            ax.plot([x] * len(vals), vals.clip(upper=YMAX), "o", color="#9fc3ee", ms=4)
            if r.val < YMAX:
                ax.plot([x - 0.3, x + 0.3], [r.val] * 2, color="#2a78d6", lw=2)
            else:                                   # diverged: arrow at the top with the value
                ax.annotate(f"{r.val:.0f}", (x, YMAX), xytext=(x, YMAX - 2.2), ha="center", fontsize=6, color="#d03b3b",
                            arrowprops=dict(arrowstyle="->", color="#d03b3b", lw=0.8))
            ticks.append(x)
            labels.append(re.sub(r"(arch|activation|fourier_features|fourier_scale)", "", r.arm).replace("_", " ").strip())
            x += 1
        sep.append(x - 0.5)
        ax.text((x - len(s) / 2) - 0.5, 1.02 + 0.05 * (len(sep) % 2), f, transform=ax.get_xaxis_transform(), ha="center",
                fontsize=7, color="#52514e")
        x += 0.6
    ax.axhspan(ref_seeds.min(), ref_seeds.max(), color="#eb6834", alpha=0.15, lw=0, label="reference: range of its 2 seeds")
    ax.axhline(ref.val, color="#eb6834", lw=1, ls="--", label="reference: mean")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_ylim(0, YMAX)
    ax.set_ylabel("validation loss error [%] (mean of 3 OPs)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.9))

    ax2 = fig.add_axes([0.73, 0.3, 0.24, 0.6])
    om = d[d.factor.isin(["omega x omega0", "reference"])].copy()
    M = np.full((6, 6), np.nan)
    for i, w0 in enumerate(OMEGAS):
        for j, w in enumerate(OMEGAS):
            s = om[np.isclose(om.omega, w) & np.isclose(om.omega0, w0)]
            if len(s):
                M[i, j] = s.val_w_relmae.mean()
    im = ax2.imshow(M, origin="lower", cmap="viridis_r", vmin=np.nanmin(M), vmax=min(np.nanmax(M), 3 * np.nanmin(M)))
    for i in range(6):
        for j in range(6):
            if np.isfinite(M[i, j]):
                ax2.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                         color="white" if M[i, j] > np.nanmedian(M) else "black")
    lab = ["1e-3", "1e-2", "0.1", "1", "10", "100"]
    ax2.set_xticks(range(6))
    ax2.set_xticklabels(lab)
    ax2.set_yticks(range(6))
    ax2.set_yticklabels(lab)
    ax2.set_xlabel("ω (hidden layers), initial")
    ax2.set_ylabel("ω0 (first layer), initial")
    fig.colorbar(im, ax=ax2, shrink=0.8, label="validation loss error [%]")
    fig.savefig(os.path.join(CAMP, "screening.png"), dpi=600, bbox_inches="tight")
    pd.set_option("display.width", 200)
    print(g[g.factor != "omega x omega0"][["factor", "arm", "val", "seeds", "ADP", "OP1", "OP2", "delta_vs_ref", "train_min"]]
          .round(2).to_string(index=False))
    print("\nomega grid (rows omega0, cols omega):\n", pd.DataFrame(M, index=lab, columns=lab).round(1))


if __name__ == "__main__":
    main()
