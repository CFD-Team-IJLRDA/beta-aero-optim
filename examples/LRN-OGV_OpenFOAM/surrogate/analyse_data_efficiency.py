"""
Data-efficiency study of the surrogate-assisted optimisation (report: reports/data_efficiency.md).

Inputs: the optimisation + CFD verification folders sopt/<arm>_N<N>_s<k>_o<o>/ (arm = inr or gp; budget N CFD blades; k = budget
subset, i.e. which N DoE blades are simulated; o = optimiser seed), the INR runs runs/eff_N<N>_s<k>_m<seed>/, the brute-force
run <AERO_RUNS>/optim_nsga2/ and the DoE <AERO_RUNS>/doe1000_bladegen/.

Final front of a run = non-dominated, CFD-feasible blades among the N budget blades and its CFD-verified blades.
"doe" arm = the N budget blades alone (what the budget gives without any surrogate).
Metrics (losses relative to the baseline CFD loss): best L_ADP and L_OP [%], blades beating the baseline in both, hypervolume with
the baseline as reference point (fraction of the brute-force one), gap = relative loss reduction (both losses) the front still needs
to dominate the brute-force front [%].

Outputs in reports/: data_efficiency_fronts.csv (one row per front), data_efficiency_accuracy.csv (one row per surrogate),
data_efficiency_metrics.png, data_efficiency_fronts.png, data_efficiency_accuracy.png (600 DPI).
Usage (from the work folder): ~/miniconda3/envs/ipynb/bin/python <this folder>/analyse_data_efficiency.py
"""
import glob
import json
import os
import sys
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
HERE = _repo.WORK                 # data, runs, sopt: the surrogate work folder (aero_optim/surrogate/paths.py)
R = _repo.RUNS
OUT = os.path.join(HERE, "reports")
OPS = ("ADP", "OP1", "OP2")
COL = {"inr": "#2a78d6", "gp": "#eb6834", "doe": "#8a8983"}
LAB = {"inr": "INR optimisation + CFD check", "gp": "GP optimisation + CFD check", "doe": "best of the N simulated DoE blades (no surrogate)"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
CORE_H_PER_BLADE = 0.6                                  # medium mesh, 3 OPs (timings/timings.md)


def nondominated(F):
    F = np.asarray(F)
    return np.array([i for i in range(len(F)) if not any((F[j] <= F[i]).all() and (F[j] < F[i]).any() for j in range(len(F)))], int)


def hv(F, ref=(1., 1.)):
    F = np.array([f for f in F if f[0] < ref[0] and f[1] < ref[1]])
    if not len(F):
        return 0.
    area, best = 0., ref[1]
    for x, y in F[np.argsort(F[:, 0])]:
        if y < best:
            area, best = area + (ref[0] - x) * (best - y), y
    return area


def gap(F, B):
    return 100 * (max(min(max(a[0] / b[0], a[1] / b[1]) for a in F) for b in B) - 1.)


def front_line(ax, F, **kw):
    """Pareto front: its blades sorted by L_ADP, joined by straight lines (the lines only guide the eye)."""
    F = F[np.argsort(F[:, 0])]
    return ax.plot(F[:, 0], F[:, 1], **kw)


def main():
    os.makedirs(OUT, exist_ok=True)
    doe = pd.read_csv(f"{R}/doe1000_bladegen/doe_results.csv", dtype={"blade": str})
    b0 = doe.loc[doe.blade == "baseline", ["L_ADP", "L_OP"]].to_numpy()[0]
    hist = pd.read_csv(f"{R}/optim_nsga2/history.csv")
    BF = pd.read_csv(f"{R}/optim_nsga2/pareto.csv")[["L_ADP", "L_OP"]].to_numpy() / b0
    hv_bf = hv(BF)
    BFu = BF[(BF < 1).all(1)]                       # the 8 brute-force Pareto blades better than the baseline in both losses
    D = hist[(hist.source == "doe") & hist.feasible][["L_ADP", "L_OP"]].to_numpy() / b0
    DF = D[nondominated(D)]

    def metrics(F):
        return dict(n_front=len(F), best_L_ADP_pct=100 * (F[:, 0].min() - 1), best_L_OP_pct=100 * (F[:, 1].min() - 1),
                    n_beat_baseline=int((F < 1).all(1).sum()), hv_frac=hv(F) / hv_bf, gap_pct=gap(F, BF), gap_useful_pct=gap(F, BFu),
                    best_ADP_no_worse_OP_pct=100 * (F[F[:, 1] <= 1, 0].min() - 1) if (F[:, 1] <= 1).any() else np.nan,
                    best_OP_no_worse_ADP_pct=100 * (F[F[:, 0] <= 1, 1].min() - 1) if (F[:, 0] <= 1).any() else np.nan)

    rows, fronts, acc, seen_doe, pairs, feas = [], {}, [], set(), {}, {}
    for run in sorted(glob.glob(os.path.join(HERE, "sopt", "*_N*_s*_o*"))):
        m = re.match(r"(inr|inr287|gp|grad|gradk1|grad2|grad2k1)_N(\d+)_s(\d+)_o(\d+)$", os.path.basename(run))
        vtag = "better" if os.path.isfile(os.path.join(run, "verification_better.csv")) else "cfd"
        if not m or not os.path.isfile(os.path.join(run, f"verification_{vtag}.csv")):
            continue
        arm, n, k, o = m[1], int(m[2]), int(m[3]), int(m[4])
        h = pd.read_csv(os.path.join(run, "history.csv"))
        known = h[(h.source == "cfd") & h.feasible][["L_ADP", "L_OP"]].to_numpy() / b0
        v = pd.read_csv(os.path.join(run, f"verification_{vtag}.csv"))
        ok = v[v.status.eq("ok")]
        ver = ok[ok.cfd_feasible][["cfd_L_ADP", "cfd_L_OP"]].to_numpy() / b0
        F = np.vstack([known, ver]) if len(ver) else known
        F = F[nondominated(F)]
        rj, vj = json.load(open(os.path.join(run, "run.json"))), json.load(open(os.path.join(run, f"verification_{vtag}.json")))
        rows.append(dict(arm=arm, N=n, subset=k, opt_seed=o, check=vtag, cfd_blades=n + len(v), verified=len(v), verified_feasible=int(v.cfd_feasible.sum()),
                         **metrics(F), loss_err_verified_pct=100 * np.mean([((ok[f"pred_{p}_w"] - ok[f"cfd_{p}_w"]).abs() / ok[f"cfd_{p}_w"]).mean() for p in OPS]),
                         angle_err_verified_deg=np.mean([(ok[f"pred_{p}_angle"] - ok[f"cfd_{p}_angle"]).abs().mean() for p in OPS]),
                         **{f"{q}_{c}_pct": f(100 * (ok[f"pred_{c}"] - ok[f"cfd_{c}"]) / ok[f"cfd_{c}"]) for c in ("L_ADP", "L_OP")
                            for q, f in (("bias", np.mean), ("mae", lambda e: np.mean(np.abs(e))))},
                         pred_ok_cfd_infeasible=int((~ok.cfd_feasible).sum()),
                         fit_min=rj.get("surrogate_fit_s", np.nan) / 60, optim_s=rj["optimisation_wall_s"], verify_min=vj["wall_s"] / 60,
                         cfd_core_h=CORE_H_PER_BLADE * (n + len(v))))
        fronts[(arm, n, k, o)] = F
        feas[(arm, n, k, o)] = ok.cfd_feasible.to_numpy(bool)
        pairs[(arm, n, k, o)] = (ok[["pred_L_ADP", "pred_L_OP"]].to_numpy() / b0, ok[["cfd_L_ADP", "cfd_L_OP"]].to_numpy() / b0)
        if (n, k) not in seen_doe and len(known):
            seen_doe.add((n, k))
            K = known[nondominated(known)]
            rows.append(dict(arm="doe", N=n, subset=k, opt_seed=0, cfd_blades=n, **metrics(K), cfd_core_h=CORE_H_PER_BLADE * n))
            fronts[("doe", n, k, 0)] = K
    T = pd.DataFrame(rows).sort_values(["arm", "N", "subset", "opt_seed"])
    T.round(4).to_csv(os.path.join(OUT, "data_efficiency_fronts.csv"), index=False)
    ref = {"brute force (1816 CFD blades)": metrics(BF), "full DoE (1000 CFD blades)": metrics(DF)}
    print(pd.DataFrame(ref).T.round(2).to_string())
    g = T.groupby(["arm", "N"])
    print(g[["best_ADP_no_worse_OP_pct", "best_OP_no_worse_ADP_pct", "gap_useful_pct", "hv_frac", "best_L_ADP_pct", "best_L_OP_pct", "n_beat_baseline", "loss_err_verified_pct"]].agg(["median", "min", "max"]).round(2).to_string())
    print(g.size().rename("fronts").to_string())

    # --- accuracy on the 88 test blades: INR 3-seed ensemble (eval files of the runs) vs GP fitted on the same budget
    from gp_baseline import GPModel, table           # noqa: E402  (aero_optim.surrogate.data set by set_data below)
    from aero_optim.surrogate import data as Dd                        # noqa: E402
    blade_err = []                                    # per test blade and operating point, pooled over subsets
    for run0 in sorted(glob.glob(os.path.join(HERE, "runs", "eff_N*_s*_m0"))):
        m = re.match(r"eff_N(\d+)_s(\d+)_m0$", os.path.basename(run0))
        n, k = int(m[1]), int(m[2])
        ev = [os.path.join(HERE, "runs", f"eff_N{n}_s{k}_m{s}", "eval_test_qoi.csv") for s in range(3)]
        if not all(os.path.isfile(e) for e in ev):
            continue
        E = [pd.read_csv(e) for e in ev]
        e = E[0].copy()
        e["w_pred"] = np.mean([x.w_pred for x in E], 0)
        e["phi_pred"] = np.mean([x.phi_pred for x in E], 0)
        blade_err += [("inr", n, x) for x in 100 * ((e.w_pred - e.w_avg).abs() / e.w_avg).to_numpy()]
        acc.append(dict(arm="inr", N=n, subset=k, loss_err_pct=100 * ((e.w_pred - e.w_avg).abs() / e.w_avg).mean(),
                        angle_err_deg=(e.phi_pred - e.phi_avg).abs().mean(),
                        fit_min=sum(json.load(open(os.path.join(HERE, "runs", f"eff_N{n}_s{k}_m{s}", "summary.json")))["train_wall_s"] for s in range(3)) / 60))
        Dd.set_data(f"data_subsets/N{n}_s{k}")
        sp = Dd.splits()
        X, Q = table(sp["train"] + sp["val"])
        Xt, Qt = table(sp["test"])
        P = GPModel(X, Q).predict(Xt)
        blade_err += [("gp", n, x) for x in 100 * np.abs(P[..., 0] - Qt[..., 0]).ravel() / Qt[..., 0].ravel()]
        acc.append(dict(arm="gp", N=n, subset=k, loss_err_pct=100 * np.mean(np.abs(P[..., 0] - Qt[..., 0]) / Qt[..., 0]),
                        angle_err_deg=np.mean(np.abs(P[..., 1] - Qt[..., 1]))))
    A = pd.DataFrame(acc).sort_values(["arm", "N", "subset"])
    A.round(4).to_csv(os.path.join(OUT, "data_efficiency_accuracy.csv"), index=False)
    print(A.groupby(["arm", "N"])[["loss_err_pct", "angle_err_deg"]].agg(["median", "min", "max"]).round(3).to_string())

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK})

    def style(a):
        a.grid(color=GRID, lw=0.6)
        a.set_axisbelow(True)

    # --- figure 1: metrics vs CFD budget (each dot = one front; line = median)
    NS = sorted(T[T.arm != "doe"].N.unique())
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.7))
    for a, (col, lab, refv) in zip(ax, (("best_ADP_no_worse_OP_pct", "best L_ADP gain, L_OP not worse than baseline [%]", "best_ADP_no_worse_OP_pct"),
                                        ("best_OP_no_worse_ADP_pct", "best L_OP gain, L_ADP not worse than baseline [%]", "best_OP_no_worse_ADP_pct"),
                                        ("hv_frac", "hypervolume / brute force", "hv_frac"))):
        for arm, dx in (("doe", 0.94), ("gp", 0.97), ("inr", 1.03)):
            t = T[(T.arm == arm) & T.N.isin(NS)]
            if not len(t):
                continue
            a.scatter(t.N * dx, t[col], s=14, color=COL[arm], alpha=0.55, lw=0)
            med = t.groupby("N")[col].median()
            a.plot(med.index * dx, med.values, color=COL[arm], lw=2, marker="o", ms=5, label=LAB[arm])
        a.axhline(ref["brute force (1816 CFD blades)"][refv], color=INK, lw=1, ls="--")
        a.axhline(ref["full DoE (1000 CFD blades)"][refv], color=MUTED, lw=1, ls=":")
        a.set_xscale("log")
        a.set_xticks(NS)
        a.set_xticklabels(NS)
        a.minorticks_off()
        a.set_xlabel("CFD budget N (blades used to build the surrogate)")
        a.set_ylabel(lab)
        style(a)
    for a_ in ax:
        k_ = a_.get_lines()[-2].get_ydata()[0], a_.get_lines()[-1].get_ydata()[0]
        a_.text(NS[0] * 0.9, k_[0], "brute force", va="bottom", ha="left", fontsize=6.5)
        a_.text(NS[0] * 0.9, k_[1], "full DoE", va="bottom", ha="left", fontsize=6.5, color=MUTED)
    h_, l_ = ax[0].get_legend_handles_labels()
    h_ += [plt.Line2D([], [], color=INK, lw=1, ls="--"), plt.Line2D([], [], color=MUTED, lw=1, ls=":")]
    l_ += ["brute-force optimisation (1816 CFD blades)", "full DoE (1000 CFD blades)"]
    fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(OUT, "data_efficiency_metrics.png"), dpi=600)
    plt.close(fig)

    # --- figure 2: the fronts, one column per N; top = whole front, bottom = zoom on the blades that beat the baseline
    NR = [n for n in NS if (T[(T.arm == "inr") & (T.N == n)].shape[0] > 1)]        # budgets with repetitions
    med = {}                                                                      # median-hypervolume repetition per arm and N
    for (arm, n), t in T[T.arm.isin(["inr", "gp"])].groupby(["arm", "N"]):
        r = t.iloc[(t.hv_frac - t.hv_frac.median()).abs().argsort().iloc[0]]
        med[(arm, n)] = (r.subset, r.opt_seed)
    fig, ax = plt.subplots(2, len(NR), figsize=(3.8 * len(NR), 6.4), sharey="row", sharex="row")
    ax = np.atleast_2d(ax).reshape(2, -1)
    for j, n in enumerate(NR):
        for i, (xl, yl) in enumerate((((0.85, 1.06), (0.9, 1.7)), ((0.855, 0.95), (0.92, 1.02)))):
            a = ax[i, j]
            a.scatter(*D.T, s=4, color="#cfcdc4", lw=0, label="DoE blades (feasible)", zorder=1)
            for (arm, nn, k, o), F in fronts.items():
                if nn == n and arm in ("inr", "gp"):
                    bold = med[(arm, n)] == (k, o)
                    front_line(a, F, color=COL[arm], lw=1.8 if bold else 0.6, alpha=1 if bold else 0.3, zorder=3 if bold else 2)
                    if bold:
                        a.scatter(*F.T, s=14, color=COL[arm], lw=0, zorder=3)
            front_line(a, BF, color=INK, lw=1.4, zorder=4)
            a.scatter(*BF.T, s=12, color=INK, lw=0, zorder=5)
            a.plot(1, 1, marker="*", ms=9, color=INK, mec="white", mew=0.6, ls="none", zorder=6)
            a.set_xlim(*xl)
            a.set_ylim(*yl)
            style(a)
            if i == 0:
                a.set_title(f"N = {n} CFD blades", fontsize=9)
                a.add_patch(plt.Rectangle((0.855, 0.92), 0.095, 0.10, fill=False, ec=MUTED, lw=0.8, ls="--"))
            a.set_xlabel("L_ADP / baseline")
        ax[0, 0].set_ylabel("L_OP / baseline")
        ax[1, 0].set_ylabel("L_OP / baseline (zoom)")
    hs = [plt.Line2D([], [], color=INK, lw=1.4, marker="o", ms=3, label="brute-force front (1816 CFD blades)"),
          plt.Line2D([], [], color=COL["inr"], lw=1, marker="o", ms=3, label="INR + CFD check (bold: median run; faint: the 6 subset/seed runs)"),
          plt.Line2D([], [], color=COL["gp"], lw=1, marker="o", ms=3, label="GP + CFD check (bold: median run; faint: the 6 subset/seed runs)"),
          plt.Line2D([], [], color="#cfcdc4", marker="o", ls="none", ms=3, label="DoE blades (feasible)"),
          plt.Line2D([], [], color=INK, marker="*", ls="none", ms=8, label="baseline")]
    fig.legend(handles=hs, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(OUT, "data_efficiency_fronts.png"), dpi=600)
    plt.close(fig)

    # --- figure 4: predicted front vs the same blades re-computed with CFD (median run per arm and N)
    fig, ax = plt.subplots(2, len(NR), figsize=(3.8 * len(NR), 6.6), sharex=True, sharey=True)
    for (i, arm), (j, n) in ((x, y) for x in enumerate(("inr", "gp")) for y in enumerate(NR)):
        a = ax[i, j]
        a.scatter(*D.T, s=4, color="#cfcdc4", lw=0, zorder=1)
        front_line(a, BF, color=INK, lw=1.2, zorder=2)
        if True:
            P, C = pairs[(arm, n, *med[(arm, n)])]
            for p, c in zip(P, C):
                a.plot([p[0], c[0]], [p[1], c[1]], color=COL[arm], lw=0.6, alpha=0.6, zorder=3)
            a.scatter(*P.T, s=14, facecolor="white", edgecolor=COL[arm], lw=0.9, zorder=4)
            a.scatter(*C.T, s=14, color=COL[arm], lw=0, zorder=5)
            front_line(a, P[nondominated(P)], color=COL[arm], lw=1.3, ls="--", zorder=4)
            Cf = C[feas[(arm, n, *med[(arm, n)])]]
            front_line(a, Cf[nondominated(Cf)], color=COL[arm], lw=1.8, zorder=5)
        a.plot(1, 1, marker="*", ms=9, color=INK, mec="white", mew=0.6, ls="none", zorder=6)
        a.set_xlim(0.855, 1.01)
        a.set_ylim(0.9, 1.08)
        a.set_title(f"{arm.upper()}, N = {n} CFD blades (median run)", fontsize=9)
        a.set_xlabel("L_ADP / baseline")
        a.set_ylabel("L_OP / baseline")
        style(a)
    hs = [plt.Line2D([], [], color=INK, lw=1.2, label="brute-force front"),
          plt.Line2D([], [], marker="o", ls="--", mfc="white", mec=INK, color=INK, ms=4, label="predicted by the surrogate (dashed: predicted front)"),
          plt.Line2D([], [], marker="o", ls="-", color=INK, ms=4, label="same blades, CFD (solid: CFD front, feasible blades)"),
          plt.Line2D([], [], color="#cfcdc4", marker="o", ls="none", ms=3, label="DoE blades")]
    fig.legend(handles=hs, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(OUT, "data_efficiency_pred_vs_cfd.png"), dpi=600)
    plt.close(fig)

    # --- figure 5: gradient-based vs NSGA-II on the same surrogate (INR, N = 200, subset 0, seed 1)
    G = [(a_, l_) for a_, l_ in (("inr", "NSGA-II"), ("grad2", "SLSQP, κ = 0 (mean)"), ("grad2k1", "SLSQP, κ = 1 (mean + 1 std)"))
         if (a_, 200, 0, 1) in pairs]
    if len(G) > 1:
        fig, ax = plt.subplots(2, len(G), figsize=(3.8 * len(G), 6.8), sharex="row", sharey="row")
        for (i, (xl, yl)), (j, (arm, lab)) in ((x, y) for x in enumerate((((0.85, 1.06), (0.9, 1.7)), ((0.855, 0.97), (0.9, 1.1))))
                                              for y in enumerate(G)):
            a = ax[i, j]
            P, C = pairs[(arm, 200, 0, 1)]
            Cf = C[feas[(arm, 200, 0, 1)]]
            a.scatter(*D.T, s=4, color="#cfcdc4", lw=0, zorder=1)
            front_line(a, BF, color=INK, lw=1.2, zorder=2)
            for p_, c_ in zip(P, C):
                a.plot([p_[0], c_[0]], [p_[1], c_[1]], color=COL["inr"], lw=0.6, alpha=0.6, zorder=3)
            a.scatter(*P.T, s=14, facecolor="white", edgecolor=COL["inr"], lw=0.9, zorder=4)
            a.scatter(*C.T, s=14, color=COL["inr"], lw=0, zorder=5)
            front_line(a, P[nondominated(P)], color=COL["inr"], lw=1.3, ls="--", zorder=4)
            front_line(a, Cf[nondominated(Cf)], color=COL["inr"], lw=1.8, zorder=5)
            a.plot(1, 1, marker="*", ms=9, color=INK, mec="white", mew=0.6, ls="none", zorder=6)
            a.set_xlim(*xl)
            a.set_ylim(*yl)
            if i == 0:
                a.set_title(lab, fontsize=9)
                a.add_patch(plt.Rectangle((0.855, 0.9), 0.115, 0.2, fill=False, ec=MUTED, lw=0.8, ls="--"))
            a.set_xlabel("L_ADP / baseline")
            a.set_ylabel("L_OP / baseline" + (" (zoom)" if i else ""))
            style(a)
        hs = [plt.Line2D([], [], color=INK, lw=1.2, label="brute-force front"),
              plt.Line2D([], [], marker="o", ls="--", mfc="white", mec=COL["inr"], color=COL["inr"], ms=4, label="predicted by the INR (dashed: predicted front)"),
              plt.Line2D([], [], marker="o", ls="-", color=COL["inr"], ms=4, label="same blades, CFD (solid: CFD front, feasible blades)")]
        fig.legend(handles=hs, loc="lower center", ncol=3, fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(os.path.join(OUT, "gradient_vs_nsga.png"), dpi=600)
        plt.close(fig)

    # --- figure 3: accuracy on the test blades vs N (dots: subsets, line: median)
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.0))
    for a, (col, lab) in zip(ax, (("loss_err_pct", "loss error on test blades [%]"), ("angle_err_deg", "outflow-angle error on test blades [deg]"))):
        for arm, dx in (("gp", 0.97), ("inr", 1.03)):
            t = A[A.arm == arm]
            a.scatter(t.N * dx, t[col], s=14, color=COL[arm], alpha=0.55, lw=0)
            med = t.groupby("N")[col].median()
            a.plot(med.index * dx, med.values, color=COL[arm], lw=2, marker="o", ms=5, label={"inr": "INR (3-seed ensemble)", "gp": "GP"}[arm])
        a.set_xscale("log")
        Ns = sorted(A.N.unique())
        a.set_xticks(Ns)
        a.set_xticklabels(Ns)
        a.minorticks_off()
        a.set_ylim(bottom=0)
        a.set_xlabel("CFD budget N")
        a.set_ylabel(lab)
        style(a)
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "data_efficiency_accuracy.png"), dpi=600)

if __name__ == "__main__":
    main()
