"""
Step 2 evaluation of a trained INR on the test / pareto / baseline blades.

Per blade and operating point:
- field NRMSE per channel (physical units, paper metric) vs the CFD fields the run was trained on
  (data/: last iteration; data_mean/: time-averaged over the MP sampling window);
- QoIs from the INR: the network is evaluated along MP1/MP2 (mapped by T_bal o T_geom), then the same mixed-out
  loss and outflow angle routine as the CFD post-processing (aero_optim.simulator.cascade_qoi.plane_qois);
- two references: 'snap' = the same routine on those CFD fields linearly interpolated on MP1/MP2
  (what the INR was trained on), and 'avg' = the solver's time-averaged QoIs of the same dataset.
Writes runs/<name>/eval_<set>.csv, eval_summary.md and parity / field figures (600 DPI).

Usage: ~/miniconda3/envs/ogv_torch/bin/python eval_inr.py --name <run> [--gpu 1] [--checkpoint best_qoi_model]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
from aero_optim.simulator.cascade_qoi import plane_qois  # noqa: E402
from aero_optim.surrogate.data import COORDS, INLET_ANGLE, TARGETS, Normalizer, balance, embed, load_balance, load_blade, periodic_interpolator, set_data, splits  # noqa: E402
from aero_optim.surrogate.siren import build_model  # noqa: E402

HERE = _repo.WORK                 # data, runs, sopt: the surrogate work folder (aero_optim/surrogate/paths.py)
OPS = ["ADP", "OP1", "OP2"]


def plane_df(f):
    return pd.DataFrame({"p": f[:, 0], "rho": f[:, 1], "Ux": f[:, 2], "Uy": f[:, 3]})


class Surrogate:
    def __init__(self, run_dir, device, checkpoint="best_model.pt"):
        cfg = json.load(open(os.path.join(run_dir, "config.json")))
        set_data(cfg.get("data", "data"))
        self.norm = Normalizer.load(os.path.join(run_dir, "normalizer.npz"))
        self.coords = cfg.get("coords", "plain")
        self.model = build_model(cfg, COORDS[self.coords], 6, len(TARGETS)).to(device)
        self.model.load_state_dict(torch.load(os.path.join(run_dir, checkpoint), map_location=device))
        self.model.eval()
        self.device = device
        self.bal = load_balance()

    @torch.no_grad()
    def __call__(self, pod, op, X_geom):
        xh = embed(balance(X_geom, self.bal), self.coords, self.bal)
        P = np.tile(np.r_[pod, INLET_ANGLE[op]], (len(xh), 1))
        t = lambda a, k: torch.from_numpy(self.norm(k, a).astype(np.float32)).to(self.device)
        y = self.model(t(P, "P"), t(xh, "X")).cpu().numpy()
        return self.norm.inverse_Y(y)


def evaluate_set(sur, names):
    rows = []
    for name in names:
        d = load_blade(name)
        for k, op in enumerate(OPS):
            true = d["fields"][k][:, :5]
            pred = sur(d["pod"], k, d["X_geom"])
            r = {"blade": name, "op": op}
            rng = true.max(0) - true.min(0)
            r.update({f"nrmse_{t}": v for t, v in zip(TARGETS, np.sqrt(((true - pred) ** 2).mean(0)) / rng)})
            q_pred = plane_qois(plane_df(sur(d["pod"], k, d["mp1_ref"])), plane_df(sur(d["pod"], k, d["mp2_ref"])))
            interp = periodic_interpolator(d["X_phys"], true)
            q_snap = plane_qois(plane_df(interp(d["mp1_phys"])), plane_df(interp(d["mp2_phys"])))
            r.update(w_pred=q_pred["MixedoutLossCoef"], w_snap=q_snap["MixedoutLossCoef"], w_avg=d["qoi"][k, 0],
                     phi_pred=q_pred["OutflowAngle"], phi_snap=q_snap["OutflowAngle"], phi_avg=d["qoi"][k, 1])
            rows.append(r)
    return pd.DataFrame(rows)


def summary(df, label):
    lines = [f"### {label} ({df.blade.nunique()} blades)", "",
             "| OP | NRMSE p | rho | Ux | Uy | nut | loss: INR vs snapshot (rel. MAE) | INR vs averaged CFD | snapshot vs averaged | angle: INR vs snapshot (MAE) | INR vs averaged |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for op in OPS + ["all"]:
        s = df if op == "all" else df[df.op == op]
        rel = lambda a, b: f"{100 * np.mean(np.abs(s[a] - s[b]) / s[b]):.2f} %"
        lines.append(f"| {op} | " + " | ".join(f"{s[f'nrmse_{t}'].mean():.2e}" for t in TARGETS) + " | "
                     f"{rel('w_pred', 'w_snap')} | {rel('w_pred', 'w_avg')} | {rel('w_snap', 'w_avg')} | "
                     f"{np.mean(np.abs(s.phi_pred - s.phi_snap)):.3f}° | {np.mean(np.abs(s.phi_pred - s.phi_avg)):.3f}° |")
    return "\n".join(lines) + "\n"


def parity(df, out, tag):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    cols = {"ADP": "#2a78d6", "OP1": "#eb6834", "OP2": "#1baf7a"}
    for ax, (q, lab) in zip(axes, [("w", "mixed-out loss w"), ("phi", "outflow angle [deg]")]):
        for op in OPS:
            s = df[df.op == op]
            ax.scatter(s[f"{q}_avg"], s[f"{q}_pred"], s=10, color=cols[op], label=op, lw=0)
        lo = min(df[f"{q}_avg"].min(), df[f"{q}_pred"].min())
        hi = max(df[f"{q}_avg"].max(), df[f"{q}_pred"].max())
        ax.plot([lo, hi], [lo, hi], color="#52514e", lw=0.8)
        ax.set_xlabel(f"CFD (time-averaged) {lab}")
        ax.set_ylabel(f"INR {lab}")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"{tag}: INR vs CFD", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"parity_{tag}.png"), dpi=600)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--checkpoint", default="best_model", choices=["best_model", "best_qoi_model"],
                    help="best validation field MSE (default) or best validation QoI error; files of the latter get a _qoi suffix")
    args = ap.parse_args()
    run = os.path.join(HERE, "runs", args.name)
    sur = Surrogate(run, torch.device(f"cuda:{args.gpu}"), f"{args.checkpoint}.pt")
    sfx = "_qoi" if args.checkpoint == "best_qoi_model" else ""
    sp = splits()
    text = [f"# Evaluation of {args.name} ({args.checkpoint})", ""]
    for s in ("test", "pareto", "baseline"):
        df = evaluate_set(sur, sp[s])
        df.to_csv(os.path.join(run, f"eval_{s}{sfx}.csv"), index=False)
        text.append(summary(df, s))
        if s != "baseline":
            parity(df, run, s + sfx)
    open(os.path.join(run, f"eval_summary{sfx}.md"), "w").write("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
