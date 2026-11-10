"""
Diagnostic figures of the INR surrogate (4-model ensemble of the seed study) vs the time-averaged CFD (data_mean/).
The INR predicts p, rho, Ux, Uy, nut; the total pressure p0 is derived from the predicted p, rho, U.

1. Pass over the 88 test blades (+ baseline) x 3 OPs: CFD (cell values linearly interpolated on MP1/MP2) and INR on the
   measurement lines -> mixed-out loss, per-field line errors (RMS, pitchwise mean = offset, fluctuation), mass-averaged
   total pressures. Written to figures/predictions/test_line_errors.csv; aggregated in line_error_stats.csv.
2. Chosen blades = median and two largest |ensemble ADP loss error| test blades + baseline; per blade and OP:
   lines_<blade>_<OP>.png, lines_err_<blade>_<OP>.png, fields_<blade>_<OP>.png.
3. mp1_error_summary.png over all test blades and OPs (3 panels).

"Cut": each measurement line crosses the periodic boundary of the reference (T_geom) domain once, where y_ref jumps
from pitch to 0; the INR is not periodic there. Statistics marked 'excl_cut' leave out +-2 % of the pitch
(+-CUT_PTS line points) around that crossing; 'cut_from_cfd' losses use the CFD values there instead of the INR.

Usage: ~/miniconda3/envs/ogv_torch/bin/python plot_predictions.py [--gpu 1]
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402
from scipy.interpolate import LinearNDInterpolator  # noqa: E402
from scipy.spatial import Delaunay  # noqa: E402

import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
HERE = _repo.WORK                 # data, runs, sopt: the surrogate work folder (aero_optim/surrogate/paths.py)
sys.path.insert(0, _repo.SRC)
torch.set_num_threads(2)
from aero_optim.simulator.cascade_qoi import plane_qois  # noqa: E402
from eval_inr import Surrogate, plane_df  # noqa: E402
from aero_optim.surrogate.data import load_blade, set_data, splits  # noqa: E402
from aero_optim.surrogate.foam import CascadeMesh  # noqa: E402
from aero_optim.surrogate.recut import PITCH, Recut  # noqa: E402

RUNS = ["mean_s1", "mean_s2", "mean_noclip_s0", "mean_noclip_s1"]
OPS = ["ADP", "OP1", "OP2"]
GAMMA = 1005.0 / (1005.0 - 8314.46 / 28.9)
FIELDS = ["p", "rho", "Ux", "Uy", "p0"]                       # line quantities (p0 derived)
UNIT = {"p": "Pa", "rho": "kg/m³", "Ux": "m/s", "Uy": "m/s", "p0": "Pa"}
SYM = {"p": "p", "rho": "ρ", "Ux": "Ux", "Uy": "Uy", "p0": "p0 (derived)"}
OUT = os.path.join(HERE, "figures", "predictions")
CUT_PTS = 20          # line points (of 1001) on each side of the periodic-boundary crossing = +-2 % of the pitch
ZOOM_MARGIN = 0.005   # [m] the zoom colour range uses the cells with x <= x_LE - ZOOM_MARGIN (excludes the LE stagnation)
ERR_MAX = 5.0         # [%] cap of the full-domain error maps (|INR - CFD| / CFD range)
ERR_MAX_ZOOM = 1.0    # [%] cap of the upstream-zoom error maps (same normalisation)
NLINE = 1001
DPI = 600
C_CFD, C_INR, C_MP1, C_MP2 = "#0b0b0b", "#2a78d6", "#eb6834", "#2a78d6"
C_OP = {"ADP": "#2a78d6", "OP1": "#eb6834", "OP2": "#1baf7a"}
CUT_LABEL = "location where the line crosses the periodic boundary of the reference domain"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5})


def with_p0(f):
    """(..., >=4) [p, rho, Ux, Uy, ...] -> (..., 5) [p, rho, Ux, Uy, p0] (isentropic total pressure, derived)."""
    p, rho, ux, uy = f[..., 0], f[..., 1], f[..., 2], f[..., 3]
    m2 = (ux ** 2 + uy ** 2) * rho / (GAMMA * p)
    return np.stack([p, rho, ux, uy, p * (1 + 0.5 * (GAMMA - 1) * m2) ** (GAMMA / (GAMMA - 1))], -1)


def mass_avg(f, j):
    """Mass-flux (rho Ux) weighted pitchwise average of column j of a (n, 5) line array."""
    w = f[:, 1] * f[:, 2]
    return np.sum(f[:, j] * w) / np.sum(w)


def loss(m1, m2):
    return plane_qois(plane_df(m1), plane_df(m2))["MixedoutLossCoef"]


class Ensemble:
    def __init__(self, device):
        self.models = [Surrogate(os.path.join(HERE, "runs", r), device) for r in RUNS]
        set_data("data_mean")

    def __call__(self, pod, k, X):
        return np.stack([m(pod, k, X) for m in self.models])        # (4, N, 5) physical


def cut_index(d, mp):
    """First line point after the periodic boundary of the reference domain (where y_ref jumps from pitch to 0)."""
    return int(np.argmax(np.abs(np.diff(d[f"{mp}_ref"][:, 1])))) + 1


def lines(ens, d, k, interp):
    """
    CFD and INR on MP1/MP2, keeping the points where the CFD interpolation is defined.
    'near' flags the points within +-2 % of the pitch of the periodic-boundary crossing; 'abs_dp0' is |INR - CFD| of
    the derived p0 on all 1001 line points (NaN where dropped).
    """
    out = {}
    for mp in ("mp1", "mp2"):
        c = interp(d[f"{mp}_phys"])
        ok = ~np.isnan(c).any(1)
        m = ens(d["pod"], k, d[f"{mp}_ref"][ok])
        y = d[f"{mp}_phys"][:, 1]
        i = cut_index(d, mp)
        near = (np.abs(np.arange(len(y)) - i) <= CUT_PTS)[ok]
        full = np.full(len(y), np.nan)
        full[ok] = np.abs(with_p0(m.mean(0))[:, 4] - with_p0(c[ok])[:, 4])
        out[mp] = dict(y=(y[ok] - y[0]) / PITCH, cfd=c[ok], models=m, ens=m.mean(0), near=near,
                       y_cut=(y[i] - y[0]) / PITCH, abs_dp0=full)
    return out


def cut_from_cfd(L, mp):
    """Ensemble line with the points near the periodic-boundary crossing replaced by the CFD values."""
    e = L[mp]["ens"].copy()
    e[L[mp]["near"]] = L[mp]["cfd"][L[mp]["near"]]
    return e


# ------------------------------------------------------------------ pass over the test blades
def line_stats(ens, names):
    rows, cache, prof = [], {}, []
    for name in names:
        d = load_blade(name)
        tri = Delaunay(d["X_phys"])
        for k, op in enumerate(OPS):
            true = d["fields"][k][:, :5]
            L = lines(ens, d, k, LinearNDInterpolator(tri, true))
            cache[(name, op)] = L
            w_cfd = loss(L["mp1"]["cfd"], L["mp2"]["cfd"])
            w_ens = loss(L["mp1"]["ens"], L["mp2"]["ens"])
            r = {"blade": name, "op": op, "w_cfd": w_cfd, "w_avg": d["qoi"][k, 0], "w_ens": w_ens,
                 "w_err_rel_pct": 100 * (w_ens / w_cfd - 1)}
            for i, run in enumerate(RUNS):
                r[f"w_{run}"] = loss(L["mp1"]["models"][i], L["mp2"]["models"][i])
            f1, f2 = cut_from_cfd(L, "mp1"), cut_from_cfd(L, "mp2")
            r["w_ens_cut_from_cfd"] = loss(f1, f2)
            r["w_err_rel_cut_from_cfd_pct"] = 100 * (r["w_ens_cut_from_cfd"] / w_cfd - 1)
            rng = np.ptp(with_p0(true), 0)
            for mp, fixed in (("mp1", f1), ("mp2", f2)):
                c, e = with_p0(L[mp]["cfd"]), with_p0(L[mp]["ens"])
                keep = ~L[mp]["near"]
                for tag, err in (("", e - c), ("_excl_cut", (e - c)[keep])):
                    for j, f in enumerate(FIELDS):
                        r[f"{mp}_{f}_rms_pct{tag}"] = 100 * np.sqrt(np.mean(err[:, j] ** 2)) / rng[j]
                        r[f"{mp}_{f}_mean{tag}"] = err[:, j].mean()
                        r[f"{mp}_{f}_std{tag}"] = err[:, j].std()
                        r[f"{mp}_{f}_offset_share{tag}"] = err[:, j].mean() ** 2 / np.mean(err[:, j] ** 2)
                r[f"{mp}_p0m_cfd"], r[f"{mp}_p0m_ens"] = mass_avg(c, 4), mass_avg(e, 4)
                r[f"{mp}_p0m_ens_cut_from_cfd"] = mass_avg(with_p0(fixed), 4)
                r[f"{mp}_pm_cfd"], r[f"{mp}_pm_ens"] = mass_avg(c, 0), mass_avg(e, 0)
                r[f"{mp}_y_cut"] = L[mp]["y_cut"]
                if name != "baseline":
                    prof.append(dict(mp=mp, op=op, pct=100 * L[mp]["abs_dp0"] / rng[4], y_cut=L[mp]["y_cut"]))
            rows.append(r)
        print(f"  {name}: ADP loss error {rows[-3]['w_err_rel_pct']:+.1f} %", flush=True)
    return pd.DataFrame(rows), cache, prof


def aggregate(df):
    """Per line, field and scope: quartiles of the RMS error (% of range), median offset / fluctuation, offset share."""
    rows = []
    for mp in ("mp1", "mp2"):
        for f in FIELDS:
            for tag, scope in (("", "all points"), ("_excl_cut", "excluding ±2 % of the pitch around the crossing")):
                rms = df[f"{mp}_{f}_rms_pct{tag}"]
                rows.append({"line": mp.upper(), "field": f, "scope": scope, "rms_pct_p25": rms.quantile(0.25),
                             "rms_pct_median": rms.median(), "rms_pct_p75": rms.quantile(0.75),
                             "abs_mean_err_median": df[f"{mp}_{f}_mean{tag}"].abs().median(),
                             "std_err_median": df[f"{mp}_{f}_std{tag}"].median(),
                             "offset_share_median": df[f"{mp}_{f}_offset_share{tag}"].median(), "unit": UNIT[f]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ line figures
def line_figure(L, blade, op, r, err_mode):
    """Pitchwise coordinate on the horizontal axis, quantity (or INR - CFD) on the vertical axis."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.6), sharex=True)
    for i, mp in enumerate(("mp1", "mp2")):
        y, keep = L[mp]["y"], ~L[mp]["near"]
        c, e, ms = with_p0(L[mp]["cfd"]), with_p0(L[mp]["ens"]), with_p0(L[mp]["models"])
        for j, f in enumerate(FIELDS):
            ax = axes[i, j]
            ax.axvline(L[mp]["y_cut"], color="#8a8986", lw=0.8, ls=":", label=CUT_LABEL)
            if err_mode:
                lo, hi = (ms[..., j] - c[:, j]).min(0), (ms[..., j] - c[:, j]).max(0)
                ee = e[:, j] - c[:, j]
                ax.fill_between(y, lo, hi, color=C_INR, alpha=0.2, lw=0, label="range of the 4 models")
                ax.axhline(0, color=C_CFD, lw=0.8)
                ax.plot(y, ee, color=C_INR, lw=1.2, label="ensemble − CFD")
                ax.axhline(ee.mean(), color=C_INR, lw=0.8, ls="--", label="pitchwise mean of ensemble − CFD (offset)")
                ek = ee[keep]
                ax.set_title(f"mean {ee.mean():+.3g}, RMS {np.sqrt(np.mean(ee ** 2)):.3g}\n"
                             f"away from the crossing: mean {ek.mean():+.3g}, RMS {np.sqrt(np.mean(ek ** 2)):.3g}",
                             fontsize=6.5)
                ax.set_ylabel(f"Δ{SYM[f]} [{UNIT[f]}]", fontsize=8)
                vals = np.r_[lo[keep], hi[keep], 0.0]
            else:
                lo, hi = ms[..., j].min(0), ms[..., j].max(0)
                ax.fill_between(y, lo, hi, color=C_INR, alpha=0.2, lw=0, label="range of the 4 models")
                ax.plot(y, c[:, j], color=C_CFD, lw=1.2, label="CFD")
                ax.plot(y, e[:, j], color=C_INR, lw=1.2, ls="--", label="INR ensemble")
                ax.set_ylabel(f"{SYM[f]} [{UNIT[f]}]", fontsize=8)
                vals = np.r_[lo[keep], hi[keep], c[:, j]]
            a, b = vals.min(), vals.max()
            pad = 0.08 * (b - a) if b > a else 1e-6 * max(abs(a), 1)
            ax.set_ylim(a - pad, b + pad)
            ax.tick_params(axis="y", labelsize=7)
            ax.yaxis.get_offset_text().set_fontsize(7)
            if i == 1:
                ax.set_xlabel("(y − y₀) / pitch")
            if j == 0:
                ax.set_ylabel(f"{mp.upper()} ({'x = −0.020 m, upstream' if mp == 'mp1' else 'x = 0.087 m, downstream'})"
                              f"\n{ax.get_ylabel()}", fontsize=8)
    axes[0, 0].set_xlim(0, 1)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=len(l), frameon=False)
    kind = "INR − CFD on the measurement planes" if err_mode else "measurement planes"
    fig.suptitle(f"{blade}, {op}: {kind}. Mixed-out loss w: CFD {r['w_cfd']:.4f}, INR ensemble {r['w_ens']:.4f} "
                 f"({r['w_err_rel_pct']:+.1f} %; {r['w_err_rel_cut_from_cfd_pct']:+.1f} % with CFD values within ±2 % "
                 f"of the pitch of the periodic-boundary crossing)\nVertical range set by the data away from that "
                 f"crossing (the local spike there may be clipped). p0 is derived from the predicted p, ρ, U."
                 + (" 'Away from the crossing': without ±2 % of the pitch around it." if err_mode else ""), fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, f"{'lines_err' if err_mode else 'lines'}_{blade}_{op}.png"), dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------------------ field figures
def masked_tri(X, inside):
    tri = Triangulation(X[:, 0], X[:, 1])
    tri.set_mask(~inside(X[tri.triangles].mean(axis=1)))
    return tri


def field_figure(ens, d, k, blade, op, mesh):
    X = d["X_phys"]
    cfd = with_p0(d["fields"][k][:, :5])
    inr = with_p0(ens(d["pod"], k, d["X_geom"]).mean(0))
    rng = np.ptp(cfd, 0)
    err_pct = 100 * np.abs(inr - cfd) / rng                    # pointwise error, NRMSE normalisation
    b = mesh.bottom
    pitch_path = Path(np.vstack([b, (b + [0, PITCH])[::-1]]))
    wall_path = Path(mesh.wall)
    inside = lambda p: pitch_path.contains_points(p) & ~wall_path.contains_points(p)
    x_le = mesh.wall[:, 0].min()
    zone = X[:, 0] <= x_le
    tri, tri_z = masked_tri(X, inside), masked_tri(X[zone], inside)
    ext = lambda P: (np.ptp(P[:, 0]), np.ptp(P[:, 1]))
    (wf, hf), (wz, hz) = ext(X), ext(X[zone])
    rows = [("p", 0), ("Ux", 2), ("Uy", 3), ("p0", 4)]
    fig = plt.figure(figsize=(18, 10.5), layout="constrained")
    gs = fig.add_gridspec(4, 6, width_ratios=[wf / hf] * 3 + [wz / hz] * 3)
    rc = Recut(mesh)
    xs_up = np.linspace(rc.x_in, rc.le[0], 200)
    xs_dn = np.linspace(rc.te[0], rc.x_out, 200)
    colour_sel = [np.ones(len(X), bool), X[zone, 0] <= x_le - ZOOM_MARGIN]
    for i, (f, j) in enumerate(rows):
        for part, (T, sel) in enumerate(((tri, slice(None)), (tri_z, zone))):
            cv, iv, ev = cfd[sel, j], inr[sel, j], err_pct[sel, j]
            cs_ = colour_sel[part]
            lo, hi = min(cv[cs_].min(), iv[cs_].min()), max(cv[cs_].max(), iv[cs_].max())
            emax = ERR_MAX if part == 0 else ERR_MAX_ZOOM
            nrmse = np.sqrt(np.mean((iv - cv) ** 2)) / rng[j] * 100
            axs = [fig.add_subplot(gs[i, 3 * part + c]) for c in range(3)]
            for c, (ax, val) in enumerate(zip(axs, (cv, iv, ev))):
                if c < 2:
                    cs = ax.tricontourf(T, val, levels=np.linspace(lo, hi, 41), cmap="turbo", extend="both")
                    if c == 0:
                        for xs in (xs_up, xs_dn):
                            ax.plot(xs, rc.cut(xs), ls=":", color="white", lw=0.8, zorder=4)
                else:
                    cs_e = ax.tricontourf(T, val, levels=np.linspace(0, emax, 26), cmap="magma_r", extend="max")
                ax.fill(mesh.wall[:, 0], mesh.wall[:, 1], color="#d9d9d9", ec="#52514e", lw=0.5, zorder=3)
                for mp in ("mp1", "mp2"):
                    q = d[f"{mp}_phys"]
                    ax.plot(q[:, 0], q[:, 1], ls="--", color="k", lw=0.7, zorder=4)
                ax.set_aspect("equal")
                ax.grid(False)
                xz, yz = X[sel, 0], X[sel, 1]
                ax.set_xlim(xz.min(), xz.max())
                ax.set_ylim(yz.min(), yz.max())
                ax.tick_params(labelsize=6)
                if i == len(rows) - 1:
                    ax.set_xlabel("x [m]", fontsize=8)
            if part == 0:
                axs[0].set_title("CFD", fontsize=8)
                axs[1].set_title("INR ensemble", fontsize=8)
                axs[2].set_title(f"|INR − CFD| / range [%]\nNRMSE {nrmse:.2f} %", fontsize=8)
            else:
                axs[0].set_title("CFD, zoom", fontsize=7)
                axs[1].set_title("INR, zoom", fontsize=7)
                axs[2].set_title(f"error, zoom\nNRMSE {nrmse:.2f} %", fontsize=7)
            axs[0].set_ylabel(f"{SYM[f]}\ny [m]", fontsize=8)
            cb = fig.colorbar(cs, ax=axs[:2], shrink=0.9, pad=0.01, aspect=25)
            cb.set_label(f"{SYM[f]} [{UNIT[f]}]", fontsize=7)
            cb.ax.tick_params(labelsize=6)
            cb = fig.colorbar(cs_e, ax=axs[2], shrink=0.9, pad=0.01, aspect=25)
            cb.set_label(f"|Δ{SYM[f]}| [% of CFD range]", fontsize=7)
            cb.ax.tick_params(labelsize=6)
    fig.suptitle(f"{blade}, {op}: CFD vs INR ensemble (4 models); p0 is derived from the predicted p, ρ, U. "
                 f"Dashed: MP1 and MP2; dotted (CFD panels): periodic boundary of the reference domain.\n"
                 f"Error = |INR − CFD| / (max − min of the CFD field over the domain) × 100, capped at {ERR_MAX:.0f} % "
                 f"(domain) and {ERR_MAX_ZOOM:.0f} % (zoom); NRMSE = RMSE / range × 100 over the domain or the zoom region.\nZoom: inlet to LE; field colour ranges from the cells with "
                 f"x ≤ x_LE − {1e3 * ZOOM_MARGIN:.0f} mm (LE stagnation saturated)", fontsize=10)
    fig.savefig(os.path.join(OUT, f"fields_{blade}_{op}.png"), dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------------------ summary figure
def summary_figure(df, prof):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    y = df.w_err_rel_pct
    d01 = df.mp1_p0m_cfd - df.mp2_p0m_cfd
    x1 = 100 * (df.mp1_p0m_ens - df.mp1_p0m_cfd) / d01
    x2 = -100 * (df.mp2_p0m_ens - df.mp2_p0m_cfd) / d01
    r_op = lambda x: {op: np.corrcoef(x[df.op == op], y[df.op == op])[0, 1] for op in OPS}
    r1, r2 = np.corrcoef(x1, y)[0, 1], np.corrcoef(x2, y)[0, 1]
    core = np.abs(x2) <= 30                                 # leaves out a few high-loss OP1 cases
    r2c = np.corrcoef(x2[core], y[core])[0, 1]
    titles = [f"(a) The loss error follows the MP1 inlet total-pressure error (r = {r1:.2f})",
              f"(b) The MP2 error does not explain it (r = {r2c:.2f} without the {(~core).sum()} OP1 outliers on the right)"]
    for ax, x, xlab, title in (
            (axes[0], x1, "MP1 (upstream): (p̄01,INR − p̄01,CFD) / (p̄01 − p̄02)CFD [%]", titles[0]),
            (axes[1], x2, "MP2 (downstream): −(p̄02,INR − p̄02,CFD) / (p̄01 − p̄02)CFD [%]", titles[1])):
        rr = r_op(x)
        for op in OPS:
            s = df.op == op
            ax.scatter(x[s], y[s], s=10, lw=0, color=C_OP[op], label=f"{op} (r = {rr[op]:.2f})")
        lim = np.abs(np.r_[x1, x2, y]).max() * 1.05
        ax.plot([-lim, lim], [-lim, lim], color="#52514e", lw=0.8, ls="--", label="y = x")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(xlab, fontsize=8)
        ax.set_ylabel("ensemble loss error (w_INR / w_CFD − 1) [%]", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
    ax = axes[2]
    s = np.linspace(0, 1, NLINE)
    for mp, col in (("mp1", C_MP1), ("mp2", C_MP2)):
        a = np.array([p["pct"] for p in prof if p["mp"] == mp])
        ax.fill_between(s, *np.nanpercentile(a, [25, 75], 0), color=col, alpha=0.25, lw=0)
        ax.plot(s, np.nanmedian(a, 0), color=col, lw=1.0, label=f"{mp.upper()}: median of the 264 cases, 25–75 % band")
        yc = np.median([p["y_cut"] for p in prof if p["mp"] == mp])
        ax.axvline(yc, color=col, ls=":", lw=1.0,
                   label=f"location where {mp.upper()} crosses the periodic\nboundary of the reference domain")
    ax.set_xlim(0, 1)
    ax.set_xlabel("(y − y₀) / pitch", fontsize=8)
    ax.set_ylabel("|Δp0 (derived)| [% of the CFD p0 range over the domain]", fontsize=8)
    fix = df.w_err_rel_cut_from_cfd_pct.abs().mean()
    ax.set_title(f"(c) MP1 p0 error peaks at the periodic boundary; loss impact {y.abs().mean():.1f} → {fix:.1f} %",
                 fontsize=9)
    ax.set_ylim(0, None)
    ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"INR ensemble (4 models), {df.blade.nunique()} test blades × 3 OPs. p̄0: mass-flux-averaged total "
                 "pressure on a line (p0 derived from the predicted p, ρ, U); loss: mixed-out loss coefficient", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mp1_error_summary.png"), dpi=DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    set_data("data_mean")
    ens = Ensemble(torch.device(f"cuda:{args.gpu}"))
    sp = splits()
    print("pass over the test blades and the baseline")
    df, cache, prof = line_stats(ens, sp["test"] + sp["baseline"])
    df.to_csv(os.path.join(OUT, "test_line_errors.csv"), index=False)
    test = df[df.blade != "baseline"].reset_index(drop=True)
    aggregate(test).to_csv(os.path.join(OUT, "line_error_stats.csv"), index=False)
    adp = test[test.op == "ADP"].assign(a=lambda s: s.w_err_rel_pct.abs()).sort_values("a")
    median = adp.iloc[len(adp) // 2].blade
    chosen = [median] + list(adp.blade.iloc[-2:][::-1]) + ["baseline"]
    print("chosen blades:", chosen)

    summary_figure(test, prof)
    meta = {b["name"]: b["case"] for b in json.load(open(os.path.join(HERE, "data_mean", "meta.json")))["blades"]}
    for blade in chosen:
        d = load_blade(blade)
        mesh = CascadeMesh(os.path.join(meta[blade], "mesh", "constant", "polyMesh"))
        for k, op in enumerate(OPS):
            r = df[(df.blade == blade) & (df.op == op)].iloc[0]
            for err_mode in (False, True):
                line_figure(cache[(blade, op)], blade, op, r, err_mode)
            field_figure(ens, d, k, blade, op, mesh)
            print(f"  figures {blade} {op}", flush=True)
    json.dump({"chosen": chosen}, open(os.path.join(OUT, "chosen_blades.json"), "w"))


if __name__ == "__main__":
    main()
