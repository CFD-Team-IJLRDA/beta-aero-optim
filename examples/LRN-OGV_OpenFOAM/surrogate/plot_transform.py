"""
Step 1 figures (600 DPI) for one blade, in checks/figures/<blade>/:
  1_mesh_original.png   physical domain: cell centres, inlet/outlet, periodic boundaries, blade sides, cut lines, LE/TE
  2_mesh_recut.png      re-cut domain (blade on the new periodic boundaries), coloured by Mach
  3_<OP>_<domain>_{scatter,contour}.png   Mach in the physical, geometry-aligned (T_geom) and balanced (T_bal)
                        domains; 'contour' = filled tricontour (Delaunay on the cell centres, triangles outside
                        the fluid masked), turbo colormap; the balanced domain is drawn as the unit square.
  balance_maps.png      the training-averaged T_bal maps
Also prints the check Ma = |U| / sqrt(gamma p / rho).

Usage: ~/miniconda3/envs/ipynb/bin/python plot_transform.py [blade=baseline] [op=ADP]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
HERE = _repo.WORK                 # data, runs, sopt: the surrogate work folder (aero_optim/surrogate/paths.py)
sys.path.insert(0, _repo.SRC)
from aero_optim.surrogate.foam import CascadeMesh  # noqa: E402
from aero_optim.surrogate.recut import PITCH, Recut  # noqa: E402

OPS = ["ADP", "OP1", "OP2"]
DPI = 600
C = dict(inlet="#2a78d6", outlet="#d03b3b", upper="#1a8a3a", lower="#eda100", suction="#d55181", pressure="#1baf7a",
         cut="#0b0b0b", interior="#b9b3d6")
plt.rcParams.update({"font.size": 10})


def balance(X, bal):
    return np.c_[np.interp(X[:, 0], bal["grid_x"], bal["F_x"]), np.interp(X[:, 1], bal["grid_y"], bal["F_y"])]


def masked_triangulation(X, inside):
    """Delaunay triangulation of the points, masking triangles whose centroid fails inside(centroids)."""
    tri = Triangulation(X[:, 0], X[:, 1])
    cen = X[tri.triangles].mean(axis=1)
    tri.set_mask(~inside(cen))
    return tri


def field_figures(X, ma, inside, name, title, outdir, square=False, xlabel="x [m]", ylabel="y [m]", extra=None):
    w, h = (6.5, 6.5) if square else (10, 10 * np.ptp(X[:, 1]) / np.ptp(X[:, 0]) + 1.2)
    for kind in ("scatter", "contour"):
        fig, ax = plt.subplots(figsize=(w, max(h, 3.2)))
        if kind == "scatter":
            m = ax.scatter(X[:, 0], X[:, 1], c=ma, s=0.6, lw=0, cmap="turbo", vmin=0, vmax=0.9, rasterized=True)
        else:
            m = ax.tricontourf(masked_triangulation(X, inside), ma, levels=np.linspace(0, 0.9, 61), cmap="turbo", extend="max")
        if extra:
            extra(ax)
        ax.set_aspect("equal")
        if square:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        fig.colorbar(m, ax=ax, shrink=0.8, label="Mach number", ticks=np.arange(0, 1.0, 0.1))
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{name}_{kind}.png"), dpi=DPI)
        plt.close(fig)


def main():
    blade = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    op = sys.argv[2] if len(sys.argv) > 2 else "ADP"
    k = OPS.index(op)
    meta_case = {b["name"]: b["case"] for b in __import__("json").load(open(os.path.join(HERE, "data", "meta.json")))["blades"]}
    d = np.load(os.path.join(HERE, "data", "blades", f"{blade}.npz"))
    bal = np.load(os.path.join(HERE, "data", "balance.npz"))
    outdir = os.path.join(HERE, "checks", "figures", blade)
    os.makedirs(outdir, exist_ok=True)
    mesh = CascadeMesh(os.path.join(meta_case[blade], "mesh", "constant", "polyMesh"))
    rc = Recut(mesh)
    ma = d["fields"][k, :, 5]
    Xp, Xc, Xg, Xh = d["X_phys"], rc.apply(d["X_phys"]), d["X_geom"], balance(d["X_geom"], bal)
    up, lo = rc.upper_side, rc.lower_side
    xs = np.linspace(rc.x_in, rc.x_out, 800)

    # 1. original mesh
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(Xp[:, 0], Xp[:, 1], s=0.4, color=C["interior"], lw=0, label="cell centres (fluid)", rasterized=True)
    b = mesh.bottom
    ax.plot(b[:, 0], b[:, 1], ".", ms=2.5, color=C["lower"], label="periodic lower")
    ax.plot(b[:, 0], b[:, 1] + PITCH, ".", ms=2.5, color=C["upper"], label="periodic upper")
    ax.plot([rc.x_in] * 2, [np.interp(rc.x_in, b[:, 0], b[:, 1]), np.interp(rc.x_in, b[:, 0], b[:, 1]) + PITCH], color=C["inlet"], lw=3, label="inlet")
    ax.plot([rc.x_out] * 2, [np.interp(rc.x_out, b[:, 0], b[:, 1]), np.interp(rc.x_out, b[:, 0], b[:, 1]) + PITCH], color=C["outlet"], lw=3, label="outlet")
    ax.plot(up[:, 0], up[:, 1], color=C["suction"], lw=2, label="suction side (upper surface)")
    ax.plot(lo[:, 0], lo[:, 1], color=C["pressure"], lw=2, label="pressure side (lower surface)")
    xl, xr = xs[xs < rc.le[0]], xs[xs > rc.te[0]]
    ax.plot(xl, rc.cut(xl), "--", color=C["cut"], lw=1.2, label="cut: periodic lower shifted through LE / TE")
    ax.plot(xr, rc.cut(xr), "--", color=C["cut"], lw=1.2)
    for x, ls, lab in ((rc.le[0], "--", "leading edge (min x)"), (rc.te[0], ":", "trailing edge (max x)")):
        ax.axvline(x, color="#52514e", ls=ls, lw=1, label=lab)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{blade}: original OpenFOAM domain ({len(Xp)} cells) and periodic cut", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "1_mesh_original.png"), dpi=DPI)
    plt.close(fig)

    # 2. re-cut domain coloured by Mach
    fig, ax = plt.subplots(figsize=(12, 7))
    sc = ax.scatter(Xc[:, 0], Xc[:, 1], c=ma, s=0.5, lw=0, cmap="turbo", vmin=0, vmax=0.9, rasterized=True)
    ax.plot(xs, rc.lower(xs), color=C["lower"], lw=2, label="new periodic lower (cut + suction side)")
    ax.plot(xs, rc.upper(xs), color=C["upper"], lw=2, label="new periodic upper (cut + pressure side, + pitch)")
    ax.plot([rc.x_in] * 2, [rc.lower(rc.x_in), rc.upper(rc.x_in)], color=C["inlet"], lw=3, label="inlet")
    ax.plot([rc.x_out] * 2, [rc.lower(rc.x_out), rc.upper(rc.x_out)], color=C["outlet"], lw=3, label="outlet")
    for x, ls, lab in ((rc.le[0], "--", "leading edge"), (rc.te[0], ":", "trailing edge")):
        ax.axvline(x, color="#52514e", ls=ls, lw=1, label=lab)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.colorbar(sc, ax=ax, shrink=0.7, label=f"Mach number ({op})")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{blade}: re-cut domain, the blade lies on the periodic boundaries; both boundaries are functions y(x)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "2_mesh_recut.png"), dpi=DPI)
    plt.close(fig)

    # 3. Mach in the three domains
    bpath = Path(np.vstack([b, (b + [0, PITCH])[::-1]]))
    blade_path = Path(np.vstack([up, lo[::-1]]))

    def inside_phys(p):
        return bpath.contains_points(p) & ~blade_path.contains_points(p)

    def mp_lines(key):
        def draw(ax):
            for mp, col in (("mp1", "white"), ("mp2", "white")):
                p = d[f"{mp}_phys"] if key == "phys" else (d[f"{mp}_ref"] if key == "geom" else balance(d[f"{mp}_ref"], bal))
                ax.plot(p[:, 0], p[:, 1], ".", ms=0.8, color=col)
        return draw

    field_figures(Xp, ma, inside_phys, f"3_{op}_a_physical", f"{blade}, {op}: physical domain (MP1, MP2 in white)",
                  outdir, extra=mp_lines("phys"))
    field_figures(Xg, ma, lambda p: np.ones(len(p), bool), f"3_{op}_b_geometry_aligned",
                  f"{blade}, {op}: T_geom, re-cut + RBF morph onto [x_in, x_out] x [0, pitch]", outdir,
                  xlabel="x_geom [m]", ylabel="y_geom [m]", extra=mp_lines("geom"))
    field_figures(Xh, ma, lambda p: np.ones(len(p), bool), f"3_{op}_c_balanced",
                  f"{blade}, {op}: T_bal, balanced unit square", outdir, square=True,
                  xlabel="x_hat", ylabel="y_hat", extra=mp_lines("hat"))

    # balance maps
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, kk, lab in ((axes[0], "x", "x_geom [m]"), (axes[1], "y", "y_geom [m]")):
        g = bal[f"grid_{kk}"]
        ax.plot(g, bal[f"F_{kk}"], color="#2a78d6", label="training-averaged eCDF")
        ax.plot(g, (g - g[0]) / np.ptp(g), ":", color="#52514e", label="uniform (no balancing)")
        ax.set_xlabel(lab)
        ax.set_ylabel(f"{kk}_hat = F_{kk}({kk}_geom)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "checks", "balance_maps.png"), dpi=DPI)
    plt.close(fig)

    f = d["fields"][k]
    ma_derived = np.hypot(f[:, 2], f[:, 3]) / np.sqrt(1.4 * f[:, 0] / f[:, 1])
    print(f"{blade} {op}: figures in {outdir}; max |Ma_solver - |U|/sqrt(gamma p/rho)| = {np.abs(ma_derived - f[:, 5]).max():.2e}")


if __name__ == "__main__":
    main()
