import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts folder: _repo, train_inr, ...
import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
import numpy as np, torch
from aero_optim.surrogate.foam import CascadeMesh
from aero_optim.surrogate.recut import Recut, PITCH
from aero_optim.surrogate.morph import RBFMorph, geometric_targets
m = CascadeMesh("/mnt/Data_B/beta-aero-optim-runs/doe1000_bladegen/OPENFOAM/openfoam_g1_c0/mesh/constant/polyMesh")
r = Recut(m); X = r.apply(m.centroids); pts, lab = r.boundary()
tgt = geometric_targets(r, pts, lab, X_le=r.le[0], X_te=r.te[0], wall_by_arclength=True)
mo = RBFMorph(pts, tgt, radius=0.1, kernel="wendland_c2", device="cuda")
Y = mo(X).cpu().numpy(); J = mo.jacobian_det(X).cpu().numpy()
o = np.flatnonzero((Y[:, 1] < -1e-9) | (Y[:, 1] > PITCH + 1e-9))
print("outside:", [(f"{1e3*X[i,0]:.3f}", f"{1e3*X[i,1]:.3f}", f"{1e3*Y[i,1]:.4f}") for i in o])
k = np.argsort(J)[:8]
print("lowest det J at (x, y mm):", [(f"{1e3*X[i,0]:.3f}", f"{1e3*X[i,1]:.3f}", f"{J[i]:.4f}") for i in k])
print("det J quantiles:", np.quantile(J, [0, 0.001, 0.01, 0.5]).round(4))
print("wall vertices near LE (mm):", (1e3*r.upper_side[:4]).round(3).tolist(), (1e3*r.lower_side[:4]).round(3).tolist())
print("cells with det J < 0.05:", int((J < 0.05).sum()), " < 0.2:", int((J < 0.2).sum()), "of", len(J))
low = J < 0.05
print("x range of det<0.05 cells (mm):", (1e3*X[low, 0].min()).round(2), (1e3*X[low, 0].max()).round(2))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sc = axes[0].scatter(Y[:, 0], Y[:, 1], c=np.log10(np.clip(J, 1e-3, None)), s=0.3, cmap="viridis"); axes[0].set_title("reference rectangle, color = log10 det J"); plt.colorbar(sc, ax=axes[0])
z = (Y[:, 0] > -0.003) & (Y[:, 0] < 0.004)
axes[1].scatter(Y[z, 0], Y[z, 1], c=np.log10(np.clip(J[z], 1e-3, None)), s=1, cmap="viridis"); axes[1].set_title("zoom at the LE (x_ref in -3..4 mm)")
for a in axes: a.set_xlabel("x_ref"); a.set_ylabel("y_ref")
fig.tight_layout(); fig.savefig("checks/morph_baseline_detJ.png", dpi=200)
