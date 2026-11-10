import sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts folder: _repo, train_inr, ...
import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
import numpy as np, torch
from aero_optim.surrogate.foam import CascadeMesh
from aero_optim.surrogate.recut import Recut, PITCH
from aero_optim.surrogate.morph import RBFMorph, geometric_targets
t0 = time.time()
m = CascadeMesh("/mnt/Data_B/beta-aero-optim-runs/doe1000_bladegen/OPENFOAM/openfoam_g1_c0/mesh/constant/polyMesh")
print(f"read {len(m.centroids)} cells, {len(m.wall)} wall vertices, x_in {m.x_in:.4f} x_out {m.x_out:.4f} ({time.time()-t0:.1f}s)")
r = Recut(m)
X = r.apply(m.centroids)
print("LE", r.le, "TE", r.te, "points outside re-cut domain:", r.check(X))
pts, lab = r.boundary()
ARC = True
tgt = geometric_targets(r, pts, lab, X_le=r.le[0], X_te=r.te[0], wall_by_arclength=ARC)
ARC = True
dev = "cuda" if torch.cuda.is_available() else "cpu"
for kern in ["wendland_c2", "wendland_c4"]:
    for R in [0.05, 0.1, 0.2, 0.4]:
        mo = RBFMorph(pts, tgt, radius=R, kernel=kern, device=dev)
        Y = mo(X).cpu().numpy()
        J = mo.jacobian_det(X).cpu().numpy()
        out = np.mean((Y[:, 1] < -1e-9) | (Y[:, 1] > PITCH + 1e-9) | (Y[:, 0] < m.x_in - 1e-9) | (Y[:, 0] > m.x_out + 1e-9))
        print(f"{kern} R={R}: boundary residual {mo.residual:.1e} m, det J min {J.min():.3f} max {J.max():.2f}, "
              f"det<=0: {(J<=0).sum()}, outside rectangle: {100*out:.3f}%")
