"""Why the direct RBF solve: condition numbers and boundary residuals, old (ridge on normal equations, Wendland C10,
radius 6.5) vs new (direct square solve, Wendland C2 + affine term, radius 0.1 m), on the baseline's boundary points."""
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts folder: _repo, train_inr, ...
import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
import numpy as np, torch
from aero_optim.surrogate.foam import CascadeMesh
from aero_optim.surrogate.recut import Recut
from aero_optim.surrogate.morph import RBFMorph, geometric_targets
m = CascadeMesh("/mnt/Data_B/beta-aero-optim-runs/doe1000_bladegen/OPENFOAM/openfoam_g1_c0/mesh/constant/polyMesh")
r = Recut(m); pts, lab = r.boundary(); tgt = geometric_targets(r, pts, lab, r.le[0], r.te[0])
c = torch.tensor(pts, dtype=torch.float64); d = torch.tensor(tgt - pts, dtype=torch.float64)
D = torch.cdist(c, c)
c10 = lambda xi: torch.clamp(1 - xi, min=0) ** 12 * (128 * xi**5 + 40 * xi**4 + 6 * xi**2 + xi + 1)
c2 = lambda xi: torch.clamp(1 - xi, min=0) ** 4 * (4 * xi + 1)
print(f"{len(pts)} boundary points (= RBF centres), square system {len(pts)}x{len(pts)}")
for name, k, R in (("old: C10, R=6.5 m", c10, 6.5), ("C2, R=6.5 m", c2, 6.5), ("new: C2, R=0.1 m", c2, 0.1)):
    Phi = k(D / R)
    ev = torch.linalg.eigvalsh(Phi)
    print(f"{name}: Phi symmetric, eigenvalues min {ev.min():.2e} max {ev.max():.2e}, cond(Phi) {ev.max()/ev.min():.2e}, "
          f"cond(Phi^T Phi) {(ev.max()/ev.min())**2:.2e}, rank {int(torch.linalg.matrix_rank(Phi))}")
    th = torch.linalg.solve(Phi.T @ Phi + 1e-6 * torch.eye(len(c), dtype=torch.float64), Phi.T @ d)   # old ridge
    print(f"   ridge (normal eq., lambda=1e-6): max boundary error {(Phi @ th - d).abs().max():.2e} m")
mo = RBFMorph(pts, tgt, radius=0.1, kernel="wendland_c2")
print(f"new direct solve with affine term: max boundary error {mo.residual:.2e} m")
