"""
T_geom: RBF morphing of the re-cut domain onto the rectangle [x_in, x_out] x [0, pitch] (PyTorch).

Boundary targets:
    lower boundary (x, lower(x)) -> (X(x), 0),  upper boundary (x, upper(x)) -> (X(x), pitch)
    inlet / outlet points        -> (x_in / x_out, y - lower(x_in / x_out))
with X(x) piecewise linear so that every blade's LE and TE land on the same reference abscissae X_LE, X_TE.

Displacements are interpolated with a compactly supported Wendland kernel plus an affine polynomial
(exact reproduction of constant/linear displacements), solving the square interpolation system directly
(LU on the small saddle-point system): boundary points land exactly on their targets.
This replaces the ridge/normal-equations solve of the previous jax code, which squares the condition number.
"""
import numpy as np
import torch

from .recut import PITCH

KERNELS = {
    "wendland_c2": lambda xi: torch.clamp(1 - xi, min=0) ** 4 * (4 * xi + 1),
    "wendland_c4": lambda xi: torch.clamp(1 - xi, min=0) ** 6 * (35 * xi ** 2 + 18 * xi + 3) / 3,
    "wendland_c6": lambda xi: torch.clamp(1 - xi, min=0) ** 8 * (32 * xi ** 3 + 25 * xi ** 2 + 8 * xi + 1),
}


def x_alignment(x, x_in, x_le, x_te, x_out, X_le, X_te):
    """Piecewise-linear map of x sending (x_in, x_le, x_te, x_out) to (x_in, X_le, X_te, x_out)."""
    return np.interp(x, [x_in, x_le, x_te, x_out], [x_in, X_le, X_te, x_out])


class RBFMorph:
    def __init__(self, centers, targets, radius=0.25, kernel="wendland_c2", device="cpu", dtype=torch.float64):
        self.kernel = KERNELS[kernel]
        self.radius = radius
        self.device, self.dtype = device, dtype
        c = torch.as_tensor(centers, device=device, dtype=dtype)
        d = torch.as_tensor(targets, device=device, dtype=dtype) - c
        n = len(c)
        phi = self.kernel(torch.cdist(c, c) / radius)
        P = torch.cat([torch.ones(n, 1, device=device, dtype=dtype), c], dim=1)
        A = torch.zeros(n + 3, n + 3, device=device, dtype=dtype)
        A[:n, :n], A[:n, n:], A[n:, :n] = phi, P, P.T
        rhs = torch.zeros(n + 3, 2, device=device, dtype=dtype)
        rhs[:n] = d
        sol = torch.linalg.solve(A, rhs)
        self.centers, self.w, self.beta = c, sol[:n], sol[n:]
        self.residual = float((self.displacement(c) - d).abs().max())   # boundary interpolation error [m]

    def displacement(self, x):
        x = torch.as_tensor(x, device=self.device, dtype=self.dtype)
        phi = self.kernel(torch.cdist(x, self.centers) / self.radius)
        P = torch.cat([torch.ones(len(x), 1, device=self.device, dtype=self.dtype), x], dim=1)
        return phi @ self.w + P @ self.beta

    def __call__(self, x, chunk=20000):
        x = torch.as_tensor(x, device=self.device, dtype=self.dtype)
        return torch.cat([xc + self.displacement(xc) for xc in torch.split(x, chunk)])

    def jacobian_det(self, x, chunk=5000):
        """det(I + grad(delta)) at the points x (bijectivity check: must be > 0 everywhere)."""
        out = []
        for xc in torch.split(torch.as_tensor(x, device=self.device, dtype=self.dtype), chunk):
            xc = xc.clone().requires_grad_(True)
            m = xc + self.displacement(xc)
            gx = torch.autograd.grad(m[:, 0].sum(), xc, retain_graph=True)[0]
            gy = torch.autograd.grad(m[:, 1].sum(), xc)[0]
            out.append((gx[:, 0] * gy[:, 1] - gx[:, 1] * gy[:, 0]).detach())
        return torch.cat(out)


def _arclength_fraction(side):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(side, axis=0), axis=1))]
    return s / s[-1]


def geometric_targets(recut, pts, labels, X_le, X_te, pitch=PITCH, wall_by_arclength=True):
    """
    Boundary targets of T_geom for the boundary samples returned by Recut.boundary().
    Cut lines: X(x) (piecewise linear, LE/TE aligned). Blade surfaces: by default placed by normalized
    arc length between X_le and X_te, so the rounded LE/TE keep their size instead of being squeezed
    (placing them by x compresses ~1.5 mm of nose arc into ~0.1 mm: det J ~ 1e-4).
    """
    X = x_alignment(pts[:, 0], recut.x_in, recut.le[0], recut.te[0], recut.x_out, X_le, X_te)
    if wall_by_arclength:
        for side, lab, shift in ((recut.upper_side, "lower", 0.0), (recut.lower_side, "upper", pitch)):
            on = labels == lab
            idx = np.flatnonzero(on)
            # the blade vertices are the block of boundary samples equal to the side vertices
            d = np.linalg.norm(pts[idx][:, None, :] - (side + [0, shift])[None, :, :], axis=2)
            hit = d.min(axis=1) < 1e-12
            X[idx[hit]] = X_le + _arclength_fraction(side)[d[hit].argmin(axis=1)] * (X_te - X_le)
    tgt = np.c_[X, np.zeros(len(pts))]
    tgt[labels == "upper", 1] = pitch
    for lab in ("inlet", "outlet"):
        m = labels == lab
        tgt[m, 1] = pts[m, 1] - recut.lower(pts[m, 0])
    return tgt
