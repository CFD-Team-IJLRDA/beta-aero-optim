"""
Geometric constraints of aero_optim.optim.cascade_foam.GeometryConstraints in torch, so that their gradients with respect to
the POD coefficients come from automatic differentiation (used by the gradient-based optimiser and the KKT analysis).
"""
import numpy as np
import torch


def _interp(x, xp, fp):
    """torch np.interp (xp increasing, clamped), differentiable in x, xp and fp."""
    i = torch.clamp(torch.searchsorted(xp.detach().contiguous(), x.detach().contiguous()), 1, len(xp) - 1)
    t = torch.clamp((x - xp[i - 1]) / (xp[i] - xp[i - 1]), 0, 1)
    return fp[i - 1] + t * (fp[i] - fp[i - 1])


class TorchGeometry:
    """
    The geometric constraints of aero_optim.optim.cascade_foam.GeometryConstraints (t_max/c, x_tmax/c_ax, A/c^2, x_cg/c_ax, r_LE,
    BladeGen family) in torch, so that their gradients w.r.t. the POD coefficients come from autodiff. Same formulas as
    aero_optim.geom (shoelace area and centroid, thickness on interpolated surfaces, LE circle test, nearest DoE blade); the discrete
    choices (LE/TE vertices, nearest points, max) are taken at the current blade and are piecewise constant.
    """
    def __init__(self, ref, pod, numpy_geometry, device="cpu"):
        self.dd = dict(dtype=torch.float64, device=device)
        self.pod = {k: torch.as_tensor(getattr(pod, k), **self.dd) for k in ("base", "mean", "modes")}
        self.ref, self.tol, self.r_le_min = ref, numpy_geometry.tol, numpy_geometry.r_le_min
        self.scale = torch.as_tensor(numpy_geometry.scale, **self.dd)
        self.train = torch.as_tensor(np.asarray(numpy_geometry.tree.data), **self.dd)          # standardised training coefficients
        self.max_dist = numpy_geometry.max_dist

    def profile(self, c):
        return self.pod["base"] + (self.pod["mean"] + self.pod["modes"] @ c).reshape(self.pod["base"].shape)

    @staticmethod
    def _sides(p):
        pn = p.detach().cpu().numpy()
        le, te = int(np.argmin(pn[:, 0])), int(np.argmax(pn[:, 0]))
        s, e = min(le, te), max(le, te)
        mid, wrap = list(range(s, e + 1)), list(range(e, len(pn))) + list(range(0, s + 1))
        a, b = (mid, wrap) if pn[mid, 1].max() > pn[wrap, 1].max() else (wrap, mid)
        return p[a], p[b], le, te                                                   # upper, lower

    def metrics(self, p):
        up, lo, le, te = self._sides(p)
        c, c_ax = torch.linalg.norm(p[le] - p[te]), p[te, 0] - p[le, 0]
        q = torch.roll(p, 1, 0)                                                     # p[i - 1]
        cross = q[:, 0] * p[:, 1] - p[:, 0] * q[:, 1]
        area = 0.5 * cross.sum()
        x_cg = ((q[:, 0] + p[:, 0]) * cross).sum() / (6 * area)
        grids = []
        for s_ in (up, lo):
            s_ = s_[torch.argsort(s_[:, 0].detach())]
            x0, x1 = s_[:, 0].min(), s_[:, 0].max()
            n = int(np.ceil(float((x1 - x0 - 1e-6).detach() / (0.0025 * c.detach()))))
            x = x0 + 1e-6 + torch.arange(n, **self.dd) * 0.0025 * c
            grids.append(torch.stack([x, _interp(x, s_[:, 0], s_[:, 1])], 1))
        gu, gl = grids
        j = torch.cdist(gu.detach(), gl.detach()).argmin(1)                         # nearest lower point (choice, no gradient)
        dmin = torch.linalg.norm(gu - gl[j], dim=1)
        k = int(dmin.detach().argmax())
        camber_x = 0.5 * (gu[k, 0] + gl[j[k], 0])
        le_x = torch.minimum(gu[:, 0].min(), gl[:, 0].min())
        return {"t_max/c": dmin[k] / c, "x_tmax/c_ax": (camber_x - le_x) / c_ax, "A/c2": torch.abs(area) / c ** 2,
                "x_cg/c_ax": x_cg / c_ax, "c": c}

    def le_radius(self, p, r, max_offset=1.4):
        pn = p.detach().cpu().numpy()
        te = int(np.argmax(pn[:, 0]))
        le = int(np.argmax(np.linalg.norm(pn - pn[te], axis=1)))
        n, seg = len(pn), np.linalg.norm(np.diff(pn, axis=0, append=pn[:1]), axis=1)
        rr, ahead, behind, ka, kb = float(r.detach()), 0., 0., le, le
        while ahead < 2 * rr:
            ahead += seg[ka % n]
            ka += 1
        while behind < 2 * rr:
            kb -= 1
            behind += seg[kb % n]
        dvec = 0.5 * (p[ka % n] + p[kb % n]) - p[le]
        dvec = dvec / torch.linalg.norm(dvec)
        ab = torch.roll(p, -1, 0) - p
        best = None
        for f in np.linspace(1, max_offset, 9):
            x = p[le] + f * r * dvec
            t = torch.clamp(((x - p) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
            dist = torch.linalg.norm(p + t[:, None] * ab - x, dim=1).min()
            best = dist if best is None else torch.maximum(best, dist)
        return r - best

    def __call__(self, c):
        """g (6,) as torch: the 4 relative geometric constraints, r_LE and the BladeGen family (<= 0: satisfied)."""
        p = self.profile(c)
        m = self.metrics(p)
        g = [torch.abs(m[k] - self.ref[k]) / self.ref[k] - self.tol[k] for k in ("t_max/c", "x_tmax/c_ax", "A/c2", "x_cg/c_ax")]
        r = self.r_le_min * m["c"]
        g.append(self.le_radius(p, r) / r)
        g.append(torch.linalg.norm(self.train - c / self.scale, dim=1).min() / self.max_dist - 1.)
        return torch.stack(g)
