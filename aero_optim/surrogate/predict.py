"""
Surrogate QoI predictor for NEW blades (no mesh, no CFD): POD coefficients -> blade profile -> T_geom of the two
measurement lines (periodic re-cut + RBF morph from the profile alone, same code as the dataset) -> T_bal + embedding ->
INR ensemble on MP1/MP2 -> mixed-out loss and outflow angle per operating point (same plane_qois as the CFD post-processing).

Validation (13 Nov): the profile-only mapping changes the predicted loss by 0.06-0.17 % on average (max 0.43 %) and the
angle by 0.001 deg w.r.t. the mapping built from the CFD mesh, much less than the model error.

Usage:
    P = Predictor(["runs/a", "runs/b", "runs/c"], pod_basis="/.../doe1000_bladegen/pod/basis.npz")
    q = P.qois(profiles, pods)      # (n, n_models, 3 OPs, 2) = [loss, angle]
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from aero_optim.simulator.cascade_qoi import plane_qois

from . import data as D
from .data import COORDS, INLET_ANGLE, RAMP, TARGETS, Normalizer, balance, embed, load_balance
from .foam import CascadeMesh
from .morph import RBFMorph, geometric_targets
from .paths import WORK as HERE                    # datasets (meta.json) live in the surrogate work folder
from .recut import PITCH, Recut
from .siren import build_model

X_LE, X_TE = -0.00011, 0.06725                       # reference LE / TE abscissae of T_geom (build_dataset.py)
MP = ((-0.02, -0.0285704, 0.0118176), (0.087, 0.000001, 0.040389))
RBF = dict(radius=0.1, kernel="wendland_c2")
OPS = ["ADP", "OP1", "OP2"]


class _Domain:
    """Stand-in for CascadeMesh: the fixed domain (periodic line, inlet, outlet) plus the blade wall of a candidate."""
    def __init__(self, base, wall):
        self.x_in, self.x_out, self.bottom, self.wall = base.x_in, base.x_out, base.bottom, wall


def _df(f):
    return pd.DataFrame({"p": f[:, 0], "rho": f[:, 1], "Ux": f[:, 2], "Uy": f[:, 3]})


class Predictor:
    def __init__(self, run_dirs, data="data_mean", checkpoint="best_qoi_model.pt", device="cuda:1", baseline_case=None):
        D.set_data(data)
        meta = json.load(open(os.path.join(HERE, data, "meta.json")))["blades"]
        case = baseline_case or next(b["case"] for b in meta if b["name"] == "baseline")
        self.base = CascadeMesh(os.path.join(case, "mesh", "constant", "polyMesh"))
        self.device = torch.device(device)
        self.models = []
        for rd in run_dirs:
            cfg = json.load(open(os.path.join(rd, "config.json")))
            D.set_data(cfg.get("data", data))
            model = build_model(cfg, COORDS[cfg.get("coords", "plain")], 6, len(TARGETS)).to(self.device)
            model.load_state_dict(torch.load(os.path.join(rd, checkpoint), map_location=self.device))
            model.eval()
            self.models.append((model, Normalizer.load(os.path.join(rd, "normalizer.npz")), cfg.get("coords", "plain"),
                                dict(load_balance())))
        D.set_data(data)

    def lines(self, profile):
        """MP1 and MP2 sample points of a blade in the reference domain, (1001, 2) each."""
        rc = Recut(_Domain(self.base, np.asarray(profile)[:, :2]))
        pts, lab = rc.boundary()
        morph = RBFMorph(pts, geometric_targets(rc, pts, lab, X_LE, X_TE), **RBF)
        return [morph(rc.apply(np.c_[np.full(1001, x), np.linspace(y0, y1, 1001)])).numpy() for x, y0, y1 in MP]

    @torch.no_grad()
    def qois(self, profiles, pods):
        """(n, n_models, 3, 2): [mixed-out loss, outflow angle] per blade, model and operating point."""
        n = len(profiles)
        L = np.stack([np.stack(self.lines(p)) for p in profiles])             # (n, 2 lines, 1001, 2)
        out = np.zeros((n, len(self.models), 3, 2))
        for m, (model, norm, coords, bal) in enumerate(self.models):
            X = np.stack([embed(balance(L[i, j], bal), coords, bal) for i in range(n) for j in range(2)])   # (2n, 1001, k)
            Xt = torch.from_numpy(norm("X", X.reshape(-1, X.shape[-1])).astype(np.float32)).to(self.device)
            for k in range(3):
                P = np.repeat(np.c_[np.asarray(pods), np.full(n, INLET_ANGLE[k])], 2 * 1001, axis=0).reshape(n, 2 * 1001, 6)
                Pt = torch.from_numpy(norm("P", P.reshape(-1, 6)).astype(np.float32)).to(self.device)
                # rows: blade-major, then line, then point -> same order as X.reshape(-1, k)
                y = norm.inverse_Y(model(Pt, Xt).cpu().numpy()).reshape(n, 2, 1001, -1)
                for i in range(n):
                    q = plane_qois(_df(y[i, 0]), _df(y[i, 1]))
                    out[i, m, k] = (q["MixedoutLossCoef"], q["OutflowAngle"])
        return out


# ---------------------------------------------------------------------------------------------------------------------
# Differentiable version (torch end to end): POD coefficients -> profile -> re-cut -> T_geom of the MP lines -> T_bal ->
# embedding -> INR ensemble -> mixed-out loss and outflow angle. Same maths as Recut / geometric_targets / RBFMorph /
# plane_qois (numpy), so gradients d(loss, angle)/d(coefficients) come from autograd. Geometry in float64.
GAMMA = 1.4


def _interp(x, xp, fp):
    """torch np.interp (xp increasing, clamped at the ends); differentiable in x, xp and fp."""
    i = torch.clamp(torch.searchsorted(xp.detach().contiguous(), x.detach().contiguous()), 1, len(xp) - 1)
    x0, x1, f0, f1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    t = torch.clamp((x - x0) / (x1 - x0), 0, 1)
    return f0 + t * (f1 - f0)


def _mixedout(rho, rhou, rhov, p):
    m, r, pb = rhou.mean(), rho.mean(), p.mean()
    u, v = m / r, rhov.mean() / r
    xm, ym, g = m * u + pb, m * v, GAMMA
    E = m * g / (g - 1) * pb / r + m / 2 * (u ** 2 + v ** 2)
    Q = 1 / m ** 2 * (1 - 2 * g / (g - 1))
    L = 2 / m ** 2 * (g / (g - 1) * xm - xm)
    C = 1 / m ** 2 * (xm ** 2 + ym ** 2) - 2 * E / m
    pb = (-L - torch.sqrt(L ** 2 - 4 * Q * C)) / 2 / Q
    return pb, pb * ((g - 1) / g * E / m * r / pb) ** (g / (g - 1))


def _plane_qois(f1, f2):
    """(loss, angle) from fields (M, >=4) = p, rho, Ux, Uy on MP1 and MP2 (torch version of cascade_qoi.plane_qois)."""
    s = [(f[:, 1], f[:, 1] * f[:, 2], f[:, 1] * f[:, 3], f[:, 0]) for f in (f1, f2)]
    p1, p01 = _mixedout(*s[0])
    _, p02 = _mixedout(*s[1])
    return (p01 - p02) / (p01 - p1), torch.rad2deg(torch.atan(s[1][2].mean() / s[1][1].mean()))


class DiffPredictor(Predictor):
    """
    qois_torch(c) -> (n_models, 3 OPs, [loss, angle]) as a differentiable function of the POD coefficients c (torch, (5,)).
    The re-cut uses the same discrete choices as Recut (LE/TE vertices, sides, points moved by one pitch), taken from the
    current blade; they are piecewise constant, so they do not enter the gradient.
    """
    N_CUT, N_END = 60, 20

    def __init__(self, run_dirs, pod, **kw):
        super().__init__(run_dirs, **kw)
        dd = dict(dtype=torch.float64, device=self.device)
        self.dd = dd
        self.pod = {k: torch.as_tensor(getattr(pod, k), **dd) for k in ("base", "mean", "modes")}
        self.bottom = torch.as_tensor(self.base.bottom, **dd)
        self.x_in, self.x_out = float(self.base.x_in), float(self.base.x_out)
        self.mp = [torch.stack([torch.full((1001,), x, **dd), torch.linspace(y0, y1, 1001, **dd)], 1) for x, y0, y1 in MP]
        meta = json.load(open(os.path.join(HERE, kw.get("data", "data_mean"), "meta.json")))
        self.ext = []
        for model, norm, coords, bal in self.models:
            b = {k: torch.as_tensor(v, **dd) for k, v in bal.items()}
            ext = tuple(float(np.interp(x, bal["grid_x"], bal["F_x"])) for x in (meta["x_le_ref"], meta["x_te_ref"]))
            st = {k: tuple(torch.as_tensor(a, **dd) for a in v) for k, v in norm.stats.items()}
            self.ext.append((b, ext, st))

    def profile(self, c):
        return self.pod["base"] + (self.pod["mean"] + self.pod["modes"] @ c).reshape(self.pod["base"].shape)

    def lines_torch(self, c):
        """MP1, MP2 sample points in the reference (T_geom) domain, differentiable in c."""
        W = self.profile(c)
        Wn = W.detach().cpu().numpy()
        n, le, te = len(Wn), int(np.argmin(Wn[:, 0])), int(np.argmax(Wn[:, 0]))
        ia = [(le + k) % n for k in range((te - le) % n + 1)]
        ib = [(le - k) % n for k in range((le - te) % n + 1)]
        ya = np.interp(0.5 * (Wn[ia[0], 0] + Wn[ia[-1], 0]), Wn[ia, 0], Wn[ia, 1])
        yb = np.interp(0.5 * (Wn[ib[0], 0] + Wn[ib[-1], 0]), Wn[ib, 0], Wn[ib, 1])
        up, lo = (W[ia], W[ib]) if ya > yb else (W[ib], W[ia])       # upper / lower blade surface, LE -> TE
        LE, TE, B, pitch = up[0], up[-1], self.bottom, PITCH
        d_le = LE[1] - _interp(LE[0:1], B[:, 0], B[:, 1])[0]
        d_te = TE[1] - _interp(TE[0:1], B[:, 0], B[:, 1])[0]

        def cut(x, s):
            return torch.where(x < LE[0], _interp(x, B[:, 0], B[:, 1]) + d_le,
                               torch.where(x > TE[0], _interp(x, B[:, 0], B[:, 1]) + d_te, _interp(x, s[:, 0], s[:, 1])))
        lower = lambda x: cut(x, up)                                                       # noqa: E731
        upper = lambda x: cut(x, lo) + pitch                                               # noqa: E731
        k = torch.arange(self.N_CUT, **self.dd)
        xl = self.x_in + (LE[0] - self.x_in) * k / self.N_CUT
        xr = TE[0] + (self.x_out - TE[0]) * (k + 1) / self.N_CUT
        yi = torch.linspace(0, pitch, self.N_END + 2, **self.dd)[1:-1]
        xi, xo = torch.full_like(yi, self.x_in), torch.full_like(yi, self.x_out)
        pts = torch.cat([torch.stack([xl, lower(xl)], 1), up, torch.stack([xr, lower(xr)], 1),
                         torch.stack([xl, upper(xl)], 1), lo + torch.tensor([0., pitch], **self.dd), torch.stack([xr, upper(xr)], 1),
                         torch.stack([xi, lower(xi) + yi], 1), torch.stack([xo, lower(xo) + yi], 1)])
        Xle, Xte = X_LE, X_TE
        align = lambda x: _interp(x, torch.stack([torch.tensor(self.x_in, **self.dd), LE[0], TE[0],   # noqa: E731
                                                  torch.tensor(self.x_out, **self.dd)]),
                                  torch.tensor([self.x_in, Xle, Xte, self.x_out], **self.dd))

        def arc(s):
            a = torch.cat([torch.zeros(1, **self.dd), torch.cumsum(torch.linalg.norm(s[1:] - s[:-1], dim=1), 0)])
            return Xle + a / a[-1] * (Xte - Xle)
        z = lambda m: torch.zeros(m, **self.dd)                                            # noqa: E731
        tx = torch.cat([align(xl), arc(up), align(xr), align(xl), arc(lo), align(xr), xi, xo])
        ty = torch.cat([z(2 * self.N_CUT + len(up)), torch.full((2 * self.N_CUT + len(lo),), pitch, **self.dd),
                        yi, yi])
        tgt = torch.stack([tx, ty], 1)
        morph = RBFMorph(pts, tgt, device=self.device, **RBF)
        out = []
        for L in self.mp:
            below = (L[:, 1] < cut(L[:, 0], lo)).detach()
            L = L + torch.stack([torch.zeros_like(L[:, 0]), below * pitch], 1)
            out.append(L + morph.displacement(L))
        return out

    def qois_torch(self, c):
        Ls = self.lines_torch(c)
        res = []
        for (model, norm, coords, _), (b, (x_le, x_te), st) in zip(self.models, self.ext):
            feats = []
            for L in Ls:
                xh = torch.stack([_interp(L[:, 0], b["grid_x"], b["F_x"]), _interp(L[:, 1], b["grid_y"], b["F_y"])], 1)
                if coords == "periodic":
                    x, y = xh[:, 0], xh[:, 1]
                    t = torch.clamp(torch.minimum((x - (x_le - RAMP)) / RAMP, ((x_te + RAMP) - x) / RAMP), 0, 1)
                    xh = torch.stack([x, torch.sin(2 * np.pi * y), torch.cos(2 * np.pi * y), t * t * (3 - 2 * t) * y], 1)
                feats.append((xh - st["X"][0]) / st["X"][1])
            X = torch.cat(feats).float()
            per_op = []
            for k in range(3):
                P = torch.cat([c, torch.tensor([INLET_ANGLE[k]], **self.dd)])
                P = ((P - st["P"][0]) / st["P"][1]).float().expand(len(X), -1)
                y = model(P, X).double() * st["Y"][1] + st["Y"][0]
                per_op.append(torch.stack(_plane_qois(y[:1001], y[1001:])))
            res.append(torch.stack(per_op))
        return torch.stack(res)
