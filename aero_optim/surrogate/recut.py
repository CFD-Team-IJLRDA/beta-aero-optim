"""
Periodic re-cut of the cascade passage so that the blade lies on the (new) periodic boundaries.

Original domain: one pitch between the lower periodic line B(x) and B(x) + pitch, with the blade inside.
Cut curve C(x):
    x < x_LE : B(x) shifted vertically to pass through the leading edge (LE = min-x wall vertex)
    x_LE..x_TE: lower blade surface
    x > x_TE : B(x) shifted vertically to pass through the trailing edge (TE = max-x wall vertex)
Every point below C(x) is moved up by one pitch (periodicity), so the new domain lies between
    lower(x) = C(x) with the *upper* blade surface between LE and TE,
    upper(x) = C(x) + pitch (lower blade surface + pitch between LE and TE).
Both boundaries are functions y(x) because each blade surface is strictly monotonic in x (checked).
"""
import numpy as np

PITCH = 0.04039


def split_wall(wall):
    """**Returns** (upper, lower) blade surfaces from the closed wall loop, each from LE to TE (x increasing)."""
    n = len(wall)
    le, te = int(np.argmin(wall[:, 0])), int(np.argmax(wall[:, 0]))
    a = wall[[(le + k) % n for k in range((te - le) % n + 1)]]
    b = wall[[(le - k) % n for k in range((le - te) % n + 1)]]
    for s in (a, b):
        if not np.all(np.diff(s[:, 0]) > 0):
            raise ValueError("blade surface not strictly monotonic in x: boundaries would not be functions")
    return (a, b) if np.interp(0.5 * (a[0, 0] + a[-1, 0]), a[:, 0], a[:, 1]) > \
        np.interp(0.5 * (b[0, 0] + b[-1, 0]), b[:, 0], b[:, 1]) else (b, a)


class Recut:
    def __init__(self, mesh, pitch=PITCH):
        self.pitch = pitch
        self.x_in, self.x_out = mesh.x_in, mesh.x_out
        self.upper_side, self.lower_side = split_wall(mesh.wall)
        self.le, self.te = self.upper_side[0], self.upper_side[-1]
        b = mesh.bottom
        self.bottom = b
        self.d_le = self.le[1] - np.interp(self.le[0], b[:, 0], b[:, 1])
        self.d_te = self.te[1] - np.interp(self.te[0], b[:, 0], b[:, 1])

    def cut(self, x, side="lower"):
        """C(x): the cut curve passing along the lower ('lower') or upper ('upper') blade surface."""
        x = np.asarray(x, float)
        s = self.lower_side if side == "lower" else self.upper_side
        y = np.where(x < self.le[0], np.interp(x, self.bottom[:, 0], self.bottom[:, 1]) + self.d_le,
                     np.where(x > self.te[0], np.interp(x, self.bottom[:, 0], self.bottom[:, 1]) + self.d_te,
                              np.interp(x, s[:, 0], s[:, 1])))
        return y

    def lower(self, x):
        """New lower boundary: cut lines + upper blade surface."""
        return self.cut(x, "upper")

    def upper(self, x):
        """New upper boundary: (cut lines + lower blade surface) + pitch."""
        return self.cut(x, "lower") + self.pitch

    def apply(self, xy):
        """**Returns** points moved into the re-cut domain (points below the cut move up one pitch)."""
        out = np.array(xy, float, copy=True)
        below = out[:, 1] < self.cut(out[:, 0], "lower")
        out[below, 1] += self.pitch
        return out

    def check(self, xy):
        """Fraction of points outside [lower(x), upper(x)] after the re-cut (should be 0)."""
        lo, up = self.lower(xy[:, 0]), self.upper(xy[:, 0])
        return float(np.mean((xy[:, 1] < lo - 1e-12) | (xy[:, 1] > up + 1e-12)))

    def boundary(self, n_cut=60, n_end=20):
        """
        **Returns** boundary sample points of the re-cut domain (x, y) and their labels:
        'lower' / 'upper' (cut lines resampled + blade wall vertices), 'inlet', 'outlet'.
        """
        xl = np.linspace(self.x_in, self.le[0], n_cut, endpoint=False)
        xr = np.linspace(self.te[0], self.x_out, n_cut + 1)[1:]
        lower = np.vstack([np.c_[xl, self.lower(xl)], self.upper_side, np.c_[xr, self.lower(xr)]])
        upper = np.vstack([np.c_[xl, self.upper(xl)], self.lower_side + [0, self.pitch], np.c_[xr, self.upper(xr)]])
        yi = np.linspace(0, self.pitch, n_end + 2)[1:-1]
        inlet = np.c_[np.full_like(yi, self.x_in), self.lower(self.x_in) + yi]
        outlet = np.c_[np.full_like(yi, self.x_out), self.lower(self.x_out) + yi]
        pts = np.vstack([lower, upper, inlet, outlet])
        labels = np.array(["lower"] * len(lower) + ["upper"] * len(upper) + ["inlet"] * len(inlet) + ["outlet"] * len(outlet))
        return pts, labels
