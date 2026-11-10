"""Are the re-cut boundaries functions y(x)? For each blade: split the profile at LE (min x) and TE (max x),
check that x is strictly monotonic along both sides, and report the LE/TE positions."""
import glob, os
import numpy as np, pandas as pd
rows = []
for f in sorted(glob.glob("/mnt/Data_B/beta-aero-optim-runs/doe1000_bladegen/profiles/blade_*.dat")):
    p = np.loadtxt(f, skiprows=2)[:, :2]
    n = len(p)
    le, te = int(np.argmin(p[:, 0])), int(np.argmax(p[:, 0]))
    a = p[[(le + k) % n for k in range((te - le) % n + 1)]]          # LE -> TE one way round
    b = p[[(le - k) % n for k in range((le - te) % n + 1)]]          # LE -> TE the other way
    dxa, dxb = np.diff(a[:, 0]), np.diff(b[:, 0])
    upper, lower = (a, b) if a[len(a) // 2, 1] > b[len(b) // 2, 1] else (b, a)
    rows.append(dict(blade=os.path.basename(f)[6:-4], n_side_a=len(a), n_side_b=len(b),
                     mono_a=bool(np.all(dxa > 0)), mono_b=bool(np.all(dxb > 0)),
                     min_dx_a=dxa.min(), min_dx_b=dxb.min(),
                     x_le=p[le, 0], y_le=p[le, 1], x_te=p[te, 0], y_te=p[te, 1]))
d = pd.DataFrame(rows); d.to_csv("functional_boundaries.csv", index=False)
print(len(d), "blades; both sides strictly monotonic in x:", int((d.mono_a & d.mono_b).sum()))
bad = d[~(d.mono_a & d.mono_b)]
print(bad.head(10).to_string(index=False))
for c in ["x_le", "y_le", "x_te", "y_te"]:
    print(f"{c}: min {1e3*d[c].min():.3f} mm, max {1e3*d[c].max():.3f} mm, spread {1e3*(d[c].max()-d[c].min()):.3f} mm")
