"""Cascade quantities of interest from OpenFOAM line samples at the measurement planes.

Port of the WOLF post-processing in examples/MultifidelityOptimization/RANS_bruteForce/execute.py:
mixed-out loss (A. Prasad 2004, doi:10.1115/1.1928289), mass-flux weighted "MP" loss and flow angles.
Each plane is sampled uniformly over exactly one pitch, so plain means are pitchwise averages.
"""
import glob
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GAMMA = 1.4
QOI_NAMES = ["MixedoutLossCoef", "MPLossCoef", "OutflowAngle", "InflowAngle"]


def mixedout_state(rho: np.ndarray, rhou: np.ndarray, rhov: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """
    **Returns** the mixed-out static and total pressure (p_bar, p0_bar) of a measurement plane.
    Identical to `compute_mixedout_qty` of the WOLF pipeline.
    """
    m_bar = np.nanmean(rhou)
    rho_bar = np.nanmean(rho)
    p_bar = np.nanmean(p)
    u_bar = m_bar / rho_bar
    v_bar = np.nanmean(rhov) / rho_bar
    x_mom = m_bar * u_bar + p_bar
    y_mom = m_bar * v_bar
    g = GAMMA
    E = m_bar * g / (g - 1) * p_bar / rho_bar + m_bar / 2. * (u_bar**2 + v_bar**2)
    Q = 1 / m_bar**2 * (1 - 2 * g / (g - 1))
    L = 2 / m_bar**2 * (g / (g - 1) * x_mom - x_mom)
    C = 1 / m_bar**2 * (x_mom**2 + y_mom**2) - 2 * E / m_bar
    p_bar = (-L - np.sqrt(L**2 - 4 * Q * C)) / 2 / Q  # subsonic root
    # T0/T with R cancelled: T_bar = p_bar/(rho_bar R), T0_bar = (g-1)/(g R) E/m_bar
    t0_over_t = (g - 1) / g * E / m_bar * rho_bar / p_bar
    return p_bar, p_bar * t0_over_t**(g / (g - 1))


def total_pressure(rho: np.ndarray, U2: np.ndarray, p: np.ndarray) -> np.ndarray:
    """**Returns** the local isentropic total pressure, using a^2 = gamma p / rho."""
    mach2 = rho * U2 / (GAMMA * p)
    return p * (1 + (GAMMA - 1) / 2 * mach2)**(GAMMA / (GAMMA - 1))


def plane_qois(mp1: pd.DataFrame, mp2: pd.DataFrame) -> dict[str, float]:
    """
    **Returns** the QoIs from one pair of plane samples with columns rho, Ux, Uy, p.
    """
    planes = {}
    for name, df in (("MP1", mp1), ("MP2", mp2)):
        rho = df["rho"].to_numpy()
        ux, uy = df["Ux"].to_numpy(), df["Uy"].to_numpy()
        p = df["p"].to_numpy()
        planes[name] = dict(rho=rho, rhou=rho * ux, rhov=rho * uy, p=p,
                            p0=total_pressure(rho, ux**2 + uy**2, p))

    p1, p01 = mixedout_state(*(planes["MP1"][k] for k in ("rho", "rhou", "rhov", "p")))
    _, p02 = mixedout_state(*(planes["MP2"][k] for k in ("rho", "rhou", "rhov", "p")))

    def flux_avg(plane, key):
        return np.nansum(plane[key] * plane["rhou"]) / np.nansum(plane["rhou"])

    mp_p1 = flux_avg(planes["MP1"], "p")
    mp_p01 = flux_avg(planes["MP1"], "p0")
    mp_p02 = flux_avg(planes["MP2"], "p0")

    def angle(plane):
        return np.degrees(np.arctan(np.nanmean(plane["rhov"]) / np.nanmean(plane["rhou"])))

    return {
        "MixedoutLossCoef": (p01 - p02) / (p01 - p1),
        "MPLossCoef": (mp_p01 - mp_p02) / (mp_p01 - mp_p1),
        "OutflowAngle": angle(planes["MP2"]),
        "InflowAngle": angle(planes["MP1"]),
    }


def read_set(time_dir: str, set_name: str) -> pd.DataFrame:
    """
    **Returns** one OpenFOAM `sets` sample (csv format) as a DataFrame with columns
    rho, Ux, Uy, p (plus whatever else was sampled), merging the per-field-type files.
    """
    files = sorted(glob.glob(os.path.join(time_dir, f"{set_name}*.csv")))
    if not files:
        raise FileNotFoundError(f"no {set_name} samples in {time_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    for comp, name in ((0, "Ux"), (1, "Uy")):
        for col in (f"U_{comp}", f"U_{'xyz'[comp]}", f"U:{comp}"):
            if col in df.columns:
                df[name] = df[col]
                break
    return df


def qoi_history(sets_dir: str) -> pd.DataFrame:
    """
    **Returns** the QoIs at every sampled iteration found in sets_dir
    (e.g. <case>/postProcessing/MP), indexed by iteration.
    """
    rows = {}
    for tdir in glob.glob(os.path.join(sets_dir, "*")):
        try:
            it = int(float(os.path.basename(tdir)))
        except ValueError:
            continue
        rows[it] = plane_qois(read_set(tdir, "MP1"), read_set(tdir, "MP2"))
    if not rows:
        raise FileNotFoundError(f"no samples found in {sets_dir}")
    return pd.DataFrame.from_dict(rows, orient="index").sort_index().rename_axis("iteration")


def run_quality(history: pd.DataFrame, max_rel_std: float = 5., max_drift: float = 1.) -> dict:
    """
    **Returns** whether the time-averaged loss of one run is usable: it must be positive, its
    oscillation (std / mean) below max_rel_std % and the drift of its mean between the two halves
    of the sampling window below max_drift %. Keys: rel_std_%, drift_%, reason ("" if usable).
    """
    w = history["MixedoutLossCoef"]
    split = (w.index[0] + w.index[-1] + 1) // 2
    mean = w.mean()
    rel_std = 100 * w.std(ddof=0) / abs(mean)
    drift = 100 * abs(w[w.index >= split].mean() - w[w.index < split].mean()) / abs(mean)
    if not mean > 0:
        reason = "non-positive loss"
    elif rel_std >= max_rel_std:
        reason = f"loss oscillation {rel_std:.1f}% >= {max_rel_std}%"
    elif drift >= max_drift:
        reason = f"loss drift {drift:.2f}% >= {max_drift}%"
    else:
        reason = ""
    return {"rel_std_%": rel_std, "drift_%": drift, "reason": reason}


def summarize(history: pd.DataFrame) -> dict[str, float]:
    """
    **Returns** mean, standard deviation and variance of each QoI over the sampled iterations.
    """
    out = {}
    for q in QOI_NAMES:
        out[q] = history[q].mean()
        out[f"{q}_std"] = history[q].std(ddof=0)
        out[f"{q}_var"] = history[q].var(ddof=0)
    out["n_samples"] = len(history)
    return out
