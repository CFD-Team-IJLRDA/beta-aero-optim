"""BladeGen blade generation and a POD shape basis built from BladeGen blades.

Each BladeGen blade is resampled on the baseline's points by relative arc length: both blades are
split at the leading and trailing edges, and a baseline point lying at a given fraction of its side
maps to the same fraction of the same side of the BladeGen blade. The POD is fitted on the
resulting (x, y) displacements, so any coefficient vector gives a blade with exactly the baseline's
point count, ordering and clustering, ready for the mesh template.
(The previous DLR_POD_2D kept x fixed and interpolated y, which folds the profile at the
near-vertical leading edge for ~12 % of the blades; normal offsets fail when the stagger rotates
the trailing edge by more than its own radius.)
"""
import logging
import os
import re
import subprocess
from multiprocessing import Pool

import numpy as np
from scipy.stats import qmc

logger = logging.getLogger(__name__)

TEC_FILE = os.path.join("Output", "Profile_mergedGestaf.tec")
TEC_ZONE = 'ZONE T="NURB_Profil_gestaf0"'


def write_progen(template_text: str, params: dict[str, float], path: str) -> None:
    """**Writes** a single-section progen.input from template_text with params substituted."""
    text = template_text
    for key, value in params.items():
        text, n = re.subn(rf"(?m)^({re.escape(key)}\s+)\S+", rf"\g<1>{value:.10g}", text)
        if n != 1:
            raise KeyError(f"parameter {key} found {n} times in the BladeGen template")
    with open(path, "w") as f:
        f.write(text)


def read_tec_profile(path: str) -> np.ndarray:
    """**Returns** the open blade profile (N, 2) of BladeGen's staggered output file."""
    pts, in_zone = [], False
    for line in open(path):
        if TEC_ZONE in line:
            in_zone = True
            continue
        if in_zone and line.strip().startswith("ZONE"):
            break
        if in_zone:
            try:
                pts.append([float(v) for v in line.split()[:2]])
            except (ValueError, IndexError):
                continue
    pts = np.array(pts)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


def run_bladegen(exe: str, template_text: str, params: dict[str, float], folder: str) -> np.ndarray:
    """
    **Runs** BladeGen in folder and **returns** the blade profile (metres).
    BladeGen exits with code 100 even on success, so success is judged by its output file.
    """
    os.makedirs(folder, exist_ok=True)
    write_progen(template_text, params, os.path.join(folder, "progen.input"))
    subprocess.run([os.path.abspath(exe)], cwd=folder, capture_output=True)
    tec = os.path.join(folder, TEC_FILE)
    if not os.path.isfile(tec):
        raise RuntimeError(f"BladeGen produced no {TEC_FILE} in {folder}")
    return read_tec_profile(tec)


def signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def edge_indices(points: np.ndarray) -> tuple[int, int]:
    """**Returns** (LE, TE) indices: TE = largest x, LE = point farthest from the TE."""
    te = int(np.argmax(points[:, 0]))
    le = int(np.argmax(np.linalg.norm(points - points[te], axis=1)))
    return le, te


def _sides(points: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    **Returns** for the two sides (LE -> TE, then TE -> LE, following the point order) the
    point indices and their cumulative arc length normalised to [0, 1] on that side.
    """
    n = len(points)
    le, te = edge_indices(points)
    first = [(le + k) % n for k in range((te - le) % n + 1)]
    second = [(te + k) % n for k in range((le - te) % n + 1)]
    fractions = []
    for idx in (first, second):
        seg = np.linalg.norm(np.diff(points[idx], axis=0), axis=1)
        s = np.concatenate(([0.], np.cumsum(seg)))
        fractions.append(s / s[-1])
    return [np.array(first), np.array(second)], fractions


def arclength_resample(profile: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    **Returns** profile (open, any point count) resampled onto base's points (N, 2): each base
    point takes the position at the same relative arc length on the same side of profile.
    """
    if np.sign(signed_area(profile)) != np.sign(signed_area(base)):
        profile = profile[::-1]
    base_idx, base_frac = _sides(base)
    prof_idx, prof_frac = _sides(profile)
    out = np.full(base.shape, np.nan)
    for side in (0, 1):
        pts = profile[prof_idx[side]]
        for k in (0, 1):
            out[base_idx[side], k] = np.interp(base_frac[side], prof_frac[side], pts[:, k])
    return out


def _dataset_worker(args):
    i, exe, template_text, params, folder, base = args
    try:
        profile = run_bladegen(exe, template_text, params, folder)
        return i, (arclength_resample(profile, base) - base).ravel(), None
    except Exception as e:
        return i, None, str(e)


class BladeGenPOD:
    """
    POD basis of blade shapes as displacements of the baseline points:
    profile = base + reshape(mean + modes @ coeffs, (N, 2)).

    Modes are orthonormal (coefficients in metres), ordered by decreasing energy;
    `bounds` are the per-mode min/max coefficients of the training blades.
    """
    def __init__(self, base: np.ndarray, mean: np.ndarray, modes: np.ndarray, bounds: np.ndarray,
                 energy: np.ndarray):
        self.base, self.mean, self.modes, self.bounds, self.energy = base, mean, modes, bounds, energy

    @property
    def n_modes(self) -> int:
        return self.modes.shape[1]

    def reconstruct(self, coeffs: np.ndarray) -> np.ndarray:
        """**Returns** the open profile (N, 2) of a coefficient vector."""
        return self.base + (self.mean + self.modes @ np.asarray(coeffs)).reshape(self.base.shape)

    def project(self, displacement: np.ndarray) -> np.ndarray:
        """**Returns** the coefficients of a blade given by its displacement (N, 2) from base."""
        return self.modes.T @ (np.ravel(displacement) - self.mean)

    def save(self, path: str):
        np.savez(path, base=self.base, mean=self.mean, modes=self.modes, bounds=self.bounds,
                 energy=self.energy)

    @classmethod
    def load(cls, path: str) -> "BladeGenPOD":
        d = np.load(path)
        return cls(d["base"], d["mean"], d["modes"], d["bounds"], d["energy"])

    @classmethod
    def fit(cls, base: np.ndarray, D: np.ndarray, n_modes: int) -> "BladeGenPOD":
        """**Fits** the basis on the flattened displacements D (n_blades, 2N) of the training blades."""
        mean = D.mean(axis=0)
        U, s, _ = np.linalg.svd((D - mean).T, full_matrices=False)
        energy = s**2 / np.sum(s**2)
        modes = U[:, :n_modes]
        coeffs = (D - mean) @ modes
        bounds = np.column_stack((coeffs.min(axis=0), coeffs.max(axis=0)))
        return cls(base, mean, modes, bounds, energy)


def build_dataset(
        exe: str, template_path: str, param_bounds: dict[str, list[float]], baseline: np.ndarray,
        workdir: str, n_samples: int, seed: int, n_jobs: int
) -> dict[str, np.ndarray]:
    """
    **Generates** n_samples BladeGen blades from a Latin hypercube over param_bounds, resamples
    them on the baseline points and returns params, flattened displacements D (n, 2N) and
    failure flags. workdir must be new or empty, so results from other bounds or seeds are
    never reused.
    """
    if os.path.isdir(workdir) and os.listdir(workdir):
        raise FileExistsError(f"{workdir} is not empty: use a fresh folder for a new dataset")
    os.makedirs(workdir, exist_ok=True)
    keys = list(param_bounds)
    lo, hi = np.array([param_bounds[k] for k in keys]).T
    params = qmc.scale(qmc.LatinHypercube(d=len(keys), seed=seed).random(n_samples), lo, hi)
    template_text = open(template_path).read()
    jobs = [(i, exe, template_text, dict(zip(keys, params[i])), os.path.join(workdir, f"blade_{i:04d}"),
             baseline) for i in range(n_samples)]
    D = np.full((n_samples, baseline.size), np.nan)
    errors = [""] * n_samples
    with Pool(n_jobs) as pool:
        for i, d, err in pool.imap_unordered(_dataset_worker, jobs):
            if err is None:
                D[i] = d
            else:
                errors[i] = err
    failed = np.array([bool(e) for e in errors])
    if failed.any():
        logger.warning(f"{failed.sum()} / {n_samples} BladeGen blades failed, e.g.: "
                       f"{next(e for e in errors if e)}")
    return dict(keys=np.array(keys), params=params, base=baseline, D=D, failed=failed,
                errors=np.array(errors))
