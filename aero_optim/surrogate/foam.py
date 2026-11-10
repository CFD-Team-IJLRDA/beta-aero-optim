"""Minimal readers for the 2D (one-cell-thick) OpenFOAM cascade cases: cell centroids, boundary geometry, fields."""
import os
import re

import numpy as np

_LIST = re.compile(r"\n(\d+)\s*\n\(")

GAMMA = 1005.0 / (1005.0 - 8314.46 / 28.9)   # Cp / Cv of the case thermophysicalProperties (perfect gas)


def _read_list(path):
    """**Returns** the first top-level list of an ascii OpenFOAM file as a string body and its length."""
    txt = open(path).read()
    m = _LIST.search(txt)
    return txt, int(m.group(1)), m.end()


def read_scalar_list(path):
    txt, n, start = _read_list(path)
    return np.array(txt[start:].split(")", 1)[0].split()[:n], dtype=float)


def read_vector_list(path, ncomp=3):
    txt, n, start = _read_list(path)
    body = txt[start:]
    vals = re.findall(r"\(([^()]+)\)", body[: body.find("\n)\n") + 1] if "\n)\n" in body else body)
    return np.array([v.split()[:ncomp] for v in vals[:n]], dtype=float)


def read_faces(path):
    txt, n, start = _read_list(path)
    return [list(map(int, f.split())) for f in re.findall(r"\d+\(([\d ]+)\)", txt[start:])[:n]]


def read_boundary(path):
    txt = open(path).read()
    patches = {}
    for name, body in re.findall(r"\n\s*(\w+)\s*\n\s*\{([^}]*)\}", txt):
        sf = re.search(r"startFace\s+(\d+)", body)
        nf = re.search(r"nFaces\s+(\d+)", body)
        if sf and nf:
            patches[name] = (int(sf.group(1)), int(nf.group(1)))
    return patches


class CascadeMesh:
    """
    2D view of a one-cell-thick cascade polyMesh:
    - centroids (n_cells, 2): cell centroids (= centroids of the front-face polygons)
    - wall (n, 2): blade wall vertices on the front plane, ordered along the closed contour
    - bottom (n, 2): lower periodic boundary vertices on the front plane, sorted by x
    - x_in, x_out: inlet / outlet abscissae
    """
    def __init__(self, polymesh_dir):
        pts = read_vector_list(os.path.join(polymesh_dir, "points"))
        faces = read_faces(os.path.join(polymesh_dir, "faces"))
        owner = read_scalar_list(os.path.join(polymesh_dir, "owner")).astype(int)
        patches = read_boundary(os.path.join(polymesh_dir, "boundary"))
        z0 = pts[:, 2].min()
        on_front = np.isclose(pts[:, 2], z0)

        start, nf = patches["front"]
        n_cells = owner.max() + 1
        self.centroids = np.zeros((n_cells, 2))
        for k in range(start, start + nf):
            poly = pts[faces[k], :2]
            x, y = poly[:, 0], poly[:, 1]
            xs, ys = np.roll(x, -1), np.roll(y, -1)
            cross = x * ys - xs * y
            a = 0.5 * cross.sum()
            self.centroids[owner[k]] = [((x + xs) * cross).sum() / (6 * a), ((y + ys) * cross).sum() / (6 * a)]

        self.wall = self._ordered_loop(pts, faces, patches["airfoil"], on_front)
        # lower / upper periodic lines chosen by position: in the gmsh template the patch called "bottom"
        # is geometrically the upper line
        a = self._front_vertices(pts, faces, patches["bottom"], on_front)
        b = self._front_vertices(pts, faces, patches["top"], on_front)
        self.bottom, self.top = (a, b) if a[:, 1].mean() < b[:, 1].mean() else (b, a)
        self.x_in = self._front_vertices(pts, faces, patches["inlet"], on_front)[:, 0].mean()
        self.x_out = self._front_vertices(pts, faces, patches["outlet"], on_front)[:, 0].mean()

    @staticmethod
    def _front_vertices(pts, faces, patch, on_front):
        start, nf = patch
        idx = sorted({i for k in range(start, start + nf) for i in faces[k] if on_front[i]})
        v = pts[idx, :2]
        return v[np.argsort(v[:, 0])]

    @staticmethod
    def _ordered_loop(pts, faces, patch, on_front):
        """Chains the wall edges (front-plane vertex pairs of the wall faces) into one closed loop."""
        start, nf = patch
        edges = [tuple(i for i in faces[k] if on_front[i]) for k in range(start, start + nf)]
        nbrs = {}
        for a, b in edges:
            nbrs.setdefault(a, []).append(b)
            nbrs.setdefault(b, []).append(a)
        first = edges[0][0]
        loop, prev, cur = [first], None, first
        while True:
            nxt = [v for v in nbrs[cur] if v != prev][0]
            if nxt == first:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        assert len(loop) == len(edges), "wall is not a single closed loop"
        return pts[loop, :2]


def last_time(case):
    return str(max(int(d) for d in os.listdir(case) if d.isdigit()))


def read_fields(case, names=("p", "rho", "U", "nut", "Ma"), mean=False):
    """
    **Returns** {name: array} of cell-centred fields at the last iteration (U -> Ux, Uy).
    mean=True reads the fieldAverage means (pMean, rhoMean, UMean, nutMean) under the same names;
    'Ma' is then derived from the means, |U| / sqrt(gamma p / rho), since there is no MaMean.
    """
    t = last_time(case)
    out = {}
    for n in names:
        if mean and n == "Ma":
            continue
        path = os.path.join(case, t, n + "Mean" if mean else n)
        if n == "U":
            u = read_vector_list(path)
            out["Ux"], out["Uy"] = u[:, 0], u[:, 1]
        else:
            out[n] = read_scalar_list(path)
    if mean and "Ma" in names:
        out["Ma"] = np.hypot(out["Ux"], out["Uy"]) / np.sqrt(GAMMA * out["p"] / out["rho"])
    return out
