import logging
import os
import re
import subprocess

logger = logging.getLogger(__name__)

# LRN-OGV cascade mesh recipe (authored by Mattia, used by the validated OpenFOAM reference cases).
DEFAULT_CASCADE_TEMPLATE = os.path.join(os.path.dirname(__file__), "templates", "cascade_mattia.geo")

_POINT_RE = re.compile(r'^(Point\()(\d+)(\) = \{)([^,]+),([^,]+),([^,]+)(,[^}]+\};)$')


def write_cascade_geo(template_path: str, points: list[tuple[float, float]], out_path: str) -> None:
    """
    **Writes** a new gmsh `.geo` file to out_path by substituting the coordinates of the
    first len(points) `Point(N) = {x, y, z, size};` entries of template_path with points,
    leaving everything else (domain, boundary-layer/background fields, transfinite
    curves, extrude, physical groups) untouched.

    - template_path (str): path to a reference cascade `.geo` file (e.g. `cascade_mattia.geo`)
      whose first len(points) points define the blade profile, in the exact order and
      winding the file's Spline/BSpline definitions expect.
    - points (list[tuple[float, float]]): the (x, y) blade profile coordinates to
      substitute in, same length, order and winding as the template's own profile.
    """
    n = len(points)
    with open(template_path, "r") as f:
        lines = f.readlines()

    n_replaced = 0
    new_lines = []
    for line in lines:
        m = _POINT_RE.match(line.rstrip("\n"))
        if m and 1 <= int(m.group(2)) <= n:
            x, y = points[int(m.group(2)) - 1]
            new_lines.append(f"{m.group(1)}{m.group(2)}{m.group(3)}"
                              f"{x:.12e},{y:.12e},{m.group(6)}{m.group(7)}\n")
            n_replaced += 1
        else:
            new_lines.append(line)

    if n_replaced != n:
        raise RuntimeError(
            f"expected to replace {n} Point(..) entries in {template_path}, replaced {n_replaced}"
        )

    with open(out_path, "w") as f:
        f.writelines(new_lines)


def build_mesh(geo_path: str, msh_path: str) -> None:
    """
    **Runs** gmsh on geo_path (CLI, matching the proven `run_mesh.sh` recipe) and writes
    the resulting MSH2 ASCII mesh to msh_path.
    """
    logger.info(f"gmsh {geo_path} -> {msh_path}")
    subprocess.run(
        ["gmsh", os.path.abspath(geo_path), "-3", "-format", "msh2", "-o", os.path.abspath(msh_path)],
        check=True, capture_output=True, text=True,
    )


def cascade_mattia_patch_types(pitch: float = 0.04039) -> dict[str, dict[str, str]]:
    """
    **Returns** the OpenFOAM boundary patch-type overrides matching `cascade_mattia.geo`'s
    own physical-surface naming (`front`/`back`/`top`/`bottom`/`airfoil`), as prototyped by
    hand in `run_mesh.sh`: `front`/`back` (the z-extrusion faces) become `empty`, `top`/
    `bottom` (the pitchwise periodic faces) become `cyclicAMI`, `airfoil` becomes `wall`.
    `inlet`/`outlet` are left as gmshToFoam's default `patch` type.
    """
    return {
        "front": {"type": "empty"},
        "back": {"type": "empty"},
        "top": {
            "type": "cyclicAMI",
            "neighbourPatch": "bottom",
            "transform": "translational",
            "separationVector": f"(0 {pitch:.10g} 0)",
        },
        "bottom": {
            "type": "cyclicAMI",
            "neighbourPatch": "top",
            "transform": "translational",
            "separationVector": f"(0 {-pitch:.10g} 0)",
        },
        "airfoil": {"type": "wall"},
    }
