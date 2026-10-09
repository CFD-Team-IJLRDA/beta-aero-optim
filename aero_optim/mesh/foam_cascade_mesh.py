import logging
import os
import re
import subprocess

from aero_optim.utils import from_dat

logger = logging.getLogger(__name__)

# LRN-OGV cascade mesh recipe (authored by Mattia, used by the validated OpenFOAM reference cases).
DEFAULT_CASCADE_TEMPLATE = os.path.join(os.path.dirname(__file__), "templates", "cascade_mattia.geo")

_POINT_RE = re.compile(r'^(Point\()(\d+)(\) = \{)([^,]+),([^,]+),([^,]+)(,[^}]+\};)$')


def write_cascade_geo(
        template_path: str, points: list[tuple[float, float]], out_path: str, n_blade: int = 322
) -> None:
    """
    **Writes** a new gmsh `.geo` file to out_path by substituting the coordinates of the
    blade points `Point(1..n_blade)` of template_path with points, leaving everything else
    (domain, boundary-layer/background fields, transfinite curves, extrude, physical groups)
    untouched.

    - template_path (str): path to a reference cascade `.geo` file (e.g. `cascade_mattia.geo`).
    - points (list[tuple[float, float]]): the (x, y) blade profile coordinates, same order and
      winding as the template's own profile.
    - n_blade (int): number of blade points in the template (322 for `cascade_mattia.geo`).
    """
    if len(points) != n_blade:
        raise RuntimeError(f"expected {n_blade} blade points, got {len(points)}")
    n = n_blade
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


class CascadeTemplateMesh:
    """
    Blade mesher with the interface `Optimizer.mesh()` expects (`get_meshfile`, `build_mesh`,
    `write_mesh`): the blade profile file is substituted into the cascade `.geo` template and
    meshed with gmsh into an MSH2 file for OpenFOAM.

    Config: `config["mesh"]` may set `template` (a path, or the name of a file in mesh/templates/;
    default: cascade_mattia.geo), `header`
    (profile file header lines, default 2) and `scale` (profile scaling to metres, default 1).
    """
    def __init__(self, config: dict, datfile: str = ""):
        mesh_config = config.get("mesh", {})
        self.dat_file: str = datfile if datfile else config["study"]["file"]
        self.outdir: str = config["study"]["outdir"]
        self.outfile: str = os.path.splitext(os.path.basename(self.dat_file))[0]
        self.template: str = mesh_config.get("template", DEFAULT_CASCADE_TEMPLATE)
        if not os.path.isfile(self.template):     # a bare name refers to a template shipped in mesh/templates/
            self.template = os.path.join(os.path.dirname(DEFAULT_CASCADE_TEMPLATE), self.template)
        self.header: int = mesh_config.get("header", 2)
        self.scale: float = mesh_config.get("scale", 1.)

    def get_meshfile(self, mesh_dir: str) -> str:
        return os.path.join(mesh_dir, self.outfile + ".msh")

    def build_mesh(self):
        """Nothing to prepare: meshing happens in `write_mesh`."""

    def write_mesh(self, mesh_dir: str = "") -> str:
        mesh_dir = mesh_dir or self.outdir
        os.makedirs(mesh_dir, exist_ok=True)
        pts = from_dat(self.dat_file, self.header, self.scale)
        geo_file = os.path.join(mesh_dir, self.outfile + ".geo")
        write_cascade_geo(self.template, [(p[0], p[1]) for p in pts], geo_file)
        build_mesh(geo_file, self.get_meshfile(mesh_dir))
        return self.get_meshfile(mesh_dir)


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
