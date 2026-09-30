import logging
import os
import re
import shutil
import subprocess

logger = logging.getLogger(__name__)


def _patch_boundary_file(boundary_file: str, patch_types: dict[str, dict[str, str]]) -> None:
    """
    **Rewrites** an OpenFOAM `constant/polyMesh/boundary` file, replacing the entries of
    each patch named in patch_types with the given key/value pairs, while preserving its
    existing `nFaces`/`startFace`.
    """
    with open(boundary_file, "r") as f:
        content = f.read()

    for patch_name, entries in patch_types.items():
        pattern = (r"([ \t]+)" + re.escape(patch_name) + r"[ \t]*\n"
                   r"([ \t]*\{)([^}]*)(\})")

        def repl(m: re.Match, entries: dict[str, str] = entries) -> str:
            indent, ob, body, cb = m.group(1), m.group(2), m.group(3), m.group(4)
            nf_m = re.search(r"nFaces\s+\d+;", body)
            sf_m = re.search(r"startFace\s+\d+;", body)
            if not nf_m or not sf_m:
                raise RuntimeError(f"could not find nFaces/startFace for patch '{patch_name}'")
            ei = " " * 8
            new_body = "\n" + "".join(f"{ei}{k:<15} {v};\n" for k, v in entries.items())
            new_body += f"{ei}{nf_m.group(0)}\n{ei}{sf_m.group(0)}\n{indent}"
            return f"{indent}{patch_name}\n{ob}{new_body}{cb}"

        new_content, n_sub = re.subn(pattern, repl, content, count=1, flags=re.DOTALL)
        if n_sub == 0:
            raise RuntimeError(f"patch '{patch_name}' not found in {boundary_file}")
        content = new_content

    with open(boundary_file, "w") as f:
        f.write(content)


def convert_to_foam(
        msh_file: str, case_dir: str, patch_types: dict[str, dict[str, str]], init_fields: bool = True
) -> None:
    """
    **Converts** a gmsh `.msh` (v2 ASCII) mesh into the OpenFOAM case at case_dir, patches
    the resulting boundary file per patch_types, validates it with checkMesh, and, if
    init_fields, (re)initializes `0/` from `0.org/` with cell-centre fields.

    - msh_file (str): path to the gmsh `.msh` (v2 ASCII) mesh, e.g. as written by
      `foam_cascade_mesh.build_mesh()`.
    - case_dir (str): path to an OpenFOAM case directory that already contains `system/`,
      `constant/` (without `constant/polyMesh`, which this function creates) and, if
      init_fields, `0.org/`.
    - patch_types (dict): patch name -> OpenFOAM boundary entry overrides, e.g. as returned
      by `foam_cascade_mesh.cascade_mattia_patch_types()`.
    """
    msh_file = os.path.abspath(msh_file)
    polymesh_dir = os.path.join(case_dir, "constant", "polyMesh")
    if os.path.isdir(polymesh_dir):
        shutil.rmtree(polymesh_dir)

    logger.info(f"gmshToFoam {msh_file} in {case_dir}")
    subprocess.run(
        ["gmshToFoam", msh_file], cwd=case_dir, check=True, capture_output=True, text=True
    )

    _patch_boundary_file(os.path.join(polymesh_dir, "boundary"), patch_types)

    logger.info(f"checkMesh in {case_dir}")
    result = subprocess.run(["checkMesh"], cwd=case_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"checkMesh failed in {case_dir}:\n{result.stdout}\n{result.stderr}")
    with open(os.path.join(case_dir, "log.checkMesh"), "w") as f:
        f.write(result.stdout)
    failed = re.search(r"Failed (\d+) mesh checks", result.stdout)
    if failed:
        # e.g. gmsh could not recover a surface edge and left unassigned faces (defaultFaces)
        raise RuntimeError(f"checkMesh: {failed.group(0)} in {case_dir}, see log.checkMesh")
    if not init_fields:
        return

    zero_dir = os.path.join(case_dir, "0")
    org_dir = os.path.join(case_dir, "0.org")
    if os.path.isdir(zero_dir):
        shutil.rmtree(zero_dir)
    shutil.copytree(org_dir, zero_dir)

    logger.info(f"initializing cell-centre fields in {case_dir}")
    subprocess.run(
        ["postProcess", "-func", "writeCellCentres", "-time", "0"],
        cwd=case_dir, check=True, capture_output=True, text=True
    )
