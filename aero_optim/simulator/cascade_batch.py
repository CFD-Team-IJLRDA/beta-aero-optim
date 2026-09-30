"""Evaluates a batch of cascade blades: profile file -> gmsh template mesh -> OpenFOAM at every operating point."""
import logging
import os
import time

import numpy as np

from aero_optim.geom import self_intersects
from aero_optim.mesh.foam_cascade_mesh import CascadeTemplateMesh
from aero_optim.simulator.openfoam import OpenFOAMSimulator

logger = logging.getLogger(__name__)


def run_blades(
        config: dict, sim: OpenFOAMSimulator, blades: list[tuple[str, int, int, np.ndarray]],
        outdir: str, budget: int
) -> dict[tuple[int, int], str]:
    """
    **Meshes and simulates** blades (name, gid, cid, profile) with at most `budget` solver
    processes at once, **waits** for all of them and **returns** each blade's status:
    "ok" (simulated; see sim.df_dict[gid][cid] for per-OP results), "invalid_geometry"
    (self-intersecting profile) or "mesh_failed". Finished blades are reloaded, not rerun.

    Profiles go to <outdir>/profiles/blade_<name>.dat and meshes to <outdir>/MESH.
    """
    profile_dir, mesh_dir = os.path.join(outdir, "profiles"), os.path.join(outdir, "MESH")
    os.makedirs(profile_dir, exist_ok=True)
    n_ops = len(sim.ops)
    status = {}
    try:
        for name, gid, cid, profile in blades:
            if self_intersects(profile):
                status[(gid, cid)] = "invalid_geometry"
                logger.warning(f"blade {name}: self-intersecting profile, skipped")
                continue
            dat = os.path.join(profile_dir, f"blade_{name}.dat")
            np.savetxt(dat, profile, header=f"blade {name}\nx y [m]")
            mesher = CascadeTemplateMesh(config, dat)
            try:
                meshfile = mesher.get_meshfile(mesh_dir)
                if not os.path.isfile(meshfile):
                    meshfile = mesher.write_mesh(mesh_dir)
            except Exception as e:
                status[(gid, cid)] = "mesh_failed"
                logger.error(f"blade {name}: meshing failed: {e}")
                continue
            while sim.monitor_sim_progress() + n_ops > budget:
                time.sleep(2)
            sim.execute_sim(meshfile, gid, cid)
            status[(gid, cid)] = "ok"
        while sim.monitor_sim_progress() > 0:
            time.sleep(5)
    except KeyboardInterrupt:
        sim.kill_all()
        raise
    return status
