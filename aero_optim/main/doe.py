"""Design of experiments: BladeGen + POD blades evaluated with OpenFOAM at every operating point.

Usage: doe -c openfoam_config.json [--pilot N]

Everything is written under config["study"]["outdir"] and the run can be resumed: finished
blades are reloaded from their qois.csv, existing meshes and the POD basis are reused.
"""
import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc

from aero_optim.mesh.foam_cascade_mesh import CascadeTemplateMesh
from aero_optim.shape.bladegen_pod import BladeGenPOD, build_dataset
from aero_optim.simulator.openfoam import OpenFOAMSimulator
from aero_optim.utils import from_dat

logger = logging.getLogger("doe")
DOE_GID, BASELINE_GID = 0, 1


def self_intersects(profile: np.ndarray) -> bool:
    """**Returns** True if the closed polygon through profile's points crosses itself."""
    p = profile
    q = np.roll(profile, -1, axis=0)
    n = len(p)

    def orient(a, b, c):
        return np.sign((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
                       - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))

    i, j = np.triu_indices(n, k=2)
    keep = ~((i == 0) & (j == n - 1))  # first and last segments share a point
    i, j = i[keep], j[keep]
    d1 = orient(p[i], q[i], p[j])
    d2 = orient(p[i], q[i], q[j])
    d3 = orient(p[j], q[j], p[i])
    d4 = orient(p[j], q[j], q[i])
    return bool(np.any((d1 * d2 < 0) & (d3 * d4 < 0)))


def load_or_build_pod(config: dict, outdir: str, baseline: np.ndarray) -> BladeGenPOD:
    pod_config, bg = config["pod"], config["bladegen"]
    pod_dir = os.path.join(outdir, "pod")
    basis_file = os.path.join(pod_dir, "basis.npz")
    if os.path.isfile(basis_file):
        logger.info(f"POD basis loaded from {basis_file}")
        return BladeGenPOD.load(basis_file)
    t0 = time.time()
    ds = build_dataset(
        bg["exe"], bg["template"], bg["param_bounds"], baseline, os.path.join(pod_dir, "bladegen"),
        pod_config.get("n_samples", 1000), pod_config.get("seed", 123),
        pod_config.get("n_jobs", max(1, (os.cpu_count() or 2) // 2)),
    )
    np.savez(os.path.join(pod_dir, "dataset.npz"), **ds)
    ok = ~ds["failed"] & np.array([np.all(np.isfinite(d)) and not self_intersects(baseline + d.reshape(-1, 2))
                                   for d in ds["D"]])
    pod = BladeGenPOD.fit(baseline, ds["D"][ok], pod_config.get("n_modes", 5))
    pod.save(basis_file)
    base_err = np.linalg.norm(pod.reconstruct(pod.project(np.zeros_like(baseline))) - baseline, axis=1)
    cum = np.cumsum(pod.energy)
    report = [
        f"BladeGen blades: {len(ds['D'])}, used for POD: {ok.sum()} "
        f"({ds['failed'].sum()} BladeGen failures, {(~ok).sum() - ds['failed'].sum()} self-intersecting)",
        "energy per mode (%): " + " ".join(f"{100 * e:.3f}" for e in pod.energy[:10]),
        "cumulative (%):      " + " ".join(f"{100 * c:.3f}" for c in cum[:10]),
        f"{pod.n_modes} modes capture {100 * cum[pod.n_modes - 1]:.3f} %",
        f"real baseline reconstructed with {pod.n_modes} modes: max error "
        f"{1e6 * np.abs(base_err).max():.0f} um, mean {1e6 * np.abs(base_err).mean():.0f} um",
        "coefficient bounds: " + "; ".join(f"[{lo:.5f}, {hi:.5f}]" for lo, hi in pod.bounds),
        f"built in {time.time() - t0:.0f} s",
    ]
    with open(os.path.join(pod_dir, "pod_report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    logger.info("POD basis built:\n" + "\n".join(report))
    return pod


def doe_samples(config: dict, outdir: str, pod: BladeGenPOD) -> pd.DataFrame:
    path = os.path.join(outdir, "doe_samples.csv")
    if os.path.isfile(path):
        return pd.read_csv(path, index_col=0)
    doe = config["doe"]
    unit = qmc.LatinHypercube(d=pod.n_modes, seed=doe.get("seed", 1)).random(doe.get("n_samples", 1000))
    coeffs = qmc.scale(unit, pod.bounds[:, 0], pod.bounds[:, 1])
    df = pd.DataFrame(coeffs, columns=[f"c{k + 1}" for k in range(pod.n_modes)]).rename_axis("sample")
    df.to_csv(path)
    return df


def results_row(df_dict_entry: dict[str, pd.DataFrame]) -> dict:
    row = {}
    for op, df in df_dict_entry.items():
        r = df.iloc[0]
        row[f"{op}_status"] = r["status"]
        for q, short in (("MixedoutLossCoef", "w"), ("OutflowAngle", "angle"),
                         ("MPLossCoef", "w_mp"), ("InflowAngle", "inflow_angle")):
            row[f"{op}_{short}"] = r.get(q, np.nan)
            row[f"{op}_{short}_std"] = r.get(f"{q}_std", np.nan)
    if {"ADP_w", "OP1_w", "OP2_w"} <= row.keys():
        row["L_ADP"] = row["ADP_w"]
        row["L_OP"] = 0.5 * (row["OP1_w"] + row["OP2_w"])
    return row


def run(config: dict, pilot: int | None = None):
    outdir = config["study"]["outdir"]
    os.makedirs(outdir, exist_ok=True)
    baseline = np.array(from_dat(config["study"]["file"], 2, 1))[:, :2]
    pod = load_or_build_pod(config, outdir, baseline)
    samples = doe_samples(config, outdir, pod)
    if pilot:
        samples = samples.iloc[:pilot]
    budget = config["doe"].get("budget", 96)

    sim = OpenFOAMSimulator(config)
    n_ops = len(sim.ops)
    profile_dir, mesh_dir = os.path.join(outdir, "profiles"), os.path.join(outdir, "MESH")
    os.makedirs(profile_dir, exist_ok=True)

    blades = [("baseline", BASELINE_GID, 0, baseline, pod.project(np.zeros_like(baseline)))]
    blades += [(f"{i:04d}", DOE_GID, int(i), pod.reconstruct(c), c.to_numpy()) for i, c in samples.iterrows()]
    status = {}
    try:
        for name, gid, cid, profile, _ in blades:
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
        while sim.monitor_sim_progress() > 0:
            time.sleep(5)
    except KeyboardInterrupt:
        sim.kill_all()
        raise

    rows = []
    for name, gid, cid, _, coeffs in blades:
        row = {"blade": name, "status": status.get((gid, cid), "ok")}
        row.update({f"c{k + 1}": v for k, v in enumerate(coeffs)})
        if row["status"] == "ok":
            row.update(results_row(sim.df_dict[gid][cid]))
        rows.append(row)
    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(outdir, "doe_results.csv"), index=False)
    logger.info(f"{(results.status == 'ok').sum()} / {len(results)} blades simulated, "
                f"results in {os.path.join(outdir, 'doe_results.csv')}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True, help="path to the JSON config")
    parser.add_argument("--pilot", type=int, default=None, help="only run the first N DoE samples")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path = os.path.abspath(args.config)
    config = json.load(open(config_path))
    os.chdir(os.path.dirname(config_path))
    run(config, args.pilot)


if __name__ == "__main__":
    main()
