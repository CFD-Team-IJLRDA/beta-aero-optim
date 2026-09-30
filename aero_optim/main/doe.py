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

from aero_optim.geom import self_intersects
from aero_optim.shape.bladegen_pod import BladeGenPOD, build_dataset
from aero_optim.simulator.cascade_batch import run_blades
from aero_optim.simulator.openfoam import OpenFOAMSimulator
from aero_optim.utils import from_dat

logger = logging.getLogger("doe")
DOE_GID, BASELINE_GID = 0, 1


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
    """**Returns** the DoE samples: POD coefficients c1..cN, plus the BladeGen parameters of each
    blade when config["doe"]["sampling"] is "bladegen".

    - "pod_box" (default): Latin hypercube over the box of the training coefficients. Its corners
      lie outside the BladeGen cloud, so many blades there are not BladeGen blades.
    - "bladegen": the POD training blades themselves (a Latin hypercube over the BladeGen
      parameter bounds), each meshed as its N-mode reconstruction.
    """
    path = os.path.join(outdir, "doe_samples.csv")
    if os.path.isfile(path):
        return pd.read_csv(path, index_col=0)
    doe = config["doe"]
    columns = [f"c{k + 1}" for k in range(pod.n_modes)]
    if doe.get("sampling", "pod_box") == "bladegen":
        ds = np.load(os.path.join(outdir, "pod", "dataset.npz"))
        ok = ~ds["failed"] & np.all(np.isfinite(ds["D"]), axis=1)
        coeffs = np.array([pod.project(d) for d in ds["D"][ok]])
        df = pd.DataFrame(ds["params"][ok], columns=ds["keys"], index=np.flatnonzero(ok))
        df[columns] = coeffs
        df = df.iloc[:doe.get("n_samples", len(df))]
    else:
        unit = qmc.LatinHypercube(d=pod.n_modes, seed=doe.get("seed", 1)).random(doe.get("n_samples", 1000))
        df = pd.DataFrame(qmc.scale(unit, pod.bounds[:, 0], pod.bounds[:, 1]), columns=columns)
    df = df.rename_axis("sample")
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
    coeff_cols = [f"c{k + 1}" for k in range(pod.n_modes)]
    blades = [("baseline", BASELINE_GID, 0, baseline, dict(zip(coeff_cols, pod.project(np.zeros_like(baseline)))))]
    blades += [(f"{i:04d}", DOE_GID, int(i), pod.reconstruct(s[coeff_cols].to_numpy()), s.to_dict())
               for i, s in samples.iterrows()]
    status = run_blades(config, sim, [b[:4] for b in blades], outdir, budget)

    rows = []
    for name, gid, cid, _, inputs in blades:
        row = {"blade": name, "status": status[(gid, cid)]}
        row.update(inputs)
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
