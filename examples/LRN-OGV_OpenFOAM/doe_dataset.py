"""Turns a finished DoE into a documented training dataset.

Usage: python doe_dataset.py -c openfoam_config.json [--outdir DIR] [--delete-rejected]

Reads <outdir>/doe_results.csv and every run's qoi_history.csv, classifies each run, samples
the MP1 Mach number on the last iteration and writes, under <outdir>/dataset/:
    dataset.csv   one row per kept blade (good at every operating point)
    rejected.csv  one row per rejected blade, with the reason
    dataset.json  dataset card: provenance, CFD setup, QoI definitions, quality criterion, columns
With --delete-rejected, the case, mesh and profile files of rejected blades are removed.
The OpenFOAM environment must be sourced (postProcess is called for the MP1 Mach number).
"""
import argparse
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from aero_optim.simulator.cascade_qoi import GAMMA, mixedout_state, run_quality  # noqa: E402

# a run is "good" if its time-averaged loss is positive, oscillates by less than MAX_REL_STD and
# its mean moves by less than MAX_DRIFT between the two halves of the averaging window
MAX_REL_STD, MAX_DRIFT = 5.0, 1.0  # %

MP1_DICT = """FoamFile { version 2.0; format ascii; class dictionary; object mp1Dict; }
functions
{
    MP1check
    {
        type sets; libs (sampling); setFormat csv; interpolationScheme cellPoint;
        fields (p U rho);
        sets { MP1 { type uniform; axis xyz; start (-0.02 -0.0285704 0.005);
                     end (-0.02 0.0118176 0.005); nPoints 1001; } }
    }
}
"""


def mp1_mach(case: str, time: int, dict_path: str) -> float:
    """**Returns** the mixed-out Mach number at MP1 on iteration `time` of an OpenFOAM case."""
    out = glob.glob(os.path.join(case, "postProcessing", "MP1check", str(time), "*.csv"))
    if not out:
        subprocess.run(["postProcess", "-case", case, "-time", str(time), "-dict", dict_path,
                        "-fields", "(p U rho)"], check=True, capture_output=True)
        out = glob.glob(os.path.join(case, "postProcessing", "MP1check", str(time), "*.csv"))
    d = pd.read_csv(out[0])
    p1, p01 = mixedout_state(d.rho.values, d.rho.values * d.U_0.values, d.rho.values * d.U_1.values, d.p.values)
    return float(np.sqrt(2 / (GAMMA - 1) * ((p01 / p1)**((GAMMA - 1) / GAMMA) - 1)))


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except OSError:
        return "unknown"


def columns_doc(ops: list[str], params: list[str], n_modes: int) -> dict:
    doc = {
        "blade": "blade id: 'baseline' or the 4-digit DoE sample number",
        "geometry": "'real_baseline' (the reference blade itself) or 'pod_reconstruction' "
                    "(profile = POD mean + sum_k c_k * mode_k)",
        "profile": "path of the 322-point profile (x y in m), relative to the DoE folder",
        "case": "path of the OpenFOAM case folder (mesh + one sub-case per operating point), relative to the DoE folder",
    }
    doc.update({f"c{k + 1}": f"coefficient of POD mode {k + 1} [m] (for the baseline: projection of the real blade)"
                for k in range(n_modes)})
    doc.update({p: f"BladeGen parameter {p} (Latin hypercube over the bounds in dataset.json)" for p in params})
    for op in ops:
        doc.update({
            f"{op}_w": f"{op}: mixed-out total-pressure loss coefficient (P01 - P02)/(P01 - p1), mixed-out states "
                       "at MP1 and MP2, time-averaged over the sampling window [-]",
            f"{op}_w_std": f"{op}: standard deviation of the loss over the sampling window (iterative oscillation "
                           "amplitude, not the error of the mean) [-]",
            f"{op}_w_mp": f"{op}: same loss with mass-flux-averaged total pressures (excludes downstream mixing) [-]",
            f"{op}_angle_out": f"{op}: outflow angle atan(mean rho v / mean rho u) at MP2 [deg]",
            f"{op}_angle_out_std": f"{op}: standard deviation of the outflow angle over the window [deg]",
            f"{op}_angle_in": f"{op}: inflow angle at MP1 [deg]",
            f"{op}_Ma1": f"{op}: mixed-out Mach number at MP1, last iteration [-]",
            f"{op}_rel_std_%": f"{op}: {op}_w_std / {op}_w [%]",
            f"{op}_drift_%": f"{op}: |mean loss second half - first half of the window| / mean [%]",
            f"{op}_window": f"{op}: iterations averaged over (later than the default for rescued runs, "
                            "see dataset.json)",
        })
    doc["L_ADP"] = "objective 1: ADP_w"
    doc["L_OP"] = "objective 2: (OP1_w + OP2_w) / 2"
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--outdir", help="DoE folder, if not the one in the config")
    parser.add_argument("--delete-rejected", action="store_true")
    parser.add_argument("--jobs", type=int, default=32)
    args = parser.parse_args()
    config_path = os.path.abspath(args.config)
    config = json.load(open(config_path))
    outdir, sim = args.outdir or config["study"]["outdir"], config["simulator"]
    ops = list(sim["operating_points"])
    dsdir = os.path.join(outdir, "dataset")
    os.makedirs(dsdir, exist_ok=True)
    dict_path = os.path.join(dsdir, "mp1Dict")
    with open(dict_path, "w") as f:
        f.write(MP1_DICT)

    results = pd.read_csv(os.path.join(outdir, "doe_results.csv"), dtype={"blade": str})
    samples = pd.read_csv(os.path.join(outdir, "doe_samples.csv"), index_col=0)
    params = [c for c in samples.columns if not (c[0] == "c" and c[1:].isdigit())]
    n_modes = sum(c[0] == "c" and c[1:].isdigit() for c in samples.columns)
    pod = np.load(os.path.join(outdir, "pod", "basis.npz"))

    def case_of(blade):
        return f"OPENFOAM/openfoam_g1_c0" if blade == "baseline" else f"OPENFOAM/openfoam_g0_c{int(blade)}"

    def assess(row):
        out = {"reason": row.status if row.status != "ok" else ""}
        if out["reason"]:
            return out
        if all(row[f"{op}_status"] != "ok" for op in ops):
            log = os.path.join(outdir, case_of(row.blade), "mesh", "log.checkMesh")
            if os.path.isfile(log) and "Failed" in open(log).read():
                out["reason"] = "meshing failed (invalid gmsh mesh, all runs crashed at start)"
                return out
        reasons = []
        for op in ops:
            case = os.path.join(outdir, case_of(row.blade), op)
            hist_file = os.path.join(case, "qoi_history.csv")
            if row[f"{op}_status"] != "ok" or not os.path.isfile(hist_file):
                reasons.append(f"{op}: run failed")
                continue
            q = run_quality(pd.read_csv(hist_file, index_col=0), MAX_REL_STD, MAX_DRIFT)
            out[f"{op}_rel_std_%"], out[f"{op}_drift_%"] = q["rel_std_%"], q["drift_%"]
            if q["reason"]:
                reasons.append(f"{op}: {q['reason']}")
            else:
                last = max(int(d) for d in os.listdir(case) if d.isdigit())
                out[f"{op}_Ma1"] = mp1_mach(case, last, dict_path)
        out["reason"] = "; ".join(reasons)
        return out

    with ThreadPoolExecutor(args.jobs) as pool:
        assessed = pd.DataFrame(list(pool.map(assess, (r for _, r in results.iterrows()))))
    results = pd.concat([results.drop(columns=[c for c in results.columns if c in assessed]), assessed], axis=1)
    kept, rejected = results[results.reason == ""].copy(), results[results.reason != ""].copy()

    kept["geometry"] = np.where(kept.blade == "baseline", "real_baseline", "pod_reconstruction")
    kept["profile"] = kept.blade.map(lambda b: f"profiles/blade_{b}.dat")
    kept["case"] = kept.blade.map(case_of)
    for p in params:
        kept[p] = kept.blade.map(lambda b: np.nan if b == "baseline" else samples.at[int(b), p])
    rename = {f"{op}_{a}": f"{op}_{b}" for op in ops for a, b in
              (("angle", "angle_out"), ("angle_std", "angle_out_std"), ("inflow_angle", "angle_in"))}
    kept = kept.rename(columns=rename)
    doc = columns_doc(ops, params, n_modes)
    kept = kept[[c for c in doc if c in kept.columns]]
    kept.to_csv(os.path.join(dsdir, "dataset.csv"), index=False)
    run_cols = [f"{op}_{q}" for op in ops for q in ("w", "rel_std_%", "drift_%")]
    rejected[["blade", "reason"] + [f"c{k + 1}" for k in range(n_modes)]
             + [c for c in run_cols if c in rejected]].to_csv(os.path.join(dsdir, "rejected.csv"), index=False)

    sampling = "bladegen" if params else "pod_box"
    card = {
        "name": os.path.basename(outdir.rstrip("/")),
        "description": "2D (pseudo-3D, one cell span) RANS evaluations of LRN-OGV compressor cascade blades at "
                       "three operating points, for surrogate / ML training and optimisation.",
        "created": datetime.date.today().isoformat(),
        "code_commit": git_commit(),
        "counts": {"blades_in_doe": len(results), "kept": len(kept), "rejected": len(rejected)},
        "geometry": {
            "parameterisation": f"{n_modes} POD modes of BladeGen blade shapes (displacements of the 322 baseline "
                                "points, arc-length resampled)",
            "pod_energy_captured_%": round(100 * float(np.sum(pod["energy"][:n_modes])), 3),
            "pod_training_set": "1000 BladeGen blades, Latin hypercube over the 13 bounds below",
            "bladegen_param_bounds": config["bladegen"]["param_bounds"],
            "sampling": {
                "pod_box": "Latin hypercube over the min/max box of the training POD coefficients; "
                           "part of the blades lie outside the BladeGen family",
                "bladegen": "the 1000 BladeGen training blades themselves, each meshed as its "
                            f"{n_modes}-mode reconstruction",
            }[sampling],
            "baseline": "the real LRN-OGV blade (baseline_322.dat), meshed as is",
            "files": "pod/basis.npz (base, mean, modes, bounds, energy), profiles/blade_<id>.dat",
        },
        "cfd": {
            "solver": f"OpenFOAM v2406 {sim['exec_cmd']}, steady, kOmegaSSTLM transition model",
            "mesh": "gmsh template cascade_mattia.geo (quad boundary layer), one pitch with cyclicAMI periodicity",
            "boundary_conditions": "inlet: fixed velocity vector and static temperature; outlet: fixed static "
                                   "pressure (per operating point, tuned once on the baseline)",
            "operating_points": sim["operating_points"],
            "iterations": sim["end_time"],
            "averaging_window": [sim["sample_start"], sim["end_time"]],
            "rescued_runs": "runs rejected after the default window were continued (doe_rescue.py) and "
                            "re-averaged over a later window of the same length, given in <OP>_window; "
                            "those still failing the criterion were rejected",
            "measurement_planes": {"MP1": "x = -0.020 m, one pitch", "MP2": "x = 0.087 m, one pitch"},
        },
        "quality_criterion": {
            "kept_if": f"at every operating point: time-averaged loss > 0, loss std / mean < {MAX_REL_STD}%, "
                       f"drift of the mean between the two halves of the window < {MAX_DRIFT}%",
            "rationale": "the average must be stationary and well defined; thresholds are engineering choices",
            "rejected_reasons": rejected.reason.str.replace(r"\d+(\.\d+)?%", "x%", regex=True)
                                .str.split("; ").explode().value_counts().to_dict(),
        },
        "columns": {c: doc[c] for c in kept.columns},
    }
    with open(os.path.join(dsdir, "dataset.json"), "w") as f:
        json.dump(card, f, indent=2)
    print(f"kept {len(kept)} / {len(results)} blades -> {dsdir}")
    print(json.dumps(card["quality_criterion"]["rejected_reasons"], indent=1))

    if args.delete_rejected:
        for b in rejected.blade:
            if b == "baseline":
                continue
            shutil.rmtree(os.path.join(outdir, case_of(b)), ignore_errors=True)
            for f in glob.glob(os.path.join(outdir, "MESH", f"blade_{b}.*")) + \
                    [os.path.join(outdir, "profiles", f"blade_{b}.dat")]:
                if os.path.isfile(f):
                    os.remove(f)
        print(f"deleted the files of {len(rejected)} rejected blades")


if __name__ == "__main__":
    main()
