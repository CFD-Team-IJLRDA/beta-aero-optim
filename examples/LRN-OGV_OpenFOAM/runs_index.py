"""
Writes /mnt/Data_B/beta-aero-optim-runs/INDEX.json: the run folders grouped by theme, each with a one-line description, its
documents (.md) and its figures (png/pdf/gif, grouped by sub-folder). Open it in an editor that folds JSON to collapse groups.
Case trees (OPENFOAM, MESH, profiles, postProcessing), tensorboard logs and per-training-run plots are skipped (counted only).

Descriptions and themes live in THEMES below: add a folder there when it is created. Folders not listed land in "unsorted".
Usage: python3 runs_index.py [root]
"""
import json
import os
import sys
import time
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/Data_B/beta-aero-optim-runs"
SKIP = {"OPENFOAM", "MESH", "profiles", "postProcessing", "tb", "__pycache__", "blades", "mp_aug", "trials"}
FIG = (".png", ".pdf", ".gif", ".jpg", ".svg")
THEMES = {
    "1. RANS solver (OpenFOAM): validation, mesh, URANS": {
        "2026-09_openfoam_validation": "29 Sep. First OpenFOAM runs of the baseline (3 OPs) and a deformed blade, before the pipeline.",
        "2026-09_step3_baseline": "29 Sep. Baseline through the pipeline at 3 OPs; RANS vs WOLF vs LES table.",
        "mesh_study": "7–12 Oct. Mesh study (coarse to extra-fine, wall refinements, wake block); justification of the wall×2 standard.",
        "urans_tests": "6–12 Oct. URANS (rhoPimpleFoam) tests; v4 from converged RANS is stable.",
        "les_blade_rans": "6 Oct. Steady RANS of the blade of the colleague's LES, for a like-for-like RANS/LES comparison.",
        "notes": "Reference notes: OpenFOAM setup (solver, schemes, BCs, loss definitions), Jean Zay cheat sheet, constrained optimisation (KKT, SLSQP).",
    },
    "2. DoE and parametrisation (BladeGen + POD)": {
        "doe1000_bladegen": "30 Sep–1 Oct. 1000 BladeGen blades on the medium mesh (880 kept); holds the POD basis used everywhere.",
        "doe1000_wall2": "9–12 Oct. Same 1000 blades on the wall×2 mesh standard (976 kept).",
        "slides_doe": "2 Oct. Slide material of the DoE.",
    },
    "3. Brute-force optimisation (NSGA-II, RANS)": {
        "optim_nsga2": "6–7 Oct. NSGA-II 32 × 30 on the medium mesh; 20 Pareto blades; reference for the surrogate study.",
        "optim_nsga2_s2": "5 Nov. Same brute-force run with optimiser seed 2 (corrsin, 90 runs at once), to measure the brute force's own spread.",
        "optim_nsga2_s3": "5 Nov. Same with seed 3 (murphy); stopped early to make room for the random-start run.",
        "optim_nsga2_random": "5–6 Nov. Same NSGA-II started from 32 random DoE blades instead of the best 32: the fair reference for the surrogate study (864 CFD blades, same front). Started on murphy, finished on corrsin.",
        "optim_nsga2_lhs_s2": "10 Nov. NSGA-II started from 32 new Latin-hypercube BladeGen blades (init \"lhs\"), seed 2; spread of the random-start reference. launch_lhs_seeds.sh starts both LHS seeds.",
        "optim_nsga2_lhs_s3": "10 Nov. Same with seed 3.",
        "slides_optim": "7 Oct. Slide material: Pareto front, blades A/B/C, meshes, Mach fields, GIF.",
    },
    "4. Surrogate (INR) and surrogate-assisted optimisation": {
        "surrogate_ogv": "Since 7 Oct. INR surrogate (data, code, trainings, HPO, data-efficiency study). Start with README.md, RESULTS.md, reports/.",
        "cfd_sopt": "3 Nov. CFD cases of the verification of surrogate-optimised blades (one folder per front; results are in surrogate_ogv/sopt/).",
    },
    "5. LES (MUSICAA)": {
        "musicaa_sandbox": "Since 7 Oct. MUSICAA LES on corrsin: baseline ADP LES, handover document.",
        "MUSICAA_results_Laura": "Colleague's LES of an optimised blade (statistics only) + post-processing.",
        "les_instantaneous": "13 Oct. Instantaneous LES fields (several colormaps), GIFs.",
    },
    "6. Presentations, reports and cost": {
        "report_2026_11": "10 Nov. LaTeX summary report of the whole work (report.tex/.pdf, 10 pages); make_figures.py draws its figures.",
        "mateiral_figure_block_4_poster_oct2026": "13 Oct. Poster figures: T_geom / T_bal method in 4 panels, colormap variants.",
        "timings": "Timings of CFD, meshing and surrogate for the speed-up figures (timings.md).",
    },
}
CONVENTIONS = [
    "Diary: DIARY.md (what was done when). This file: where things are.",
    "CFD case layout: MESH/ (gmsh per blade), OPENFOAM/openfoam_g<gid>_c<cid>/<OP>/ (OP in ADP, OP1, OP2; log.solver, qoi_history.csv, time folders), profiles/ (blade .dat); qois.csv per case.",
    "g0 = DoE blades (c<cid> = blade index); g1_c0 = baseline inside the DoEs; in optimisations gid = generation, cid = candidate.",
    "Medium-mesh standard (DoE, brute force): 6000 iterations, QoIs averaged over 4000–6000. wall×2 standard: 10000 iterations, averaged over 8000–10000.",
    "Losses: L_ADP = mixed-out loss at ADP, L_OP = mean of OP1 and OP2. Do not ls/du the OPENFOAM trees casually (huge).",
    "Deleted 7 Oct (user decision): doe1000/ (POD-box DoE) and optim_validation/.",
]


def scan(folder):
    docs, figs, skipped = [], defaultdict(list), 0
    for d, dirs, files in os.walk(folder):
        rel = os.path.relpath(d, folder)
        skipped += sum(1 for x in dirs if x in SKIP)
        dirs[:] = sorted(x for x in dirs if x not in SKIP and not x.startswith("."))
        per_run = rel.startswith("runs" + os.sep) and rel.count(os.sep) == 1 and os.path.basename(folder) == "surrogate_ogv"
        for f in sorted(files):
            p = f if rel == "." else os.path.join(rel, f)
            if f.endswith(".md"):
                if not per_run:
                    docs.append(p)
            elif f.lower().endswith(FIG):
                if per_run:
                    figs["(per-training-run plots, counted only)"].append(None)
                else:
                    figs["." if rel == "." else rel].append(f)
    figs = {k: (len(v) if k.startswith("(") else v) for k, v in sorted(figs.items())}
    return docs, figs


def main():
    out = {"_about": f"Index of {ROOT}, generated {time.strftime('%d %b %Y')} by "
                     "/mnt/Data_B/beta-aero-optim/examples/LRN-OGV_OpenFOAM/runs_index.py. Paths are relative to the folder.",
           "_conventions": CONVENTIONS}
    listed = set()
    for theme, folders in THEMES.items():
        out[theme] = {}
        for name, about in folders.items():
            path = os.path.join(ROOT, name)
            if not os.path.isdir(path):
                continue
            listed.add(name)
            docs, figs = scan(path)
            out[theme][name] = {"about": about, "docs": docs, "figures": figs}
    rest = [x for x in sorted(os.listdir(ROOT)) if os.path.isdir(os.path.join(ROOT, x)) and x not in listed and not x.startswith(".")]
    if rest:
        out["unsorted"] = {x: {"about": "not described yet"} for x in rest}
    json.dump(out, open(os.path.join(ROOT, "INDEX.json"), "w"), indent=1, ensure_ascii=False)
    n = sum(len(v) if isinstance(v, list) else v for t in out.values() if isinstance(t, dict)
            for f in t.values() if isinstance(f, dict) and "figures" in f for v in f["figures"].values())
    print(f"INDEX.json: {len(listed)} folders, {n} figures, unsorted: {rest}")


if __name__ == "__main__":
    main()
