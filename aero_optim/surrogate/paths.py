"""
Locations used by the surrogate code. The code lives in the repository; data, trained models and optimisation runs live in
a work folder outside it, by default <AERO_RUNS>/surrogate_ogv.

Environment overrides (e.g. on murphy or Jean Zay):
- AERO_RUNS: root of the runs area (default /mnt/Data_B/beta-aero-optim-runs);
- SURROGATE_WORK: the surrogate work folder (default $AERO_RUNS/surrogate_ogv), holding data*/, data_subsets/, runs/, sopt/, ...
Relative paths given on the command line of the scripts (e.g. --models runs/a) are relative to the current directory:
run the scripts from the work folder, as in examples/LRN-OGV_OpenFOAM/surrogate/README.md.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.environ.get("AERO_RUNS", "/mnt/Data_B/beta-aero-optim-runs")
WORK = os.environ.get("SURROGATE_WORK", os.path.join(RUNS, "surrogate_ogv"))

EXAMPLE = os.path.join(REPO, "examples", "LRN-OGV_OpenFOAM")
BASELINE_DAT = os.path.join(EXAMPLE, "baseline_322.dat")       # baseline blade profile
DOE = os.path.join(RUNS, "doe1000_bladegen")                    # BladeGen DoE (medium mesh): doe_results.csv, OPENFOAM/
POD_DIR = os.path.join(DOE, "pod")                              # POD basis used everywhere (basis.npz, dataset.npz)
CFD_SOPT = os.path.join(RUNS, "cfd_sopt")                       # CFD cases of the verified surrogate fronts
