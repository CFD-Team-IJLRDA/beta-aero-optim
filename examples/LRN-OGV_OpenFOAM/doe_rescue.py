"""Continues the rejected runs of a DoE and re-averages them over a later window.

Usage: python doe_rescue.py -c openfoam_config.json [--outdir DIR] [--extra 4000] [--budget 96]

Run after doe_dataset.py (it reads <outdir>/dataset/rejected.csv). Every run rejected for its
time-average (oscillation, drift or non-positive loss) is continued for --extra iterations
from its last saved state and re-averaged over a window of the original length ending at the
new last iteration (e.g. 6000 -> 10000, window 8000-10000). Meshing failures are not retried.
Per run, the old history is kept as qoi_history_<start>-<end>.csv; qois.csv and doe_results.csv
are updated and get a <OP>_window column. Then run doe_dataset.py again to re-classify.
The OpenFOAM environment must be sourced.
"""
import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from aero_optim.main.doe import results_row  # noqa: E402
from aero_optim.simulator.cascade_qoi import qoi_history, summarize  # noqa: E402

logger = logging.getLogger("doe_rescue")


def case_of(blade: str) -> str:
    return "OPENFOAM/openfoam_g1_c0" if blade == "baseline" else f"OPENFOAM/openfoam_g0_c{int(blade)}"


def run_control(case: str) -> dict[str, int]:
    txt = open(os.path.join(case, "system", "include", "runControl")).read()
    return {k: int(re.search(rf"{k}\s+(\d+)", txt).group(1)) for k in ("endTime", "sampleStart", "sampleInterval")}


def finish(case: str, old: dict[str, int], new: dict[str, int], op: str) -> bool:
    """**Re-averages** a continued run; **returns** False if it failed."""
    sets_dir = os.path.join(case, "postProcessing", "MP")
    try:
        history = qoi_history(sets_dir)
    except FileNotFoundError:
        return False
    history = history[history.index >= new["sampleStart"]]
    old_hist = os.path.join(case, "qoi_history.csv")
    shutil.move(old_hist, os.path.join(case, f"qoi_history_{old['sampleStart']}-{old['endTime']}.csv"))
    history.to_csv(old_hist)
    shutil.rmtree(sets_dir, ignore_errors=True)
    summary_file = os.path.join(os.path.dirname(case), "qois.csv")
    summary = pd.read_csv(summary_file, index_col=0)
    for k, v in summarize(history).items():
        summary.loc[op, k] = v
    summary.loc[op, "status"] = "ok"
    summary.loc[op, "window"] = f"{new['sampleStart']}-{new['endTime']}"
    summary.to_csv(summary_file)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--outdir", help="DoE folder, if not the one in the config")
    parser.add_argument("--extra", type=int, default=4000, help="iterations to add")
    parser.add_argument("--budget", type=int, default=96, help="maximum concurrent runs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = json.load(open(os.path.abspath(args.config)))
    outdir, sim = args.outdir or config["study"]["outdir"], config["simulator"]
    ops = list(sim["operating_points"])

    rejected = pd.read_csv(os.path.join(outdir, "dataset", "rejected.csv"), dtype={"blade": str})
    todo = []
    for _, r in rejected.iterrows():
        for op in ops:
            if f"{op}: loss" in r.reason or f"{op}: non-positive" in r.reason:
                case = os.path.join(outdir, case_of(r.blade), op)
                if not any(re.fullmatch(r"qoi_history_\d+-\d+\.csv", f) for f in os.listdir(case)):  # not yet continued
                    todo.append((r.blade, op))
    logger.info(f"{len(todo)} runs of {rejected.blade.nunique()} rejected blades to continue by {args.extra} iterations")

    running, done, failed = [], [], []
    queue = list(todo)
    while queue or running:
        while queue and len(running) < args.budget:
            blade, op = queue.pop(0)
            case = os.path.join(outdir, case_of(blade), op)
            old = run_control(case)
            new = {**old, "endTime": old["endTime"] + args.extra,
                   "sampleStart": old["endTime"] + args.extra - (old["endTime"] - old["sampleStart"])}
            with open(os.path.join(case, "system", "include", "runControl"), "w") as f:
                f.write(f"endTime {new['endTime']};\nsampleStart {new['sampleStart']};\n"
                        f"sampleInterval {new['sampleInterval']};\n")
            with open(os.path.join(case, f"log.solver_{old['endTime']}-{new['endTime']}"), "wb") as log:
                proc = subprocess.Popen(sim.get("exec_cmd", "rhoSimpleFoam").split(), cwd=case, stdout=log,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                                        env={**os.environ, "PWD": os.path.abspath(case)})
            running.append((blade, op, case, old, new, proc))
        still = []
        for item in running:
            blade, op, case, old, new, proc = item
            if proc.poll() is None:
                still.append(item)
            elif proc.returncode == 0 and finish(case, old, new, op):
                done.append((blade, op))
            else:
                failed.append((blade, op))
                logger.error(f"blade {blade} {op}: continuation failed")
        if len(still) < len(running):
            logger.info(f"{len(done)} done, {len(failed)} failed, {len(still)} running, {len(queue)} queued")
        running = still
        time.sleep(5)

    # update doe_results.csv from the blades' qois.csv
    results = pd.read_csv(os.path.join(outdir, "doe_results.csv"), dtype={"blade": str})
    window = f"{sim['sample_start']}-{sim['end_time']}"
    for op in ops:
        if f"{op}_window" not in results:
            results[f"{op}_window"] = window
    for blade in sorted({b for b, _ in done}):
        summary = pd.read_csv(os.path.join(outdir, case_of(blade), "qois.csv"), index_col=0)
        idx = results.index[results.blade == blade][0]
        row = results_row({op: summary.loc[[op]].reset_index(drop=True) for op in summary.index})
        for k, v in row.items():
            results.loc[idx, k] = v
        for op in ops:
            if "window" in summary and isinstance(summary.at[op, "window"], str):
                results.loc[idx, f"{op}_window"] = summary.at[op, "window"]
    results.to_csv(os.path.join(outdir, "doe_results.csv"), index=False)
    logger.info(f"continued {len(done)} runs ({len(failed)} failed); now re-run doe_dataset.py")


if __name__ == "__main__":
    main()
