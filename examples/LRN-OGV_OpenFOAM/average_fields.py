"""Adds time-averaged fields to finished OpenFOAM cascade cases (for field surrogates).

Usage: python average_fields.py -c openfoam_config.json --cases-file cases.txt [--extra 2000] [--budget 96]

cases.txt lists blade case folders (each with ADP/ OP1/ OP2/ sub-cases), one per line. Every run is restarted
from its last saved iteration for --extra iterations with the template's controlDict (fieldAverage enabled),
so pMean, UMean, rhoMean, TMean and nutMean are averaged over exactly the iterations on which MP1/MP2 are
sampled. The previous time folder is kept (purgeWrite 0, one write at the end). Per blade, the QoIs of the new
window are written to qois_fieldavg.csv (same columns as qois.csv, plus 'window'). The MP post-processing runs in
a pool of --post-workers processes so that solver launches are not held up. Resumable: blades with
qois_fieldavg.csv are skipped, runs already post-processed are reused and runs whose solve ended but were not
post-processed are only post-processed. The OpenFOAM environment must be sourced.
"""
import argparse
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from aero_optim.simulator.cascade_qoi import qoi_history, summarize  # noqa: E402

logger = logging.getLogger("average_fields")


def last_time(case):
    return max(int(d) for d in os.listdir(case) if d.isdigit())


def prepare(case, template_controldict, extra, interval):
    t0 = last_time(case)
    shutil.copy(template_controldict, os.path.join(case, "system", "controlDict"))
    cd = open(os.path.join(case, "system", "controlDict")).read()
    cd = re.sub(r"purgeWrite\s+\d+;", "purgeWrite      0;", cd)
    cd = re.sub(r"writeInterval\s+\d+;", f"writeInterval   {extra};", cd, count=1)
    open(os.path.join(case, "system", "controlDict"), "w").write(cd)
    with open(os.path.join(case, "system", "include", "runControl"), "w") as f:
        f.write(f"endTime {t0 + extra};\nsampleStart {t0};\nsampleInterval {interval};\n")
    shutil.rmtree(os.path.join(case, "postProcessing", "MP"), ignore_errors=True)
    return t0, t0 + extra


def previous(case):
    """**Returns** ("done", t0, t1) if post-processed, ("solved", t0, t1) if solved only, else (None, 0, 0)."""
    for f in glob.glob(os.path.join(case, "qoi_history_fieldavg_*.csv")):
        t0, t1 = map(int, f.rsplit("_", 1)[1][:-4].split("-"))
        return "done", t0, t1
    for f in glob.glob(os.path.join(case, "log.solver_fieldavg_*")):
        t0, t1 = map(int, f.rsplit("_", 1)[1].split("-"))
        if os.path.isdir(os.path.join(case, str(t1))) and open(f).read().rstrip().endswith("End"):
            return "solved", t0, t1
    return None, 0, 0


def finish(case, t0, t1):
    f = os.path.join(case, f"qoi_history_fieldavg_{t0}-{t1}.csv")
    if os.path.isfile(f):
        row = summarize(pd.read_csv(f, index_col=0))
        row.update(status="ok", window=f"{t0 + 1}-{t1}")
        return row
    hist = qoi_history(os.path.join(case, "postProcessing", "MP"))
    hist = hist[hist.index > t0]
    hist.to_csv(os.path.join(case, f"qoi_history_fieldavg_{t0}-{t1}.csv"))
    shutil.rmtree(os.path.join(case, "postProcessing", "MP"), ignore_errors=True)
    row = summarize(hist)
    row.update(status="ok", window=f"{t0 + 1}-{t1}")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--cases-file", required=True)
    ap.add_argument("--extra", type=int, default=2000)
    ap.add_argument("--budget", type=int, default=96)
    ap.add_argument("--post-workers", type=int, default=16)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg_path = os.path.abspath(args.config)
    config = json.load(open(cfg_path))
    template = os.path.join(os.path.dirname(cfg_path), config["simulator"]["ref_input"], "system", "controlDict")
    ops = list(config["simulator"]["operating_points"])
    interval = config["simulator"].get("sample_interval", 1)
    blades = [line.strip() for line in open(args.cases_file) if line.strip()]

    pool = ProcessPoolExecutor(args.post_workers)
    queue, posts, results, failed = [], [], {}, 0
    for b in blades:
        if os.path.isfile(os.path.join(b, "qois_fieldavg.csv")):
            continue
        for op in ops:
            case = os.path.join(b, op)
            state, t0, t1 = previous(case)
            if state:
                posts.append((b, op, pool.submit(finish, case, t0, t1)))
            else:
                queue.append((b, op))
    logger.info(f"{len(blades)} blades, {len(queue)} runs to do, {len(posts)} to post-process only")
    running = []
    while queue or running or posts:
        while queue and len(running) < args.budget:
            b, op = queue.pop(0)
            case = os.path.join(b, op)
            t0, t1 = prepare(case, template, args.extra, interval)
            with open(os.path.join(case, f"log.solver_fieldavg_{t0}-{t1}"), "wb") as log:
                proc = subprocess.Popen(config["simulator"].get("exec_cmd", "rhoSimpleFoam").split(), cwd=case,
                                        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        start_new_session=True, env={**os.environ, "PWD": os.path.abspath(case)})
            running.append((b, op, case, t0, t1, proc))
        still = []
        for item in running:
            b, op, case, t0, t1, proc = item
            if proc.poll() is None:
                still.append(item)
            elif proc.returncode == 0:
                posts.append((b, op, pool.submit(finish, case, t0, t1)))
            else:
                posts.append((b, op, None))
        running = still
        pending, n_done = [], 0
        for b, op, fut in posts:
            if fut is not None and not fut.done():
                pending.append((b, op, fut))
                continue
            try:
                row = fut.result() if fut is not None else {"status": "failed"}
            except Exception as e:  # missing samples etc.
                row = {"status": f"failed: {e}"}
            failed += row["status"] != "ok"
            n_done += 1
            results.setdefault(b, {})[op] = row
            if len(results[b]) == len(ops):
                pd.DataFrame.from_dict(results[b], orient="index").to_csv(os.path.join(b, "qois_fieldavg.csv"))
        posts = pending
        if n_done:
            logger.info(f"{sum(len(v) for v in results.values())} runs done ({failed} failed), {len(running)} running, "
                        f"{len(posts)} post-processing, {len(queue)} queued")
        time.sleep(5)
    pool.shutdown()

if __name__ == "__main__":
    main()
