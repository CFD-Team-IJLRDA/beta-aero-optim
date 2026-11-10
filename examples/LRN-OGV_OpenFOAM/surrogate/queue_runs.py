"""
Runs a list of INR trainings on corrsin's GPUs, a few at a time, safely (the machine is shared with CFD and cannot be
restarted remotely): at most --per-gpu runs per GPU, and a run starts only if at least --min-free-gb of RAM is available
(each run holds its data in host RAM while loading, ~10 GB). After training, each run is evaluated with both checkpoints
(best val field MSE and best val QoI). A failing run is logged and skipped; the queue goes on.

Jobs file (JSON): [{"name": "per_s0", "args": "--coords periodic --seed 0"}, ...]; every job gets --data data_mean
unless it sets --data. Logs: runs/<name>/train.log, runs/<name>/eval.log; queue log: --log.

Usage: nohup ~/miniconda3/envs/ogv_torch/bin/python queue_runs.py jobs.json [--per-gpu 2] [--gpus 0,1] &
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time

import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
HERE = _repo.WORK                 # data, runs, sopt: the surrogate work folder (aero_optim/surrogate/paths.py)
PY = sys.executable


def free_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1e6
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--per-gpu", type=int, default=2)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--min-free-gb", type=float, default=40)
    ap.add_argument("--log", default=os.path.join(HERE, "logs", "queue.log"))
    args = ap.parse_args()
    jobs = json.load(open(args.jobs))
    gpus = [int(g) for g in args.gpus.split(",")]
    log = open(args.log, "a")
    say = lambda m: (log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}\n"), log.flush())
    running = {}                                   # name -> (proc, gpu)
    say(f"queue {args.jobs}: {len(jobs)} jobs, gpus {gpus}, {args.per_gpu} per GPU")
    while jobs or running:
        for name, (proc, gpu) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[name]
            say(f"{name} finished (exit {proc.returncode})")
        while jobs:
            load = {g: sum(1 for _, gg in running.values() if gg == g) for g in gpus}
            gpu = min(gpus, key=lambda g: load[g])
            if load[gpu] >= args.per_gpu:
                break
            if free_gb() < args.min_free_gb:
                say(f"waiting: only {free_gb():.0f} GB RAM available")
                break
            job = jobs.pop(0)
            name, extra = job["name"], job["args"]
            out = os.path.join(HERE, "runs", name)
            if os.path.isfile(os.path.join(out, "summary.json")):
                say(f"{name} already trained, skipped")
                continue
            os.makedirs(out, exist_ok=True)
            data = "" if "--data" in extra else " --data data_mean"
            train = f"{PY} {_repo.SRC}/train_inr.py --name {name} --gpu {gpu}{data} {extra}"
            evals = " && ".join(f"{PY} {_repo.SRC}/eval_inr.py --name {name} --gpu {gpu} --checkpoint {c}" for c in ("best_model", "best_qoi_model"))
            cmd = f"{train} > {shlex.quote(out)}/train.log 2>&1 && ( {evals} ) > {shlex.quote(out)}/eval.log 2>&1"
            running[name] = (subprocess.Popen(["bash", "-c", cmd], cwd=HERE, stdin=subprocess.DEVNULL), gpu)
            say(f"{name} started on GPU {gpu} ({free_gb():.0f} GB RAM available): {extra}")
            time.sleep(60)                         # let it load its data before checking RAM again
        time.sleep(30)
    say("queue done")


if __name__ == "__main__":
    main()
