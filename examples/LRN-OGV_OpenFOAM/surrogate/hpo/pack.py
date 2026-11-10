"""
Runs a list of INR trials inside one Slurm job (or locally): K trials at a time on each visible GPU, each a separate
`train_inr.py` process (CUDA graphs, test set not monitored). Resumable and shareable between jobs: a trial with
summary.json is skipped, and a trial being run by another job is claimed with a lock file (atomic O_EXCL create).

The job logs nvidia-smi every 10 s to <campaign>/jobs/gpu_<jobid>.csv and, at the end, writes one line per GPU
with its mean utilisation to <campaign>/jobs/efficiency.csv (the > 50 % rule on Jean Zay).

Trials file: JSON list of dicts (name, factor, seed + train_inr options; see space.py).
Usage: python hpo/pack.py <campaign_dir> <trials.json> [--per-gpu 4] [--stop-after-min 105]
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts folder: _repo, train_inr, ...
import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
HERE = _repo.WORK                 # data, runs, results: the surrogate work folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from space import to_argv  # noqa: E402


def gpus():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return list(range(len(vis.split(","))))
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    return list(range(len([l for l in out.splitlines() if l.startswith("GPU")])))


def claim(trial_dir):
    os.makedirs(trial_dir, exist_ok=True)
    if os.path.isfile(os.path.join(trial_dir, "summary.json")):
        return False
    try:
        os.close(os.open(os.path.join(trial_dir, ".claimed"), os.O_CREAT | os.O_EXCL))
        return True
    except FileExistsError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("trials")
    ap.add_argument("--per-gpu", type=int, default=4)
    ap.add_argument("--stop-after-min", type=float, default=105, help="start no new trial after this (job time limit margin)")
    ap.add_argument("--extra", default="--test-every 0", help="options added to every trial")
    args = ap.parse_args()
    camp = os.path.abspath(args.campaign)
    tdir, jdir = os.path.join(camp, "trials"), os.path.join(camp, "jobs")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(jdir, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}")
    smi_file = os.path.join(jdir, f"gpu_{job}.csv")
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw",
                            "--format=csv,noheader,nounits", "-l", "10"], stdout=open(smi_file, "w"), stderr=subprocess.DEVNULL)
    log = open(os.path.join(jdir, f"pack_{job}.log"), "a")
    say = lambda m: (log.write(f"{time.strftime('%H:%M:%S')} {m}\n"), log.flush())
    queue = json.load(open(args.trials))
    devs, running, t0 = gpus(), {}, time.time()
    say(f"job {job}: {len(queue)} trials in the list, GPUs {devs}, {args.per_gpu} per GPU")
    while True:
        for name, (proc, g) in list(running.items()):
            if proc.poll() is not None:
                del running[name]
                say(f"{name} done (exit {proc.returncode}) on GPU {g}")
        late = (time.time() - t0) / 60 > args.stop_after_min
        while queue and not late:
            load = {g: sum(1 for _, gg in running.values() if gg == g) for g in devs}
            g = min(devs, key=lambda d: load[d])
            if load[g] >= args.per_gpu:
                break
            t = queue.pop(0)
            if not claim(os.path.join(tdir, t["name"])):
                continue
            cfg = {k: v for k, v in t.items() if k not in ("name", "factor")}
            argv = [sys.executable, os.path.join(_repo.SRC, "train_inr.py"), "--name", t["name"], "--out-root", tdir,
                    "--gpu", str(g), "--data", "data_mean"] + to_argv(cfg) + args.extra.split()
            json.dump(t, open(os.path.join(tdir, t["name"], "trial.json"), "w"), indent=1)
            running[t["name"]] = (subprocess.Popen(argv, cwd=HERE, stdout=open(os.path.join(tdir, t["name"], "train.log"), "w"),
                                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL), g)
            say(f"{t['name']} started on GPU {g}")
        if not running and (not queue or late):
            break
        time.sleep(15)
    smi.terminate()
    # efficiency summary of this job
    import pandas as pd
    try:
        d = pd.read_csv(smi_file, header=None, names=["t", "gpu", "util", "mem", "power"])
        eff = d.groupby("gpu").agg(util_mean=("util", "mean"), power_mean=("power", "mean"), samples=("util", "size")).reset_index()
        eff.insert(0, "job", job)
        f = os.path.join(jdir, "efficiency.csv")
        eff.to_csv(f, mode="a", header=not os.path.isfile(f), index=False)
        say(f"GPU utilisation: {eff.util_mean.round(0).tolist()} %")
    except Exception as e:
        say(f"efficiency summary failed: {e}")
    say("job done")


if __name__ == "__main__":
    main()
