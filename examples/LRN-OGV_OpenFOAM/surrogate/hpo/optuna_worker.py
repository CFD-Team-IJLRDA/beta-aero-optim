"""
Optuna worker for phase 3. Several workers (K per GPU, many jobs) share one study through a journal file on $WORK
(JournalFileBackend with an open-file lock: safe on Lustre, unlike SQLite).

Strategy: TPE sampler (multivariate, grouped for the conditional SIREN/MLP branches; 30 random start-up trials),
objective = validation mixed-out-loss error (mean over the 3 OPs, best over epochs), reported every --qoi-every epochs;
successive-halving pruner (ASHA-style: rungs at 25, 50, 100 epochs, keeps the best third).
Each trial trains in this process (data loaded once and cached) and writes the usual run folder.

Usage (one per slot): python hpo/optuna_worker.py <campaign_dir> --gpu G [--stop-after-min 105] [--n-trials 300]
"""
import argparse
import gc
import os
import sys
import time

import optuna
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts folder: _repo, train_inr, ...
import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
HERE = _repo.WORK                 # data, runs, results: the surrogate work folder
sys.path.insert(0, _repo.SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_inr  # noqa: E402
from space import suggest, suggest_wide, to_argv  # noqa: E402


def study(campaign, sampler="tpe", prune=True, seed=None, population=40):  # seed None: each worker draws its own (a shared seed made all workers repeat the same trials)
    """sampler: tpe (Bayesian, 30 random start-up trials), random, nsga2 (genetic algorithm: population 40, crossover and
    mutation, single objective), cmaes. prune=False: every trial runs to the end (no early stopping)."""
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(
        os.path.join(campaign, "study.journal"),
        lock_obj=optuna.storages.journal.JournalFileOpenLock(os.path.join(campaign, "study.journal"))))
    return optuna.create_study(
        study_name="inr", storage=storage, direction="minimize", load_if_exists=True,
        sampler={"tpe": lambda: optuna.samplers.TPESampler(multivariate=True, group=True, n_startup_trials=30, seed=None),
                 "random": lambda: optuna.samplers.RandomSampler(seed=seed),
                 "nsga2": lambda: optuna.samplers.NSGAIISampler(population_size=population, seed=seed),
                 "cmaes": lambda: optuna.samplers.CmaEsSampler(seed=seed)}[sampler](),
        pruner=(optuna.pruners.SuccessiveHalvingPruner(min_resource=25, reduction_factor=3, min_early_stopping_rate=0)
                if prune else optuna.pruners.NopPruner()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--stop-after-min", type=float, default=95)
    ap.add_argument("--n-trials", type=int, default=300, help="total trials of the study (all workers)")
    ap.add_argument("--sampler", default="tpe", choices=["tpe", "random", "nsga2", "cmaes"])
    ap.add_argument("--no-prune", action="store_true", help="every trial runs to the end")
    ap.add_argument("--space", default="narrow", choices=["narrow", "wide"], help="narrow = phase 3 (suggest), wide = phase 3b (suggest_wide)")
    ap.add_argument("--population", type=int, default=40, help="nsga2 population size (= number of workers is a good choice)")
    ap.add_argument("--extra", default="", help="options appended to every trial (smoke tests, e.g. '--train-subset 20 --epochs 10')")
    args = ap.parse_args()
    camp = os.path.abspath(args.campaign)
    tdir = os.path.join(camp, "trials")
    os.makedirs(tdir, exist_ok=True)
    s = study(camp, args.sampler, not args.no_prune, population=args.population)
    space = suggest_wide if args.space == "wide" else suggest
    t0 = time.time()
    while (time.time() - t0) / 60 < args.stop_after_min:
        done = [t for t in s.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
                                                   optuna.trial.TrialState.RUNNING)]
        if len(done) >= args.n_trials:
            break
        trial = s.ask()
        cfg = space(trial)
        name = f"t{trial.number:04d}"
        argv = ["--name", name, "--out-root", tdir, "--gpu", str(args.gpu), "--data", "data_mean", "--test-every", "0",
                "--qoi-every", "5"] + to_argv(cfg) + args.extra.split()
        try:
            summary = train_inr.main(argv, trial)
            s.tell(trial, summary["best_val_w_relmae"])
        except train_inr.Pruned:
            s.tell(trial, state=optuna.trial.TrialState.PRUNED)
        except Exception as e:                                   # e.g. divergence, OOM: record as failed, go on
            print(f"{name} failed: {e!r}", flush=True)
            s.tell(trial, state=optuna.trial.TrialState.FAIL)
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
