"""
Data folder for the data-efficiency study: a CFD budget of N DoE blades (train + validation), everything else as data_mean.

- pool = the train + validation blades of data_mean (791 DoE blades, with CFD); a random order (seed) is drawn and the
  first N blades are the budget: the first round(0.15 N) of them are the validation set (used to select the checkpoint,
  so they cost CFD too), the rest the training set. The 87 test blades, the 20 Pareto blades and the baseline of
  data_mean are kept as evaluation sets (not counted in the budget).
- blades/, mp_aug/ and meta.json are links/copies of data_mean; T_bal (balance.npz) is refitted on the N-blade training set
  (as it would be with only that data).

Usage: ~/miniconda3/envs/ipynb/bin/python make_data_subset.py --n 100 [--seed 0]  -> data_subsets/N100_s0/
"""
import argparse
import json
import os
import sys
import shutil

import numpy as np

import _repo  # noqa: E402,F401  (repository on sys.path; SRC, WORK)
import fit_balance as FB

HERE = _repo.WORK                 # data, runs, sopt: the surrogate work folder (aero_optim/surrogate/paths.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default="data_mean")
    args = ap.parse_args()
    out = os.path.join(HERE, "data_subsets", f"N{args.n}_s{args.seed}")
    os.makedirs(out, exist_ok=True)
    sp = json.load(open(os.path.join(HERE, args.base, "splits.json")))
    pool = sorted(sp["train"] + sp["val"])
    order = list(np.random.default_rng(args.seed).permutation(pool))
    sub = order[:args.n]
    n_val = max(8, round(0.15 * args.n))
    new = {"val": sorted(sub[:n_val]), "train": sorted(sub[n_val:]), "test": sp["test"], "pareto": sp["pareto"],
           "baseline": sp["baseline"], "seed": args.seed, "budget_cfd_blades": args.n}
    json.dump(new, open(os.path.join(out, "splits.json"), "w"), indent=1)
    for name in ("blades", "mp_aug"):
        link = os.path.join(out, name)
        if not os.path.lexists(link):
            os.symlink(os.path.join("..", "..", args.base, name), link)
    for f in ("meta.json", "quality.csv"):
        shutil.copy(os.path.join(HERE, args.base, f), os.path.join(out, f))
    X = [np.load(os.path.join(HERE, args.base, "blades", f"{n}.npz"))["X_geom"] for n in new["train"]]
    lo, hi = np.min([x.min(axis=0) for x in X], axis=0), np.max([x.max(axis=0) for x in X], axis=0)
    grids = [np.linspace(lo[k], hi[k], FB.N_GRID) for k in range(2)]
    F = [np.mean([FB.ecdf_on_grid(x[:, k], grids[k]) for x in X], axis=0) for k in range(2)]
    F = [(1 - FB.EPS) * f + FB.EPS * np.linspace(0, 1, FB.N_GRID) for f in F]
    for f in F:
        f[0], f[-1] = 0.0, 1.0
        assert np.all(np.diff(f) > 0)
    np.savez(os.path.join(out, "balance.npz"), grid_x=grids[0], F_x=F[0], grid_y=grids[1], F_y=F[1])
    print(f"{os.path.basename(out)}: budget {args.n} = train {len(new['train'])} + val {len(new['val'])}; test {len(new['test'])}, pareto {len(new['pareto'])}")


if __name__ == "__main__":
    main()
