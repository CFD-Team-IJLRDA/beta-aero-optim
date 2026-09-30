"""Brute-force RANS optimisation of the LRN-OGV cascade with OpenFOAM (NSGA-II on the POD coefficients).

Usage: optim-foam -c openfoam_config.json [--n-gen N]

See aero_optim/optim/cascade_foam.py for the problem definition and the "optim" config entries.
Re-running the same command resumes the optimisation: finished blades are reloaded.
"""
import argparse
import json
import logging
import os

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from aero_optim.optim.cascade_foam import CascadeFoamProblem

logger = logging.getLogger("optim_foam")


def run(config: dict, n_gen: int | None = None):
    opt = config["optim"]
    problem = CascadeFoamProblem(config)
    pop_size = opt.get("pop_size", 32)
    algorithm = NSGA2(pop_size=pop_size, sampling=problem.initial_population(pop_size),
                      eliminate_duplicates=True)
    n_gen = n_gen or opt.get("n_gen", 30)
    # pymoo counts the (already evaluated) initial population as generation 1
    res = minimize(problem, algorithm, ("n_gen", n_gen + 1), seed=opt.get("seed", 1), verbose=True)
    logger.info(f"done: {problem.gen - 1} generations, Pareto front in {os.path.join(problem.outdir, 'pareto.csv')}")
    return res, problem


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True, help="path to the JSON config")
    parser.add_argument("--n-gen", type=int, default=None, help="number of generations (overrides the config)")
    args = parser.parse_args()
    config_path = os.path.abspath(args.config)
    config = json.load(open(config_path))
    os.chdir(os.path.dirname(config_path))
    os.makedirs(config["optim"]["outdir"], exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(config["optim"]["outdir"], "optim.log"))])
    run(config, args.n_gen)


if __name__ == "__main__":
    main()
