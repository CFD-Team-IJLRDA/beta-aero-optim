import argparse
import copy
import dill as pickle
import numpy as np
import os
import sys
import time

from aero_optim.mf_sm.mf_models import get_model, get_sampler, MultiObjectiveModel
from aero_optim.utils import check_config, mv_filelist

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cascade_mf.main_mf_sm import (set_mo_DOE, save_results, print) # noqa


def main():
    """
    This script trains  and saves a single-fidelity surrogate model from the high-fidelity samples
    generated during the Bayesian optimization (see main_mf_sm.py).

    Note: for this experiment we are not considering the outflow angle. Once the model is saved,
    the NSGA-II algorithm can be executed with it:
    $ optim -c cascade_mf.json
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", type=str, help="/path/to/lconfig.json")
    parser.add_argument("-v", "--verbose", type=int, help="logger verbosity level", default=3)
    args = parser.parse_args()

    t0 = time.time()

    # intit problem
    sm_config, _, _ = check_config(args.config, optim=True)
    # study
    pod = sm_config["study"]["ffd_type"] == "ffd_pod_2d"
    outdir = sm_config["study"]["outdir"]
    # ffd
    ffd_config = sm_config["ffd"]
    # optim
    optim_config = sm_config["optim"]
    seed = optim_config.get("seed", 123)

    # get bound
    n_design = optim_config["n_design"] if not pod else ffd_config["pod_ncontrol"]
    bound = sm_config["optim"]["bound"]
    bound = ([bound[0]] * n_design, [bound[-1]] * n_design)

    # update bound in case of rotation
    if ffd_config.get("rotation", False):
        rot_bound = ffd_config["rot_bound"]
        bound = (bound[0] + [rot_bound[0]], bound[-1] + [rot_bound[-1]])
        n_design += 1

    # get models
    print("MFSM: model selection..")
    model_name = sm_config["optim"]["model_name"]
    # loss model
    model_loss_ADP = get_model(model_name, n_design, sm_config, outdir, seed)
    model_loss_OP = copy.deepcopy(model_loss_ADP)
    model_loss = MultiObjectiveModel([model_loss_ADP, model_loss_OP])
    # outflow angle model
    model_angle_ADP = copy.deepcopy(model_loss_ADP)
    model_angle_OP1 = copy.deepcopy(model_loss_ADP)
    model_angle_OP2 = copy.deepcopy(model_loss_ADP)
    model_angle = MultiObjectiveModel([model_angle_ADP, model_angle_OP1, model_angle_OP2])

    # get training data
    x_hf = np.loadtxt(sm_config["optim"]["hf_candidates"])
    hf_w = np.loadtxt(sm_config["optim"]["hf_fitnesses"])
    
    # training loss model
    print("MFSM: training model(s)..")
    set_mo_DOE(
        model_loss,
        x_lf=x_hf, y_lf=[hf_w[:, 0], hf_w[:, 1]],
        x_hf=x_hf, y_hf=[hf_w[:, 0], hf_w[:, 1]]
    )
    model_loss.train()
    print(f"MFSM: finished training loss model after {time.time() - t0} seconds")
    # saves mf-sm
    with open(os.path.join(outdir, "model_loss.pkl"), "wb") as handle:
        pickle.dump(model_loss, handle)
    print(f"MFSM: model saved to {outdir}")


if __name__ == '__main__':
    main()
