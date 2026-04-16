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
from cascade_mf.main_mf_sm import (set_mo_DOE, save_results, print, # noqa
                                   compute_bayesian_infill, execute_single_gen)


def execute_infill(
        X: np.ndarray, config: str,
        n_design: int, outdir: str,
        ite: int, fidelity: str,
        QoI: str, CoI: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    **Executes** infill candidates and returns their associated losses and outflow angles.
    """
    name = f"{fidelity}_infill_{ite}"
    df_dict = execute_single_gen(
        X=X,
        config=config,
        outdir=os.path.join(outdir, name),
        name=name,
        n_design=n_design
    )
    # losses
    loss_ADP = np.array(
        [df_dict[0][cid]["ADP"][QoI].dropna().iloc[-1] for cid in range(len(df_dict[0]))]
    )
    loss_OP = np.array(
        [0.5 * (df_dict[0][cid]["OP1"][QoI].dropna().iloc[-1]
                + df_dict[0][cid]["OP2"][QoI].dropna().iloc[-1])
            for cid in range(len(df_dict[0]))]
    )
    assert len(loss_ADP) == len(np.atleast_2d(X))
    # angles
    angle_ADP = np.array([df_dict[0][cid]["ADP"][CoI].dropna().iloc[-1]
                          for cid in range(len(df_dict[0]))])
    angle_OP1 = np.array([df_dict[0][cid]["OP1"][CoI].dropna().iloc[-1]
                          for cid in range(len(df_dict[0]))])
    angle_OP2 = np.array([df_dict[0][cid]["OP2"][CoI].dropna().iloc[-1]
                          for cid in range(len(df_dict[0]))])
    return [loss_ADP, loss_OP], [angle_ADP, angle_OP1, angle_OP2]


def main():
    """
    This script performs a multi-fidelity RANS/LES optimization with Bayesian infills.

    Note: in the result dictionaries the number of values per variable may changes which is
          why .dropna() is used to make sure iloc[-1] gives the expected value in the DataFrame.
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

    # get sampler and lf / hf DOEs
    print("MFSM: sampler selection..")
    # FFD sampling
    mf_sampler = get_sampler(
        n_design, bound, seed, model_loss.requires_nested_doe
    )
    x_lf, x_hf = mf_sampler.sample_mf(sm_config["optim"]["n_lf"], sm_config["optim"]["n_hf"])

    # generate hf_DOE
    print("MFSM: HF DOE computation..")
    hf_dir = os.path.join(outdir, "hf_doe")
    hf_dict = execute_single_gen(
        X=x_hf,
        config=sm_config["optim"]["hf_config"],
        outdir=hf_dir,
        name="hf_doe",
        n_design=sm_config["optim"]["n_design"]
    )
    print(f"MFSM: HF DOE computation finished after {time.time() - t0} seconds")

    # generate lf_DOE
    print("MFSM: LF DOE computation..")
    lf_dir = os.path.join(outdir, "lf_doe")
    lf_dict = execute_single_gen(
        X=x_lf,
        config=sm_config["optim"]["lf_config"],
        outdir=lf_dir,
        name="lf_doe",
        n_design=sm_config["optim"]["n_design"]
    )
    print(f"MFSM: LF DOE computation finished after {time.time() - t0} seconds")

    # compute DOEs
    # lf_DOE
    # QoIs
    QoI = sm_config["optim"]["QoI"]
    lf_w_ADP = np.array([lf_dict[0][cid]["ADP"][QoI].dropna().iloc[-1]
                         for cid in range(len(lf_dict[0]))])
    lf_w_OP = np.array(
        [0.5 * (lf_dict[0][cid]["OP1"][QoI].dropna().iloc[-1]
                + lf_dict[0][cid]["OP2"][QoI].dropna().iloc[-1])
         for cid in range(len(lf_dict[0]))]
    )
    # CoIs
    CoI = sm_config["optim"]["CoI"]
    lf_a_ADP = np.array([lf_dict[0][cid]["ADP"][CoI].dropna().iloc[-1]
                         for cid in range(len(lf_dict[0]))])
    lf_a_OP1 = np.array([lf_dict[0][cid]["OP1"][CoI].dropna().iloc[-1]
                         for cid in range(len(lf_dict[0]))])
    lf_a_OP2 = np.array([lf_dict[0][cid]["OP2"][CoI].dropna().iloc[-1]
                         for cid in range(len(lf_dict[0]))])

    # hf_DOE
    # QoIs
    hf_w_ADP = np.array([hf_dict[0][cid]["ADP"][QoI].dropna().iloc[-1]
                         for cid in range(len(hf_dict[0]))])
    hf_w_OP = np.array(
        [0.5 * (hf_dict[0][cid]["OP1"][QoI].dropna().iloc[-1]
                + hf_dict[0][cid]["OP2"][QoI].dropna().iloc[-1])
         for cid in range(len(hf_dict[0]))]
    )
    # CoIs
    hf_a_ADP = np.array([hf_dict[0][cid]["ADP"][CoI].dropna().iloc[-1]
                         for cid in range(len(hf_dict[0]))])
    hf_a_OP1 = np.array([hf_dict[0][cid]["OP1"][CoI].dropna().iloc[-1]
                         for cid in range(len(hf_dict[0]))])
    hf_a_OP2 = np.array([hf_dict[0][cid]["OP2"][CoI].dropna().iloc[-1]
                         for cid in range(len(hf_dict[0]))])

    # addition of lf baseline results
    # QoI
    x_lf = np.vstack([x_lf, np.zeros(x_lf.shape[-1])])
    lf_w_ADP = np.append(lf_w_ADP, sm_config["optim"]["bsl_lf_w_ADP"])
    lf_w_OP = np.append(lf_w_OP, sm_config["optim"]["bsl_lf_w_OP"])
    # CoI
    lf_a_ADP = np.append(lf_a_ADP, sm_config["optim"]["bsl_lf_a_ADP"])
    lf_a_OP1 = np.append(lf_a_OP1, sm_config["optim"]["bsl_lf_a_OP1"])
    lf_a_OP2 = np.append(lf_a_OP2, sm_config["optim"]["bsl_lf_a_OP2"])
    # addition of hf baseline results
    # QoI
    x_hf = np.vstack([x_hf, np.zeros(x_hf.shape[-1])])
    hf_w_ADP = np.append(hf_w_ADP, sm_config["optim"]["bsl_hf_w_ADP"])
    hf_w_OP = np.append(hf_w_OP, sm_config["optim"]["bsl_hf_w_OP"])
    # CoI
    hf_a_ADP = np.append(hf_a_ADP, sm_config["optim"]["bsl_hf_a_ADP"])
    hf_a_OP1 = np.append(hf_a_OP1, sm_config["optim"]["bsl_hf_a_OP1"])
    hf_a_OP2 = np.append(hf_a_OP2, sm_config["optim"]["bsl_hf_a_OP2"])

    # training loss model
    print("MFSM: training model(s)..")
    set_mo_DOE(model_loss, x_lf=x_lf, y_lf=[lf_w_ADP, lf_w_OP], x_hf=x_hf, y_hf=[hf_w_ADP, hf_w_OP])
    model_loss.train()
    print(f"MFSM: finished training loss model after {time.time() - t0} seconds")
    # saves mf-sm
    with open(os.path.join(outdir, "model_loss.pkl"), "wb") as handle:
        pickle.dump(model_loss, handle)
    print(f"MFSM: model saved to {outdir}")

    # set DOE angle model
    set_mo_DOE(
        model_angle,
        x_lf=x_lf,
        y_lf=[lf_a_ADP, lf_a_OP1, lf_a_OP2], x_hf=x_hf, y_hf=[hf_a_ADP, hf_a_OP1, hf_a_OP2]
    )

    # MFSM based optimization with adaptive infill
    print("MFSM: surrogate model based optimization..")
    nite = sm_config["optim"]["infill_nb"]
    infill_size = sm_config["optim"]["infill_lf_size"]
    infill_nb_gen = sm_config["optim"]["infill_nb_gen"]
    infill_regularization = sm_config["optim"].get("regularization", False)
    for ite in range(nite):
        outdir_ite = os.path.join(outdir, outdir.split("/")[-1] + f"_{ite}")
        # optimization
        # compute Bayesian infill
        x_lf_infill = compute_bayesian_infill(
            model_loss,
            infill_size,
            infill_nb_gen,
            infill_regularization,
            n_design,
            bound,
            seed,
        )
        # execute hf infill
        x_hf_infill = x_lf_infill[0]
        y_hf_infill_loss, y_hf_infill_angle = execute_infill(
            x_hf_infill,
            config=sm_config["optim"]["hf_config"],
            n_design=sm_config["optim"]["n_design"],
            outdir=outdir_ite,
            ite=ite,
            fidelity="high",
            QoI=QoI,
            CoI=CoI
        )
        print(f"MFSM: hf infill candidate {x_hf_infill} with fitness {y_hf_infill_loss}")
        # execute lf infill
        y_lf_infill_loss, y_lf_infill_angle = execute_infill(
            x_lf_infill,
            config=sm_config["optim"]["lf_config"],
            n_design=sm_config["optim"]["n_design"],
            outdir=outdir_ite,
            ite=ite,
            fidelity="low",
            QoI=QoI,
            CoI=CoI
        )
        # update models
        # loss DOE + training
        set_mo_DOE(model_loss, x_lf_infill, y_lf_infill_loss, x_hf_infill, y_hf_infill_loss)
        model_loss.train()
        # angle DOE
        set_mo_DOE(model_angle, x_lf_infill, y_lf_infill_angle, x_hf_infill, y_hf_infill_angle)
        # save updated loss model
        mv_filelist([os.path.join(outdir, "model_loss.pkl")],
                    [os.path.join(outdir, f"model_loss_{ite}.pkl")])
        with open(os.path.join(outdir, "model_loss.pkl"), "wb") as handle:
            pickle.dump(model_loss, handle)
        print(f"MFSM: surrogate model based optimization {ite + 1}/{nite}"
              f" finished after {time.time() - t0} seconds")
    # train angle model
    model_angle.train()
    # save angle model
    with open(os.path.join(outdir, "model_angle.pkl"), "wb") as handle:
        pickle.dump(model_angle, handle)
    # save datasets and final loss model
    save_results(model_loss, outdir)


if __name__ == '__main__':
    main()
