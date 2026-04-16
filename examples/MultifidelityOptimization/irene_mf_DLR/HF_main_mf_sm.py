import argparse
import copy
import dill as pickle
import functools
import numpy as np
import os
import sys
import time
import pandas as pd
from typing      import Any

from aero_optim.mf_sm.mf_models import get_model, get_sampler, MfDNN, MfSMT, MultiObjectiveModel
from aero_optim.utils import check_config, mv_filelist
from aero_optim.ffd.ffd import FFD_POD_2D, DLR_POD_2D

from aero_optim.utils import (check_config, check_dir, check_file,
                              cp_filelist, mv_filelist, replace_in_file)
from feasible_constrained_acquisition import *
import subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

print = functools.partial(print, flush=True)

def get_mo_model(
        model_name: str, n_design: int, config: dict, outdir: str, seed: int
) -> MultiObjectiveModel | MfDNN:
    if model_name == "mfsmt":
        model_ADP      = get_model(model_name, n_design, config, outdir, seed)
        model_OP       = copy.deepcopy(model_ADP)
        model_BETA_ADP = copy.deepcopy(model_ADP)
        model_BETA_OP1 = copy.deepcopy(model_ADP)
        model_BETA_OP2 = copy.deepcopy(model_ADP)
        assert not isinstance(model_ADP, MfDNN) and not isinstance(model_OP, MfDNN) and not isinstance(model_BETA_ADP, MfDNN) and not isinstance(model_BETA_OP1, MfDNN) and not isinstance(model_BETA_OP2, MfDNN)
        return MultiObjectiveModel([model_ADP, model_OP, model_BETA_ADP, model_BETA_OP1, model_BETA_OP2])
    elif model_name == "mfdnn":
        model = get_model(model_name, n_design, config, outdir, seed)
        assert isinstance(model, MfDNN)
        return model
    else:
        raise Exception(f"{model_name} is currently not supported")

def set_mo_DOE(
        model: MfDNN | MultiObjectiveModel,
        x_lf: np.ndarray,
        y_lf: list[np.ndarray],
        x_hf: np.ndarray,
        y_hf: list[np.ndarray]
):
    if isinstance(model, MfDNN):
        model.set_DOE(x_lf=x_lf, x_hf=x_hf, y_lf=np.column_stack(y_lf), y_hf=np.column_stack(y_hf))
    elif isinstance(model, MultiObjectiveModel):
        model.set_DOE(x_lf=x_lf, y_lf=y_lf, x_hf=x_hf, y_hf=y_hf)
    else:
        raise Exception(f"{type(model)} is currently not supported")


def save_results(model: MfDNN | MultiObjectiveModel, outdir: str):
    if isinstance(model, MfDNN):
        assert model.x_lf_DOE is not None and model.x_hf_DOE is not None
        assert model.y_lf_DOE is not None and model.y_hf_DOE is not None
        np.savetxt(os.path.join(outdir, "lf_candidates.txt"), model.x_lf_DOE)
        np.savetxt(os.path.join(outdir, "lf_fitnesses.txt"), model.y_lf_DOE)
        np.savetxt(os.path.join(outdir, "hf_candidates.txt"), model.x_hf_DOE)
        np.savetxt(os.path.join(outdir, "hf_fitnesses.txt"), model.y_hf_DOE)
    elif isinstance(model, MultiObjectiveModel):
        assert model.models[0].x_lf_DOE is not None and model.models[0].x_hf_DOE is not None
        np.savetxt(os.path.join(outdir, "lf_candidates.txt"), model.models[0].x_lf_DOE)
        np.savetxt(os.path.join(outdir, "hf_candidates.txt"), model.models[0].x_hf_DOE)
        assert model.models[0].y_lf_DOE is not None and model.models[0].y_hf_DOE is not None
        assert model.models[1].y_lf_DOE is not None and model.models[1].y_hf_DOE is not None
        y_lf = np.column_stack([model.models[0].y_lf_DOE, model.models[1].y_lf_DOE])
        y_hf = np.column_stack([model.models[0].y_hf_DOE, model.models[1].y_hf_DOE])
        np.savetxt(os.path.join(outdir, "lf_fitnesses.txt"), y_lf)
        np.savetxt(os.path.join(outdir, "hf_fitnesses.txt"), y_hf)
    else:
        raise Exception(f"{type(model)} is currently not supported")
    

def compute_bayesian_infill(
        model: MfDNN | MultiObjectiveModel,
        ffd : FFD_2D,
        infill_lf_size: int,
        infill_nb_gen: int,
        infill_regularization: bool,
        n_design: int,
        bound: list[Any],
        seed: int,
) -> np.ndarray:
    """
    **Computes** the low fidelity Bayesian infill candidates.
    """
    assert isinstance(model, MultiObjectiveModel)
    # Probability of Improvement
    if infill_regularization:
        infill_lf = maximize_RegCrit(
            MPI_acquisition_function, model, n_design, bound, seed, infill_nb_gen
        )
    else:
        infill_lf = maximize_MPI_BO_constrained(model, ffd, n_design, bound, seed, infill_nb_gen)

    n_infill = 1
    print('Finished MPI BO infill')

    infill_lf_LCB_1 = minimize_LCB_constrained(model, ffd, 0, n_design, bound, seed, infill_nb_gen)
    print('Finished LCB 1 infill')
    infill_lf_LCB_2 = minimize_LCB_constrained(model, ffd, 1, n_design, bound, seed, infill_nb_gen)
    print('Finished LCB 2 infill')

    if not(infill_lf_LCB_1 in infill_lf):
        n_infill += 1
        infill_lf = np.vstack((infill_lf, infill_lf_LCB_1))
    if not(infill_lf_LCB_2 in infill_lf):
        n_infill += 1
        infill_lf = np.vstack((infill_lf, infill_lf_LCB_2))

    # max-min Euclidean Distance
    current_DOE = model.get_DOE()
    current_DOE = np.vstack((current_DOE, infill_lf))
    for i in range(infill_lf_size - 3):
        infill_lf_ED = maximize_ED_constrained(current_DOE, ffd, n_design, bound, seed, infill_nb_gen)
        infill_lf = np.vstack((infill_lf, infill_lf_ED))
        current_DOE = np.vstack((current_DOE, infill_lf_ED))
        print(f'Finished ED {i+1} infill')
    return infill_lf

def execute_single_gen(
        X: np.ndarray, config: str, outdir: str, name: str, n_design: int = 0
) -> dict[int, dict[int, pd.DataFrame]]:
    """
    **Executes** a single generation of candidates.
    """
    X = np.round(X, decimals=12)
    check_file(config)
    check_dir(outdir)
    cp_filelist([config], [outdir])
    config_path = os.path.join(outdir, config)
    custom_doe = os.path.join(outdir, f"{name}.txt")
    np.savetxt(custom_doe, np.atleast_2d(X), fmt="%.10e")
    # updates @outdir, @n_design, @doe_size, @custom_doe
    # Note: @n_design is the number of FFD control points even when using POD
    config_args = {
        "@outdir": outdir,
        "@n_design": f"{n_design if n_design else np.atleast_2d(X).shape[1]}",
        "@doe_size": f"{np.atleast_2d(X).shape[0]}",
        "@custom_doe": f"{custom_doe}"
    }
    replace_in_file(config_path, config_args)
    # execute single generation
    exec_cmd = ["optim", "-c", f"{config_path}", "-v", "3", "--pymoo"]
    subprocess.run(exec_cmd, env=os.environ, stdin=subprocess.DEVNULL, check=True)
    # load results
    with open(os.path.join(outdir, "df_dict.pkl"), "rb") as handle:
        df_dict = pickle.load(handle)
    return df_dict

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
    loss_ADP = np.array([df_dict[0][cid]["ADP"][QoI].dropna().iloc[-1] for cid in range(len(df_dict[0]))])
    loss_OP = np.array([0.5 * (df_dict[0][cid]["OP1"][QoI].dropna().iloc[-1] + df_dict[0][cid]["OP2"][QoI].dropna().iloc[-1])
            for cid in range(len(df_dict[0]))]
    )
    assert len(loss_ADP) == len(np.atleast_2d(X))
    # angles
    angle_ADP = np.array([df_dict[0][cid]["ADP"][CoI].dropna().iloc[-1] for cid in range(len(df_dict[0]))])
    angle_OP1 = np.array([df_dict[0][cid]["OP1"][CoI].dropna().iloc[-1] for cid in range(len(df_dict[0]))])
    angle_OP2 = np.array([df_dict[0][cid]["OP2"][CoI].dropna().iloc[-1] for cid in range(len(df_dict[0]))])

    return [loss_ADP, loss_OP, angle_ADP, angle_OP1, angle_OP2]


def main():
    """
    This script performs a multi-fidelity RANS/LES optimization with Bayesian infills.

    Note: in the result dictionaries the number of values per variable may changes which is
          why .dropna() is used to make sure iloc[-1] gives the expected value in the DataFrame.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", type=str, help="/path/to/lconfig.json")
    parser.add_argument("-p", "--pod", action="store_true", help="perform POD data reduction")
    parser.add_argument("-v", "--verbose", type=int, help="logger verbosity level", default=3)
    args = parser.parse_args()    

    t0 = time.time()

    # intit problem
    sm_config, _, _ = check_config(args.config, optim=True)
    optim_config = sm_config["optim"]
    ffd_config = sm_config["ffd"]
    outdir = sm_config["study"]["outdir"]

    QoI = sm_config["optim"]["QoI"]
    CoI = sm_config["optim"]["CoI"]

    seed = optim_config.get("seed", 123)
    pod = 'pod' in sm_config["study"]["ffd_type"]

    # # get bound
    n_design = optim_config["n_design"] if not pod else ffd_config["pod_ncontrol"]
    
    # bound = sm_config["optim"]["bound"]
    # bound = ([bound[0]] * n_design, [bound[-1]] * n_design)

    # print(n_design)
    # print(bound)
    # input()
    
    if pod and ('dlr' in sm_config["study"]['ffd_type']):
        sm_config["ffd"]["dat_file"] = sm_config["study"]["file"]
        ffd = DLR_POD_2D(**sm_config["ffd"])
        n_design = ffd.pod_ncontrol
        bound = ffd.get_bound()
    elif pod and ('ffd' in sm_config["study"]['ffd_type']):
        sm_config["ffd"]["dat_file"] = sm_config["study"]["file"]
        ffd = FFD_POD_2D(
            dat_file=sm_config["study"]["file"],
            pod_ncontrol=sm_config["ffd"]["pod_ncontrol"],
            ffd_ncontrol=sm_config["optim"]["n_design"],
            ffd_dataset_size=sm_config["ffd"]["ffd_dataset_size"],
            ffd_bound=sm_config["optim"]["bound"]
        )
        n_design = ffd.pod_ncontrol
        bound = ffd.get_bound()
    elif 'dlr' in sm_config["study"]['ffd_type']:
        ffd = DLR_2D(**sm_config["ffd"])
        n_design = sm_config["optim"]["n_design"]
        ul_bounds = np.array(list(sm_config["ffd"]["param_bounds"].values()))
        bound    = (ul_bounds[:,0].tolist(), ul_bounds[:,1].tolist())
        print(bound)
    else:
        ffd = FFD_2D(**sm_config["ffd"])
        n_design = sm_config["optim"]["n_design"]
        bound = sm_config["optim"]["bound"]

    # update bound in case of rotation
    if ffd_config.get("rotation", False):
        rot_bound = ffd_config["rot_bound"]
        bound = (bound[0] + [rot_bound[0]], bound[-1] + [rot_bound[-1]])
        n_design += 1

    # get models
    print("MFSM: model selection..")
    model_total = get_mo_model(sm_config["optim"]["model_name"], n_design, sm_config, outdir, seed)

    # get sampler and lf / hf DOEs
    print("MFSM: sampler selection..")
    # FFD sampling
    mf_sampler = get_sampler(
        n_design, bound, seed, model_total.requires_nested_doe
    )
    x_lf, x_hf = mf_sampler.sample_mf(sm_config["optim"]["n_lf"], sm_config["optim"]["n_hf"])

    # try:
    #     with open(os.path.join(outdir, "model_baseline.pkl"), "rb") as f:
    #         model_total = pickle.load(f)
    #     if not(os.path.isfile(os.path.join(outdir, f"model.pkl"))):
    #         cp_filelist([os.path.join(outdir, "model_baseline.pkl")],
    #                     [os.path.join(outdir, f"model.pkl")])
    #     print("model_baseline.pkl exists, therefore loaded!")
    # except:

    # generate hf_DOE
    print("MFSM: HF DOE computation..")
    hf_dir = os.path.join(outdir, "hf_doe")
    # try: 
    #     with open(os.path.join(hf_dir, "df_dict.pkl"), "rb") as handle:
    #         hf_dict = pickle.load(handle)
    #     print("HF Simulations already done, therefore loaded!")
    # except:
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
    # try:
    #     with open(os.path.join(lf_dir, "df_dict.pkl"), "rb") as handle:
    #         lf_dict = pickle.load(handle)
    #     print("LF Simulations already done, therefore loaded!")
    # except:
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
    lf_w_ADP = np.array([lf_dict[0][cid]["ADP"][QoI].dropna().iloc[-1] for cid in range(len(lf_dict[0]))])
    lf_w_OP = np.array([0.5 * (lf_dict[0][cid]["OP1"][QoI].dropna().iloc[-1] + lf_dict[0][cid]["OP2"][QoI].dropna().iloc[-1])
        for cid in range(len(lf_dict[0]))])

    lf_a_ADP = np.array([lf_dict[0][cid]["ADP"][CoI].dropna().iloc[-1] for cid in range(len(lf_dict[0]))])
    lf_a_OP1 = np.array([lf_dict[0][cid]["OP1"][CoI].dropna().iloc[-1] for cid in range(len(lf_dict[0]))])
    lf_a_OP2 = np.array([lf_dict[0][cid]["OP2"][CoI].dropna().iloc[-1] for cid in range(len(lf_dict[0]))])

    # hf_DOE
    hf_w_ADP = np.array([hf_dict[0][cid]["ADP"][QoI].dropna().iloc[-1] for cid in range(len(hf_dict[0]))])
    hf_w_OP = np.array([0.5 * (hf_dict[0][cid]["OP1"][QoI].dropna().iloc[-1]+ hf_dict[0][cid]["OP2"][QoI].dropna().iloc[-1])
        for cid in range(len(hf_dict[0]))])

    # CoIs
    hf_a_ADP = np.array([hf_dict[0][cid]["ADP"][CoI].dropna().iloc[-1] for cid in range(len(hf_dict[0]))])
    hf_a_OP1 = np.array([hf_dict[0][cid]["OP1"][CoI].dropna().iloc[-1] for cid in range(len(hf_dict[0]))])
    hf_a_OP2 = np.array([hf_dict[0][cid]["OP2"][CoI].dropna().iloc[-1] for cid in range(len(hf_dict[0]))])

#     # addition of lf baseline results
#     # QoI
#     x_lf = np.vstack([x_lf, np.zeros(x_lf.shape[-1])])
#     lf_w_ADP = np.append(lf_w_ADP, sm_config["optim"]["bsl_lf_w_ADP"])
#     lf_w_OP = np.append(lf_w_OP, sm_config["optim"]["bsl_lf_w_OP"])
#
#     # CoI
#     lf_a_ADP = np.append(lf_a_ADP, sm_config["optim"]["bsl_lf_a_ADP"])
#     lf_a_OP1 = np.append(lf_a_OP1, sm_config["optim"]["bsl_lf_a_OP1"])
#     lf_a_OP2 = np.append(lf_a_OP2, sm_config["optim"]["bsl_lf_a_OP2"])
#
#     # addition of hf baseline results
#
#     # QoI
#     x_hf = np.vstack([x_hf, np.zeros(x_hf.shape[-1])])
#     hf_w_ADP = np.append(hf_w_ADP, sm_config["optim"]["bsl_hf_w_ADP"])
#     hf_w_OP = np.append(hf_w_OP, sm_config["optim"]["bsl_hf_w_OP"])
#
#     # CoI
#     hf_a_ADP = np.append(hf_a_ADP, sm_config["optim"]["bsl_hf_a_ADP"])
#     hf_a_OP1 = np.append(hf_a_OP1, sm_config["optim"]["bsl_hf_a_OP1"])
#     hf_a_OP2 = np.append(hf_a_OP2, sm_config["optim"]["bsl_hf_a_OP2"])

    # training loss model
    print("MFSM: training model(s)..")
    set_mo_DOE(model_total, x_lf=x_lf, y_lf=[lf_w_ADP, lf_w_OP, lf_a_ADP, lf_a_OP1, lf_a_OP2], 
                            x_hf=x_hf, y_hf=[hf_w_ADP, hf_w_OP, hf_a_ADP, hf_a_OP1, hf_a_OP2])
    model_total.train()
    print(f"MFSM: finished training loss model after {time.time() - t0} seconds")
    # saves mf-sm
    with open(os.path.join(outdir, "model_baseline.pkl"), "wb") as handle:
        pickle.dump(model_total, handle)
    with open(os.path.join(outdir, "model.pkl"), "wb") as handle:
        pickle.dump(model_total, handle)
    print(f"MFSM: model saved to {outdir}")

    # MFSM based optimization with adaptive infill
    print("MFSM: surrogate model based optimization..")
    nite = sm_config["optim"]["infill_nb"]
    infill_size = sm_config["optim"]["infill_lf_size"]
    infill_nb_gen = sm_config["optim"]["infill_nb_gen"]
    infill_regularization = sm_config["optim"].get("regularization", False)
    
    for ite in range(nite):
        outdir_ite = os.path.join(outdir, outdir.split("/")[-1] + f"_{ite}")
        print(os.path.join(outdir, f"model_{ite}.pkl"))
        print(os.path.isfile(os.path.join(outdir, f"model_{ite}.pkl")))
        if os.path.isfile(os.path.join(outdir, f"model_{ite}.pkl")):
            with open(os.path.join(outdir, f"model.pkl"), "rb") as f:
                model_total = pickle.load(f)
            print(f"model_{ite}.pkl exists, therefore model.pkl has been loaded!")

        else:

            # compute Bayesian infill
            x_lf_infill = compute_bayesian_infill(
                model_total,
                ffd,
                infill_size,
                infill_nb_gen,
                infill_regularization,
                n_design,
                bound,
                seed,
            )

            # execute hf infill

            x_hf_infill = x_lf_infill[0]
            y_hf_infill = execute_infill(
                x_hf_infill,
                config=sm_config["optim"]["hf_config"],
                n_design=sm_config["optim"]["n_design"],
                outdir=outdir_ite,
                ite=ite,
                fidelity="high",
                QoI=QoI,
                CoI=CoI
            )

            print(f"MFSM: hf infill candidate {x_hf_infill} with fitness {y_hf_infill}")
            
            # execute lf infill
            y_lf_infill = execute_infill(
                x_lf_infill,
                config=sm_config["optim"]["lf_config"],
                n_design=sm_config["optim"]["n_design"],
                outdir=outdir_ite,
                ite=ite,
                fidelity="low",
                QoI=QoI,
                CoI=CoI
            )
            print(f"MFSM: lf infill candidate {x_lf_infill} with fitness {y_lf_infill}")

            # update model
            set_mo_DOE(model_total, x_lf_infill, y_lf_infill, x_hf_infill, y_hf_infill)
            model_total.train()
            # save updated model
            mv_filelist([os.path.join(outdir, "model.pkl")],
                        [os.path.join(outdir, f"model_{ite}.pkl")])

        with open(os.path.join(outdir, "model.pkl"), "wb") as handle:
            pickle.dump(model_total, handle)
        print(f"MFSM: surrogate model based optimization {ite + 1}/{nite}"
              f" finished after {time.time() - t0} seconds")
        
    # save datasets and final loss model
    save_results(model_total, outdir)


if __name__ == '__main__':
    main()
