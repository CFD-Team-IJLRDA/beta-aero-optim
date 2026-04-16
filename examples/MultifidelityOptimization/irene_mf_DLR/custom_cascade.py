import dill as pickle
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

from aero_optim.mf_sm.mf_infill import compute_pareto
from aero_optim.simulator.simulator import Simulator
from aero_optim.utils import check_file

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cascade_mf.custom_cascade import get_hf_DOE # noqa
from cascade_wolf_base.custom_cascade_wolf import CustomEvolution as WolfCustomEvolution # noqa
from cascade_wolf_base.custom_cascade_wolf import CustomOptimizer as WolfCustomOptimizer # noqa

logger = logging.getLogger()


class CustomSimulator(Simulator):
    """
    Custom class similar to the one defined in custom_cascade.py of cascade_mf.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.model_loss = self.set_model(config["simulator"]["model_loss_file"])
        self.model_angle = self.set_model(config["simulator"]["model_angle_file"])

    def process_config(self):
        logger.info("processing config..")
        if "model_loss_file" not in self.config["simulator"]:
            raise Exception(f"ERROR -- no <model_loss_file> entry in {self.config['simulator']}")
        if "model_angle_file" not in self.config["simulator"]:
            raise Exception(f"ERROR -- no <model_angle_file> entry in {self.config['simulator']}")

    def set_solver_name(self):
        self.solver_name = self.config["optim"]["model_name"]

    def set_model(self, model_file: str):
        check_file(model_file)
        with open(model_file, "rb") as handle:
            model = pickle.load(handle)
        return model

    def execute_sim(self, candidates: list[float] | np.ndarray, gid: int = 0):
        logger.info(f"execute simulations g{gid} with {self.solver_name}")
        QoI: list[np.ndarray] = self.model_loss.evaluate(np.array(candidates))
        CoI: list[np.ndarray] = self.model_angle.evaluate(np.array(candidates))
        self.df_dict[gid] = {
            cid: pd.DataFrame(
                {"loss_ADP": QoI[0][cid],
                 "loss_OP": QoI[1][cid],
                 "angle_ADP": CoI[0][cid],
                 "angle_OP1": CoI[1][cid],
                 "angle_OP2": CoI[2][cid]}) for cid in range(len(candidates))}


class CustomOptimizer(WolfCustomOptimizer):
    """
    Mix of custom_cascade.py of cascade_mf and custom_cascade_wolf.py of cascade_wolf_base
    """
    def set_gmsh_mesh_class(self):
        self.MeshClass = None

    def set_inner(self):
        super().set_inner()
        self.bsl_w_ADP = self.config["optim"].get("bsl_hf_w_ADP")
        self.bsl_w_OP = self.config["optim"].get("bsl_hf_w_OP")

    def _evaluate(self, X: np.ndarray, out: np.ndarray, *args, **kwargs):
        """
        Slightly modified version of cascade_wolf_base.custom_cascade_wolf.CustomOptimizer._evaluate
        to handle the fact that the surrogate directly predicts w_OP.
        """
        gid = self.gen_ctr
        self.feasible_cid[gid] = []

        # compute candidates geometric constraints and execute feasible candidates only
        geom_constraints = self.execute_constrained_candidates(X, gid)

        # update candidates fitness
        # Note: this time only the first value in the dataframe should be read
        for cid in range(len(X)):
            if cid in self.feasible_cid[gid]:
                logger.debug(f"gid, cid: {gid}, {cid}, "
                             f"df_dict[gid][cid]: {self.simulator.df_dict[gid][cid]}")
                loss_ADP = self.simulator.df_dict[gid][cid]["loss_ADP"].iloc[0]
                loss_OP = self.simulator.df_dict[gid][cid]["loss_OP"].iloc[0]
                logger.info(f"g{gid}, c{cid}: w_ADP = {loss_ADP}, w_OP = {loss_OP}")
                self.J.append([loss_ADP, loss_OP])
            else:
                self.J.append([float("nan"), float("nan")])

        # compute candidates angle constraints
        if not self.constraint:
            angle_constraints = [[-1.] * 3 for _ in range(len(X))]
        else:
            angle_constraints = []
            for cid in range(len(X)):
                outflow_angle_ADP = self.simulator.df_dict[gid][cid]["angle_ADP"].iloc[0]
                outflow_angle_OP1 = self.simulator.df_dict[gid][cid]["angle_OP1"].iloc[0]
                outflow_angle_OP2 = self.simulator.df_dict[gid][cid]["angle_OP2"].iloc[0]
                angle_constraints.append(
                    [self.angle_ADP[0] - outflow_angle_ADP if outflow_angle_ADP < self.angle_ADP[0]
                     else outflow_angle_ADP - self.angle_ADP[1],
                     self.angle_OP1[0] - outflow_angle_OP1 if outflow_angle_OP1 < self.angle_OP1[0]
                     else outflow_angle_OP1 - self.angle_OP1[1],
                     self.angle_OP2[0] - outflow_angle_OP2 if outflow_angle_OP2 < self.angle_OP2[0]
                     else outflow_angle_OP2 - self.angle_OP2[1]]
                )
                logger.debug(f"g{gid}, c{cid} ADP outflow angle: ({outflow_angle_ADP})")
                if angle_constraints[-1][0] > 0:
                    logger.info(f"g{gid}, c{cid} ADP outflow angle: constraint violation")
                logger.debug(f"g{gid}, c{cid} OP1 outflow angle: ({outflow_angle_OP1})")
                if angle_constraints[-1][1] > 0:
                    logger.info(f"g{gid}, c{cid} OP1 outflow angle: constraint violation")
                logger.debug(f"g{gid}, c{cid} OP2 outflow angle: ({outflow_angle_OP2})")
                if angle_constraints[-1][2] > 0:
                    logger.info(f"g{gid}, c{cid} OP2 outflow angle: constraint violation")

        out["F"] = np.vstack(self.J[-self.doe_size:])
        self._observe(out["F"])
        out["G"] = np.column_stack([geom_constraints, np.vstack(angle_constraints)])
        self.gen_ctr += 1

    def execute_constrained_candidates(self, candidates: np.ndarray, gid: int) -> np.ndarray:
        """
        **Executes** feasible candidates only and **waits** for them to finish.
        """
        logger.info(f"evaluating candidates of generation {self.gen_ctr}..")
        self.ffd_profiles.append([])
        self.inputs.append([])
        constraint = []
        for cid, cand in enumerate(candidates):
            self.inputs[gid].append(np.array(cand))
            ffd_file, ffd_profile = self.deform(cand, gid, cid)
            self.ffd_profiles[gid].append(ffd_profile)
            logger.info(f"candidate g{gid}, c{cid} constraint computation..")
            constraint.append(self.apply_candidate_constraints(ffd_profile, gid, cid))
            # only mesh and execute feasible candidates
            if len([v for v in constraint[cid] if v > 0.]) == 0:
                self.feasible_cid[gid].append(cid)
            else:
                logger.info(f"unfeasible candidate g{gid}, c{cid}")

        self.execute_candidates(candidates, gid)
        return np.vstack(constraint)

    def execute_candidates(self, candidates, gid: int):
        logger.info(f"evaluating candidates of generation {gid}..")
        self.simulator.execute_sim(candidates, gid)

    def _observe(self, pop_fitness: np.ndarray):
        """
        **Plots** some results each time a generation has been evaluated:</br>
        > the simulations residuals,</br>
        > the candidates fitnesses,</br>
        > the baseline and deformed profiles.
        """
        gid = self.gen_ctr

        # plot settings
        baseline: np.ndarray = self.ffd.pts
        profiles: list[np.ndarray] = self.ffd_profiles[gid]
        cmap = mpl.colormaps[self.cmap].resampled(self.doe_size)
        colors = cmap(np.linspace(0, 1, self.doe_size))
        # subplot construction
        fig = plt.figure(figsize=(16, 16))
        ax1 = plt.subplot(2, 1, 1)  # profiles
        ax2 = plt.subplot(2, 1, 2)  # fitness (loss_ADP vs loss_OP)
        plt.subplots_adjust(wspace=0.25)
        ax1.plot(baseline[:, 0], baseline[:, 1], color="k", lw=2, ls="--", label="baseline")
        # loop over candidates through the last generated profiles
        logger.debug(f"pop fitness: {pop_fitness} and shape {pop_fitness.shape}")
        for cid in self.feasible_cid[gid]:
            ax1.plot(profiles[cid][:, 0], profiles[cid][:, 1], color=colors[cid], label=f"c{cid}")
            ax2.scatter(pop_fitness[cid, 0], pop_fitness[cid, 1],
                        color=colors[cid], label=f"c{cid}")
        ax2.scatter(self.bsl_w_ADP, self.bsl_w_OP, marker="*", color="red", label="baseline")
        # legend and title
        fig.suptitle(
            f"Generation {gid} results", size="x-large", weight="bold", y=0.93
        )
        # top
        ax1.set_title("FFD profiles", weight="bold")
        ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        # bottom
        ax2.set_title(f"{self.QoI} ADP vs {self.QoI} OP", weight="bold")
        ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax2.set_xlabel('$w_\\text{ADP}$')
        ax2.set_ylabel('$w_\\text{OP}$')
        # save figure as png
        fig_name = f"pymoo_g{gid}.png"
        logger.info(f"saving {fig_name} to {self.figdir}")
        plt.savefig(os.path.join(self.figdir, fig_name), bbox_inches='tight')
        plt.close()

    def final_observe(self, best_candidates: np.ndarray):
        """
        **Plots** convergence progress by plotting the fitness values
        obtained with the successive generations.
        """
        logger.info(f"plotting populations statistics after {self.gen_ctr} generations..")

        # plot construction
        _, ax = plt.subplots(figsize=(8, 8))
        gen_fitness = np.vstack(self.J)

        # plotting data: last gen. pareto, y_hf infill, hf baseline
        nsga = compute_pareto(gen_fitness[-self.doe_size:, 0], gen_fitness[-self.doe_size:, 1])
        ax.plot(nsga[:, 0], nsga[:, 1],
                color="forestgreen", label=f"g{len(gen_fitness) // self.doe_size - 1}")
        w_ADP_hf, w_OP_hf = get_hf_DOE(self.simulator.model_loss)
        ax.scatter(w_ADP_hf, w_OP_hf,
                   marker="s", color="k", facecolors="none", label="hf DOE & infill")
        ax.scatter(self.bsl_w_ADP, self.bsl_w_OP, marker="*", color="red", label="hf baseline")
        ax.plot()
        ax.set_axisbelow(True)
        plt.grid(True, color="grey", linestyle="dashed")

        # legend and title
        ax.set_title(f"Optimization evolution ({self.gen_ctr} g. x {self.doe_size} c.)")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_xlabel('$w_\\text{ADP}$')
        ax.set_ylabel('$w_\\text{OP}$')

        # save figure as png
        fig_name = f"pymoo_optim_g{self.gen_ctr}_c{self.doe_size}.png"
        logger.info(f"saving {fig_name} to {self.outdir}")
        plt.savefig(os.path.join(self.outdir, fig_name), bbox_inches='tight')
        plt.close()


class CustomEvolution(WolfCustomEvolution):
    """
    Same custom class as the one defined in cascade_wolf and cascade_adap.
    """
