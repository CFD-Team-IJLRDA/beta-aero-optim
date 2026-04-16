
import numpy as np
import pandas as pd
from scipy.stats            import norm
from scipy.spatial.distance import cdist
from pymoo.core.problem import Problem

from aero_optim.mf_sm.mf_infill import (compute_pareto, AcquisitionFunctionProblem, optimize_acquisition_function)
from aero_optim.mf_sm.mf_models import MultiObjectiveModel

# Configurazione feasibility thresholds (modificabili manualmente)
FEASIBILITY_BOUNDS = {
    'ADP': 0.8,  # theta_ADP bounds
    'OP1': 1.5,      # theta_OP1 bounds  
    'OP2': 1.5       # theta_OP2 bounds
}

def probability_feasibility(x: np.ndarray, model: MultiObjectiveModel) -> np.ndarray:
    """
    Computes the probability of feasibility for a Gaussian variable.
    P(u <= threshold) = 1 - P(u > threshold) = 1 - Phi((u - threshold) / std)
    where Phi is the CDF of the standard normal distribution.
    """

    u_beta_ADP = model.models[2].evaluate(x)
    u_beta_OP1 = model.models[3].evaluate(x)
    u_beta_OP2 = model.models[4].evaluate(x)

    std_beta_ADP = model.models[2].evaluate(x) + 1e-6
    std_beta_OP1 = model.models[3].evaluate(x) + 1e-6
    std_beta_OP2 = model.models[4].evaluate(x) + 1e-6

    prob_feas_ADP = 1-2.0*norm.cdf((u_beta_ADP+FEASIBILITY_BOUNDS["ADP"])/std_beta_ADP)
    prob_feas_OP1 = 1-2.0*norm.cdf((u_beta_OP1+FEASIBILITY_BOUNDS["OP1"])/std_beta_OP1)
    prob_feas_OP2 = 1-2.0*norm.cdf((u_beta_OP2+FEASIBILITY_BOUNDS["OP2"])/std_beta_OP2)


    return (prob_feas_ADP*prob_feas_OP1*prob_feas_OP2).flatten()

def MPI_acquisition_function_constrained(x: np.ndarray, model: MultiObjectiveModel) -> np.ndarray:
    """
    Bi-objective Minimal Probability of Improvement:
    see A. A. Rahat (2017): https://dl.acm.org/doi/10.1145/3071178.3071276
    """
    assert model.models[0].y_hf_DOE is not None
    assert model.models[1].y_hf_DOE is not None
    pareto = compute_pareto(model.models[0].y_hf_DOE, model.models[1].y_hf_DOE)

    u1 = model.models[0].evaluate(x)
    u2 = model.models[1].evaluate(x)
    std1 = model.models[0].evaluate_std(x) + 1e-6
    std2 = model.models[1].evaluate_std(x) + 1e-6

    MPI = np.min(np.column_stack(
        [1 - norm.cdf((u1 - pp[0]) / std1) * norm.cdf((u2 - pp[1]) / std2)
            for pp in pareto]
    ), axis=1)
    
    return MPI*probability_feasibility(x, model)

def maximize_MPI_BO_constrained(
        model: MultiObjectiveModel,
        n_var: int, bound: list, seed: int, n_gen: int = 100
) -> np.ndarray:
    """
    Bi-objective Minimal Probability of Improvement maximization.
    """
    problem = AcquisitionFunctionProblem(MPI_acquisition_function_constrained, model, n_var, bound, min=False)
    return optimize_acquisition_function(problem, seed, n_gen=n_gen).X

def LCB_acquisition_function_constrained(x: np.ndarray, model: MultiObjectiveModel, idx: int, alpha: float = 1) -> np.ndarray:
    """
    Lower Confidence Bound acquisition function.
    """
    return (model.models[idx].evaluate(x) - alpha * model.models[idx].evaluate_std(x)).flatten()*(1-probability_feasibility(x, model))

def minimize_LCB_constrained(
        model: MultiObjectiveModel,
        idx: int,
        n_var: int, bound: list, seed: int, n_gen: int = 100
) -> np.ndarray:
    """
    Lower Confidence Bound minimization function.
    """
    print(idx)
    lcb_func = lambda x,_: LCB_acquisition_function_constrained(x, model, idx)
    problem = AcquisitionFunctionProblem(lcb_func, model, n_var, bound)
    return optimize_acquisition_function(problem, seed, n_gen=n_gen).X

def ED_acquisition_function_constrained(x: np.ndarray, DOE: np.ndarray) -> np.ndarray:
    """
    Euclidean Distance:
    see X. Zhang et al. (2021): https://doi.org/10.1016/j.cma.2020.113485
    """
    f1 = np.min(cdist(np.atleast_2d(x), DOE, "euclidean"), axis=1)
    f1 = np.expand_dims(f1, axis=1)
    assert f1.shape == (x.shape[0], 1), f"f1 shape {f1.shape} x shape {x.shape}"
    return f1
 
class EDProblem_constrained(Problem):
    
    """
    Euclidean Distance problem.
    """
    def __init__(self, DOE: np.ndarray, n_var: int, bound: list):
        super().__init__(n_var=n_var, n_obj=1, xl=bound[0], xu=bound[-1])
        self.DOE = DOE
        # print(DOE)
        # print(DOE.shape)

    def _evaluate(self, x: np.ndarray, out: np.ndarray, *args, **kwargs):
        out["F"] = -ED_acquisition_function_constrained(x, self.DOE)


def maximize_ED_constrained(
        DOE: np.ndarray, n_var: int, bound: list, seed: int, n_gen: int = 100
) -> np.ndarray:
    """
    Euclidean distance maximization function.

    Inputs:
        DOE (np.ndarray): the full low- and high-fidelity DOE.
    """
    problem = EDProblem_constrained(DOE, n_var=n_var, bound=bound)
    return optimize_acquisition_function(problem, seed, n_gen=n_gen).X
