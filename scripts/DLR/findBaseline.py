import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from scipy.spatial.distance import cdist    
from pathlib import Path

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from aero_optim.ffd.ffd import DLR_POD_2D

# Define optimization problem
class ProfileMatchingProblem(Problem):
    def __init__(self, dlr_obj, baseline_profile, n_var):
        self.dlr = dlr_obj
        self.baseline_x = baseline_profile[:, 0]  # Target x coordinates
        self.baseline_y = baseline_profile[:, 1]  # Target y coordinates
        
        super().__init__(
            n_var=n_var,  # 5 design variables (Delta vector)
            n_obj=1,  # Single objective: minimize difference
            n_constr=0,  # No constraints
            xl=-10,  # Lower bounds
            xu=10    # Upper bounds
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate the objective for each candidate in X"""
        F = np.zeros((X.shape[0], 1))
        
        for i, delta in enumerate(X):
            try:
                # Apply FFD with current delta
                profile = self.dlr.apply_ffd(delta)[:-2, :]
                profile_x = profile[:, 0]
                profile_y = profile[:, 1]
                
                # Compute weighted mean squared error
                if len(profile_y) == len(self.baseline_y):
                    # Create weights: higher weight for x<10 and x>60
                    weights = np.ones(len(self.baseline_x))
                    # weights[(self.baseline_x < 10) | (self.baseline_x > 60)] = 1.0
                    
                    # Weighted MSE
                    squared_errors = (profile_y - self.baseline_y)**2 + (profile_x - self.baseline_x)**2
                    mse = np.mean(weights * squared_errors)
                else:
                    mse = 1e6  # Penalty for size mismatch
                
                F[i, 0] = np.log10(mse)
            except Exception as e:
                print(f"Error evaluating delta {delta}: {e}")
                F[i, 0] = 1e6  # Large penalty for failed evaluations
        
        out["F"] = F
        

design_sensitivity = {'BetaLE':[111.6727,131.6727],
    'BetaTE':[58.0394,70.0394],
    'BetaST':[88,92],
    'x2SS':[0.0122990,0.0522990],
    'x3SS':[0.31,0.40],
    'y3SS':[0.177428,0.217428],
    'x4SS':[0.81321,0.89321],
    'm2DS':[0.16946,0.239465],
    'd2DS':[0.035454,0.055454],
    'd3DS':[0.020002,0.080002],
    'd4DS':[0.0013360,0.0017360],
    'Dmax_approx':[0.8,1.2],
    'rTE':[0.004,0.01136]}

# design_sensitivity = {
#                 "BetaLE":[118,129],
#                 "BetaTE":[60,66],
#                 "BetaST":[88,92],
#                 "x2SS":[0.013,0.033],
#                 "x3SS":[0.31,0.40],
#                 "y3SS":[0.192,0.21],
#                 "x4SS":[0.818,0.86],
#                 "m2DS":[0.127,0.18],
#                 "d2DS":[0.035,0.048],
#                 "d3DS":[0.022,0.051],
#                 "d4DS":[0.014,0.017],
#                 "Dmax_approx":[0.8,1.1],
#                 "rTE":[0.00495,0.0080]},
    

baseline_file = '/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/RANS_bruteForce/ogv1c.dat'
# best_ADP = '/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_6/high_infill_6/FFD/ogv1c_g0_c0.dat'
# best_to  = '/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_0/high_infill_0/FFD/ogv1c_g0_c0.dat'
# best_OP  = '/home/mciarlatani/GPROfficial/beta-aero-optim/Optimization/irene_mf_DLR/output_paper/output_paper_2/high_infill_2/FFD/ogv1c_g0_c0.dat'
POD_DIR = '/home/mciarlatani/GPROfficial/beta-aero-optim/scripts/DLR/POD_Dataset_RANS/'

# baseline_file = best_ADP

seed =  123
n_modes = 5
    
baseline = np.genfromtxt(baseline_file, skip_header=2, skip_footer=1, delimiter=' ')

dlr = DLR_POD_2D(baseline_file,  POD_DIR, '/home/mciarlatani/bin/BladeGenerator.exe', design_sensitivity, n_modes, 1000, 123, scale = 1000, perturb_POD='Baseline')

# Run optimization
print("Starting optimization to match baseline profile...")
problem = ProfileMatchingProblem(dlr, baseline, n_modes)

algorithm = NSGA2(
    pop_size=300,
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 300)

res = minimize(
    problem,
    algorithm,
    termination,
    seed=seed,
    verbose=True
)

print("\nOptimization completed!")
print(f"Best Delta found: {res.X}")
print(f"MSE: {res.F[0]}")

# Get the best profile
best_delta = res.X
profile = dlr.apply_ffd(best_delta)[:-2, :]
# profile = dlr.apply_ffd([0,0,0,0,0])[:-2, :]

print(profile.shape)
print(baseline.shape)

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Profiles comparison
ax1.plot(profile[:,0], profile[:,1], 'o-', color='tab:blue', label='Optimized', linewidth=2)
ax1.plot(baseline[:,0], baseline[:,1], 'o-', color='tab:orange', label='Baseline', linewidth=2, alpha=0.7)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Profile Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Y-coordinate difference
if len(profile[:, 1]) == len(baseline[:, 1]):
    y_diff = profile[:, 1] - baseline[:, 1]
    ax2.plot(baseline[:, 0], y_diff, 'o-', color='red', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Y difference (Optimized - Baseline)')
    ax2.set_title(f'Y-coordinate Error (MSE = {res.F[0]:.6e})')
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()