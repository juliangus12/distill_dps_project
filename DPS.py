import os
import pickle
import numpy as np
from scipy.optimize import bisect
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV  # Hypervolume indicator
from pymoo.operators.sampling.rnd import FloatRandomSampling    
from pymoo.operators.crossover.sbx import SBX  # Simulated Binary Crossover
from pymoo.operators.mutation.pm import PM  # Polynomial Mutation
from pymoo.termination import get_termination
from LakeProblem import LakeProblem  # Import your custom lake problem

# -------------------------- NSGA-II Optimization -------------------------- #

algorithm = NSGA2(
    pop_size=100,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),  # Corrected to use SBX instance
    mutation=PM(eta=20),  # Corrected to use PM instance
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 100)
problem = LakeProblem()

# Perform the optimization
res = minimize(problem, algorithm, termination, seed=1, save_history=True, verbose=True)

# Calculate the hypervolume
hv_indicator = HV(ref_point=np.array([1.0, 1.0, 1.0, 1.0]))
best_hv = hv_indicator.do(res.F)

# Get the best solution and save the state-action pairs
best_solution = res.X[np.argmax(res.F[:, 0])]  # Assumes maximizing objective 0

# Save the RBF parameters to a file
C, R, W = best_solution[0:2], best_solution[2:4], best_solution[4:6]
with open('rbf_params.pkl', 'wb') as f:
    pickle.dump((C, R, W), f)

print(f"Best hypervolume: {best_hv}")
print("RBF parameters saved to 'rbf_params.pkl'")
