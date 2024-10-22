import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV  # Hypervolume indicator
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import FloatRandomSampling  # Add this
import pickle
from LakeProblem import LakeProblem  # Import your custom lake problem

# NSGA-II algorithm setup
algorithm = NSGA2(  
    pop_size=100, 
    sampling=FloatRandomSampling(),  # Use pymoo's random sampling
    crossover='real_sbx', 
    mutation='real_pm', 
    eliminate_duplicates=True
)

# Number of generations after which NSGA-II terminates 
termination = get_termination("n_gen", 1)

# Define the LakeProblem class for the optimization problem
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
