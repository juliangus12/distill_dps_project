import numpy as np
from scipy.optimize import bisect
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators import hv  # Hypervolume indicator
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX  # Simulated Binary Crossover
from pymoo.operators.mutation.pm import PM  # Polynomial Mutation
from pymoo.termination import get_termination
import pickle  # For saving state-action pairs

# --------------------------- Global Variables ---------------------------- #

# Number of years for lake model simulation
nYears = 100

# Number of different random samples used to simulate natural phosphorus inflows
nSamples = 100

# Inertia threshold for changes in pollution control policy (control action Y)
inertia_threshold = -0.02

# Minimum reliability threshold, this value must be greater than 85%
reliability_threshold = 0.85

# Lake model parameters
b = 0.42  # Decay rate of phosphorus in the lake (natural loss)
q = 2  # Recycling rate of phosphorus in the lake
alpha = 0.4  # Weight given to economic benefits from pollution control actions
delta = 0.98  # Discount factor applied to future benefits (for economic evaluation)

# Number of Radial Basis Functions (RBFs) used in the policy definition
n = 2  

# Number of decision variables: Each RBF has 3 variables (center, radius, weight)
nvars = n * 3

# Number of objectives we are optimizing: Economic benefit, phosphorus, inertia, reliability
nobjs = 4

# One constraint on the reliability, it must be > 85%
nconstrs = 1

# Matrix to hold pre-computed natural phosphorus inflow data for different states of the world (SOW)
nat_flowmat = np.zeros((10000, nYears))

# -------------------------- Helper Functions ----------------------------- #

# Define the root function for calculating critical phosphorus concentration (pCrit)
# This function comes from the lake model and helps determine the critical threshold
def root_function(x, q, b):
    return (x**q) / (1 + x**q) - b * x

# Use bisection method to solve for pCrit: The critical phosphorus level where the lake can flip to a polluted state
pCrit = bisect(root_function, 0.01, 1.0, args=(q, b))

# This function computes the policy using Radial Basis Functions (RBFs) to decide pollution control actions (Y)
# based on the current lake phosphorus state (lake_state) and the RBF parameters (C, R, W)
def RBFpolicy(lake_state, C, R, W):
    """Implements the RBF policy calculation as described."""
    Y = 0  # Initialize the pollution control action Y to zero
    for i in range(len(C)):  # Loop over each RBF
        r_value = np.isscalar(R[i]) if isinstance(R[i], np.ndarray) else R[i]
        if r_value > 0:  # Avoid division by zero
            # Compute RBF contribution to the pollution control action Y
            Y += W[i] * ((abs(lake_state - C[i]) / r_value) ** 3)
    # Ensure that Y is constrained between 0.01 and 0.1
    return max(0.01, min(0.1, Y))

# -------------------------- Lake Model Problem Class -------------------------- #

# Define the lake model as a multi-objective optimization problem
class LakeProblem(Problem):

    def __init__(self):
        # Initialize the problem with the number of variables, objectives, and constraints
        super().__init__(n_var=nvars,
                         n_obj=nobjs,
                         n_constr=nconstrs,
                         # Lower bounds for RBF parameters
                         xl=np.array([-2.0] * n + [0.0] * n + [0.0] * n),
                         # Upper bounds for RBF parameters
                         xu=np.array([2.0] * n + [2.0] * n + [1.0] * n))

    # Define the function that evaluates the objective values and constraints for the population
    def _evaluate(self, X, out, *args, **kwargs):
        # X is the population of individuals, where each individual contains decision variables
        F = []  # Store objectives for each individual
        G = []  # Store constraint violations for each individual
        state_action_pairs = []  # To store state-action pairs for distillation

        # Loop over each individual (a set of decision variables)
        for individual in X:
            # Extract the RBF parameters (centers C, radii R, weights W) from the individual
            C = individual[0:n]
            R = individual[n:2*n]
            W = individual[2*n:3*n]
            # Normalize the weights to ensure they sum to 1
            W = W / np.sum(W) if np.sum(W) != 0 else np.full(n, 1.0/n)

            # Initialize arrays to store results for each sample
            average_annual_P = np.zeros(nYears)  # Average phosphorus concentration
            discounted_benefit = np.zeros(nSamples)  # Economic benefits
            yrs_inertia_met = np.zeros(nSamples)  # Number of years inertia threshold is met
            yrs_pCrit_met = np.zeros(nSamples)  # Number of years lake phosphorus is below pCrit

            # Loop over the samples to simulate different states of the world
            for s in range(nSamples):
                nat_flow = nat_flowmat[s, :]  # Natural phosphorus inflow for the sample
                lake_state = np.zeros(nYears + 1)  # Initialize lake phosphorus concentration
                Y = np.zeros(nYears)  # Initialize pollution control actions
                Y[0] = RBFpolicy(lake_state[0], C, R, W)  # Calculate the first pollution control action

                # Simulate the lake for each year
                for i in range(nYears):
                    # Update lake phosphorus concentration using the model
                    lake_state[i + 1] = lake_state[i] * (1 - b) + \
                                        (lake_state[i]**q) / (1 + lake_state[i]**q) + \
                                        Y[i] + nat_flow[i]
                    # Track the phosphorus concentration over time
                    average_annual_P[i] += lake_state[i + 1] / nSamples
                    # Compute discounted economic benefits over time
                    discounted_benefit[s] += alpha * Y[i] * (delta ** i)

                    # Log the state-action pair for distillation
                    state_action_pairs.append((lake_state[i], Y[i]))  # (state, action)

                    # Check if the inertia threshold is violated
                    if i >= 1 and (Y[i] - Y[i - 1]) > inertia_threshold:
                        yrs_inertia_met[s] += 1

                    # Check if phosphorus level is below the critical threshold
                    if lake_state[i + 1] < pCrit:
                        yrs_pCrit_met[s] += 1

                    # Compute the next pollution control action
                    if i < (nYears - 1):
                        Y[i + 1] = RBFpolicy(lake_state[i + 1], C, R, W)

            # Compute the four objectives: negative economic benefit, phosphorus concentration, inertia, reliability
            objs = [-np.mean(discounted_benefit), 
                    np.max(average_annual_P),
                    -np.mean(yrs_inertia_met) / (nYears - 1),
                    -np.mean(yrs_pCrit_met) / nYears]

            # Constraint: Reliability must be greater than the threshold
            constrs = [max(0.0, reliability_threshold - (-objs[3]))]

            # Store objectives and constraints for this individual
            F.append(objs)
            G.append(constrs)

        # Set the objective and constraint arrays for the population
        out["F"] = np.array(F)
        out["G"] = np.array(G)

        # Save state-action pairs for distillation
        with open('state_action_pairs.pkl', 'wb') as f:
            pickle.dump(state_action_pairs, f)
# -------------------------- NSGA-II Optimization -------------------------- #

# Define the NSGA-II algorithm with the specified parameters
algorithm = NSGA2(
    pop_size=100,  # Population size of 100
    sampling=FloatRandomSampling(),  # Randomly sample initial solutions
    crossover=SBX(prob=0.9, eta=15),  # Simulated binary crossover
    mutation=PM(eta=20),  # Polynomial mutation
    eliminate_duplicates=True  # Eliminate duplicate solutions
)

# Terminate after 250 generations
termination = get_termination("n_gen", 250)

# Instantiate the lake problem
problem = LakeProblem()

# Run the optimization using NSGA-II
res = minimize(problem,
               algorithm,
               termination,
               seed=1,  # Random seed for reproducibility
               save_history=True,
               verbose=True)

# -------------------------- Hypervolume Calculation -------------------------- #

# Compute the hypervolume of the solutions using the reference point [1.0, 1.0, 1.0, 1.0]
hv_indicator = hv.HV(ref_point=np.array([1.0, 1.0, 1.0, 1.0]))
# Print the final hypervolume value
print("Hypervolume:", hv_indicator.do(res.F))
print("State-action pairs saved to 'state_action_pairs.pkl'.")

