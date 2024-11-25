import math
import random
import platypus
import numpy as np
import shelve
from rhodium import *
from pareto import *
from scipy.optimize import brentq as root

# Define constants for the lake model
num_years = 100  # Number of years in the simulation
num_samples = 100  # Number of samples for Monte Carlo simulations
inertia_threshold = -0.02  # Threshold for inertia objective
reliability_threshold = 0.85  # Minimum reliability constraint
num_rbfs = 2  # Number of Radial Basis Functions (RBFs)
num_vars = num_rbfs * 3  # Total number of RBF-related variables

# Define model parameters
benefit_scale = 0.4  # Scaling factor for economic benefit (alpha)
discount_rate = 0.98  # Discount rate for economic benefits (delta)
phosphorus_loss_rate = 0.42  # Lake phosphorus loss rate (b)
phosphorus_recycling_rate = 2.0  # Lake phosphorus recycling rate (q)
inflow_mean = 0.03  # Mean of natural inflows (mu)
inflow_variance = 10e-5  # Variance of natural inflows (sigma)

class CubicDPSLever(Lever):
    def __init__(self, name, num_rbfs = num_rbfs, center_bounds=(0, 1), radius_bounds=(0, 1)):
        super().__init__(name)
        self.num_rbfs = num_rbfs
        self.center_bounds = center_bounds
        self.radius_bounds = radius_bounds

    def to_variables(self):
        """
        Converts the RBF parameters to decision variables for optimization.
        
        Returns:
            list: Decision variables for the optimization algorithm.
        """
        vars_list = []
        for _ in range(self.num_rbfs):
            vars_list += [platypus.Real(self.center_bounds[0], self.center_bounds[1])]  # Center
            vars_list += [platypus.Real(self.radius_bounds[0], self.radius_bounds[1])]  # Radius
            vars_list += [platypus.Real(0, 1)]  # Weight
        return vars_list

    def from_variables(self, variables):
        """
        Converts decision variables back into a dictionary format for RBF policy parameters.

        Parameters:
            variables (list): List of decision variables representing RBF parameters.

        Returns:
            dict: A dictionary containing the RBF parameters for the policy.
        """
        policy = {"num_rbfs": self.num_rbfs, "rbf_params": []}
        for i in range(self.num_rbfs):
            policy["rbf_params"].append({
                "center": variables[i * 3 + 0],
                "radius": variables[i * 3 + 1],
                "weight": variables[i * 3 + 2]
            })
        weight_sum = sum([rbf["weight"] for rbf in policy["rbf_params"]])
        for rbf in policy["rbf_params"]:
            rbf["weight"] /= weight_sum  # Normalize weights to sum to 1
        return policy

class DPSExpertPolicy:
    def __init__(self, C, R, W, observation_space, action_space):
        self.vars = []
        for c, r, w in zip(C, R, W):
            self.vars.extend([c, r, w])
        self.observation_space = observation_space
        self.action_space = action_space

    def predict(self, obs, state=None, mask=None, deterministic=False):
        """
        Predict method compatible with Stable Baselines 3 policies.
        """
        # Ensure obs is a NumPy array
        obs = np.array(obs)
        #print(f"DEBUG: DPSExpertPolicy.predict received obs={obs}, shape={obs.shape}, state={state}")

        # Handle edge cases for shapes
        if obs.ndim == 2 and obs.shape[1] == 1:
            obs = obs.flatten()  # Convert (N, 1) -> (N,)
        elif obs.ndim == 1 and obs.shape == (1,):
            obs = obs[0]  # Convert (1,) -> scalar

        # Debugging corrected shape
        #print(f"DEBUG: Corrected observation shape: {obs.shape}")

        # Single observation: compute action
        if obs.ndim == 0:  # Scalar case
            action = compute_policy_output(obs, self.vars)
            #print(f"DEBUG: Single action computed: {action}")
            return np.array([action]), None  # Return consistent array
        elif obs.ndim == 1:  # Batch case
            actions = np.array([compute_policy_output(o, self.vars) for o in obs])
            #print(f"DEBUG: Batch actions computed: {actions}")
            return actions, None
        else:
            raise ValueError(f"Unexpected observation shape after correction: {obs.shape}")



    def __call__(self, obs, state=None, mask=None):
        action, _ = self.predict(obs)
        return action


def compute_policy_output(lake_state, vars):
    """
    Computes the policy output Y based on the current lake state and policy parameters.

    Parameters:
        lake_state (float): Current phosphorus level in the lake.
        vars (list): Flattened list of RBF parameters [C1, R1, W1, C2, R2, W2, ...].

    Returns:
        float: The controlled phosphorus inflow based on the RBF policy.
    """
    n = len(vars) // 3  # Number of RBFs
    m = 1  # Single input dimension for lake state
    C = np.zeros([n, m])
    B = np.zeros([n, m])
    W = np.zeros(n)

    # Extract centers, radii, and weights from vars
    for i in range(n):
        W[i] = vars[(i + 1) * (2 * m + 1) - 1]
        for j in range(m):
            C[i, j] = vars[(2 * m + 1) * i + 2 * j]
            B[i, j] = vars[(2 * m + 1) * i + 2 * j + 1]

    # Normalize weights to sum to 1
    total_weight = sum(W)
    if total_weight != 0.0:
        W = W / total_weight
    else:
        W = np.ones(n) / n  # Avoid division by zero

    # Evaluate the RBF policy
    Y = 0
    for i in range(n):
        for j in range(m):
            if B[i, j] != 0:
                Y += W[i] * ((abs(lake_state - C[i, j]) / B[i, j]) ** 3)

    # Constrain output
    return min(0.1, max(Y, 0.01))
# Function to retrieve main simulation constants for lake problem
def get_simulation_constants():
    """
    Returns the main constants for the lake problem simulation.

    Returns:
        tuple: (benefit_scale, discount_rate, phosphorus_loss_rate, phosphorus_recycling_rate, inflow_mean, inflow_variance)
    """
    return benefit_scale, discount_rate, phosphorus_loss_rate, phosphorus_recycling_rate, inflow_mean, inflow_variance

# Function to retrieve model parameters
def get_model_parameters():
    """
    Returns the model parameters for lake problem simulations.

    Returns:
        tuple: (num_years, num_samples, inertia_threshold, reliability_threshold, num_rbfs, num_vars)
    """
    return num_years, num_samples, inertia_threshold, reliability_threshold, num_rbfs, num_vars

def unpack_solutions(solutions_file):
    """
    Loads RBF parameters and objective values from a text file containing solutions and objectives.

    Parameters:
        solutions_file (str): Path to the text file containing Pareto solutions and objectives.

    Returns:
        tuple:
            np.ndarray: RBF parameter matrix (num_solutions x num_parameters).
            np.ndarray: Objective values matrix (num_solutions x num_objectives).
    """
    rbf_parameters = []
    objective_values = []
    reading_solutions = False
    reading_objectives = False

    with open(solutions_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            if line.startswith("Pareto Solutions:"):
                reading_solutions = True
                reading_objectives = False
                continue
            elif line.startswith("Pareto Objectives:"):
                reading_solutions = False
                reading_objectives = True
                continue

            # Parse solutions or objectives
            if reading_solutions:
                rbf_parameters.append([float(value) for value in line.split(",")])
            elif reading_objectives:
                objective_values.append([float(value) for value in line.split(",")])

    # Convert to numpy arrays
    rbf_parameters = np.array(rbf_parameters)
    objective_values = np.array(objective_values)

    return rbf_parameters, objective_values

def find_pareto_frontier(solutions, epsilons=None, filter=False):
    """
    Finds the Pareto-optimal solutions from a list of solutions.
    Optionally filters solutions for high economic benefit and reliability.

    Parameters:
        solutions (list): List of solutions from optimization.
        epsilons (list of float): Epsilon values for epsilon-nondominated sorting.
            If None, defaults to a very small epsilon for each objective.
        filter (bool): If True, filters the Pareto frontier for high economic benefit and reliability.

    Returns:
        tuple:
            np.ndarray: Array of Pareto-optimal RBF parameters.
            np.ndarray: Array of objective values for Pareto-optimal solutions.
    """
    rbf_parameters, objective_values = [], []

    # Process each solution to extract RBF parameters and objectives
    for solution in solutions:
        policy = solution["policy"]
        rbf_parameters.append(
            [param for rbf in policy["rbf_params"] for param in (rbf["center"], rbf["radius"], rbf["weight"])]
        )
        objective_values.append(
            [solution["avg_max_phosphorus"], solution["avg_utility"], solution["avg_inertia"], solution["avg_reliability"]]
        )

    rbf_parameters = np.array(rbf_parameters).T
    objective_values = np.array(objective_values)

    # Perform epsilon-nondominated sorting
    if epsilons is None:
        epsilons = [1e-9] * objective_values.shape[1]

    archive = Archive(epsilons)
    for objectives, parameters in zip(objective_values, rbf_parameters.T):
        archive.sortinto(objectives.tolist(), parameters.tolist())

    pareto_solutions = np.array([tagalong for tagalong in archive.tagalongs])
    pareto_objectives = np.array([obj for obj in archive.archive])

    # Apply filtering for high economic benefit and reliability
    if filter:
        best_indices = np.lexsort((-pareto_objectives[:, 1], -pareto_objectives[:, 3]))
        pareto_solutions = pareto_solutions[best_indices]
        pareto_objectives = pareto_objectives[best_indices]

    return pareto_solutions, pareto_objectives

# Function to calculate the critical phosphorus level (Pcrit)
def calculate_pcrit(loss_rate=phosphorus_loss_rate, recycle_rate=phosphorus_recycling_rate):
    """
    Calculate the critical phosphorus level (Pcrit) based on the recycling rate and loss rate.

    Parameters:
        loss_rate (float): Phosphorus loss rate.
        recycle_rate (float): Phosphorus recycling rate.

    Returns:
        float: The critical phosphorus level (Pcrit).
    """
    return root(lambda x: x**recycle_rate / (1 + x**recycle_rate) - loss_rate * x, 0.01, 1.5)

def generate_inflow_data(mean=inflow_mean, variance=inflow_variance, num_samples=num_samples, num_years=num_years, single=False):
    """
    Generates synthetic natural inflow data for the lake model using a log-normal distribution.

    Parameters:
        mean (float): Mean of the inflow distribution.
        variance (float): Variance of the inflow distribution.
        num_samples (int): Number of samples for Monte Carlo simulations (ignored if single=True).
        num_years (int): Number of years per sample (ignored if single=True).
        single (bool): If True, generates a single inflow value; otherwise, generates a list of lists.

    Returns:
        float or list: A single inflow value (if single=True) or a list of lists (if single=False).
    """
    sigma = math.sqrt(math.log(1 + variance / mean**2))
    mu = math.log(mean) - (sigma**2 / 2)

    if single:
        # Generate a single inflow value
        u1, u2 = random.random(), random.random()
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        inflow_value = math.exp(mu + sigma * z0)
        return inflow_value

    # Generate a list of lists for inflow data
    inflow_data = []
    for _ in range(num_samples):
        inflow = []
        for _ in range(num_years):
            u1, u2 = random.random(), random.random()
            z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            inflow_value = math.exp(mu + sigma * z0)
            inflow.append(inflow_value)
        inflow_data.append(inflow)

    return inflow_data

# Function to simulate lake dynamics with given policy
def lake_model(policy, loss_rate=phosphorus_loss_rate, recycle_rate=phosphorus_recycling_rate,
               mean=inflow_mean, variance=inflow_variance, benefit=benefit_scale,
               discount=discount_rate, num_samples=num_samples, num_steps=num_years):
    """
    Simulates the lake phosphorus dynamics with given parameters and policy.

    Parameters:
        policy (dict): DPS policy with RBF parameters.
        loss_rate, recycle_rate (float): Lake parameters for phosphorus dynamics.
        mean, variance (float): Mean and variance for natural inflows.
        benefit, discount (float): Parameters for economic utility.
        num_samples, num_steps (int): Monte Carlo sample size and simulation duration.

    Returns:
        tuple: (max_phosphorus, avg_utility, avg_inertia, avg_reliability)
    """
    critical_p = calculate_pcrit(loss_rate, recycle_rate)
    lake_state = [0] * num_steps
    avg_daily_p = [0] * num_steps
    total_reliability, total_utility, total_inertia = 0, 0, 0

    inflow_data = generate_inflow_data(mean, variance, num_samples, num_steps)

    for sample in inflow_data:
        lake_state[0] = 0
        emissions = [compute_policy_output(lake_state[0], policy) for _ in range(num_steps)]

        for step in range(1, num_steps):
            inflow = sample[step - 1]
            lake_state[step] = (
                (1 - loss_rate) * lake_state[step - 1] +
                lake_state[step - 1]**recycle_rate / (1 + lake_state[step - 1]**recycle_rate) +
                emissions[step - 1] + inflow
            )
            avg_daily_p[step] += lake_state[step] / num_samples

        total_reliability += sum(1 for p in lake_state if p < critical_p) / num_steps
        total_utility += sum(benefit * emissions[i] * discount**i for i in range(num_steps))
        total_inertia += sum(1 for i in range(1, num_steps) if emissions[i] - emissions[i - 1] > inertia_threshold) / (num_steps - 1)

    return (
        max(avg_daily_p),
        total_utility / num_samples,
        total_inertia / num_samples,
        total_reliability / num_samples
    )