import math
import random
import platypus
import numpy as np
import shelve
from rhodium import *
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

# Expert policy class
class DPSExpertPolicy:
    def __init__(self, C, R, W, observation_space, action_space):
        self.C = C
        self.R = R
        self.W = W
        self.observation_space = observation_space
        self.action_space = action_space

    def output(self, obs):
        lake_state = obs[0]
        action = compute_policy_output(lake_state, self.C, self.R, self.W)
        return np.array([[action]], dtype=np.float32), None 

    def __call__(self, obs, state=None, dones=None):
        actions, state = self.predict(obs)
        return actions, state
# Function to calculate critical phosphorus level (Pcrit)
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

# Function to generate synthetic natural inflow data using a log-normal distribution
def generate_inflow_data(mean=inflow_mean, variance=inflow_variance, num_samples=num_samples, num_years=num_years):
    """
    Generates synthetic natural inflow data for the lake model using a log-normal distribution.
    
    Parameters:
        mean (float): The desired mean of the real-space distribution.
        variance (float): The desired variance of the real-space distribution.
        num_samples (int): Number of samples to generate.
        num_years (int): Number of years per sample.
        
    Returns:
        list: Generated natural inflow data as a list of lists, each with `num_years` values.
    """
    # Calculate sigma and mu for the log-normal distribution
    sigma = math.sqrt(math.log(1 + variance / mean**2))
    mu = math.log(mean) - (sigma**2 / 2)

    # Generate `num_samples` sets of `num_years` inflow values using the Box-Muller transform
    inflow_data = []
    for _ in range(num_samples):
        inflow = []
        for _ in range(num_years):
            u1, u2 = random.random(), random.random()
            z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            inflow_value = math.exp(mu + sigma * z0)  # Transform to log-normal
            inflow.append(inflow_value)
        inflow_data.append(inflow)

    return inflow_data

# Function to compute policy output Y based on the current lake state and policy parameters
def compute_policy_output(lake_state, centers, radii, weights):
    """
    Computes the policy output Y based on the current lake state and policy parameters.
    
    Parameters:
        lake_state (float): The current phosphorus level in the lake.
        centers (list of float): Centers of the RBFs.
        radii (list of float): Radii of the RBFs.
        weights (list of float): Weights of the RBFs.

    Returns:
        float: The controlled phosphorus inflow based on the RBF policy.
    """
    policy_value = 0
    for i in range(len(centers)):
        radius = radii[i]
        if radius > 0:
            policy_value += weights[i] * ((abs(lake_state - centers[i]) / radius) ** 3)
    return max(0.01, min(0.1, policy_value))  # Constrain the inflow within 0.01 and 0.1

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
        emissions = [compute_policy_output(lake_state[0], 
                                           [rbf["center"] for rbf in policy["rbf_params"]],
                                           [rbf["radius"] for rbf in policy["rbf_params"]],
                                           [rbf["weight"] for rbf in policy["rbf_params"]])
                     for _ in range(num_steps)]
        
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


def load_rbf_parameters_from_cache(cache_file):
    """
    Loads the RBF parameters (centers, radii, and weights) from a cache file.

    Parameters:
        cache_file (str): Path to the cache file.

    Returns:
        np.ndarray: A 3 x num_samples matrix where each row contains all centers, radii, 
                    and weights respectively across all samples.
    """
    centers = []
    radii = []
    weights = []

    # Open the cache file
    with shelve.open(cache_file) as cache:
        # Assuming there is only one main dataset in the cache
        for key in cache:
            dataset = cache[key]

            # Attempt to convert dataset to DataFrame
            try:
                df = dataset.as_dataframe()
            except Exception as e:
                print(f"Error converting to DataFrame: {e}")
                continue

            # Iterate over each row (solution) in the DataFrame
            for _, row in df.iterrows():
                # Access the policy and rbf_params
                policy = row['policy']
                rbf_params = policy.get('rbf_params', [])

                # Extract centers, radii, and weights
                solution_centers = [rbf['center'] for rbf in rbf_params]
                solution_radii = [rbf['radius'] for rbf in rbf_params]
                solution_weights = [rbf['weight'] for rbf in rbf_params]

                # Store values in the respective lists
                centers.extend(solution_centers)
                radii.extend(solution_radii)
                weights.extend(solution_weights)

        # Convert lists to numpy arrays and stack them as rows
        C = np.array(centers)
    R = np.array(radii)
    W = np.array(weights)

    # Reshape into a 3 x num_samples matrix if needed
    num_samples = len(centers)
    rbf_matrix = np.vstack((C, R, W)).reshape(3, num_samples)

    return rbf_matrix

def load_objective_values_from_cache(cache_file):
    """
    Loads objective values (max_phosphorus, avg_utility, avg_inertia, avg_reliability) 
    from a cache file.

    Parameters:
        cache_file (str): Path to the cache file.

    Returns:
        np.ndarray: A matrix where each row contains the objective values for each solution.
    """
    # Initialize lists to store the objective values across all samples
    max_phosphorus = []
    avg_utility = []
    avg_inertia = []
    avg_reliability = []

    # Open the cache file
    with shelve.open(cache_file) as cache:
        # Assuming there is only one main dataset in the cache
        for key in cache:
            dataset = cache[key]

            # Convert dataset to DataFrame for easy access
            try:
                df = dataset.as_dataframe()
            except Exception as e:
                print(f"Error converting to DataFrame: {e}")
                continue

            # Iterate over each row (solution) in the DataFrame
            for _, row in df.iterrows():
                # Extract objective values from each solution
                max_phosphorus.append(row['max_phosphorus'])
                avg_utility.append(row['avg_utility'])
                avg_inertia.append(row['avg_inertia'])
                avg_reliability.append(row['avg_reliability'])

    # Convert lists to a numpy array and stack them as rows
    objectives_matrix = np.vstack((max_phosphorus, avg_utility, avg_inertia, avg_reliability)).T
    
    return objectives_matrix