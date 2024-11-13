import random
import math
import os
from scipy.optimize import brentq as root

# Define constants for the model
n_years = 100  # Number of years in the simulation
n_samples = 100  # Number of samples for Monte Carlo simulations
inertia_threshold = -0.02  # Threshold for inertia objective
reliability_threshold = 0.85  # Minimum reliability constraint
n_rbf = 2
n_vars = n_rbf * 3

# Define model parameters (could be adjusted globally)
alpha = 0.4  # Scaling factor for economic benefit
delta = 0.98  # Discount rate for economic benefits
b = 0.42  # Lake phosphorus loss rate
q = 2.0  # Lake phosphorus recycling rate
real_mean = 0.03  # Real-space mean of natural inflows
real_variance = 10e-5  # Real-space variance of natural inflows

# Function to generate synthetic natural inflow data using log-normal distribution
def generate_natural_inflows(real_mean=real_mean, real_variance=real_variance, n_samples=n_samples, n_years=n_years):
    """
    Generates synthetic natural inflow data for the lake model using the log-normal distribution.
    
    Parameters:
        real_mean (float): The desired mean of the real-space distribution.
        real_variance (float): The desired variance of the real-space distribution.
        n_samples (int): Number of samples to generate.
        n_years (int): Number of years per sample.
        
    Returns:
        list: Generated natural inflow data as a list of `n_samples` lists, each with `n_years` values.
    """
    # Calculate sigma and mu for the normal distribution
    sigma = math.sqrt(math.log(real_variance / real_mean**2 + 1))
    mu = math.log(real_mean) - (sigma**2 / 2)

    # Generate `n_samples` sets of `n_years` inflow values
    nat_flowmat = []
    for _ in range(n_samples):
        inflow = []
        for _ in range(n_years):
            # Generate a single log-normal value using the Box-Muller method
            u1 = random.random()
            u2 = random.random()
            z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            z1 = mu + sigma * z0  # Normal distributed value with mean `mu` and standard deviation `sigma`
            inflow.append(math.exp(z1))  # Convert to log-normal
        nat_flowmat.append(inflow)

    return nat_flowmat

# Optionally, you can return these parameters if needed
def get_simulation_constants():
    
    """
    Returns the simulation constants for lake problem simulations.
    
    Returns:
        tuple: (alpha, delta, b, q, real_mean, real_variance)
    """
    return alpha, delta, b, q, real_mean, real_variance

# Function to return model simulation constants
def get_model_parameters():
    """
    Returns the model parameters for lake problem simulations.
    
    Returns:
        tuple: (n_years, n_samples, inertia_threshold, reliability_threshold)
    """
    return n_years, n_samples, inertia_threshold, reliability_threshold, n_rbf, n_vars

# Function to calculate critical phosphorus level (Pcrit)
def calculate_Pcrit(b=b, q=q):
    """
    Calculate the critical phosphorus level `Pcrit` based on recycling rate `q` and loss rate `b`.

    Parameters:
        b (float): Phosphorus loss rate.
        q (float): Phosphorus recycling rate.

    Returns:
        float: The critical phosphorus level `Pcrit`.
    """
    return root(lambda x: x**q / (1 + x**q) - b * x, 0.01, 1.5)

# Function to calculate RBF policy
def RBFpolicy(lake_state, C, R, W):
    """
    Computes the policy output `Y` based on the current lake state and policy parameters.

    Parameters:
        lake_state (float): The current phosphorus level in the lake.
        C (list of float): Centers of the RBFs.
        R (list of float): Radii of the RBFs.
        W (list of float): Weights of the RBFs.

    Returns:
        float: The controlled phosphorus inflow based on the RBF policy.
    """
    Y = 0
    for i in range(len(C)):
        r_value = R[i]
        if r_value > 0:
            Y += W[i] * ((abs(lake_state - C[i]) / r_value) ** 3)
    return max(0.01, min(0.1, Y))  # Constrain the inflow within 0.01 and 0.1

def simulate_lake_dynamics(n_samples, nat_flowmat, b, q, alpha, delta, inertia_threshold, reliability_threshold, C=None, R=None, W=None, emissions=None):
    """
    Simulates the phosphorus dynamics of a lake based on different policies (RBF or emission controls).

    Parameters:
        n_samples (int): Number of Monte Carlo samples.
        nat_flowmat (list): Matrix of natural inflows.
        b (float): Lake phosphorus loss rate.
        q (float): Lake phosphorus recycling rate.
        alpha (float): Scaling factor for economic benefit.
        delta (float): Discount rate for economic benefits.
        inertia_threshold (float): Inertia threshold for policy changes.
        reliability_threshold (float): Minimum reliability constraint.
        C (list): Centers for the RBF policy (if applicable).
        R (list): Radii for the RBF policy (if applicable).
        W (list): Weights for the RBF policy (if applicable).
        emissions (list): Emissions for the intertemporal approach (if applicable).

    Returns:
        tuple: Objective values and constraints:
            - avg_economic_benefit
            - max_P (max phosphorus level over time)
            - avg_inertia
            - avg_reliability
            - reliability_constraint (constraint indicator)
    """
    # Calculate critical phosphorus level `Pcrit`
    Pcrit = calculate_Pcrit(b, q)

    # Initialize variables for objectives
    average_annual_P = [0] * n_years
    discounted_benefit = [0] * n_samples
    yrs_inertia_met = [0] * n_samples
    yrs_pCrit_met = [0] * n_samples

    # Directly iterate over each inflow scenario in `nat_flowmat`
    for s in range(n_samples):
        nat_flow = nat_flowmat[s]
        lake_state = [0] * (n_years + 1)  # Initialize lake phosphorus levels
        
        # Use emissions directly as `Y` if provided, else calculate `Y` using RBF policy
        if emissions:
            Y = emissions  # Directly set `Y` to emissions list if provided
        else:
            Y = [0] * n_years  # Otherwise, initialize Y for RBF policy

        # Initialize RBF policy if applicable
        if C is not None and R is not None and W is not None and not emissions:
            Y[0] = RBFpolicy(lake_state[0], C, R, W)
        
        # Simulate lake phosphorus dynamics for each year
        for i in range(n_years):
            lake_state[i + 1] = (
                lake_state[i] * (1 - b) +
                lake_state[i] ** q / (1 + lake_state[i] ** q) +
                Y[i] +  # Use emission values or RBF policy outputs
                nat_flow[i]
            )
            average_annual_P[i] += lake_state[i + 1] / n_samples
            discounted_benefit[s] += alpha * Y[i] * (delta ** i)

            # Track inertia and reliability over time
            if i >= 1 and (Y[i] - Y[i - 1]) > inertia_threshold:
                yrs_inertia_met[s] += 1
            if lake_state[i + 1] < Pcrit:
                yrs_pCrit_met[s] += 1

    # Calculate objectives and constraints
    max_P = max(average_annual_P)
    avg_economic_benefit = -sum(discounted_benefit) / n_samples
    avg_inertia = -sum(yrs_inertia_met) / ((n_years - 1) * n_samples)
    avg_reliability = -sum(yrs_pCrit_met) / (n_years * n_samples)
    reliability_constraint = max(0.0, reliability_threshold - (-avg_reliability))

    return avg_economic_benefit, max_P, avg_inertia, avg_reliability, reliability_constraint

# Function to save the optimization results to a file
def save_optimization_results(output, output_dir, seed, n_vars, is_dps=True):
    """
    Prepares the output directory and filename, and saves the optimization results to the file.

    Parameters:
        output (list): The optimization output.
        output_dir (str): The directory to save the output.
        seed (int): The seed value to be used in the filename.
        n_vars (int): The number of variables in the optimization model.
        is_dps (bool): Flag to indicate if the optimization is for DPS or Intertemporal.
    """
    # Determine the subdirectory based on the problem type (DPS or Intertemporal)
    sub_dir = "DPS" if is_dps else "Intertemporal"
    sets_dir = os.path.join(output_dir, sub_dir, "sets")
    
    # Ensure the directory exists
    os.makedirs(sets_dir, exist_ok=True)
    
    # Prepare the filename
    filename = f"{sets_dir}/LakeDPS_S{seed}.set" if is_dps else f"{sets_dir}/LakeIT_S{seed}.set"
    
    # Save the optimization results in the specified file
    if output:
        with open(filename, "w") as file:
            for solution in output:
                values = [solution[f"C_R_W_{i}"] for i in range(n_vars)]
                objs = [solution["avg_economic_benefit"], solution["avg_max_P"], solution["avg_inertia"], solution["avg_reliability"]]
                file.write(f"Values: {values}, Objectives: {objs}\n")
