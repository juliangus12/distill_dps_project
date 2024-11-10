# Example using direct policy search (DPS) following the approach of:
#
# [1] Quinn, J. D., P. M. Reed, and K. Keller (2017). "Direct policy search for
#     robust multi-objective management of deeply uncertain socio-ecological
#     tipping points." Environmental Modelling & Software, 92:125-141.
#
# This script simulates a lake management model where phosphorus inflow is controlled
# by a Radial Basis Function (RBF) policy. The model involves multi-objective optimization
# for policy selection considering objectives of economic benefit, phosphorus concentration
# control, policy inertia, and system reliability under uncertainty.
#
# Adapted by Julian Gutierrez, Nov. 10th, 2024
# Dartmouth College
# julian.g.gutierrez.26@dartmouth.edu

import argparse 
import random  
from rhodium import Model, RealLever, UniformUncertainty, Response, optimize  
import os  
from scipy.optimize import brentq as root 


n_rbf = 2  # Number of Radial Basis Functions (RBFs) for the control policy
n_vars = n_rbf * 3  # Total number of variables (each RBF has center, radius, weight)
n_years = 100  # Number of years to simulate
n_samples = 100  # Number of Monte Carlo samples for uncertainty
inertia_threshold = -0.02  # Threshold for policy inertia (change constraint)
reliability_threshold = 0.85  
nat_flowmat = [[0] * n_years for _ in range(10000)]  # Matrix to store natural inflow data for 10000 possible states

# Load natural inflows into `nat_flowmat` from a text file
def load_natural_inflows(filename):
    """
    Reads inflow data from a file and stores it in the global matrix `nat_flowmat`.
    
    Parameters:
        filename (str): The path to the file containing inflow data.
    """
    global nat_flowmat
    with open(filename, "r") as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if line.startswith("#"):  # Skip comments
                continue
            values = [float(value) for value in line.strip().split()]
            nat_flowmat[i][:len(values)] = values  # Store the data in `nat_flowmat`

# Call function to load inflows from the specified file
load_natural_inflows("SOWs_Type6.txt")

# Define the Radial Basis Function (RBF) policy function
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

# Define the lake simulation model
def lake_problem(*C_R_W, alpha=0.4, delta=0.98, b=0.1, q=2.0, mean=0.01, stdev=0.001):
    """
    Simulates the lake model over time with policy actions based on RBF policy parameters.

    Parameters:
        C_R_W (float): A list of policy parameters for centers (C), radii (R), and weights (W).
        alpha (float): Economic benefit scaling factor.
        delta (float): Discount rate for economic benefits.
        b (float): Lake phosphorus loss rate.
        q (float): Lake phosphorus recycling rate.
        mean (float): Mean of natural inflow phosphorus.
        stdev (float): Standard deviation of natural inflow phosphorus.

    Returns:
        tuple: Objective values and constraints:
            - avg_economic_benefit
            - max_P (max phosphorus level over time)
            - avg_inertia (policy inertia)
            - avg_reliability (constraint satisfaction frequency)
            - reliability_constraint (constraint indicator)
    """
    # Split parameters into centers, radii, and weights for the RBF policy
    C = C_R_W[:n_rbf]
    R = C_R_W[n_rbf:2*n_rbf]
    W = C_R_W[2*n_rbf:3*n_rbf]
    
    # Normalize weights to ensure they sum to 1
    total_weight = sum(W)
    if total_weight != 0:
        W = [w / total_weight for w in W]
    else:
        W = [1 / n_rbf] * n_rbf

    # Calculate critical phosphorus level `Pcrit` based on recycling rate `q` and loss rate `b`
    Pcrit = root(lambda x: x**q / (1 + x**q) - b * x, 0.01, 1.5)
    
    # Initialize simulation variables
    average_annual_P = [0] * n_years  # Average annual phosphorus concentration in the lake
    discounted_benefit = [0] * n_samples  # Discounted economic benefits for each sample
    yrs_inertia_met = [0] * n_samples  # Years where inertia constraint is met
    yrs_pCrit_met = [0] * n_samples  # Years where phosphorus remains below critical level

    # Monte Carlo sampling of 100 inflow scenarios
    indices = random.sample(range(10000), n_samples)
    for s in range(n_samples):
        nat_flow = nat_flowmat[indices[s]]
        lake_state = [0] * (n_years + 1)  # Initialize lake state over simulation time
        Y = [0] * n_years  # RBF policy outputs
        Y[0] = RBFpolicy(lake_state[0], C, R, W)

        # Simulate phosphorus dynamics over the specified years
        for i in range(n_years):
            lake_state[i + 1] = lake_state[i] * (1 - b) + lake_state[i] ** q / (1 + lake_state[i] ** q) + Y[i] + nat_flow[i]
            average_annual_P[i] += lake_state[i + 1] / n_samples
            discounted_benefit[s] += alpha * Y[i] * (delta ** i)  # Calculate economic benefit

            # Check inertia and reliability constraints
            if i >= 1 and (Y[i] - Y[i - 1]) > inertia_threshold:
                yrs_inertia_met[s] += 1
            if lake_state[i + 1] < Pcrit:
                yrs_pCrit_met[s] += 1
            if i < (n_years - 1):
                Y[i + 1] = RBFpolicy(lake_state[i + 1], C, R, W)

    # Aggregate results for objectives and constraint
    avg_economic_benefit = -sum(discounted_benefit) / n_samples
    max_P = max(average_annual_P)
    avg_inertia = -sum(yrs_inertia_met) / ((n_years - 1) * n_samples)
    avg_reliability = -sum(yrs_pCrit_met) / (n_years * n_samples)
    reliability_constraint = max(0.0, reliability_threshold - (-avg_reliability))

    return avg_economic_benefit, max_P, avg_inertia, avg_reliability, reliability_constraint

# Main function to set up and execute the optimization
def main(seed, output_dir, nfe):
    """
    Initializes and runs the lake problem optimization using BorgMOEA.

    Parameters:
        seed (int): Seed for random number generation.
        output_dir (str): Directory to save output files.
        nfe (int): Number of function evaluations for optimization.
    """
    # Set the random seed for reproducibility
    random.seed(seed)

    # Define the lake model and specify levers (decision variables), responses, and uncertainties
    model = Model(lake_problem)
    model.levers = [RealLever(f"C_R_W_{i}", -2.0, 2.0) for i in range(n_vars)]
    model.responses = [
        Response("avg_economic_benefit", Response.MAXIMIZE),
        Response("max_P", Response.MINIMIZE),
        Response("avg_inertia", Response.MAXIMIZE),
        Response("avg_reliability", Response.MAXIMIZE)
    ]
    
    # Set up the uncertainties in model parameters
    model.uncertainties = [
        UniformUncertainty("b", 0.1, 0.45),
        UniformUncertainty("q", 2.0, 4.5),
        UniformUncertainty("mean", 0.01, 0.05),
        UniformUncertainty("stdev", 0.001, 0.005)
    ]
    
    # Run the optimization
    output = optimize(model, "BorgMOEA", nfe, module="pyborg", epsilons=[0.01, 0.01, 0.0001, 0.0001])

    # Prepare output directory and filename
    sets_dir = os.path.join(output_dir, "sets")
    os.makedirs(sets_dir, exist_ok=True)
    filename = f"{sets_dir}/LakeDPS_S{seed}.set"
    
    # Save optimization results in the specified file
    if output:
        with open(filename, "w") as file:
            for solution in output:
                values = [solution[f"C_R_W_{i}"] for i in range(n_vars)]
                objs = [solution["avg_economic_benefit"], solution["max_P"], solution["avg_inertia"], solution["avg_reliability"]]
                file.write(f"Values: {values}, Objectives: {objs}\n")

# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run lake problem optimization with Borg for a specific seed.")
    parser.add_argument("--seed", type=int, required=True, help="Seed for random number generation.")
    parser.add_argument("--output-dir", type=str, default=".", help="Base directory to save results.")
    parser.add_argument("--nfe", type=int, default=10000, help="Number of function evaluations.")
    
    args = parser.parse_args()
    main(args.seed, args.output_dir, args.nfe)
