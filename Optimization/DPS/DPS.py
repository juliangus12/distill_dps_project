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
# Adapted by Julian Gutierrez, Nov. 12th, 2024
# Dartmouth College
# julian.g.gutierrez.26@dartmouth.edu

import argparse
from rhodium import Model, RealLever, UniformUncertainty, Response, optimize
import sys
import os 

# Line included so that the imports in the current file's root directory are importable -- necessary for lake_problem_utils.py as well as pyborg module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lake_problem_utils import *

    
# Retrieve model constants and parameters from lake_problem_utils.py
alpha, delta, b, q, real_mean, real_variance = get_simulation_constants()
n_years, n_samples, inertia_threshold, reliability_threshold, n_rbf, n_vars = get_model_parameters()

# Define the lake simulation model (using lake_problem_utils.py)
def lake_problem(*C_R_W):
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
            - avg_max_P (max phosphorus level over time)
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

    # Generate synthetic natural inflows
    nat_flowmat = generate_natural_inflows()

    # Simulate lake dynamics using the shared function from lake_problem_utils.py
    avg_economic_benefit, avg_max_P, avg_inertia, avg_reliability, reliability_constraint = simulate_lake_dynamics(
        n_samples, nat_flowmat, b, q, alpha, delta, inertia_threshold, reliability_threshold, C=C, R=R, W=W
    )

    return avg_economic_benefit, avg_max_P, avg_inertia, avg_reliability, reliability_constraint

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
        Response("avg_max_P", Response.MINIMIZE),
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

    save_optimization_results(output, output_dir, seed, n_vars, is_dps=True)

# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run lake problem optimization with Borg for a specific seed.")
    parser.add_argument("--seed", type=int, required=True, help="Seed for random number generation.")
    parser.add_argument("--output-dir", type=str, default=".", help="Base directory to save results.")
    parser.add_argument("--nfe", type=int, default=10000, help="Number of function evaluations.")
    
    args = parser.parse_args()
    main(args.seed, args.output_dir, args.nfe)
