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
import random
import os
import sys
from rhodium import Model, RealLever, UniformUncertainty, Response, optimize

# Line included so that the imports in the current file's root directory are importable -- necessary for lake_problem_utils.py as well as pyborg module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lake_problem_utils import *

# Retrieve constants and parameters
alpha, delta, b, q, real_mean, real_variance = get_simulation_constants()
n_years, n_samples, inertia_threshold, reliability_threshold, n_rbf, n_vars = get_model_parameters()

def lake_problem(*emissions):
    """
    Lake model simulation with intertemporal decision variables representing annual emissions.
    """
    # Convert emissions tuple to a list so we can index it correctly
    emissions = list(emissions)

    # Generate natural inflows matrix
    nat_flowmat = generate_natural_inflows(real_mean=real_mean, real_variance=real_variance, n_samples=n_samples, n_years=n_years)
    
    # Simulate lake dynamics using the `simulate_lake_dynamics` function
    avg_economic_benefit, max_P, avg_inertia, avg_reliability, reliability_constraint = simulate_lake_dynamics(
        n_samples=n_samples, nat_flowmat=nat_flowmat, b=b, q=q, alpha=alpha, delta=delta,
        inertia_threshold=inertia_threshold, reliability_threshold=reliability_threshold, emissions=emissions
    )

    return avg_economic_benefit, max_P, avg_inertia, avg_reliability, reliability_constraint


# Main function to execute optimization
def main(seed, output_dir, nfe):
    """
    Runs lake problem optimization using BorgMOEA.
    """
    random.seed(seed)  # Set seed for reproducibility

    # Define the lake model, decision variables (levers), objectives, and uncertainties
    model = Model(lake_problem)
    model.levers = [RealLever(f"emission_{i}", 0.01, 0.1) for i in range(n_years)]
    model.responses = [
        Response("avg_economic_benefit", Response.MAXIMIZE),
        Response("max_P", Response.MINIMIZE),
        Response("avg_inertia", Response.MAXIMIZE),
        Response("avg_reliability", Response.MAXIMIZE)
    ]
    
    # Set up model uncertainties
    model.uncertainties = [
        UniformUncertainty("b", 0.1, 0.45),
        UniformUncertainty("q", 2.0, 4.5),
        UniformUncertainty("mean", 0.01, 0.05),
        UniformUncertainty("stdev", 0.001, 0.005)
    ]
    
    # Run the BorgMOEA optimization
    output = optimize(model, "BorgMOEA", nfe, module="pyborg", epsilons=[0.01, 0.01, 0.0001, 0.0001])

    # Save optimization results
    save_optimization_results(output, output_dir, seed, n_vars, is_dps=False)

# Parse command-line arguments and execute main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run lake problem intertemporal optimization with Borg for a specific seed.")
    parser.add_argument("--seed", type=int, required=True, help="Seed for random number generation.")
    parser.add_argument("--output-dir", type=str, default=".", help="Base directory to save results.")
    parser.add_argument("--nfe", type=int, default=10000, help="Number of function evaluations.")
    
    args = parser.parse_args()
    main(args.seed, args.output_dir, args.nfe)
