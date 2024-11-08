import argparse
import numpy as np
from rhodium import Model, RealLever, UniformUncertainty, Response, optimize
import os
from scipy.optimize import brentq as root

# Define the number of Radial Basis Functions (RBFs)
n_rbf = 2
n_vars = n_rbf * 3
n_years = 100
n_samples = 100
inertia_threshold = -0.02
reliability_threshold = 0.85
nat_flowmat = np.zeros((10000, n_years))

# Load natural inflows 
def load_natural_inflows(filename):
    global nat_flowmat
    with open(filename, "r") as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if line.startswith("#"):
                continue
            values = [float(value) for value in line.strip().split()]
            nat_flowmat[i, :len(values)] = values

# Load the inflow data once
load_natural_inflows("SOWs_Type6.txt")

# Define the RBF policy function
def RBFpolicy(lake_state, C, R, W):
    Y = 0
    for i in range(len(C)):
        r_value = R[i]
        if r_value > 0:
            Y += W[i] * ((abs(lake_state - C[i]) / r_value) ** 3)
    return max(0.01, min(0.1, Y))

# Define the lake model function using RBF policy
def lake_problem(*C_R_W, b=0.42, q=2.0, mean=0.02, stdev=0.001, alpha=0.4, delta=0.98):
    C = C_R_W[:n_rbf]
    R = C_R_W[n_rbf:2*n_rbf]
    W = C_R_W[2*n_rbf:3*n_rbf]

    Pcrit = root(lambda x: x**q / (1 + x**q) - b * x, 0.01, 1.5)
    X = np.zeros((n_years,))
    average_annual_P = np.zeros((n_years,))
    discounted_benefit = np.zeros(n_samples)
    yrs_inertia_met = np.zeros(n_samples)
    yrs_pCrit_met = np.zeros(n_samples)

    indices = np.random.choice(10000, n_samples, replace=False)
    for s in range(n_samples):
        nat_flow = nat_flowmat[indices[s]]
        lake_state = np.zeros(n_years + 1)
        Y = np.zeros(n_years)
        Y[0] = RBFpolicy(lake_state[0], C, R, W)

        for i in range(n_years):
            lake_state[i + 1] = lake_state[i] * (1 - b) + lake_state[i] ** q / (1 + lake_state[i] ** q) + Y[i] + nat_flow[i]
            average_annual_P[i] += lake_state[i + 1] / n_samples
            discounted_benefit[s] += alpha * Y[i] * (delta ** i)

            if i >= 1 and (Y[i] - Y[i - 1]) > inertia_threshold:
                yrs_inertia_met[s] += 1
            if lake_state[i + 1] < Pcrit:
                yrs_pCrit_met[s] += 1
            if i < (n_years - 1):
                Y[i + 1] = RBFpolicy(lake_state[i + 1], C, R, W)

    max_P = np.max(average_annual_P)
    avg_economic_benefit = -np.mean(discounted_benefit)
    avg_inertia = -np.mean(yrs_inertia_met) / (n_years - 1)
    avg_reliability = -np.mean(yrs_pCrit_met) / n_years

    reliability_constraint = max(0.0, reliability_threshold - (-avg_reliability))

    return avg_economic_benefit, max_P, avg_inertia, avg_reliability, reliability_constraint

# Main function to set up and run optimization
def main(seed, output_dir, nfe):
    # Set up the model
    model = Model(lake_problem)
    model.levers = [RealLever(f"C_R_W_{i}", -2.0, 2.0) for i in range(n_vars)]
    model.responses = [
        Response("avg_economic_benefit", Response.MAXIMIZE),
        Response("max_P", Response.MINIMIZE),
        Response("avg_inertia", Response.MAXIMIZE),
        Response("avg_reliability", Response.MAXIMIZE)
    ]
    model.uncertainties = [
        UniformUncertainty("b", 0.1, 0.45),
        UniformUncertainty("q", 2.0, 4.5),
        UniformUncertainty("mean", 0.01, 0.05),
        UniformUncertainty("stdev", 0.001, 0.005),
        UniformUncertainty("delta", 0.93, 0.99)
    ]

    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Run the optimization
    output = optimize(model, "BorgMOEA", nfe, module="pyborg", epsilons=[0.01, 0.01, 0.0001, 0.0001])

    # Save results to a .set file
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/LakeDPS_S{seed}.set"
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
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the .set file.")
    parser.add_argument("--nfe", type=int, default=10000, help="Number of function evaluations.")

    args = parser.parse_args()
    main(args.seed, args.output_dir, args.nfe)
