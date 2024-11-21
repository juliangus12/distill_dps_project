import os
import sys
import random  # Import the random module
from rhodium import *

# Necessary for importing the functions inside of Utils/lake_model_utils.py 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from Utils.lake_model_utils import *

# Define the directories for saving solutions and Pareto frontiers as text files
solutions_dir = os.path.join(os.path.dirname(__file__), "solutions")
pareto_dir = os.path.join(os.path.dirname(__file__), "pareto_frontiers")
os.makedirs(solutions_dir, exist_ok=True)
os.makedirs(pareto_dir, exist_ok=True)

# Get the seed from the environment variable passed by SLURM or use "default"
seed = os.environ.get("SLURM_ARRAY_TASK_ID", "default")
if seed == "default":
    print("Warning: SLURM_ARRAY_TASK_ID is not set. Using default seed value.")
else:
    seed = int(seed)  # Ensure seed is an integer
    random.seed(seed)  # Set the random seed for reproducibility
    print(f"Random seed set to: {seed}")

# File paths for the solutions and Pareto frontier text files
solutions_file = os.path.join(solutions_dir, f"{seed}_solutions.txt")
pareto_file = os.path.join(pareto_dir, f"pareto_{seed}.txt")

# Retrieve constants and model parameters
benefit_scale, discount_rate, phosphorus_loss_rate, phosphorus_recycling_rate, inflow_mean, inflow_variance = get_simulation_constants()
num_years, num_samples, inertia_threshold, reliability_threshold, num_rbfs, num_vars = get_model_parameters()

# Set up the Rhodium model using lake_model as the main function
model = Model(lake_model)

model.parameters = [
    Parameter("policy"),
    Parameter("loss_rate"),
    Parameter("recycle_rate"),
    Parameter("mean"),
    Parameter("variance"),
    Parameter("discount"),
]

model.responses = [
    Response("avg_max_phosphorus", Response.MINIMIZE),
    Response("avg_utility", Response.MAXIMIZE),
    Response("avg_inertia", Response.MAXIMIZE),
    Response("avg_reliability", Response.MAXIMIZE),
]

model.levers = [CubicDPSLever("policy", num_rbfs=num_rbfs)]

# Optimize the model
output = optimize(model, "NSGAII", 10000)

# Save solutions to a text file
with open(solutions_file, "w") as f:
    for solution in output:
        f.write(str(solution) + "\n")
print(f"Solutions saved to: {solutions_file}")

# Find Pareto frontier
pareto_solutions, pareto_objectives = find_pareto_frontier(output, filter=True)

# Save Pareto frontier to a text file
with open(pareto_file, "w") as f:
    f.write("Pareto Solutions:\n")
    for solution in pareto_solutions:
        f.write(",".join(map(str, solution)) + "\n")
    f.write("\nPareto Objectives:\n")
    for objectives in pareto_objectives:
        f.write(",".join(map(str, objectives)) + "\n")
print(f"Pareto frontier saved to: {pareto_file}")
