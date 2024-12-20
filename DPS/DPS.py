import argparse
import sys
import os
from rhodium import *
from lake_model_utils import *

# Add the project root directory to Python's module search path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Set up argument parsing
parser = argparse.ArgumentParser(description="Run DPS optimization with specified NFE and seed.")
parser.add_argument(
    "--NFE",
    type=int,
    default=10000,
    help="Number of function evaluations for optimization (default: 10000)"
)
parser.add_argument(
    "--seed",
    type=int,
    required=True,
    help="Seed value for this instance of DPS"
)
args = parser.parse_args()

# Retrieve NFE and seed values
nfe = args.NFE
seed = args.seed

# Create data directory if it doesn't exist
data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)

# Define cache file based on seed
cache_file = os.path.join(data_dir, f"{seed}.cache")

# Retrieve constants and model parameters
try:
    benefit_scale, discount_rate, phosphorus_loss_rate, phosphorus_recycling_rate, inflow_mean, inflow_variance = get_simulation_constants()
    num_years, num_samples, inertia_threshold, reliability_threshold, num_rbfs, num_vars = get_model_parameters()
except NameError as e:
    print(f"Error retrieving constants: {e}")
    sys.exit(1)

# Set up the Rhodium model using lake_problem as the main function
try:
    model = Model(lake_model)
    model.parameters = [
        Parameter("policy"), Parameter("loss_rate"), Parameter("recycle_rate"),
        Parameter("mean"), Parameter("variance"), Parameter("discount")
    ]
    model.responses = [
        Response("avg_max_phosphorus", Response.MINIMIZE),
        Response("avg_utility", Response.MAXIMIZE),
        Response("avg_inertia", Response.MAXIMIZE),
        Response("avg_reliability", Response.MAXIMIZE)
    ]
    model.levers = [CubicDPSLever("policy", num_rbfs=3)]
except Exception as e:
    print(f"Error setting up model: {e}")
    sys.exit(1)

# Run optimization with cache
try:
    setup_cache(file=cache_file)
    output = cache("dps_output", lambda: optimize(model, "NSGAII", nfe))
    print(f"Completed optimization for seed {seed} with NFE {nfe}. Results saved to {cache_file}")
except Exception as e:
    print(f"Error during optimization: {e}")
    sys.exit(1)
