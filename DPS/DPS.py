import matplotlib.pyplot as plt
from rhodium import *
from lake_model_utils import *
import sys
import os

# Retrieve constants and model parameters
benefit_scale, discount_rate, phosphorus_loss_rate, phosphorus_recycling_rate, inflow_mean, inflow_variance = get_simulation_constants()
num_years, num_samples, inertia_threshold, reliability_threshold, num_rbfs, num_vars = get_model_parameters()

# Retrieve the seed from command-line arguments (used only for naming the cache file)
seed = sys.argv[1]
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
cache_file = os.path.join(data_dir, f"{seed}.cache")

# Set up the Rhodium model using lake_problem as the main function
model = Model(lake_model)
model.parameters = [
    Parameter("policy"), Parameter("loss_rate"), Parameter("recycle_rate"),
    Parameter("mean"), Parameter("variance"), Parameter("discount")
]
model.responses = [
    Response("max_phosphorus", Response.MINIMIZE),
    Response("avg_utility", Response.MAXIMIZE),
    Response("avg_inertia", Response.MAXIMIZE),
    Response("avg_reliability", Response.MAXIMIZE)
]
model.levers = [CubicDPSLever("policy", num_rbfs=3)]

# Run optimization with cache
setup_cache(file=cache_file)
output = cache("dps_output", lambda: optimize(model, "NSGAII", 10000))

print(f"Completed optimization for seed {seed}. Results saved to {cache_file}")
