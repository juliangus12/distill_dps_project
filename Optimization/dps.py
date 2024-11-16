import os
from rhodium import *
from Utils.lake_model_utils import *  # Import utility functions from Utils

try:
    # Define the directory to store cache files
    cache_dir = os.path.join(os.path.dirname(__file__), "caches")
    os.makedirs(cache_dir, exist_ok=True)
except Exception as e:
    raise RuntimeError(f"Failed to create or access the cache directory at {cache_dir}: {e}")

try:
    # Get the seed from the environment variable passed by SLURM or use "default"
    seed = os.environ.get("SLURM_ARRAY_TASK_ID", "default")
    if seed == "default":
        print("Warning: SLURM_ARRAY_TASK_ID is not set. Using default seed value.")
    cache_file = os.path.join(cache_dir, f"{seed}.cache")
except Exception as e:
    raise RuntimeError(f"Failed to construct cache file path: {e}")

try:
    # Retrieve constants and model parameters
    benefit_scale, discount_rate, phosphorus_loss_rate, phosphorus_recycling_rate, inflow_mean, inflow_variance = get_simulation_constants()
    num_years, num_samples, inertia_threshold, reliability_threshold, num_rbfs, num_vars = get_model_parameters()
except Exception as e:
    raise RuntimeError(f"Error while retrieving simulation constants or model parameters: {e}")

try:
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
except Exception as e:
    raise RuntimeError(f"Error while setting up the Rhodium model: {e}")

try:
    # Set up cache and optimize
    setup_cache(file=cache_file)
    output = cache("dps_output", lambda: optimize(model, "NSGAII", 10000))
except Exception as e:
    raise RuntimeError(f"Error during optimization or caching: {e}")

try:
    # Print optimization results
    print(f"Cache file saved at: {cache_file}")
    print("Number of solutions discovered:", len(output))
except Exception as e:
    raise RuntimeError(f"Error while printing optimization results: {e}")

try:
    # Find and save Pareto frontier
    pareto_solutions = find_pareto_frontier(cache_file)
    pareto_save_path = os.path.join(cache_dir, f"pareto_{seed}.txt")

    with open(pareto_save_path, "w") as f:
        f.write("\n".join(map(str, pareto_solutions)))
    print(f"Pareto frontier saved at: {pareto_save_path}")
except FileNotFoundError as e:
    raise FileNotFoundError(f"Cache file not found at {cache_file}. Ensure the optimization step completed successfully: {e}")
except Exception as e:
    raise RuntimeError(f"Error while finding or saving the Pareto frontier: {e}")

try:
    # Optional: Plot the results (disabled for SLURM batch jobs)
    # Uncomment the line below for local runs
    # scatter3d(model, output)
    pass
except Exception as e:
    print(f"Warning: Error while plotting results. Skipping plot generation. Error: {e}")
