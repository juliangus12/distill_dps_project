import shelve
from lake_model_utils import *

cache_file = "lake_cache.cache"
pareto_solutions, pareto_objectives = find_pareto_frontier(cache_file)

print("Pareto-optimal RBF parameters:\n", pareto_solutions)
print("Pareto-optimal objectives:\n", pareto_objectives)
