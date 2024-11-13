import matplotlib.pyplot as plt
from rhodium import *
from lake_model_utils import *

# Retrieve constants and model parameters
benefit_scale, discount_rate, phosphorus_loss_rate, phosphorus_recycling_rate, inflow_mean, inflow_variance = get_simulation_constants()
num_years, num_samples, inertia_threshold, reliability_threshold, num_rbfs, num_vars = get_model_parameters()

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

setup_cache(file="lake_cache.cache")
output = cache("dps_output", lambda: optimize(model, "NSGAII", 10000))

print(output)
print("Number of solutions discovered:", len(output))
scatter3d(model, output)
plt.show()
