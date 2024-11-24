from lake_model_utils import (
    DPSExpertPolicy,
    generate_inflow_data,
    phosphorus_recycling_rate as q,
    phosphorus_loss_rate as b,
    benefit_scale as alpha,
    discount_rate as delta,
    inflow_mean as mu,
    inflow_variance as sigma,
    inertia_threshold,
    reliability_threshold,
    compute_policy_output
)

import numpy as np

# Hardcoded RBF parameters [C1, R1, W1, C2, R2, W2]
rbf_parameters = [0.01863850146778845,0.9999391073694877,0.5917595241260287,0.21626196382868074,0.8277986806157667,0.4082404758739714]

# Generate 1000 random lake states
random_states = np.random.uniform(0, 1, 1000)

# Count non-0.1 actions
non_01_action_count = 0
for lake_state in random_states:
    action = compute_policy_output(lake_state, rbf_parameters)
    if action != 0.1:
        non_01_action_count += 1


natural_inflow = generate_inflow_data(mean=mu, variance=sigma, single=True)  # Generate single inflow value
print(natural_inflow)

# Output results
print("--- Random Lake States Evaluation ---")
print(f"Total Random States Evaluated: {len(random_states)}")
print(f"Non-0.1 Actions Count: {non_01_action_count}")
print(f"Percentage of Non-0.1 Actions: {100 * non_01_action_count / len(random_states):.2f}%")
