import os
import pickle
import numpy as np
from scipy.optimize import bisect
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators import hv  # Hypervolume indicator
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX  # Simulated Binary Crossover
from pymoo.operators.mutation.pm import PM  # Polynomial Mutation
from pymoo.termination import get_termination
from imitation.algorithms import dagger, bc
from imitation.policies import serialize
from imitation.util import util
from imitation.data import rollout
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium.envs.registration import register
from LakeProblem import *

# -------------------------- NSGA-II Optimization -------------------------- #

algorithm = NSGA2(
    pop_size=100, 
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 5)
problem = LakeProblem()
res = minimize(problem, algorithm, termination, seed=1, save_history=True, verbose=True)

hv_indicator = hv.HV(ref_point=np.array([1.0, 1.0, 1.0, 1.0]))

# Calculate the hypervolume for each solution and find the best one
best_hv = -np.inf
best_solution = None

for i, solution in enumerate(res.F):
    solution_hv = hv_indicator.do(solution)
    if solution_hv > best_hv:
        best_hv = solution_hv
        best_solution = res.X[i]

print(f"Best hypervolume: {best_hv}")
print("State-action pairs saved to 'state_action_pairs.pkl'.")

# Extract the final solution (best C, R, W parameters from NSGA-II)
C, R, W = best_solution[0:n], best_solution[n:2*n], best_solution[2*n:3*n]

class DPSPolicy:
    """DPS Expert Policy derived from NSGA-II optimization."""
    def __init__(self, C, R, W):
        self.C = C
        self.R = R
        self.W = W

    def predict(self, obs):
        lake_state = obs[0]  # Assuming single state (lake phosphorus)
        action = RBFpolicy(lake_state, self.C, self.R, self.W)
        return [action], None  # Return action as list

# Create the DPS expert policy using the optimized parameters
expert_policy = DPSPolicy(C, R, W)

# -------------------------- DAgger Integration -------------------------- #

# Register the custom LakeEnv environment
register(
    id='LakeEnv-v1',
    entry_point='lake_env:LakeEnv',  # Register the lake environment
)

lake_env = util.make_vec_env("LakeEnv-v1", n_envs=1, rng=np.random.default_rng(42))

# Initialize BC (Behavioral Cloning) Trainer
bc_trainer = bc.BC(
    observation_space=lake_env.observation_space,
    action_space=lake_env.action_space,
    rng=np.random.default_rng(42),
)

# Initialize DAgger
dagger_trainer = dagger.SimpleDAggerTrainer(
    venv=lake_env,
    scratch_dir='dagger_scratch',  
    expert_policy=expert_policy,  # Use the DPS expert policy
    bc_trainer=bc_trainer,  # Pass the BC trainer here
    rng=np.random.default_rng(42)
)

# Train with DAgger for 50,000 timesteps
dagger_trainer.train(total_timesteps=50_000)

# Save the new state-action pairs collected using DAgger
rollouts = rollout.rollout(dagger_trainer.policy, lake_env, n_timesteps=50_000)
observations = np.array([step.obs for traj in rollouts for step in traj])
actions = np.array([step.act for traj in rollouts for step in traj])

# Save DAgger-collected data
dagger_data = list(zip(observations, actions))
with open('state_action_pairs_dagger.pkl', 'wb') as f:
    pickle.dump(dagger_data, f)

# Save the DAgger-trained policy
serialize.save_policy("models/dagger_policy", dagger_trainer.policy)
