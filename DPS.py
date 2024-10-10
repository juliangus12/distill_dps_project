# DPS.py

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
from imitation.data.rollout import make_sample_until
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
from gymnasium.envs.registration import register
from lake_env import LakeEnv
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

best_hv = -np.inf
best_solution = None

for i, solution in enumerate(res.F):
    solution_hv = hv_indicator.do(solution)
    if solution_hv > best_hv:
        best_hv = solution_hv
        best_solution = res.X[i]

print(f"Best hypervolume: {best_hv}")
print("State-action pairs saved to 'state_action_pairs.pkl'.")

C, R, W = best_solution[0:n], best_solution[n:2*n], best_solution[2*n:3*n]

# ----------------------- Modified DPSPolicy Class ----------------------- #
class DPSPolicy:
    """DPS Expert Policy derived from NSGA-II optimization."""
    def __init__(self, C, R, W, observation_space, action_space):
        self.C = C
        self.R = R
        self.W = W
        self.observation_space = observation_space
        self.action_space = action_space

    def predict(self, obs):
        lake_state = obs[0]
        action = RBFpolicy(lake_state, self.C, self.R, self.W)
        return [action], None
    
    def __call__(self, obs):
        return self.predict(obs)[0]

# Register the custom LakeEnv environment
register(
    id='LakeEnv-v1',
    entry_point='lake_env:LakeEnv',
)

# Function to create environment instances
def make_lake_env():
    return LakeEnv()

# Main execution block
if __name__ == "__main__":
    # Create the vectorized environment with the environment factory
    n_envs = 4  # Number of parallel environments
    lake_env = SubprocVecEnv([make_lake_env for _ in range(n_envs)])
    lake_env = RolloutInfoWrapper(lake_env)

    # Properly seed the SubprocVecEnv
    lake_env.seed(42)

    # Reset without passing seed (fixes TypeError)
    lake_env.reset()  

    # Create the DPS expert policy
    expert_policy = DPSPolicy(C, R, W, lake_env.observation_space, lake_env.action_space)

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
        expert_policy=expert_policy,
        bc_trainer=bc_trainer,
        rng=np.random.default_rng(42)
    )

    total_timesteps = 50_000
    timesteps_collected = 0
    sample_until = make_sample_until(min_timesteps=500)

    # Loop until desired timesteps are collected
    while timesteps_collected < total_timesteps:
        rollouts = rollout.rollout(
            policy=dagger_trainer.policy,
            venv=lake_env,
            sample_until=sample_until,
            rng=np.random.default_rng(42),
            deterministic_policy=False,
            unwrap=True
        )

        processed_rollouts = []
        for traj in rollouts:
            ep_info = traj.infos[-1].get("rollout", {})
            if ep_info:
                processed_rollouts.append(ep_info)

        dagger_trainer.extend_and_update(rollouts)
        timesteps_collected += sum(len(traj) for traj in rollouts)

    observations = np.array([step.obs for traj in rollouts for step in traj])
    actions = np.array([step.act for traj in rollouts for step in traj])

    dagger_data = list(zip(observations, actions))
    with open('state_action_pairs_dagger.pkl', 'wb') as f:
        pickle.dump(dagger_data, f)

    serialize.save_policy("models/dagger_policy", dagger_trainer.policy)
