import os
import shutil
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from imitation.algorithms import dagger, bc
from imitation.data import rollout
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.wrappers import RolloutInfoWrapper
import numpy as np
import pickle
import sys
from lake_expert import *


# Redirect terminal output to a text file
sys.stdout = open('output_log.txt', 'w')

# Load the RBF parameters generated in DPS.py
with open('rbf_params.pkl', 'rb') as f:
    C, R, W = pickle.load(f)

# Register the custom environment
gym.register(
    id='LakeEnv-v1',
    entry_point='lake_env:LakeEnv',
    max_episode_steps=100,
)

# Create a vectorized environment
n_envs = 1  # Number of parallel environments
rng = np.random.default_rng(42)  # Create a random number generator

lake_env = DummyVecEnv([lambda: RolloutInfoWrapper(FlattenObservation(gym.make("LakeEnv-v1"))) for _ in range(n_envs)])

# Instantiate the expert policy
expert_policy = DPSExpertPolicy(
    C, R, W,
    observation_space=lake_env.observation_space,
    action_space=lake_env.action_space
)

# Set up BC (Behavior Cloning) trainer with RNG
bc_trainer = bc.BC(
    observation_space=lake_env.observation_space,
    action_space=lake_env.action_space,
    rng=rng  # Provide the rng keyword argument here
)

# Set up DAgger with RNG
dagger_trainer = dagger.SimpleDAggerTrainer(
    venv=lake_env,
    expert_policy=expert_policy,
    bc_trainer=bc_trainer,
    rng=rng,  # Pass the random number generator here
    scratch_dir="dagger_scratch"
)

# Function to remove existing demos for a round
def clear_existing_demos(round_number):
    demos_dir = f"dagger_scratch/demos/round-{round_number:03d}"
    if os.path.exists(demos_dir):
        shutil.rmtree(demos_dir)
        print(f"Cleared demos for round {round_number} in {demos_dir}")

# Step 1: Collect expert demonstrations for round 0
print("Collecting expert demonstrations...")
collector = dagger_trainer.create_trajectory_collector()

# Function to generate expert demonstrations
def collect_expert_demos(collector, round_number):
    # Ensure the demo directory for the current round does not already exist
    clear_existing_demos(round_number)

    # Define how many timesteps you want to collect per round
    sample_until = rollout.make_sample_until(min_timesteps=1000)  # Adjust as needed

    # Collect rollouts from the expert policy
    rollouts = rollout.generate_trajectories(
        policy=expert_policy,
        venv=collector,
        sample_until=sample_until,
        rng=rng,
    )
    return rollouts

# Collect expert demos for round 0
rollouts = collect_expert_demos(collector, round_number=0)

# Step 2: Proceed with DAgger training after collecting demonstrations
total_timesteps = 50000
timesteps_collected = 0
round_number = 1  # Start from round 1 for DAgger updates

while timesteps_collected < total_timesteps:
    # Collect demonstrations for the current round
    print(f"Collecting expert demonstrations for round {round_number}...")
    collector = dagger_trainer.create_trajectory_collector()
    rollouts = collect_expert_demos(collector, round_number)

    # Step 3: Update DAgger policy using the newly collected expert demos
    print(f"Updating DAgger for round {round_number}...")
    dagger_trainer.extend_and_update(bc_train_kwargs=None)

    # Increment collected timesteps and round number
    timesteps_collected += sum(len(traj) for traj in rollouts)
    round_number += 1

# Save the trained learner policy
dagger_trainer.policy.save("learner_policy")

print("Learner policy saved to 'learner_policy'")

# Close the file
sys.stdout.close()
