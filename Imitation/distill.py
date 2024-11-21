import os
import sys
import argparse
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from imitation.algorithms import dagger, bc
from imitation.data import rollout
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.wrappers import RolloutInfoWrapper
from lake_gym_env import *

# Necessary for importing the functions inside of Utils/lake_model_utils.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from Utils.lake_model_utils import unpack_solutions, DPSExpertPolicy


# Argument parser to accept pareto-file as input
parser = argparse.ArgumentParser(description="Distill expert policy using DAgger.")
parser.add_argument(
    "--pareto-file",
    type=str,
    required=True,
    help="Path to the Pareto frontier text file."
)
args = parser.parse_args()

# Load the Pareto frontier from the file
print(f"Loading Pareto frontier from: {args.pareto_file}")
rbf_matrix, objectives = unpack_solutions(args.pareto_file)

# Select the first solution and objectives (best solution)
best_solution_rbf = rbf_matrix[0]
best_solution_objectives = objectives[0]

# Reshape RBF parameters for DPSExpertPolicy
num_rbfs = len(best_solution_rbf) // 3
C = best_solution_rbf[:num_rbfs]
R = best_solution_rbf[num_rbfs:2 * num_rbfs]
W = best_solution_rbf[2 * num_rbfs:]

print("Loaded best solution RBF parameters and objectives.")
print("Centers (C):", C)
print("Radii (R):", R)
print("Weights (W):", W)
print("Objectives:", best_solution_objectives)

# Register the custom environment
gym.register(
    id='LakeEnv-v1',
    entry_point='lake_gym_env:LakeEnv',
    max_episode_steps=100,
)

# Create a vectorized environment
n_envs = 1  # Number of parallel environments
rng = np.random.default_rng(42)  # Set random seed for reproducibility

# Create the environment using the RBF parameters
lake_env = DummyVecEnv([
    lambda: RolloutInfoWrapper(FlattenObservation(
        LakeEnv(C=C, R=R, W=W)  # Pass the C, R, W parameters
    )) for _ in range(n_envs)
])
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
    
    # Logging the lengths of rollouts
    for traj in rollouts:
        print(f"Trajectory length: {len(traj.acts)} actions, {len(traj.obs)} observations")
        assert len(traj.acts) == len(traj.obs) - 1, f"Mismatch in actions ({len(traj.acts)}) and observations ({len(traj.obs)})"

    return rollouts

# Collect expert demos for round 0
print("Collecting expert demonstrations...")
collector = dagger_trainer.create_trajectory_collector()
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
    timesteps_collected += sum(len(traj.acts) for traj in rollouts)
    round_number += 1

# Save the trained learner policy
dagger_trainer.policy.save("learner_policy")
print("Learner policy saved to 'learner_policy'")
