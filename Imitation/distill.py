import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from imitation.algorithms import dagger, bc
from imitation.data import rollout
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.wrappers import RolloutInfoWrapper
from lake_gym_env import LakeEnv
import shutil
import json  # To save data in a human-readable format

# Necessary for importing the functions inside of Utils/lake_model_utils.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from Utils.lake_model_utils import DPSExpertPolicy

# Hardcoded RBF Parameters
rbf_params = [0.01863850146778845,0.9999391073694877,0.5917595241260287,0.21626196382868074,0.8277986806157667,0.4082404758739714
]

# Extract C, R, and W from rbf_params
num_rbfs = len(rbf_params) // 3
C = [rbf_params[i] for i in range(0, len(rbf_params), 3)]
R = [rbf_params[i] for i in range(1, len(rbf_params), 3)]
W = [rbf_params[i] for i in range(2, len(rbf_params), 3)]

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
        LakeEnv(rbf_params=rbf_params)  # Pass flattened RBF parameters
    )) for _ in range(n_envs)
])

# Wrap DPSExpertPolicy to ensure correct output format for get_actions
class WrappedExpertPolicy:
    def __init__(self, policy, observation_space, action_space):
        self.policy = policy
        self.observation_space = observation_space
        self.action_space = action_space

    def __call__(self, obs, state=None, dones=None):
        # Normalize observation shape
        obs = np.array(obs)
        if obs.ndim == 2 and obs.shape[1] == 1:
            obs = obs.flatten()  # Convert (N, 1) -> (N,)
        elif obs.ndim == 1 and obs.shape == (1,):
            obs = obs[0]  # Convert (1,) -> scalar

        # Call the underlying policy
        action, new_state = self.policy.predict(obs, state=state)
        return action, new_state

# Instantiate the wrapped expert policy
wrapped_expert_policy = WrappedExpertPolicy(
    policy=DPSExpertPolicy(C=C, R=R, W=W, observation_space=lake_env.observation_space, action_space=lake_env.action_space),
    observation_space=lake_env.observation_space,
    action_space=lake_env.action_space
)

# Set up BC (Behavior Cloning) trainer with RNG
bc_trainer = bc.BC(
    observation_space=lake_env.observation_space,
    action_space=lake_env.action_space,
    rng=rng  # Provide the RNG for reproducibility
)

# Set up DAgger with RNG
dagger_trainer = dagger.SimpleDAggerTrainer(
    venv=lake_env,
    expert_policy=wrapped_expert_policy,
    bc_trainer=bc_trainer,
    rng=rng,  # Pass the random number generator here
    scratch_dir="dagger_scratch"
)

# Function to remove existing demos for a round
def clear_existing_demos(round_number):
    demos_dir = f"dagger_scratch/demos/round-{round_number:03d}"
    if os.path.exists(demos_dir):
        shutil.rmtree(demos_dir)

def save_trajectories_to_txt(rollouts, round_number):
    """
    Saves rollouts (observations, actions, rewards) to a .txt file for each round.

    Args:
        rollouts (list): List of rollouts from `rollout.generate_trajectories`.
        round_number (int): The current DAgger round.
    """
    output_dir = f"dagger_scratch/debugging/round-{round_number:03d}"
    os.makedirs(output_dir, exist_ok=True)

    for i, traj in enumerate(rollouts):
        file_path = os.path.join(output_dir, f"trajectory-{i:03d}.txt")
        with open(file_path, "w") as f:
            f.write("Observations:\n")
            for obs in traj.obs:
                f.write(f"{obs.tolist()}\n")
            f.write("\nActions:\n")
            for act in traj.acts:
                f.write(f"{act.tolist()}\n")
            f.write("\nRewards:\n")
            for rew in traj.rews:
                f.write(f"{rew}\n")
    print(f"Saved trajectories for round {round_number} to {output_dir}")


# Modify collect_expert_demos to save trajectories
def collect_expert_demos(collector, round_number):
    clear_existing_demos(round_number)
    sample_until = rollout.make_sample_until(min_timesteps=1000)

    rollouts = rollout.generate_trajectories(
        policy=wrapped_expert_policy,
        venv=collector,
        sample_until=sample_until,
        rng=rng,
    )

    # Save rollouts to .txt
    save_trajectories_to_txt(rollouts, round_number)

    return rollouts

# Collect expert demos for round 0
collector = dagger_trainer.create_trajectory_collector()
rollouts = collect_expert_demos(collector, round_number=0)

# Step 2: Proceed with DAgger training after collecting demonstrations
total_timesteps = 10000
timesteps_collected = 0
round_number = 1  # Start from round 1 for DAgger updates

while timesteps_collected < total_timesteps:
    # Collect demonstrations for the current round
    collector = dagger_trainer.create_trajectory_collector()
    rollouts = collect_expert_demos(collector, round_number)

    # Step 3: Update DAgger policy using the newly collected expert demos
    dagger_trainer.extend_and_update(bc_train_kwargs=None)

    # Increment collected timesteps and round number
    timesteps_collected += sum(len(traj.acts) for traj in rollouts)
    round_number += 1

# Save the trained learner policy
learner_policy_path = "learner_policy"
print(f"Saving the learner policy to {learner_policy_path}")
dagger_trainer.bc_trainer.policy.save(learner_policy_path)