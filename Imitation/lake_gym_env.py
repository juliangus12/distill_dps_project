import os
import sys
import numpy as np
from gymnasium import spaces, Env

# Necessary for importing the functions inside of Utils/lake_model_utils.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from Utils.lake_model_utils import (
    DPSExpertPolicy,
    generate_inflow_data,
    num_years as nYears,
    phosphorus_recycling_rate as q,
    phosphorus_loss_rate as b,
    benefit_scale as alpha,
    discount_rate as delta,
    inflow_mean as mu,
    inflow_variance as sigma,
    inertia_threshold,
    reliability_threshold
)

class LakeEnv(Env):
    """Custom Lake Environment that follows gym.Env"""
    metadata = {'render_modes': ['None']}

    def __init__(self, n_years=nYears, q=q, b=b, alpha=alpha, delta=delta,
                 inertia_threshold=-inertia_threshold, reliability_threshold=reliability_threshold,
                 C=None, R=None, W=None):
        super(LakeEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Box(low=np.array([0.01]), high=np.array([0.1]), shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32)

        # Initialize the DPS expert policy
        if C is None or R is None or W is None:
            raise ValueError("C, R, and W parameters for the expert policy must be provided.")
        self.expert_policy = DPSExpertPolicy(C, R, W, self.observation_space, self.action_space)

        # Initialize environment parameters
        self.n_years = n_years
        self.q = q
        self.b = b
        self.alpha = alpha
        self.delta = delta
        self.inertia_threshold = inertia_threshold
        self.reliability_threshold = reliability_threshold
        self.lake_state = 0.0  # Initial phosphorus concentration
        self.year = 0

    def step(self, learner_action):
        # Clip the learner's action and ensure it's a (1, 1) shaped array
        learner_action = np.clip(learner_action, 0.01, 0.1).reshape(1, 1)

        # Calculate lake dynamics
        P_recycling = ((self.lake_state ** self.q) / (1 + self.lake_state ** self.q))
        natural_inflow = generate_inflow_data(mean=mu, variance=sigma, single=True)
        next_lake_state = self.lake_state * (1 - self.b) + P_recycling + learner_action + natural_inflow
        next_lake_state = np.clip(next_lake_state, self.observation_space.low[0], self.observation_space.high[0])

        # Calculate reward based on imitation accuracy
        expert_action, _ = self.expert_policy.output(np.array([self.lake_state], dtype=np.float32))
        reward = float(-abs(learner_action - expert_action))

        # Update the lake state
        self.lake_state = next_lake_state

        # Check if the environment should terminate
        self.year += 1
        terminated = self.year >= self.n_years

        # Return 1D observation and 1D reward (scalar reward)
        obs = np.array([self.lake_state], dtype=np.float32)
        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)

        # Reset environment variables
        self.lake_state = 0.0  # Initial lake phosphorus concentration
        self.year = 0

        # Return 1D observation
        obs = np.array([self.lake_state], dtype=np.float32)
        return obs, {}
