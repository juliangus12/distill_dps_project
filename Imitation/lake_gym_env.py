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
                 C=None, R=None, W=None, rbf_params=None):
        super(LakeEnv, self).__init__()

        # Define action and observation space with scalar actions and array observations
        self.action_space = spaces.Box(low=0.01, high=0.1, shape=(), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=np.array([np.inf], dtype=np.float32), shape=(1,), dtype=np.float32)

        # Initialize the DPS expert policy
        if rbf_params is not None:
            # vars is a flattened list of [C1, R1, W1, C2, R2, W2, ...]
            vars = rbf_params
            n_rbf = len(vars) // 3
            C = []
            R = []
            W = []
            for i in range(n_rbf):
                C.append(vars[i * 3])
                R.append(vars[i * 3 + 1])
                W.append(vars[i * 3 + 2])
            self.expert_policy = DPSExpertPolicy(C, R, W, self.observation_space, self.action_space)
        elif C is not None and R is not None and W is not None:
            self.expert_policy = DPSExpertPolicy(C, R, W, self.observation_space, self.action_space)
        else:
            raise ValueError("Either rbf_params or C, R, and W parameters for the expert policy must be provided.")

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
        learner_action = float(np.clip(learner_action, 0.01, 0.1))
        P_recycling = ((self.lake_state ** self.q) / (1 + self.lake_state ** self.q))
        natural_inflow = generate_inflow_data(mean=mu, variance=sigma, single=True)
        next_lake_state = self.lake_state * (1 - self.b) + P_recycling + learner_action + natural_inflow
        next_lake_state = np.clip(next_lake_state, self.observation_space.low, self.observation_space.high)

        expert_action, _ = self.expert_policy.predict(
            np.array([self.lake_state], dtype=np.float32).reshape(1,)
        )
        expert_action = float(expert_action)

        reward = -abs(learner_action - expert_action)

        self.lake_state = next_lake_state
        self.year += 1
        terminated = self.year >= self.n_years

        # Normalize observation to (1,)
        obs = np.array([self.lake_state], dtype=np.float32).reshape(1,)
        #print(f"DEBUG: step output obs={obs}, shape={obs.shape}, reward={reward}, terminated={terminated}")

        return obs, reward, terminated, False, {}


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.lake_state = 0.0
        self.year = 0

        # Normalize observation to (1,)
        obs = np.array([self.lake_state], dtype=np.float32).reshape(1,)
        #print(f"DEBUG: reset output obs={obs}, shape={obs.shape}")

        return obs, {}
