import numpy as np
from LakeProblem import RBFpolicy
import pickle

# Load the RBF parameters generated in DPS.py
with open('rbf_params.pkl', 'rb') as f:
    C, R, W = pickle.load(f)

# DPSExpertPolicy in lake_agent.py

class DPSExpertPolicy:
    def __init__(self, C, R, W, observation_space, action_space):
        self.C = C
        self.R = R
        self.W = W
        self.observation_space = observation_space
        self.action_space = action_space

    def predict(self, obs):
        lake_state = obs[0]  # Assuming the observation is the lake_state
        action = RBFpolicy(lake_state, self.C, self.R, self.W)
        return np.array([[action]], dtype=np.float32), None  # Ensure (1, 1) shape

    def __call__(self, obs, state=None, dones=None):
        """Make the class callable and accept extra arguments."""
        actions, state = self.predict(obs)
        return actions, state
