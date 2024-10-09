import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LakeEnv(gym.Env):
    """Custom Lake Environment that follows gym.Env"""
    metadata = {'render_modes': ['human']}

    def __init__(self, n_years=100, q=2, b=0.42, alpha=0.4, delta=0.98, inertia_threshold=-0.02, reliability_threshold=0.85):
        super(LakeEnv, self).__init__()

        # Define action and observation space
        # We assume that the action space is the pollution control action Y between 0.01 and 0.1
        self.action_space = spaces.Box(low=np.array([0.01]), high=np.array([0.1]), dtype=np.float32)
        
        # The observation space is the lake phosphorus concentration (state) between 0 and 2
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([2]), dtype=np.float32)

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
        self.seed_value = None  # Keep track of the seed

    def step(self, action):
        """Run one timestep of the environment's dynamics."""
        Y = action[0]  # Pollution control action
        natural_inflow = np.random.uniform(0, 0.05)  # Example natural phosphorus inflow (replace with real flow data)
        
        # Update lake phosphorus concentration
        self.lake_state = self.lake_state * (1 - self.b) + (self.lake_state ** self.q) / (1 + self.lake_state ** self.q) + Y + natural_inflow

        # Reward is based on economic benefit
        reward = self.alpha * Y * (self.delta ** self.year)

        # Check if the lake's state has gone above a critical threshold
        done = self.year >= self.n_years

        # Increment year counter
        self.year += 1

        # Return observation (lake state), reward, done flag, and additional info
        return np.array([self.lake_state]), reward, done, {}

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        # Set the seed for reproducibility
        self.seed_value = seed
        if seed is not None:
            np.random.seed(seed)

        # Reset environment variables
        self.lake_state = np.random.uniform(0, 1)  # Reset lake phosphorus concentration
        self.year = 0
        
        return np.array([self.lake_state]), {}  # Return observation and info

    def render(self, mode='human', close=False):
        """Render the environment (optional)."""
        print(f"Year: {self.year}, Lake Phosphorus: {self.lake_state}")
