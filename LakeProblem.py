import os
import pickle
import numpy as np
from scipy.optimize import bisect
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators import hv  # Hypervolume indicator
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX  # Simulated Binary Crossover
from pymoo.operators.mutation.pm import PM  # Polynomial Mutation
from pymoo.termination import get_termination
from imitation.algorithms import dagger
from imitation.policies import serialize
from imitation.util import util
from imitation.data import rollout
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium.envs.registration import register

# --------------------------- Global Variables ---------------------------- #

# Number of years for lake model simulation
nYears = 100
nSamples = 100
inertia_threshold = -0.02
reliability_threshold = 0.85
b = 0.42  # Decay rate of phosphorus in the lake (natural loss)
q = 2  # Recycling rate of phosphorus in the lake
alpha = 0.4  # Weight given to economic benefits from pollution control actions
delta = 0.98  # Discount factor applied to future benefits (for economic evaluation)
n = 2  # Number of Radial Basis Functions (RBFs) used in the policy definition
nvars = n * 3
nobjs = 4
nconstrs = 1
nat_flowmat = np.zeros((10000, nYears))

# -------------------------- Helper Functions ----------------------------- #

def root_function(x, q, b):
    return (x**q) / (1 + x**q) - b * x

pCrit = bisect(root_function, 0.01, 1.0, args=(q, b))

def RBFpolicy(lake_state, C, R, W):
    """Implements the RBF policy calculation as described."""
    Y = 0  # Initialize the pollution control action Y to zero
    for i in range(len(C)):  # Loop over each RBF
        r_value = np.isscalar(R[i]) if isinstance(R[i], np.ndarray) else R[i]
        if r_value > 0:  # Avoid division by zero
            Y += W[i] * ((abs(lake_state - C[i]) / r_value) ** 3)
    return max(0.01, min(0.1, Y))

# -------------------------- Lake Model Problem Class -------------------------- #

class LakeProblem(Problem):
    def __init__(self):
        super().__init__(n_var=nvars,
                         n_obj=nobjs,
                         n_constr=nconstrs,
                         xl=np.array([-2.0] * n + [0.0] * n + [0.0] * n),
                         xu=np.array([2.0] * n + [2.0] * n + [1.0] * n))

    def _evaluate(self, X, out, *args, **kwargs):
        F, G, state_action_pairs = [], [], []

        for individual in X:
            C = individual[0:n]
            R = individual[n:2*n]
            W = individual[2*n:3*n]
            W = W / np.sum(W) if np.sum(W) != 0 else np.full(n, 1.0/n)

            average_annual_P = np.zeros(nYears)
            discounted_benefit = np.zeros(nSamples)
            yrs_inertia_met = np.zeros(nSamples)
            yrs_pCrit_met = np.zeros(nSamples)

            for s in range(nSamples):
                nat_flow = nat_flowmat[s, :]
                lake_state = np.zeros(nYears + 1)
                Y = np.zeros(nYears)
                Y[0] = RBFpolicy(lake_state[0], C, R, W)

                for i in range(nYears):
                    lake_state[i + 1] = lake_state[i] * (1 - b) + \
                                        (lake_state[i]**q) / (1 + lake_state[i]**q) + \
                                        Y[i] + nat_flow[i]
                    average_annual_P[i] += lake_state[i + 1] / nSamples
                    discounted_benefit[s] += alpha * Y[i] * (delta ** i)
                    state_action_pairs.append((lake_state[i], Y[i]))

                    if i >= 1 and (Y[i] - Y[i - 1]) > inertia_threshold:
                        yrs_inertia_met[s] += 1
                    if lake_state[i + 1] < pCrit:
                        yrs_pCrit_met[s] += 1
                    if i < (nYears - 1):
                        Y[i + 1] = RBFpolicy(lake_state[i + 1], C, R, W)

            objs = [-np.mean(discounted_benefit), 
                    np.max(average_annual_P),
                    -np.mean(yrs_inertia_met) / (nYears - 1),
                    -np.mean(yrs_pCrit_met) / nYears]
            constrs = [max(0.0, reliability_threshold - (-objs[3]))]

            F.append(objs)
            G.append(constrs)

        out["F"], out["G"] = np.array(F), np.array(G)

