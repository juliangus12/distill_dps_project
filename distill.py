import pickle
from sklearn.ensemble import GradientBoostingRegressor
from interpret.glassbox import ExplainableBoostingRegressor
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Load state-action pairs generated by DPS
with open('state_action_pairs.pkl', 'rb') as f:
    state_action_pairs = pickle.load(f)

# Split state-action pairs into X (states) and y (actions)
X = [pair[0] for pair in state_action_pairs]  # States (lake phosphorus levels)
y = [pair[1] for pair in state_action_pairs]  # Actions (pollution control actions)

# Reshape X to ensure it's 2D as expected by the models (n_samples, n_features)
X = np.array(X).reshape(-1, 1)  # Each state (phosphorus level) as a single feature

# -------------------------- Train GBM -------------------------- #

# Train a Gradient Boosting Machine to mimic the DPS policy
gbm_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gbm_model.fit(X, y)

# Save the trained GBM model
with open('gbm_model.pkl', 'wb') as f:
    pickle.dump(gbm_model, f)
print("GBM model trained and saved.")

# -------------------------- Train EBM -------------------------- #

# Train an Explainable Boosting Machine (EBM) to mimic the DPS policy
ebm_model = ExplainableBoostingRegressor()
ebm_model.fit(X, y)

# Save the trained EBM model
with open('ebm_model.pkl', 'wb') as f:
    pickle.dump(ebm_model, f)
print("EBM model trained and saved.")

# -------------------------- Train Symbolic Regression Model -------------------------- #

# Train a symbolic regression model to mimic the DPS policy
sr_model = SymbolicRegressor(generations=50, population_size=500, random_state=42, n_jobs=-1)
sr_model.fit(X, y)

# Save the trained Symbolic Regression model
with open('sr_model.pkl', 'wb') as f:
    pickle.dump(sr_model, f)
print("Symbolic Regression model trained and saved.")

# -------------------------- Hyperparameter Tuning for GBM -------------------------- #

# Define the hyperparameter grid for GBM
param_grid_gbm = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
}

# Perform RandomizedSearchCV to find the best parameters for GBM
gbm_search = RandomizedSearchCV(gbm_model, param_grid_gbm, n_iter=10, scoring='neg_mean_squared_error', random_state=42, cv=5)
gbm_search.fit(X, y)

# Get the best model and parameters for GBM
best_gbm = gbm_search.best_estimator_
print("Best GBM hyperparameters:", gbm_search.best_params_)

# Save the best GBM model
with open('best_gbm_model.pkl', 'wb') as f:
    pickle.dump(best_gbm, f)
print("Best GBM model trained and saved.")

# -------------------------- Hyperparameter Tuning for EBM -------------------------- #

# Define the hyperparameter grid for EBM
param_grid_ebm = {
    'max_bins': [128, 256],  # Number of bins used in histogram
    'max_interaction_bins': [16, 32],  # Bins for interaction terms
    'learning_rate': [0.01, 0.05, 0.1],  # Learning rate
    'n_estimators': [100, 200, 300],  # Number of boosting rounds
    'max_leaves': [3, 5, 10],  # Maximum number of leaves in each tree
}

# Perform RandomizedSearchCV for EBM
ebm_search = RandomizedSearchCV(ebm_model, param_grid_ebm, n_iter=10, scoring='neg_mean_squared_error', random_state=42, cv=5)
ebm_search.fit(X, y)

# Get the best model and parameters for EBM
best_ebm = ebm_search.best_estimator_
print("Best EBM hyperparameters:", ebm_search.best_params_)

# Save the best EBM model
with open('best_ebm_model.pkl', 'wb') as f:
    pickle.dump(best_ebm, f)
print("Best EBM model trained and saved.")

# -------------------------- Hyperparameter Tuning for Symbolic Regression -------------------------- #

# Define the hyperparameter grid for Symbolic Regression
param_grid_sr = {
    'population_size': [500, 1000, 1500],
    'generations': [50, 100, 200],  # Number of generations to evolve
    'tournament_size': [3, 5, 7],  # Tournament size for selection
    'stopping_criteria': [0.01, 0.001],  # Early stopping based on fitness
    'parsimony_coefficient': [0.001, 0.01, 0.1],  # Coefficient to control complexity
    'function_set': [['add', 'sub', 'mul', 'div'], ['add', 'sub', 'mul', 'div', 'sqrt', 'log']],
}

# Perform RandomizedSearchCV for Symbolic Regression
sr_search = RandomizedSearchCV(sr_model, param_grid_sr, n_iter=10, scoring='neg_mean_squared_error', random_state=42, cv=5)
sr_search.fit(X, y)

# Get the best model and parameters for Symbolic Regression
best_sr = sr_search.best_estimator_
print("Best Symbolic Regression hyperparameters:", sr_search.best_params_)

# Save the best Symbolic Regression model
with open('best_sr_model.pkl', 'wb') as f:
    pickle.dump(best_sr, f)
print("Best Symbolic Regression model trained and saved.")
