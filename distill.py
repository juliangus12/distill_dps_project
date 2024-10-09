import pickle
from sklearn.ensemble import GradientBoostingRegressor
from interpret.glassbox import ExplainableBoostingRegressor
from gplearn.genetic import SymbolicRegressor

# Load the DAgger-generated state-action pairs
with open('state_action_pairs_dagger.pkl', 'rb') as f:
    dagger_data = pickle.load(f)

X_train = [pair[0] for pair in dagger_data]
y_train = [pair[1] for pair in dagger_data]

# Train Gradient Boosting Machine
gbm_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gbm_model.fit(X_train, y_train)
with open('models/gbm_model_dagger.pkl', 'wb') as f:
    pickle.dump(gbm_model, f)

# Train Explainable Boosting Machine
ebm_model = ExplainableBoostingRegressor()
ebm_model.fit(X_train, y_train)
with open('models/ebm_model_dagger.pkl', 'wb') as f:
    pickle.dump(ebm_model, f)

# Train Symbolic Regression
sr_model = SymbolicRegressor(generations=50, population_size=500, random_state=42)
sr_model.fit(X_train, y_train)
with open('models/sr_model_dagger.pkl', 'wb') as f:
    pickle.dump(sr_model, f)
