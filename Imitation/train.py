import os
import pandas as pd
import numpy as np
import json
import pickle
from datasets import load_from_disk, concatenate_datasets
from sklearn.ensemble import GradientBoostingRegressor

# Load and combine datasets
data_dir = '/home/julian/Desktop/Shallow_Lake_Problem/Imitation/dagger_scratch/demos/round-009'
dataset_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
datasets = [load_from_disk(d) for d in dataset_dirs if os.path.isdir(d)]
if not datasets:
    raise ValueError("No datasets were loaded.")

df = concatenate_datasets(datasets).to_pandas()
print(f"Loaded dataset with {len(df)} trajectories.")

# Process data
def process_data(df):
    flattened_data = []
    for idx, row in df.iterrows():
        obs = [float(o[0]) for o in row['obs']]
        actions = [float(a) for a in row['acts']]
        infos = [json.loads(i) for i in row['infos']]
        rewards = [float(r) for r in row['rews']]
        for o, a, i, r in zip(obs, actions, infos, rewards):
            flattened_data.append({
                "trajectory": idx, "observation": o, "action": a,
                "P_diff": i.get("P_diff"), "natural_inflow": i.get("natural_inflow"),
                "TimeLimit_truncated": i.get("TimeLimit.truncated"), "reward": r
            })
    return pd.DataFrame(flattened_data)

df_flattened = process_data(df)
df_flattened.to_csv("processed_dataset.csv", index=False)
print("Processed dataset saved to processed_dataset.csv.")

# Train Gradient Boosting Regressor
X = np.column_stack((df_flattened['observation'], df_flattened['P_diff']))
y = df_flattened['action']
gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbm.fit(X, y)
print("Model training complete.")

# Save the model
with open("gbm_model.pkl", "wb") as f:
    pickle.dump(gbm, f)
print("Model saved to gbm_model.pkl.")
