
"""
S6E1 Stage 3 - FT-Transformer MERGE
===================================
Merges results from 3 seeds: 42, 1003, 2024.
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error

# Configuration
SEEDS = [42, 1003, 2024]
TARGET = 'exam_score'

# Load Train for scoring
if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
else:
    train_df = pd.read_csv("Dataset/train.csv")
    
y_train = train_df[TARGET].values

print(f"Merging Seeds: {SEEDS}")

oof_preds = []
sub_preds = []

for seed in SEEDS:
    oof_file = f"oof_stage3_ftt_seed{seed}.csv"
    sub_file = f"submission_stage3_ftt_seed{seed}.csv"
    
    if os.path.exists(oof_file) and os.path.exists(sub_file):
        print(f"Loading {oof_file} & {sub_file}...")
        oof = pd.read_csv(oof_file)
        sub = pd.read_csv(sub_file)
        
        oof_preds.append(oof[TARGET].values)
        sub_preds.append(sub[TARGET].values)
    else:
        print(f"⚠️ Missing files for seed {seed}! Skipping...")

if len(oof_preds) == 0:
    print("No files found!")
    exit()

# Average
avg_oof = np.mean(oof_preds, axis=0)
avg_sub = np.mean(sub_preds, axis=0)

# Score
final_rmse = np.sqrt(mean_squared_error(y_train, avg_oof))
print(f"\nFinal Merged OOF RMSE ({len(SEEDS)} seeds): {final_rmse:.5f}")

# Save
pd.DataFrame({'id': train_df['id'], 'exam_score': avg_oof}).to_csv("oof_stage3_ftt.csv", index=False)
sub[TARGET] = avg_sub
sub.to_csv("submission_stage3_ftt.csv", index=False)

print("Saved Final Files: oof_stage3_ftt.csv, submission_stage3_ftt.csv")
