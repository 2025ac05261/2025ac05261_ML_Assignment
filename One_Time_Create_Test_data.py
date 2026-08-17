# Generated from: One_Time_Create_Test_data.ipynb
# Converted at: 2026-08-17T09:11:59.652Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd

df = pd.read_csv("dataset/Credit_card.csv")

test_data = df.sample(
    frac=0.2,
    random_state=42
)

test_data.to_csv(
    "test_data.csv",
    index=False
)

print("test_data.csv created successfully")