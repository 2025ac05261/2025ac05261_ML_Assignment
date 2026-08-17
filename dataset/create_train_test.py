# Generated from: create_train_test.ipynb
# Converted at: 2026-08-17T16:52:41.700Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# dataset/create_train_test.py

import pandas as pd

from sklearn.model_selection import train_test_split

from data_preprocessing import preprocess_data


df = pd.read_csv("../Credit_card.csv")

df = preprocess_data(df)

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

train_df.to_csv(
    "train_data.csv",
    index=False
)

test_df.to_csv(
    "test_data.csv",
    index=False
)

print("Train/Test files created.")