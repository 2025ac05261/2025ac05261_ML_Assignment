# Generated from: data_preprocessing.ipynb
# Converted at: 2026-08-17T09:13:16.990Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_training_data():

    df = pd.read_csv("dataset/Credit_card.csv")

    target_column = "label"

    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def load_uploaded_test_data(uploaded_file, scaler):

    test_df = pd.read_csv(uploaded_file)

    target_column = "label"

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    X_test_scaled = scaler.transform(X_test)

    return X_test_scaled, y_test