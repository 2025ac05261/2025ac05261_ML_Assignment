# Generated from: logistic_regression.ipynb
# Converted at: 2026-08-17T09:15:12.900Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

from sklearn.linear_model import LogisticRegression
import joblib

def train_logistic(X_train, y_train):

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    joblib.dump(
        model,
        "saved_models/logistic_regression.pkl"
    )

    return model