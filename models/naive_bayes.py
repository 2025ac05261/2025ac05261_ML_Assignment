# Generated from: naive_bayes.ipynb
# Converted at: 2026-08-17T09:15:30.417Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

from sklearn.naive_bayes import GaussianNB
import joblib

def train_naive_bayes(X_train, y_train):

    model = GaussianNB()

    model.fit(X_train, y_train)

    joblib.dump(
        model,
        "saved_models/naive_bayes.pkl"
    )

    return model