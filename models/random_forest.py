# Generated from: random_forest.ipynb
# Converted at: 2026-08-17T09:15:41.584Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

from sklearn.ensemble import RandomForestClassifier
import joblib

def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(
        model,
        "saved_models/random_forest.pkl"
    )

    return model