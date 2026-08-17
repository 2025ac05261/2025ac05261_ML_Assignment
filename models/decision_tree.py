# Generated from: decision_tree.ipynb
# Converted at: 2026-08-17T09:13:50.472Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

from sklearn.tree import DecisionTreeClassifier
import joblib

def train_decision_tree(X_train, y_train):

    model = DecisionTreeClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(
        model,
        "saved_models/decision_tree.pkl"
    )

    return model