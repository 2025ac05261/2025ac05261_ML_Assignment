# Generated from: knn_classifier.ipynb
# Converted at: 2026-08-17T09:14:52.658Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

from sklearn.neighbors import KNeighborsClassifier
import joblib

def train_knn(X_train, y_train):

    model = KNeighborsClassifier(
        n_neighbors=5
    )

    model.fit(X_train, y_train)

    joblib.dump(
        model,
        "saved_models/knn.pkl"
    )

    return model