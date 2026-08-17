# models/train_all_models.py

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.tree import (
    DecisionTreeClassifier
)

from sklearn.neighbors import (
    KNeighborsClassifier
)

from sklearn.naive_bayes import (
    GaussianNB
)

from sklearn.ensemble import (
    RandomForestClassifier
)

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

sys.path.insert(0, PROJECT_ROOT)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("sys.path", sys.path[0])

MODEL_DIR = os.path.join(
PROJECT_ROOT,
"models"
)
os.makedirs(MODEL_DIR, exist_ok=True)

from dataset.data_preprocessing import (
    preprocess_data,
    fit_encoders
)

from utils.metrics import (
    evaluate_model
)


# --------------------------
# Load Data
# --------------------------

df = pd.read_csv("dataset/train_data.csv")

# --------------------------
# Features & Target
# --------------------------

X = df.drop(
    "label",
    axis=1
)

y = df["label"]

# --------------------------
# Encoding
# --------------------------

X, encoders = fit_encoders(X)

joblib.dump(
    encoders,
    os.path.join(
        MODEL_DIR,
        "encoders.pkl"
    )
)

# --------------------------
# Split
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# Logistic Regression
# ==========================

lr = LogisticRegression(
    max_iter=2000,
    random_state=42
)

lr.fit(
    X_train,
    y_train
)

joblib.dump(
    lr,
    os.path.join(
        MODEL_DIR,
        "logistic_regression_model.pkl"
    )
)
print("\nLogistic Regression")
print(
    evaluate_model(
        lr,
        X_test,
        y_test
    )
)

# ==========================
# Decision Tree
# ==========================

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(
    X_train,
    y_train
)

joblib.dump(
    dt,
    os.path.join(
        MODEL_DIR,
        "decision_tree_model.pkl"
    )
)

print("\nDecision Tree")
print(
    evaluate_model(
        dt,
        X_test,
        y_test
    )
)

# ==========================
# KNN
# ==========================

knn = KNeighborsClassifier(
    n_neighbors=5
)

knn.fit(
    X_train,
    y_train
)

joblib.dump(
    knn,
    os.path.join(
        MODEL_DIR,
        "knn_model.pkl"
    )
)


print("\nKNN")
print(
    evaluate_model(
        knn,
        X_test,
        y_test
    )
)

# ==========================
# Naive Bayes
# ==========================

nb = GaussianNB()

nb.fit(
    X_train,
    y_train
)

joblib.dump(
    nb,
    os.path.join(
        MODEL_DIR,
        "naive_bayes_model.pkl"
    )
)


print("\nNaive Bayes")
print(
    evaluate_model(
        nb,
        X_test,
        y_test
    )
)

# ==========================
# Random Forest
# ==========================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

joblib.dump(
    rf,
    os.path.join(
        MODEL_DIR,
        "random_forest_model.pkl"
    )
)

print("\nRandom Forest")
print(
    evaluate_model(
        rf,
        X_test,
        y_test
    )
)

print("\nAll Models Saved Successfully")