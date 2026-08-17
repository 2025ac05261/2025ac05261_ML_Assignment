# Generated from: app.ipynb
# Converted at: 2026-08-17T09:11:03.692Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dataset.data_preprocessing import (
    load_training_data,
    load_uploaded_test_data
)

from models.logistic_regression import train_logistic as logistic_model
from models.decision_tree import train_decision_tree as decision_model
from models.knn_classifier import train_knn as knn_model
from models.naive_bayes import train_naive_bayes as nb_model
from models.random_forest import train_random_forest as rf_model

from models.evaluation import evaluate_model

st.set_page_config(
    page_title="Credit Card Approval Classification",
    layout="wide"
)

st.title("Credit Card Approval Classification")

st.write(
    "Train models using Credit_card.csv and evaluate using uploaded test_data.csv"
)

# Load Training Dataset
X_train, y_train, scaler = load_training_data()

# Upload Test Dataset
uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

# Model Selection
model_name = st.selectbox(
    "Select Classification Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest"
    ]
)

# Train Selected Model
if model_name == "Logistic Regression":
    model = logistic_model(X_train, y_train)

elif model_name == "Decision Tree":
    model = decision_model(X_train, y_train)

elif model_name == "KNN":
    model = knn_model(X_train, y_train)

elif model_name == "Naive Bayes":
    model = nb_model(X_train, y_train)

else:
    model = rf_model(X_train, y_train)

# Evaluate on Uploaded Test Data
if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Dataset")

    st.dataframe(test_df.head())

    X_test, y_test = load_uploaded_test_data(
        uploaded_file,
        scaler
    )

    metrics, cm, report = evaluate_model(
        model,
        X_test,
        y_test
    )

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{metrics['Accuracy']:.4f}"
    )

    col2.metric(
        "AUC",
        f"{metrics['AUC']:.4f}"
    )

    col3.metric(
        "Precision",
        f"{metrics['Precision']:.4f}"
    )

    col1.metric(
        "Recall",
        f"{metrics['Recall']:.4f}"
    )

    col2.metric(
        "F1 Score",
        f"{metrics['F1 Score']:.4f}"
    )

    col3.metric(
        "MCC",
        f"{metrics['MCC']:.4f}"
    )

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    st.subheader("Classification Report")

    st.text(report)

else:

    st.info(
        "Please upload test_data.csv to evaluate the selected model."
    )