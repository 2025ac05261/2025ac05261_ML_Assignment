# app.py

import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from dataset.data_preprocessing import (
    preprocess_data,
    transform_data
)

st.set_page_config(
    page_title="Credit Card Approval",
    layout="wide"
)

st.title(
    "Credit Card Approval Prediction"
)

# -------------------------
# Upload Test Data
# -------------------------

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

# -------------------------
# Model Selection
# -------------------------

model_option = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest"
    ]
)

model_files = {

    "Logistic Regression":
    "models/logistic_regression_model.pkl",

    "Decision Tree":
    "models/decision_tree_model.pkl",

    "KNN":
    "models/knn_model.pkl",

    "Naive Bayes":
    "models/naive_bayes_model.pkl",

    "Random Forest":
    "models/random_forest_model.pkl"
}

if uploaded_file:

    df = pd.read_csv(
        uploaded_file
    )

    # preprocess

    df = preprocess_data(df)

    y_true = df["label"]

    X = df.drop(
        "label",
        axis=1
    )

    encoders = joblib.load(
        "models/encoders.pkl"
    )

    X = transform_data(
        X,
        encoders
    )

    model = joblib.load(
        model_files[model_option]
    )

    predictions = model.predict(X)

    # Metrics

    st.subheader(
        "Evaluation Metrics"
    )

    st.write(
        "Accuracy:",
        round(
            accuracy_score(
                y_true,
                predictions
            ),
            4
        )
    )

    st.write(
        "Precision:",
        round(
            precision_score(
                y_true,
                predictions
            ),
            4
        )
    )

    st.write(
        "Recall:",
        round(
            recall_score(
                y_true,
                predictions
            ),
            4
        )
    )

    st.write(
        "F1 Score:",
        round(
            f1_score(
                y_true,
                predictions
            ),
            4
        )
    )

    # Confusion Matrix

    st.subheader(
        "Confusion Matrix"
    )

    cm = confusion_matrix(
        y_true,
        predictions
    )

    st.dataframe(cm)

    # Classification Report

    st.subheader(
        "Classification Report"
    )

    report = classification_report(
        y_true,
        predictions,
        output_dict=True
    )

    st.dataframe(
        pd.DataFrame(
            report
        ).transpose()
    )