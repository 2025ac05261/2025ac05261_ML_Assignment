# Generated from: metrics.ipynb
# Converted at: 2026-08-17T16:53:31.366Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# utils/metrics.py

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


def evaluate_model(
        model,
        X_test,
        y_test):

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

    else:
        y_prob = y_pred

    metrics = {

        "Accuracy": round(
            accuracy_score(
                y_test,
                y_pred
            ),
            4
        ),

        "AUC": round(
            roc_auc_score(
                y_test,
                y_prob
            ),
            4
        ),

        "Precision": round(
            precision_score(
                y_test,
                y_pred
            ),
            4
        ),

        "Recall": round(
            recall_score(
                y_test,
                y_pred
            ),
            4
        ),

        "F1 Score": round(
            f1_score(
                y_test,
                y_pred
            ),
            4
        ),

        "MCC": round(
            matthews_corrcoef(
                y_test,
                y_pred
            ),
            4
        )
    }

    return metrics