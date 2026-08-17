# Generated from: evaluation.ipynb
# Converted at: 2026-08-17T09:14:36.651Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:,1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return metrics, cm, report