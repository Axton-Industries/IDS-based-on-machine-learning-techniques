import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    accuracy_score
)

# =========================================================
# BINARY / SCORE-BASED MODELS
# RF, XGBoost, Isolation Forest, Autoencoder
# =========================================================
def evaluate_binary(scores, y_test):

    precision, recall, thresholds = precision_recall_curve(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)

    f1 = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    best_threshold = thresholds[np.argmax(f1)]

    y_pred = (scores >= best_threshold).astype(int)

    print("\nBest threshold:", best_threshold)
    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred))

    print("\nROC-AUC:", roc_auc_score(y_test, scores))
    print("PR-AUC:", pr_auc)


# =========================================================
# MULTICLASS SUPERVISED MODELS ONLY
# =========================================================
def evaluate_multiclass(model, X_test, y_test):

    y_pred = model.predict(X_test)

    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred))

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
