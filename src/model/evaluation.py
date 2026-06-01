import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    accuracy_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# =========================================================
# BINARY / SCORE-BASED MODELS
# =========================================================
def evaluate_binary(scores, y_test):

    precision, recall, thresholds = precision_recall_curve(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)

    f1 = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    best_threshold = thresholds[np.argmax(f1)]

    y_pred = (scores >= best_threshold).astype(int)

    print("\nBest threshold:", best_threshold)

    # ---------------- CONFUSION MATRIX ----------------
    cm = confusion_matrix(y_test, y_pred)
    print("\n=== CONFUSION MATRIX ===")
    print(cm)

    cm = confusion_matrix(y_test, y_pred, normalize="true")
    print("\n=== CONFUSION MATRIX (ROW-NORMALIZED) ===")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format=".2%")
    plt.title("Confusion Matrix (Binary, % per true class)")
    plt.show()

    # ---------------- CLASSIFICATION REPORT ----------------
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred))

    # ---------------- ROC CURVE ----------------
    roc_auc = roc_auc_score(y_test, scores)
    print("\nROC-AUC:", roc_auc)

    RocCurveDisplay.from_predictions(y_test, scores)
    plt.title(f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.show()

    # ---------------- PR CURVE ----------------
    print("PR-AUC:", pr_auc)

    PrecisionRecallDisplay.from_predictions(y_test, scores)
    plt.title(f"Precision-Recall Curve (AP = {pr_auc:.4f})")
    plt.show()


# =========================================================
# MULTICLASS SUPERVISED MODELS ONLY
# =========================================================
def evaluate_multiclass(model, X_test, y_test):

    y_pred = model.predict(X_test)

    # ---------------- CONFUSION MATRIX ----------------
    cm = confusion_matrix(y_test, y_pred)
    print("\n=== CONFUSION MATRIX ===")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix (Multiclass)")
    plt.show()

    # ---------------- CLASSIFICATION REPORT ----------------
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred))

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
