import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    accuracy_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_curve,
    auc
)

def print_feature_importance(columns, importance_scores, top_n=20):
    """
    Simplest function to format, sort, and print feature importance.

    Parameters:
    -----------
    columns : list or Index
        The list of feature/column names.
    importance_scores : array-like
        The importance scores (e.g., model.feature_importances_, model.coef_,
        or custom deviation arrays).
    top_n : int, default=20
        The number of top features to display.
    """
    # Create the DataFrame using the provided columns and scores
    importance = pd.DataFrame({
        "Feature": columns,
        "Importance_Score": importance_scores
    }).sort_values(by="Importance_Score", ascending=False)

    # Print the top N features
    print(f"\nTop {top_n} Most Important Features:")
    print(importance.head(top_n).to_string(index=False))

# =========================================================
# BINARY / SCORE-BASED MODELS
# =========================================================
def get_defensive_threshold(val_scores, y_val, beta=2):
    """
    Finds the optimal threshold on validation data without touching the test set.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, val_scores)

    beta_sq = beta ** 2
    f_beta = ((1 + beta_sq) * precision[:-1] * recall[:-1]) / (
        (beta_sq * precision[:-1]) + recall[:-1] + 1e-9
    )

    # Return the absolute cutoff score
    return thresholds[np.argmax(f_beta)]



def evaluate_binary(scores, y_test, chosen_threshold):
    """
    Evaluates the test set blindly using a pre-determined threshold.
    """
    # Hard-apply the pre-calculated threshold. No looking at y_test to decide!
    y_pred = (scores >= chosen_threshold).astype(int)

    print(f"\nEvaluating with Pre-determined Threshold: {chosen_threshold:.6f}")

    # --- 1. Text Metrics & Reports ---
    print("\n=== RAW CONFUSION MATRIX ===")
    # CHANGED: Added labels=[0, 1] to keep formatting consistent even if a class is temporarily missing
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    print("\n=== CLASSIFICATION REPORT ===")
    # CHANGED: Added labels=[0, 1] to prevent ValueError crashes while keeping your exact target_names
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["Benign", "Attack"]))

    # --- 2. Visual Confusion Matrix ---
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        normalize="true",
        labels=[0, 1], # CHANGED: Explicitly tracks 0 and 1 to prevent dynamic shape errors
        display_labels=["Benign", "Attack"],
        cmap="Blues",
        values_format=".2%",
    )
    plt.title("Confusion Matrix (% per true class)")
    plt.show()

    # --- 3. Performance Curves (ROC and PR) ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve (Evaluates overall ranking power, independent of threshold)
    roc_auc = roc_auc_score(y_test, scores)
    RocCurveDisplay.from_predictions(y_test, scores, ax=ax[0])
    ax[0].axvline(x=0.01, color='r', linestyle='--', label='1% FPR Operational Limit')
    ax[0].set_title(f"ROC Curve (AUC = {roc_auc:.4f})")
    ax[0].legend()

    # Precision-Recall Curve
    pr_auc = average_precision_score(y_test, scores)
    PrecisionRecallDisplay.from_predictions(y_test, scores, ax=ax[1])
    ax[1].set_title(f"PR Curve (AP/AUC = {pr_auc:.4f})")

    plt.tight_layout()
    plt.show()



# =========================================================
# MULTICLASS SUPERVISED MODELS ONLY
# =========================================================
def evaluate_multiclass(model, X_test, y_test, benign_class_index=0):
    # 1. Handle inference safely
    if hasattr(model, "get_booster"):
        probs = model.get_booster().inplace_predict(
            X_test,
            predict_type="probability",
            iteration_range=(0, getattr(model, "best_iteration", 0) + 1)
        )
    else:
        probs = model.predict_proba(X_test)

    # 2 Extract theoretical classes from probabilities array shape
    # and get hard class predictions using the most probable classification
    n_classes = probs.shape[1]
    all_classes = np.arange(n_classes)
    y_pred = np.argmax(probs, axis=1)

    # ---------------- CONFUSION MATRIX ----------------
    cm = confusion_matrix(y_test, y_pred, labels=all_classes)
    print("\n=== CONFUSION MATRIX ===")
    print(cm)

    # Normalized by 'true' rows so tiny attack categories aren't hidden by massive benign traffic
    cm_normalized = confusion_matrix(y_test, y_pred, labels=all_classes, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=all_classes)
    disp.plot(cmap="Blues", xticks_rotation=45, values_format=".1%")
    plt.title("Normalized Confusion Matrix (Recall per Class)")
    plt.show()

    # ---------------- CLASSIFICATION REPORT ----------------
    print("\n=== CLASSIFICATION REPORT ===")
    # CHANGED: Added labels and zero_division control to avoid metric layout drops
    print(classification_report(y_test, y_pred, labels=all_classes, zero_division=0))

    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    # ---------------- MULTICLASS ROC & PR CURVES (One-vs-Rest) ----------------
    # Binarize against all theoretical classes to match probability matrix columns
    y_test_binarized = label_binarize(y_test, classes=all_classes)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot ROC Curves ---
    # Loop over all_classes and added condition to skip unrepresented classes in test split
    for i in all_classes:
        if np.sum(y_test == i) == 0:
            continue

        label_name = "Benign" if i == benign_class_index else f"Attack Class {i}"
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax[0].plot(fpr, tpr, label=f"{label_name} (AUC = {roc_auc:.4f})")

    ax[0].plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax[0].axvline(x=0.01, color='r', linestyle=':', label='1% FPR Operational Limit')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('Multiclass ROC Curve (One-vs-Rest)')
    ax[0].legend(loc='lower right')
    ax[0].grid(True, alpha=0.3)

    # --- Plot Precision-Recall Curves ---
    # Loop over all_classes and added condition to skip unrepresented classes in test split
    for i in all_classes:
        if np.sum(y_test == i) == 0:
            continue

        label_name = "Benign" if i == benign_class_index else f"Attack Class {i}"
        precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], probs[:, i])
        avg_pr = average_precision_score(y_test_binarized[:, i], probs[:, i])
        ax[1].plot(recall, precision, label=f"{label_name} (AP = {avg_pr:.4f})")

    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_title('Multiclass Precision-Recall Curve (One-vs-Rest)')
    ax[1].legend(loc='lower left')
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
