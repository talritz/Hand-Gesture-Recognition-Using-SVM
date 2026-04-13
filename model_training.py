"""
===============================================================================
Model Training Module
===============================================================================
Handles SVM model initialization and training. Integrates a custom inference
engine for the baseline evaluation to mathematically resolve One-vs-One voting ties.
"""
import time
import numpy as np
import itertools
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score


def predict_baseline_with_tie_breaking(svm_ovo_model, svm_ova_model, X):
    """
    Executes a custom voting mechanism. Uses standard OvO voting (threshold 0.0),
    but dynamically resolves voting ties by incorporating One-vs-Rest confidence scores.
    """
    estimators = svm_ovo_model.estimators_
    classes = svm_ovo_model.classes_
    pairs = list(itertools.combinations(classes, 2))

    n_samples = X.shape[0]
    votes = np.zeros((n_samples, len(classes)))

    # Phase 1: Standard OvO Voting accumulation based on a 0.0 decision threshold.
    for idx, (cls_i, cls_j) in enumerate(pairs):
        estimator = estimators[idx]
        scores = estimator.decision_function(X)

        vote_for_j = scores > 0.0
        vote_for_i = ~vote_for_j

        idx_i = np.where(classes == cls_i)[0][0]
        idx_j = np.where(classes == cls_j)[0][0]

        votes[:, idx_j] += vote_for_j
        votes[:, idx_i] += vote_for_i

    # Phase 2: OvA Tie-Breaking Mechanism.
    # Retrieve continuous distance metrics from the OvA classifier.
    ova_confidences = svm_ova_model.decision_function(X)

    # Normalize the OvA scores to act as micro-weights, ensuring they cannot
    # override decisive integer votes from the OvO phase.
    max_conf_per_sample = np.max(np.abs(ova_confidences), axis=1, keepdims=True)
    max_conf_per_sample[max_conf_per_sample == 0] = 1
    normalized_ova_confidences = ova_confidences / max_conf_per_sample

    # Add the normalized micro-weights to the integer votes.
    final_scores = votes + (normalized_ova_confidences * 0.1)

    # Determine the winning class index based on the adjusted score array.
    y_pred_custom_idx = np.argmax(final_scores, axis=1)
    return classes[y_pred_custom_idx]


def evaluate_single_kernel_ovo(kernel, params, X_train_scaled, y_train, X_val_scaled, y_val, class_labels):
    """
    Trains the primary OvO classifier and the secondary OvA classifier,
    evaluates performance using the custom inference logic, and returns metrics.
    """
    print("\n" + "-" * 55)
    print(f"--> Training '{kernel.upper()}' kernel with params: {params}...")

    # Initialize the base estimator depending on the kernel type.
    if kernel == 'linear':
        base_svm = LinearSVC(random_state=42, max_iter=10000, **params)
    else:
        base_svm = SVC(kernel=kernel, random_state=42, cache_size=1000, **params)

    # Train the primary One-vs-One classifier.
    print("    [1/2] Training OvO Classifier...")
    start_time_ovo = time.time()
    svm_ovo_model = OneVsOneClassifier(base_svm)
    svm_ovo_model.fit(X_train_scaled, y_train)
    print(f"          OvO Training took: {time.time() - start_time_ovo:.2f} seconds")

    # Train the secondary One-vs-Rest classifier designated for tie-breaking.
    print("    [2/2] Training OvA Classifier (for tie-breaking)...")
    start_time_ova = time.time()
    svm_ova_model = OneVsRestClassifier(base_svm)
    svm_ova_model.fit(X_train_scaled, y_train)
    print(f"          OvA Training took: {time.time() - start_time_ova:.2f} seconds")

    # Perform prediction utilizing the custom hybrid voting logic.
    y_val_pred = predict_baseline_with_tie_breaking(svm_ovo_model, svm_ova_model, X_val_scaled)

    # Compute evaluation metrics.
    macro_f1 = precision_recall_fscore_support(y_val, y_val_pred, average='macro', zero_division=0)[2]
    bal_acc = balanced_accuracy_score(y_val, y_val_pred)

    print(f"    Macro F1 (Baseline w/ OvA Tie-Break): {macro_f1 * 100:.2f}% | Balanced Acc: {bal_acc * 100:.2f}%")

    result_dict = {
        'Kernel': kernel.upper(),
        'Macro_F1_Base (%)': round(macro_f1 * 100, 2),
        'Balanced_Acc_Base (%)': round(bal_acc * 100, 2)
    }

    return result_dict, svm_ovo_model, svm_ova_model, y_val_pred