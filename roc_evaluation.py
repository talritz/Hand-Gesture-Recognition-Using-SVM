import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.exceptions import ConvergenceWarning

from data_loading import load_cleaned_ninapro_data
from feature_extraction import extract_all_features
import hyperparameters as hp

# --- Intel CPU Acceleration ---
from sklearnex import patch_sklearn

patch_sklearn()
# ------------------------------

# השתקת אזהרות העצירה כדי שהמסוף יהיה נקי
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def plot_roc_and_find_youden(y_val, y_score, class_labels, kernel_name):
    y_val_bin = label_binarize(y_val, classes=class_labels)

    plt.figure(figsize=(12, 8))
    print(f"\n--- Youden's J Optimal Thresholds for {kernel_name.upper()} ---")

    for i, cls in enumerate(class_labels):
        fpr, tpr, thresholds = roc_curve(y_val_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        # Youden's J Statistic calculation
        J = tpr - fpr
        best_idx = np.argmax(J)
        best_thresh = thresholds[best_idx]
        best_J = J[best_idx]

        print(f"Class {cls:<2} | Max J: {best_J:.4f} | Optimal Threshold: {best_thresh:.4f}")

        plt.plot(fpr, tpr, lw=2, label=f'Class {cls} (AUC = {roc_auc:.2f})')
        plt.scatter(fpr[best_idx], tpr[best_idx], marker='o', color='black', zorder=5)

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR) - Recall', fontsize=12)
    plt.title(f'Multi-Class ROC Curves - {kernel_name.upper()} Kernel', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.6)

    filename = f"ROC_Curve_{kernel_name.upper()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"-> Saved ROC plot: {filename}")
    plt.close()


def main():
    total_start_time = time.time()
    print("=" * 60)
    print(f"ROC & YOUDEN'S J ANALYSIS STARTED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    paths_to_check = [
        r'C:\Users\Tal\OneDrive - Afeka College Of Engineering\הקבצים של Nadav Matza - פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'C:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets'
    ]
    base_path = next((p for p in paths_to_check if os.path.exists(p)), None)

    train_subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    val_subjects = [9, 10, 11, 12]

    print("\n--- PHASE 1: Loading Data ---")
    df_train_raw = load_cleaned_ninapro_data(base_path, train_subjects, 'Train')
    df_val_raw = load_cleaned_ninapro_data(base_path, val_subjects, 'Validation')

    print("\n--- PHASE 2: Extracting Hybrid Features ---")
    df_train_features = extract_all_features(df_train_raw)
    df_val_features = extract_all_features(df_val_raw)

    columns_to_drop = ['Restimulus', 'Subject', 'dataset_type']
    X_train = df_train_features.drop(columns=columns_to_drop)
    y_train = df_train_features['Restimulus']
    X_val = df_val_features.drop(columns=columns_to_drop)
    y_val = df_val_features['Restimulus']

    class_labels = sorted(y_train.unique())

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # All 4 kernels as requested
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    print("\n--- PHASE 3: Training & ROC Extraction (OvO) ---")
    for kernel in kernels:
        print("\n" + "-" * 50)
        print(f"Processing '{kernel.upper()}' kernel...")

        if kernel == 'linear':
            base_svm = LinearSVC(random_state=42, dual=False, C=hp.SVM_C, max_iter=hp.SVM_MAX_ITER,
                                 class_weight=hp.SVM_CLASS_WEIGHT)
        else:
            base_svm = SVC(kernel=kernel, random_state=42, C=hp.SVM_C, gamma=getattr(hp, 'SVM_GAMMA', 'scale'),
                           degree=getattr(hp, 'SVM_DEGREE', 3), class_weight=hp.SVM_CLASS_WEIGHT,
                           max_iter=hp.SVM_MAX_ITER)

        svm_model = OneVsOneClassifier(base_svm)
        svm_model.fit(X_train_scaled, y_train)

        print("Extracting decision function scores...")
        y_score = svm_model.decision_function(X_val_scaled)

        plot_roc_and_find_youden(y_val, y_score, class_labels, kernel)

    total_end_time = time.time()
    mins, secs = divmod(int(total_end_time - total_start_time), 60)
    print("\n" + "=" * 60)
    print(f"ROC ANALYSIS COMPLETED IN: {mins}m {secs}s")
    print("=" * 60)


if __name__ == "__main__":
    main()