# used libraries:
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score, \
    ConfusionMatrixDisplay, roc_curve, auc
import pandas as pd
from data_loading import load_cleaned_ninapro_data
from feature_extraction import extract_all_features

# --- Intel CPU Acceleration ---
from sklearnex import patch_sklearn
patch_sklearn()
# ------------------------------

model_list = {
    'linear': {'C': 1.5},
    'poly': {'C': 8, 'gamma': 'auto', 'degree': 3},
    'rbf': {'C': 0.1, 'gamma': 'scale'},
    'sigmoid': {'C': 0.05, 'gamma': 'scale'}
}


def main():
    paths_to_check = [
        r'C:\Users\Tal\OneDrive - Afeka College Of Engineering\הקבצים של Nadav Matza - פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'C:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets'
    ]

    base_path = next((p for p in paths_to_check if os.path.exists(p)), None)
    if not base_path:
        print("Error: Dataset path not found!")
        return

    # ---------------------------------------------------------
    # --- 1. Cross-Subject Split
    # ---------------------------------------------------------
    train_subjects = [1,2,3,4,5,6,7,8]
    val_subjects = [9,10,11,12]

    print(f"\n--- Loading Train Data (Subjects: {train_subjects}) ---")
    train_data = load_cleaned_ninapro_data(base_path, train_subjects, 'Train')

    print(f"\n--- Loading Validation Data (Subjects: {val_subjects}) ---")
    val_data = load_cleaned_ninapro_data(base_path, val_subjects, 'Validation')

    print("\n--- Extracting features (6 Time-Domain Features) ---")
    train_features = extract_all_features(train_data)
    val_features = extract_all_features(val_data)

    columns_to_drop = ['Restimulus', 'Subject', 'dataset_type']
    X_train = train_features.drop(columns=columns_to_drop)
    y_train = train_features['Restimulus']

    X_val = val_features.drop(columns=columns_to_drop)
    y_val = val_features['Restimulus']

    class_labels = sorted(y_train.unique())

    # ---------------------------------------------------------
    # --- 2. Under-sampling Class 0 (Rest) for Train ONLY ---
    # ---------------------------------------------------------
    print("\n--- Under-sampling Class 0 (Rest) on Train set ---")
    rest_indices = y_train[y_train == 0].index
    move_indices = y_train[y_train != 0].index

    DESIRED_REST_SAMPLES = 2000
    rng = np.random.default_rng(seed=42)
    sampled_rest_indices = rng.choice(rest_indices, size=DESIRED_REST_SAMPLES, replace=False)

    balanced_indices = np.concatenate([sampled_rest_indices, move_indices])

    X_train = X_train.loc[balanced_indices]
    y_train = y_train.loc[balanced_indices]

    print(f"Reduced Class 0 in Train from {len(rest_indices)} to {DESIRED_REST_SAMPLES} samples.")

    # ---------------------------------------------------------
    # --- 3. Normalization (StandardScaler) ---
    # ---------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # ---------------------------------------------------------
    # --- 4. Dimensionality Reduction (PCA) ---
    # ---------------------------------------------------------
    print("\n--- Applying PCA (Retaining 85% of Variance) ---")
    pca = PCA(n_components=0.85, random_state=42)

    # מאמנים את ה-PCA רק על נתוני האימון כדי למנוע דליפת מידע!
    X_train_pca = pca.fit_transform(X_train_scaled)
    # מתמירים את נתוני הבדיקה לפי הנוסחה שלמדנו מהאימון
    X_val_pca = pca.transform(X_val_scaled)

    print(f"Original dimensions: {X_train_scaled.shape[1]}")
    print(f"Reduced dimensions after PCA: {X_train_pca.shape[1]}")

    # מעדכנים את המשתנים כדי שהמודל ירוץ על הנתונים הדחוסים
    X_train_scaled = X_train_pca
    X_val_scaled = X_val_pca

    # ---------------------------------------------------------
    # --- 5. Custom OvO Training & Pairwise Youden's J ---
    # ---------------------------------------------------------
    print("\n--- Custom One-vs-One Training with Pairwise Youden's J ---")

    for kernel, param in model_list.items():
        print(f"\nTraining Custom OvO for Kernel: {kernel.upper()}")

        models_dict = {}
        thresholds_dict = {}
        pairs = list(itertools.combinations(class_labels, 2))

        for i, (cls_a, cls_b) in enumerate(pairs):
            train_mask = y_train.isin([cls_a, cls_b])
            X_train_pair = X_train_scaled[train_mask]
            y_train_pair = y_train[train_mask]

            val_mask = y_val.isin([cls_a, cls_b])
            X_val_pair = X_val_scaled[val_mask]
            y_val_pair = y_val[val_mask]

            if kernel == 'linear':
                model = LinearSVC(random_state=42, **param, max_iter=10000)
            else:
                model = SVC(kernel=kernel, random_state=42, **param)

            model.fit(X_train_pair, y_train_pair)
            models_dict[(cls_a, cls_b)] = model

            y_val_pair_binary = np.where(y_val_pair == cls_b, 1, 0)
            scores = model.decision_function(X_val_pair)

            fpr, tpr, thresholds = roc_curve(y_val_pair_binary, scores)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            best_thresh = thresholds[best_idx]

            thresholds_dict[(cls_a, cls_b)] = best_thresh

        print("Training phase complete. All 36 classifiers optimized.")

        # --- Prediction Phase (Voting) ---
        print("Evaluating using custom Voting with optimal thresholds...")
        n_samples = X_val_scaled.shape[0]
        votes = {cls: np.zeros(n_samples) for cls in class_labels}

        for cls_a, cls_b in pairs:
            model = models_dict[(cls_a, cls_b)]
            thresh = thresholds_dict[(cls_a, cls_b)]

            scores = model.decision_function(X_val_scaled)
            votes_for_b = (scores > thresh).astype(int)
            votes_for_a = 1 - votes_for_b

            votes[cls_b] += votes_for_b
            votes[cls_a] += votes_for_a

        votes_matrix = np.column_stack([votes[cls] for cls in class_labels])
        y_val_pred_idx = np.argmax(votes_matrix, axis=1)
        y_val_pred_custom = np.array([class_labels[i] for i in y_val_pred_idx])

        # --- Metrics and Plotting ---
        bal_acc = balanced_accuracy_score(y_val, y_val_pred_custom)
        macro_f1 = precision_recall_fscore_support(y_val, y_val_pred_custom, average='macro', zero_division=0)[2]
        per_class_recall = precision_recall_fscore_support(y_val, y_val_pred_custom, average=None, zero_division=0)[1]

        print(f"\n    --- CUSTOM OvO RESULTS ({kernel.upper()}) ---")
        print(f"    Macro F1: {macro_f1 * 100:.2f}% | Balanced Acc: {bal_acc * 100:.2f}%")
        print("    Per-Class Recall:")
        for cls, recall in zip(class_labels, per_class_recall):
            print(f"      Class {cls:<2}: {recall * 100:.2f}%")

        # ---------------------------------------------------------
        # --- Confusion Matrix Generation (Normalized to Percentages) ---
        # ---------------------------------------------------------
        # normalize='true' מנרמל לפי השורות (True Labels)
        cm = confusion_matrix(y_val, y_val_pred_custom, labels=class_labels, normalize='true')

        # מכפילים ב-100 כדי לקבל מספרים של 0-100 במקום 0.0-1.0
        cm_percentage = cm * 100

        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=class_labels)

        # values_format='.1f' מבטיח שהאחוזים יוצגו עם ספרה אחת אחרי הנקודה (למשל 92.5)
        disp.plot(cmap=plt.cm.Blues, values_format='.1f', ax=ax_cm)
        plt.title(f"Normalized Confusion Matrix (%) - {kernel.upper()} Kernel (85% PCA)")

        filename = f"CM_Custom_OvO_PCA85_{kernel}_Percent.png"
        plt.savefig(filename, bbox_inches='tight', dpi=200)
        plt.close(fig_cm)
        print(f"    [+] Saved Normalized Confusion Matrix to: {filename}")


if __name__ == "__main__":
    main()