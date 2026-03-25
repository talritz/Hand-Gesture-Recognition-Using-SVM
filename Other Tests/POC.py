# used libraries:
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score, \
    ConfusionMatrixDisplay, roc_curve
import pandas as pd
import hyper_parameters as hp
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

    train_subjects = [1]
    val_subjects = [2]

    print("\n--- Loading raw data from NinaPro DB2 ---")
    train_data = load_cleaned_ninapro_data(base_path, train_subjects, 'Train')
    val_data = load_cleaned_ninapro_data(base_path, val_subjects, 'Validation')

    print("\n--- Extracting features ---")
    train_features = extract_all_features(train_data)
    val_features = extract_all_features(val_data)

    columns_to_drop = ['Restimulus', 'Subject', 'dataset_type']
    X_train = train_features.drop(columns=columns_to_drop)
    y_train = train_features['Restimulus']
    X_val = val_features.drop(columns=columns_to_drop)
    y_val = val_features['Restimulus']

    class_labels = sorted(y_train.unique())

    # ---------------------------------------------------------
    # --- Under-sampling Class 0 (Rest) for Training Data ---
    # ---------------------------------------------------------
    print("\n--- Under-sampling Class 0 (Rest) ---")

    # מציאת האינדקסים של מחלקת המנוחה ושל שאר המחוות
    rest_indices = y_train[y_train == 0].index
    move_indices = y_train[y_train != 0].index

    # הגדרת כמות דגימות המנוחה שאנחנו רוצים לשמור (למשל 2000)
    # אפשר לשנות את המספר הזה כדי לראות איך הוא משפיע על ה-Balanced Accuracy
    DESIRED_REST_SAMPLES = 200

    # שימוש ב-Seed קבוע (למשל 42) כדי לקבל את אותה "הגרלה" מפוזרת בכל הרצה
    rng = np.random.default_rng(seed=42)
    sampled_rest_indices = rng.choice(rest_indices, size=DESIRED_REST_SAMPLES, replace=False)

    # איחוד האינדקסים החדשים (ה-2000 של המנוחה + כל שאר המחוות במלואן)
    balanced_indices = np.concatenate([sampled_rest_indices, move_indices])

    # סינון ה-DataFrame המקורי לפי האינדקסים שבחרנו
    X_train = X_train.loc[balanced_indices]
    y_train = y_train.loc[balanced_indices]

    print(f"Reduced Class 0 from {len(rest_indices)} to {DESIRED_REST_SAMPLES} samples.")
    print(f"New X_train shape: {X_train.shape}")
    # ---------------------------------------------------------

    # Normalizing the data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("\n--- Custom One-vs-One Training with Pairwise Youden's J ---")

    for kernel, param in model_list.items():
        print(f"\nTraining Custom OvO for Kernel: {kernel.upper()}")

        models_dict = {}
        thresholds_dict = {}

        # 1. Training and Thresholding Phase (Pair by Pair)
        # itertools.combinations gives us all unique pairs (e.g., (0, 1), (0, 5)...)
        pairs = list(itertools.combinations(class_labels, 2))

        for cls_a, cls_b in pairs:
            # Filter train data for just these two classes
            train_mask = y_train.isin([cls_a, cls_b])
            X_train_pair = X_train_scaled[train_mask]
            y_train_pair = y_train[train_mask]

            # Filter validation data to find the optimal threshold for this pair
            val_mask = y_val.isin([cls_a, cls_b])
            X_val_pair = X_val_scaled[val_mask]
            y_val_pair = y_val[val_mask]

            # Initialize binary model
            if kernel == 'linear':
                model = LinearSVC(random_state=42, **param, max_iter=10000)
            else:
                model = SVC(kernel=kernel, random_state=42, **param)

            model.fit(X_train_pair, y_train_pair)
            models_dict[(cls_a, cls_b)] = model

            # To calculate ROC, we treat cls_b as the positive class (1) and cls_a as (0)
            # This matches sklearn's internal logic since cls_a < cls_b
            y_val_pair_binary = np.where(y_val_pair == cls_b, 1, 0)
            scores = model.decision_function(X_val_pair)

            # Find optimal threshold using Youden's J
            fpr, tpr, thresholds = roc_curve(y_val_pair_binary, scores)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            best_thresh = thresholds[best_idx]

            thresholds_dict[(cls_a, cls_b)] = best_thresh

        print("Training phase complete. All 36 classifiers optimized.")

        # 2. Prediction Phase (The Voting Mechanism)
        print("Evaluating using custom Voting with optimal thresholds...")

        n_samples = X_val_scaled.shape[0]
        # Dictionary to hold the vote count for each class for every sample
        votes = {cls: np.zeros(n_samples) for cls in class_labels}

        for cls_a, cls_b in pairs:
            model = models_dict[(cls_a, cls_b)]
            thresh = thresholds_dict[(cls_a, cls_b)]

            # Evaluate ALL validation samples on this specific binary classifier
            scores = model.decision_function(X_val_scaled)

            # If score > thresh, the model votes for cls_b. Otherwise, cls_a.
            votes_for_b = (scores > thresh).astype(int)
            votes_for_a = 1 - votes_for_b

            votes[cls_b] += votes_for_b
            votes[cls_a] += votes_for_a

        # The final prediction is the class with the maximum votes
        votes_matrix = np.column_stack([votes[cls] for cls in class_labels])
        y_val_pred_idx = np.argmax(votes_matrix, axis=1)

        # Convert index (0-8) back to actual class labels (0, 1, 5...)
        y_val_pred_custom = np.array([class_labels[i] for i in y_val_pred_idx])

        # 3. Metrics and Plotting
        bal_acc = balanced_accuracy_score(y_val, y_val_pred_custom)
        macro_f1 = precision_recall_fscore_support(y_val, y_val_pred_custom, average='macro', zero_division=0)[2]
        per_class_recall = precision_recall_fscore_support(y_val, y_val_pred_custom, average=None, zero_division=0)[1]

        print(f"\n    --- CUSTOM OvO RESULTS ({kernel.upper()}) ---")
        print(f"    Macro F1: {macro_f1 * 100:.2f}% | Balanced Acc: {bal_acc * 100:.2f}%")
        print("    Per-Class Recall:")
        for cls, recall in zip(class_labels, per_class_recall):
            print(f"      Class {cls:<2}: {recall * 100:.2f}%")

        cm = confusion_matrix(y_val, y_val_pred_custom, labels=class_labels)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)
        plt.title(f"Custom OvO Confusion Matrix - {kernel.upper()} Kernel")

        plt.show()  # Displaying to screen for POC


if __name__ == "__main__":
    main()