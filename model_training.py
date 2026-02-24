from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score
import pandas as pd
import numpy as np


def evaluate_svm_kernels(df_train_features, df_val_features):
    """
    Normalizes the data, trains SVM models using different kernels,
    evaluates them using Macro F1 and Balanced Accuracy on the Validation set,
    and prints the Per-Class Recall and Confusion Matrix for the best model.
    """
    print("\nPreparing data for modeling...")

    columns_to_drop = ['Restimulus', 'Subject', 'dataset_type']

    # 1. Separate features (X) and labels (y)
    X_train = df_train_features.drop(columns=columns_to_drop)
    y_train = df_train_features['Restimulus']

    X_val = df_val_features.drop(columns=columns_to_drop)
    y_val = df_val_features['Restimulus']

    # Get the ordered list of unique classes
    class_labels = sorted(y_train.unique())

    # 2. Data Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 3. Setup the kernels competition
    kernels_to_test = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel_val_results = {}
    trained_models = {}

    print("\n--- Phase A: SVM Kernel Evaluation (Validation Set) ---")
    print(f"Training on {len(X_train_scaled)} samples. This may take a while for non-linear kernels...")

    for kernel_name in kernels_to_test:
        print(f"\nTraining SVM with '{kernel_name}' kernel...")

        # Optimization for linear kernel
        if kernel_name == 'linear':
            svm_model = LinearSVC(random_state=42, dual=False, max_iter=10000)
        else:
            svm_model = SVC(kernel=kernel_name, random_state=42)

        # Train
        svm_model.fit(X_train_scaled, y_train)

        # Predict on validation
        y_val_pred = svm_model.predict(X_val_scaled)

        # Calculate evaluation metrics on validation
        _, _, macro_f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average='macro', zero_division=0)
        bal_acc = balanced_accuracy_score(y_val, y_val_pred)

        # Store results
        kernel_val_results[kernel_name] = {'macro_f1': macro_f1, 'balanced_accuracy': bal_acc}
        trained_models[kernel_name] = svm_model

        print(f"--> Macro F1: {macro_f1 * 100:.2f}% | Balanced Accuracy: {bal_acc * 100:.2f}%")

    # 4. Find the winning kernel based on Validation Macro F1-Score
    best_kernel = max(kernel_val_results, key=lambda k: kernel_val_results[k]['macro_f1'])

    print("\n" + "=" * 55)
    print(f"Best Kernel Selected: '{best_kernel.upper()}' (Based on highest Macro F1-Score)")
    print("=" * 55)

    # 5. Evaluate best model comprehensively on Validation set
    print(f"\n--- Phase B: Best Model Analysis (on Validation Set) ---")

    best_model = trained_models[best_kernel]
    y_val_pred_best = best_model.predict(X_val_scaled)

    # Calculate detailed validation metrics
    _, _, val_macro_f1, _ = precision_recall_fscore_support(y_val, y_val_pred_best, average='macro', zero_division=0)
    val_bal_acc = balanced_accuracy_score(y_val, y_val_pred_best)

    # Calculate Per-Class Recall
    _, val_per_class_recall, _, _ = precision_recall_fscore_support(y_val, y_val_pred_best, average=None,
                                                                    zero_division=0)

    print(f"BEST MODEL VALIDATION METRICS:")
    print(f"Macro F1-Score:    {val_macro_f1 * 100:.2f}%")
    print(f"Balanced Accuracy: {val_bal_acc * 100:.2f}%\n")

    print("Per-Class Recall:")
    for cls, recall_val in zip(class_labels, val_per_class_recall):
        print(f"  Class {cls:<2}: {recall_val * 100:.2f}%")

    # 6. Generate and print Validation Confusion Matrix
    print("\n--- Validation Confusion Matrix ---")
    cm = confusion_matrix(y_val, y_val_pred_best, labels=class_labels)

    # Create a readable pandas DataFrame
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in class_labels], columns=[f"Pred_{c}" for c in class_labels])
    print(cm_df.to_string())
    print("=" * 55)

    return kernel_val_results