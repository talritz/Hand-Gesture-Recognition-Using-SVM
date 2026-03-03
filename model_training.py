from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score
import pandas as pd

import hyperparameters as hp


def evaluate_all_kernels_ovo_test(df_train_features, df_test_features):
    print("\nPreparing data for multi-kernel OvO TEST evaluation...")

    columns_to_drop = ['Restimulus', 'Subject', 'dataset_type']
    X_train = df_train_features.drop(columns=columns_to_drop)
    y_train = df_train_features['Restimulus']
    X_test = df_test_features.drop(columns=columns_to_drop)
    y_test = df_test_features['Restimulus']

    class_labels = sorted(y_train.unique())

    # Data Normalization (Fit on Train, transform on Test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    results = []

    print(f"\nTraining all kernels using One-vs-One (OvO) strategy...")

    for kernel in kernels:
        print("\n" + "-" * 55)
        print(f"--> Training '{kernel.upper()}' kernel...")

        if kernel == 'linear':
            base_svm = LinearSVC(
                random_state=42,
                dual=False,
                C=hp.SVM_C,
                max_iter=hp.SVM_MAX_ITER,
                class_weight=hp.SVM_CLASS_WEIGHT
            )
            svm_model = OneVsOneClassifier(base_svm)
        else:
            svm_model = SVC(
                kernel=kernel,
                random_state=42,
                C=hp.SVM_C,
                gamma=getattr(hp, 'SVM_GAMMA', 'scale'),
                degree=getattr(hp, 'SVM_DEGREE', 3),
                class_weight=hp.SVM_CLASS_WEIGHT
            )

        # Train
        svm_model.fit(X_train_scaled, y_train)

        # Predict
        y_test_pred = svm_model.predict(X_test_scaled)

        # Calculate Metrics
        _, _, macro_f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='macro', zero_division=0)
        bal_acc = balanced_accuracy_score(y_test, y_test_pred)

        print(f"    Macro F1: {macro_f1 * 100:.2f}% | Balanced Acc: {bal_acc * 100:.2f}%")

        # Calculate Normalized Confusion Matrix (True rows sum to 1.0)
        cm = confusion_matrix(y_test, y_test_pred, labels=class_labels, normalize='true')
        cm_percentage = cm * 100

        # Format as percentages
        cm_df = pd.DataFrame(cm_percentage, index=[f"True_{c}" for c in class_labels],
                             columns=[f"Pred_{c}" for c in class_labels])
        cm_df_formatted = cm_df.map(lambda x: f"{x:.1f}%")

        print(f"\n    Confusion Matrix (Normalized %):")
        print(cm_df_formatted.to_string())

        results.append({
            'Kernel': kernel.upper(),
            'Macro_F1 (%)': round(macro_f1 * 100, 2),
            'Balanced_Acc (%)': round(bal_acc * 100, 2)
        })

    print("\n" + "=" * 55)
    print("🏆 ALL KERNELS OvO ANALYSIS (TEST SET) 🏆")
    print("=" * 55)

    # Sort and print final table
    results_df = pd.DataFrame(results).sort_values(by='Balanced_Acc (%)', ascending=False).reset_index(drop=True)
    print(results_df.to_string())
    print("=" * 55)

    return results_df