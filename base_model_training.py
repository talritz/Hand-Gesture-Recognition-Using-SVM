#used libraries:
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score, \
    ConfusionMatrixDisplay
import pandas as pd
import hyper_parameters as hp

def evaluate_all_kernels_ovo(df_train_features, df_val_features):
    print("\nPreparing data for multi-kernel OvO evaluation...")

    columns_to_drop = ['Restimulus', 'Subject', 'dataset_type']
    X_train = df_train_features.drop(columns=columns_to_drop)
    y_train = df_train_features['Restimulus']
    X_val = df_val_features.drop(columns=columns_to_drop)
    y_val = df_val_features['Restimulus']

    class_labels = sorted(y_train.unique())

    # Data Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    results = []

    print(f"\nTraining all kernels using One-vs-One (OvO) strategy (Baseline - Pre PCA/Youden)...")

    # שואבים את המילון ישירות מקובץ ההיפר-פרמטרים!
    for kernel, params in hp.MODEL_PARAMS.items():
        print("\n" + "-" * 55)
        print(f"--> Training '{kernel.upper()}' kernel with params: {params}...")

        if kernel == 'linear':
            base_svm = LinearSVC(random_state=42, max_iter=10000, **params)
        else:
            base_svm = SVC(kernel=kernel, random_state=42, **params)

        # Enforce One-vs-One Architecture
        svm_model = OneVsOneClassifier(base_svm)
        svm_model.fit(X_train_scaled, y_train)

        # Predict
        y_val_pred = svm_model.predict(X_val_scaled)

        # Metrics Calculation
        macro_f1 = precision_recall_fscore_support(y_val, y_val_pred, average='macro', zero_division=0)[2]
        bal_acc = balanced_accuracy_score(y_val, y_val_pred)
        per_class_recall = precision_recall_fscore_support(y_val, y_val_pred, average=None, zero_division=0)[1]

        print(f"    Macro F1: {macro_f1 * 100:.2f}% | Balanced Acc: {bal_acc * 100:.2f}%")
        print("    Per-Class Recall:")
        for cls, recall in zip(class_labels, per_class_recall):
            print(f"      Class {cls:<2}: {recall * 100:.2f}%")

        # Create and Save Matplotlib Confusion Matrix
        cm = confusion_matrix(y_val, y_val_pred, labels=class_labels)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)
        plt.title(f"Confusion Matrix - {kernel.upper()} (Baseline OvO)")

        file_name = f"CM_Baseline_{kernel.upper()}_OvO.png"
        plt.savefig(file_name, bbox_inches='tight')
        print(f"    -> Saved plot: {file_name}")
        plt.close(fig)

        results.append({
            'Kernel': kernel.upper(),
            'Macro_F1 (%)': round(macro_f1 * 100, 2),
            'Balanced_Acc (%)': round(bal_acc * 100, 2)
        })

    print("\n" + "=" * 55)
    print("ALL KERNELS OvO ANALYSIS (BASELINE)")
    print("=" * 55)
    results_df = pd.DataFrame(results).sort_values(by='Balanced_Acc (%)', ascending=False).reset_index(drop=True)
    print(results_df.to_string())
    print("=" * 55)

    return results_df