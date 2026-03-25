import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score, \
    ConfusionMatrixDisplay
import pandas as pd
import hyper_parameters as hp


def evaluate_all_kernels_ovo(X_train_scaled, y_train, X_val_scaled, y_val, class_labels, file_suffix, normalize_cm):
    results = []
    trained_models = {}

    print(f"\nTraining all kernels using One-vs-One (OvO) strategy (Baseline)...")

    for kernel, params in hp.MODEL_PARAMS.items():
        print("\n" + "-" * 55)
        print(f"--> Training '{kernel.upper()}' kernel with params: {params}...")

        if kernel == 'linear':
            base_svm = LinearSVC(random_state=42, max_iter=10000, **params)
        else:
            base_svm = SVC(kernel=kernel, random_state=42, **params)

        svm_model = OneVsOneClassifier(base_svm)
        svm_model.fit(X_train_scaled, y_train)

        trained_models[kernel] = svm_model
        y_val_pred = svm_model.predict(X_val_scaled)

        macro_f1 = precision_recall_fscore_support(y_val, y_val_pred, average='macro', zero_division=0)[2]
        bal_acc = balanced_accuracy_score(y_val, y_val_pred)

        print(f"    Macro F1: {macro_f1 * 100:.2f}% | Balanced Acc: {bal_acc * 100:.2f}%")

        if normalize_cm:
            cm = confusion_matrix(y_val, y_val_pred, labels=class_labels, normalize='true')
            cm = cm * 100
            val_format = '.1f'
            title_suffix = "(Baseline - %)"
        else:
            cm = confusion_matrix(y_val, y_val_pred, labels=class_labels)
            val_format = 'd'
            title_suffix = "(Baseline - Counts)"

        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(cmap=plt.cm.Blues, values_format=val_format, ax=ax)
        plt.title(f"Confusion Matrix - {kernel.upper()} Kernel (Baseline{file_suffix})")

        file_name = f"CM_Baseline_{kernel.upper()}{file_suffix}.png"
        plt.savefig(file_name, bbox_inches='tight')
        plt.close(fig)

        results.append({
            'Kernel': kernel.upper(),
            'Macro_F1_Base (%)': round(macro_f1 * 100, 2),
            'Balanced_Acc_Base (%)': round(bal_acc * 100, 2)
        })

    results_df = pd.DataFrame(results)
    return results_df, trained_models