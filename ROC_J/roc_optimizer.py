import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, balanced_accuracy_score, confusion_matrix, \
    ConfusionMatrixDisplay


def optimize_and_evaluate(svm_model, X_val_scaled, y_val, class_labels, kernel_name, file_suffix, normalize_cm):
    print(f"\n--- Post-ROC Optimization: {kernel_name.upper()} Kernel ---")

    y_val_score = svm_model.decision_function(X_val_scaled)
    y_val_binarized = label_binarize(y_val, classes=class_labels)

    thresholds_array = np.zeros(len(class_labels))
    plt.figure(figsize=(10, 8))

    for idx, cls in enumerate(class_labels):
        fpr, tpr, thresh = roc_curve(y_val_binarized[:, idx], y_val_score[:, idx])
        roc_auc = auc(fpr, tpr)

        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        thresholds_array[idx] = thresh[best_idx]

        line, = plt.plot(fpr, tpr, lw=2, label=f'Class {cls} (AUC = {roc_auc:.2f})')
        plt.plot(fpr[best_idx], tpr[best_idx], marker='o', color=line.get_color(),
                 markersize=8, markeredgecolor='black', markeredgewidth=1.5, ls='')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curve & Optimal Thresholds - {kernel_name.upper()} Kernel")

    roc_filename = f"ROC_Curve_{kernel_name.upper()}{file_suffix}_Youdens_J.png"
    plt.savefig(roc_filename, bbox_inches='tight')
    plt.close()

    adjusted_scores = y_val_score - thresholds_array
    y_val_pred_custom_idx = np.argmax(adjusted_scores, axis=1)
    y_val_pred_custom = np.array([class_labels[i] for i in y_val_pred_custom_idx])

    macro_f1 = precision_recall_fscore_support(y_val, y_val_pred_custom, average='macro', zero_division=0)[2]
    bal_acc = balanced_accuracy_score(y_val, y_val_pred_custom)

    # Extract optimized per-class recall
    _, recalls_opt, _, _ = precision_recall_fscore_support(y_val, y_val_pred_custom, average=None, zero_division=0)

    print(f"    New Macro F1: {macro_f1 * 100:.2f}% | New Balanced Acc: {bal_acc * 100:.2f}%")

    if normalize_cm:
        cm_custom = confusion_matrix(y_val, y_val_pred_custom, labels=class_labels, normalize='true')
        cm_custom = cm_custom * 100
        val_format = '.1f'
        title_suffix = "(Optimized - %)"
    else:
        cm_custom = confusion_matrix(y_val, y_val_pred_custom, labels=class_labels)
        val_format = 'd'
        title_suffix = "(Optimized - Counts)"

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_custom, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues, values_format=val_format, ax=ax)
    plt.title(f"Confusion Matrix (Optimized) - {kernel_name.upper()} Kernel")

    cm_filename = f"CM_Optimized_{kernel_name.upper()}{file_suffix}.png"
    plt.savefig(cm_filename, bbox_inches='tight')
    plt.close(fig)

    # Return optimized overall metrics and per-class metrics
    opt_dict = {
        'Kernel': kernel_name.upper(),
        'Macro_F1_Opt (%)': round(macro_f1 * 100, 2),
        'Balanced_Acc_Opt (%)': round(bal_acc * 100, 2)
    }
    for i, cls in enumerate(class_labels):
        opt_dict[f'Class_{cls}_Opt_Recall (%)'] = round(recalls_opt[i] * 100, 2)

    return opt_dict