import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score, \
    ConfusionMatrixDisplay
import pandas as pd

# הייבוא מהקבצים שלך
from data_loading import load_cleaned_ninapro_data
from feature_extraction import extract_all_features

# --- Intel CPU Acceleration ---
from sklearnex import patch_sklearn

patch_sklearn()
# ------------------------------

# 1. היפר-פרמטרים אופטימליים שמצאנו
model_list = {
    'linear': {'C': 1.5},
    'poly': {'C': 8, 'gamma': 'auto', 'degree': 3},
    'rbf': {'C': 0.1, 'gamma': 'scale'},
    'sigmoid': {'C': 0.05, 'gamma': 'scale'}
}

# 2. המחלקות וספי ההחלטה האופטימליים (Youden's J)
class_labels = [0, 1, 5, 6, 7, 13, 14, 17, 31]

optimal_thresholds = {
    'linear': np.array([8.3182, 7.3226, 5.3220, 5.2906, 6.3148, 2.7044, 6.3089, 5.2892, 7.3185]),
    'poly': np.array([8.3011, 7.2894, 2.7870, 5.1881, 7.2853, 1.7347, 1.7350, -0.2850, 6.3083]),
    'rbf': np.array([8.3018, 7.2899, 5.1897, 4.0852, 7.2890, 0.7202, 1.7495, 0.7257, 7.2310]),
    'sigmoid': np.array([8.3263, 6.2717, 3.7451, 2.7128, 7.2820, 0.6854, 1.6991, 1.6878, 7.2026])
}


def custom_predict(model, X_scaled, thresholds, labels):
    """
    מבצעת חיזוי על בסיס ציוני ההחלטה והספים האופטימליים.
    """
    scores = model.decision_function(X_scaled)
    adjusted_scores = scores - thresholds
    best_indices = np.argmax(adjusted_scores, axis=1)
    return np.array(labels)[best_indices]


def main():
    # מציאת נתיב הנתונים
    paths_to_check = [
        r'C:\Users\Tal\OneDrive - Afeka College Of Engineering\הקבצים של Nadav Matza - פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'C:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets'
    ]

    base_path = next((p for p in paths_to_check if os.path.exists(p)), None)
    if not base_path:
        print("Error: Dataset path not found!")
        return

    # חלוקת הנבדקים
    train_subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    test_subjects = [9, 10, 11, 12]  # נשתמש בהם כקבוצת הבדיקה הסופית

    print("\n--- Loading raw data ---")
    train_data = load_cleaned_ninapro_data(base_path, train_subjects, 'Train')
    test_data = load_cleaned_ninapro_data(base_path, test_subjects, 'Test')

    print("\n--- Extracting features ---")
    train_features = extract_all_features(train_data)
    test_features = extract_all_features(test_data)

    columns_to_drop = ['Restimulus', 'Subject', 'dataset_type']
    X_train = train_features.drop(columns=columns_to_drop)
    y_train = train_features['Restimulus']
    X_test = test_features.drop(columns=columns_to_drop)
    y_test = test_features['Restimulus']

    # נרמול
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- Training and Final Evaluation ---")

    for kernel, param in model_list.items():
        if kernel == 'linear':
            base_model = LinearSVC(random_state=42, **param, max_iter=10000)
        else:
            base_model = SVC(kernel=kernel, random_state=42, **param)

        print(f"\n=========================================")
        print(f"Current kernel: {kernel.upper()}")
        print(f"=========================================")

        # אימון המודל
        print("Training...")
        svm_model = OneVsOneClassifier(base_model)
        svm_model.fit(X_train_scaled, y_train)

        # חיזוי בעזרת הפונקציה המותאמת שלנו והספים של הקרנל הנוכחי
        print("Evaluating with optimal thresholds...")
        current_thresholds = optimal_thresholds[kernel]
        y_test_pred_optimized = custom_predict(svm_model, X_test_scaled, current_thresholds, class_labels)

        # חישוב מדדים
        per_class_recall = \
        precision_recall_fscore_support(y_test, y_test_pred_optimized, average=None, zero_division=0)[1]
        macro_f1 = precision_recall_fscore_support(y_test, y_test_pred_optimized, average='macro', zero_division=0)[2]
        bal_acc = balanced_accuracy_score(y_test, y_test_pred_optimized)

        print(f"    Macro F1: {macro_f1 * 100:.2f}% | Balanced Acc: {bal_acc * 100:.2f}%")
        print("    Per-Class Recall:")
        for cls, recall in zip(class_labels, per_class_recall):
            print(f"      Class {cls:<2}: {recall * 100:.2f}%")

        # יצירת מטריצת בלבול לתוצאות הסופיות
        cm = confusion_matrix(y_test, y_test_pred_optimized, labels=class_labels)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)
        plt.title(f"Final Confusion Matrix - {kernel.upper()} Kernel (Optimized)")

        cm_filename = f"Final_CM_{kernel.upper()}.png"
        plt.savefig(cm_filename, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    main()