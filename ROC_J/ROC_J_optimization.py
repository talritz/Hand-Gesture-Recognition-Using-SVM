#used libraries:
import os
import numpy as np
import matplotlib.pyplot as plt
from pygame.transform import threshold
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score, \
    ConfusionMatrixDisplay
import pandas as pd
import hyper_parameters as hp
from data_loading import load_cleaned_ninapro_data
from feature_extraction import extract_all_features
from model_training import evaluate_all_kernels_ovo
from sklearn.metrics import roc_curve, auc

# --- Intel CPU Acceleration ---
from sklearnex import patch_sklearn
patch_sklearn()
# ------------------------------

model_list ={
    'linear': {'C': 1.5},

    'poly': {'C': 8,'gamma': 'auto','degree': 3},

    'rbf': {'C': 0.1, 'gamma': 'scale'},

    'sigmoid': {'C': 0.05, 'gamma': 'scale'}
}


def main():

    #We are working on three different computers, so this is used to find the correct path to the database on each of them.
    paths_to_check = [
        r'C:\Users\Tal\OneDrive - Afeka College Of Engineering\הקבצים של Nadav Matza - פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'C:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets'
    ]

    base_path = next((p for p in paths_to_check if os.path.exists(p)), None)
    if not base_path:
        print("Error: Dataset path not found!")
        return


    #Defining the two groups:
    train_subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    val_subjects = [9, 10, 11, 12]

    #Model Training and evaluation pipe-line:
    print("\n--- Loading raw data from NinaPro DB2 ---")
    train_data = load_cleaned_ninapro_data(base_path, train_subjects, 'Train')
    val_data = load_cleaned_ninapro_data(base_path, val_subjects, 'Validation')

    print("\n--- Extracting features ---")
    train_features = extract_all_features(train_data)
    val_features = extract_all_features(val_data)

    #Dividing the data (features) from the answer (Restimulus)
    columns_to_drop = ['Restimulus', 'Subject', 'dataset_type']
    X_train = train_features.drop(columns=columns_to_drop)
    y_train = train_features['Restimulus']
    X_val = val_features.drop(columns=columns_to_drop)
    y_val = val_features['Restimulus']

    class_labels = sorted(y_train.unique())

    #Normalizing the data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("\n--- Multi-Kernel Training & VALIDATION Set Evaluation (OvO) ---")

    for kernel, param in model_list.items():
        if kernel == 'linear':
            base_model = LinearSVC(random_state=42, **param, max_iter=10000)
        else:
            base_model = SVC(kernel=kernel, random_state=42, **param)

        print(f"\n=========================================")
        print(f"Current kernel: {kernel.upper()}")
        print(f"=========================================")

        print("Training...")
        svm_model = OneVsOneClassifier(base_model)
        svm_model.fit(X_train_scaled, y_train)

        print("Evaluating...")
        y_val_pred = svm_model.predict(X_val_scaled)

        # Metrics Calculation
        per_class_recall = precision_recall_fscore_support(y_val, y_val_pred, average=None, zero_division=0)[1]
        macro_f1 = precision_recall_fscore_support(y_val, y_val_pred, average='macro', zero_division=0)[2]
        bal_acc = balanced_accuracy_score(y_val, y_val_pred)
        y_val_score = svm_model.decision_function(X_val_scaled)

        print(f"    Macro F1: {macro_f1 * 100:.2f}% | Balanced Acc: {bal_acc * 100:.2f}%")

        # --- Confusion Matrix ---
        cm = confusion_matrix(y_val, y_val_pred, labels=class_labels)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)
        plt.title(f"Confusion Matrix - {kernel.upper()} Kernel (OvO Strategy)")

        # שמירת מטריצת הבלבול
        cm_filename = f"Confusion_Matrix_{kernel.upper()}.png"
        plt.savefig(cm_filename, bbox_inches='tight')
        print(f"    Saved: {cm_filename}")
        plt.close(fig)  # סוגרים את הפיגורה כדי שלא יקפצו המון חלונות יחד

        # --- ROC and Youden's J ---
        y_val_binarized = label_binarize(y_val, classes=class_labels)

        # יצירת הפיגורה של ה-ROC
        plt.figure(figsize=(12, 10))  # הגדלתי מעט את הפיגורה כדי לפנות מקום לטקסט
        print("\n    Optimal Thresholds (Youden's J):")

        for idx, cls in enumerate(class_labels):
            fpr, tpr, thresh = roc_curve(y_val_binarized[:, idx], y_val_score[:, idx])
            roc_auc = auc(fpr, tpr)

            youden_j = tpr - fpr
            best_idx = np.argmax(youden_j)
            best_thresh = thresh[best_idx]

            # הדפסת הערך המספרי כדי שתוכל להעתיק לדוח או לקוד אחר
            print(f"      Class {cls:<2}: Threshold = {best_thresh:7.4f} | J-Score = {youden_j[best_idx]:.4f}")

            # שרטוט עקומת ה-ROC
            line, = plt.plot(fpr, tpr, lw=2, label=f'Class {cls} (AUC = {roc_auc:.2f})')

            # סימון נקודת העבודה האופטימלית על הגרף
            plt.scatter(fpr[best_idx], tpr[best_idx], color=line.get_color(), s=70, marker='o', edgecolors='black',
                        zorder=5)

            # --- הוספת Annotation (טקסט) ליד הנקודה ---
            # הטקסט שיופיע על הגרף (הסף מעוגל ל-3 ספרות)
            annotation_text = f"T={best_thresh:.3f}"

            # מיקום הטקסט: מעט ימינה ולמטה מהנקודה כדי לא להסתיר את הקו
            plt.annotate(
                annotation_text,
                xy=(fpr[best_idx], tpr[best_idx]),  # מיקום הנקודה המקורית
                xytext=(8, -10),  # היסט (Offset) בפיקסלים מהנקודה
                textcoords="offset points",  # שיטת המיקום
                color=line.get_color(),  # צבע הטקסט תואם לצבע הקו
                fontweight='bold',  # מודגש
                fontsize=9,  # גודל פונט מעט קטן יותר
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
                # רקע לבן חצי שקוף מאחורי הטקסט לקריאות
            )

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
        plt.legend(loc='lower right')

        # הרחבת הגבולות מעט כדי שהטקסט בפינות לא ייחתך
        plt.xlim([-0.02, 1.05])
        plt.ylim([-0.02, 1.05])

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"ROC Curve & Youden's J - {kernel.upper()} Kernel")

        # שמירת גרף ה-ROC המעודכן עם הטקסט
        roc_filename = f"ROC_Curve_{kernel.upper()}_Annotated.png"
        plt.savefig(roc_filename, bbox_inches='tight')
        print(f"    Saved: {roc_filename}")
        plt.close()  # סוגרים את הפיגורה לאחר השמירה

if __name__ == "__main__":
    main()