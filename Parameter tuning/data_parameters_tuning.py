import os
import sys
import itertools
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

# --- Intel CPU Acceleration ---
from sklearnex import patch_sklearn

patch_sklearn()
# ------------------------------

# הוספת התיקייה הראשית לנתיב כדי שנוכל לייבא את הקבצים שלנו (data_loading, feature_extraction)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loading import load_cleaned_ninapro_data
from feature_extraction import extract_all_features


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

    # נבדקים לניסוי המהיר
    train_subjects = [1]
    val_subjects = [2]

    # ==========================================
    # --- הגדרת טווח החיפוש (The Grid) ---
    # ==========================================
    margin_options = [600, 800, 1000, 1200, 1400]

    # חלון וחפיפה (שומרים על 50% חפיפה לוגית)
    window_step_options = [
        (200, 100),
        (400, 200),
        (600, 300),
        (800, 400)
    ]

    zc_options = [1e-10, 1e-09, 1e-08, 1e-07, 1e-06]
    ssc_options = [1e-14, 1e-13, 1e-12, 1e-11, 1e-10]

    # יצירת כל הקומבינציות האפשריות
    all_combinations = list(itertools.product(margin_options, window_step_options, zc_options, ssc_options))
    total_iterations = len(all_combinations)

    print(f"Starting Grid Search for Signal Parameters...")
    print(f"Total combinations to test: {total_iterations}")
    print("Model: LinearSVC (OvO) | No PCA | No Rest Undersampling\n")

    results_list = []

    # בגלל ששינוי ב-Margin דורש טעינה מחדש של הנתונים, נטען אותם רק כשצריך
    current_margin = None
    train_data_raw = None
    val_data_raw = None

    for i, (margin, (win, step), zc, ssc) in enumerate(all_combinations):
        print(f"--- Iteration {i + 1}/{total_iterations} ---")
        print(f"Margin: {margin} | Window: {win} | Step: {step} | ZC: {zc} | SSC: {ssc}")

        # 1. טעינת נתונים (נטען מחדש רק אם ה-Margin השתנה כדי לחסוך זמן!)
        if margin != current_margin:
            print(f"  [+] Loading raw data with Margin={margin}...")
            train_data_raw = load_cleaned_ninapro_data(base_path, train_subjects, 'Train', margin_samples=margin)
            val_data_raw = load_cleaned_ninapro_data(base_path, val_subjects, 'Validation', margin_samples=margin)
            current_margin = margin

        # 2. חילוץ מאפיינים
        print("  [+] Extracting features...")
        train_features = extract_all_features(train_data_raw, window_size=win, step_size=step, zc_thresh=zc,
                                              ssc_delta=ssc)
        val_features = extract_all_features(val_data_raw, window_size=win, step_size=step, zc_thresh=zc, ssc_delta=ssc)

        columns_to_drop = ['Restimulus', 'Subject', 'dataset_type']
        X_train = train_features.drop(columns=columns_to_drop)
        y_train = train_features['Restimulus']
        X_val = val_features.drop(columns=columns_to_drop)
        y_val = val_features['Restimulus']

        # 3. נרמול בלבד (ללא PCA וללא חיתוך מחלקות)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 4. אימון והערכת מודל ליניארי נקי (LinearSVC)
        print("  [+] Training Linear Model...")
        base_model = LinearSVC(C=1.5, random_state=42, max_iter=10000)
        svm_model = OneVsOneClassifier(base_model)

        try:
            svm_model.fit(X_train_scaled, y_train)
            y_pred = svm_model.predict(X_val_scaled)

            bal_acc = balanced_accuracy_score(y_val, y_pred)
            macro_f1 = precision_recall_fscore_support(y_val, y_pred, average='macro', zero_division=0)[2]

            print(f"  => Result: Balanced Acc = {bal_acc * 100:.2f}%")

        except Exception as e:
            print(f"  => Error during training: {e}")
            bal_acc, macro_f1 = 0.0, 0.0

        # שמירת התוצאות למילון
        results_list.append({
            'Margin': margin,
            'Window_Size': win,
            'Step_Size': step,
            'ZC_Thresh': zc,
            'SSC_Delta': ssc,
            'Balanced_Accuracy': round(bal_acc * 100, 2),
            'Macro_F1': round(macro_f1 * 100, 2)
        })

    # ==========================================
    # --- ייצוא ושמירת התוצאות ל-CSV ---
    # ==========================================
    print("\n==========================================")
    print("Grid Search Complete! Saving to CSV...")

    results_df = pd.DataFrame(results_list)
    # סידור הטבלה מהתוצאה הגבוהה ביותר לנמוכה ביותר
    results_df = results_df.sort_values(by='Balanced_Accuracy', ascending=False).reset_index(drop=True)

    csv_filename = "Signal_Tuning_Results.csv"
    results_df.to_csv(csv_filename, index=False)

    print(f"Successfully saved results to: {csv_filename}")

    print("\n--- Top 5 Best Configurations ---")
    print(results_df.head(5).to_string())


if __name__ == "__main__":
    main()