import os
import time
from datetime import datetime
import pandas as pd
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import ParameterGrid

# --- Intel CPU Acceleration ---
from sklearnex import patch_sklearn

patch_sklearn()
# ------------------------------

from data_loading import load_cleaned_ninapro_data
from feature_extraction import extract_all_features
import hyper_parameters as hp

# השתקת אזהרות התכנסות
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def run_dynamic_grid_search():
    total_start_time = time.time()
    print("=" * 65)
    print(f"DYNAMIC SVM GRID SEARCH TUNING STARTED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    paths_to_check = [
        r'C:\Users\Tal\OneDrive - Afeka College Of Engineering\הקבצים של Nadav Matza - פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'C:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets'
    ]
    base_path = next((p for p in paths_to_check if os.path.exists(p)), None)

    train_subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    val_subjects = [9, 10, 11, 12]

    print("\n--- PHASE 1: Loading Full Dataset ---")
    df_train_raw = load_cleaned_ninapro_data(base_path, train_subjects, 'Train')
    df_val_raw = load_cleaned_ninapro_data(base_path, val_subjects, 'Validation')

    print("\n--- PHASE 2: Extracting Hybrid Features ---")
    df_train_features = extract_all_features(df_train_raw)
    df_val_features = extract_all_features(df_val_raw)

    columns_to_drop = ['Restimulus', 'Subject', 'dataset_type']
    X_train = df_train_features.drop(columns=columns_to_drop)
    y_train = df_train_features['Restimulus']
    X_val = df_val_features.drop(columns=columns_to_drop)
    y_val = df_val_features['Restimulus']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # =====================================================================
    # אזור הגדרת הרשימות הדינמיות
    # הכנס לכאן את כל הערכים שאתה רוצה, והקוד ייצר את כל הקומבינציות האפשריות!
    # =====================================================================
    C_VALUES = [0.1, 1.0, 10.0, 40, 70, 100]
    GAMMA_VALUES = ['scale', 'auto', 0.1, 1, 10, 100]
    DEGREE_VALUES = [2, 3, 4]
    # =====================================================================

    # יצירה אוטומטית של כל הקומבינציות לכל גרעין בנפרד
    param_grid = {
        'linear': list(ParameterGrid({'C': C_VALUES})),
        'poly': list(ParameterGrid({'C': C_VALUES, 'gamma': GAMMA_VALUES, 'degree': DEGREE_VALUES})),
        'rbf': list(ParameterGrid({'C': C_VALUES, 'gamma': GAMMA_VALUES})),
        'sigmoid': list(ParameterGrid({'C': C_VALUES, 'gamma': GAMMA_VALUES}))
    }

    # חישוב סך כל הריצות מראש
    total_runs = sum(len(grid) for grid in param_grid.values())
    print(f"\nPRE-FLIGHT CHECK: You have defined {total_runs} total combinations to test.")
    print("This will take a significant amount of time. Sit back and relax!")

    best_results = []
    all_results = []

    print("\n--- PHASE 3: Grid Search Iterations ---")
    current_run = 1

    for kernel, params_list in param_grid.items():
        best_score = 0
        best_params = None

        print(f"\n" + "-" * 45)
        print(f"Tuning '{kernel.upper()}' Kernel ({len(params_list)} combinations)")
        print("-" * 45)

        for params in params_list:
            print(f"[Run {current_run}/{total_runs}] Testing {kernel.upper()} with params: {params}...")
            iteration_start = time.time()

            if kernel == 'linear':
                base_svm = LinearSVC(random_state=42, dual=False, C=params['C'], max_iter=hp.SVM_MAX_ITER,
                                     class_weight=hp.SVM_CLASS_WEIGHT)
            else:
                base_svm = SVC(kernel=kernel, random_state=42, C=params['C'], gamma=params.get('gamma', 'scale'),
                               degree=params.get('degree', 3), class_weight=hp.SVM_CLASS_WEIGHT,
                               max_iter=hp.SVM_MAX_ITER)

            svm_model = OneVsOneClassifier(base_svm)
            svm_model.fit(X_train_scaled, y_train)

            y_pred = svm_model.predict(X_val_scaled)
            bal_acc = balanced_accuracy_score(y_val, y_pred) * 100

            iter_time = time.time() - iteration_start
            print(f"   -> Balanced Accuracy: {bal_acc:.2f}% (Took {iter_time:.1f}s)")

            all_results.append({
                'Kernel': kernel.upper(),
                'Parameters': str(params),
                'Balanced_Acc (%)': round(bal_acc, 2),
                'Time (s)': round(iter_time, 1)
            })

            if bal_acc > best_score:
                best_score = bal_acc
                best_params = params

            current_run += 1

        print(f"\nBEST {kernel.upper()}: {best_params} | Score: {best_score:.2f}%")
        best_results.append({'Kernel': kernel.upper(), 'Best Params': str(best_params), 'Best Accuracy': best_score})

    print("\n" + "=" * 65)
    print("FULL GRID SEARCH RESULTS SUMMARY")
    print("=" * 65)

    full_results_df = pd.DataFrame(all_results).sort_values(by='Balanced_Acc (%)', ascending=False).reset_index(
        drop=True)
    print(full_results_df.to_string())

    csv_filename = "Grid_Search_Dynamic_Results.csv"
    full_results_df.to_csv(csv_filename, index=False)
    print(f"\nSaved detailed results to: {csv_filename}")

    total_end_time = time.time()
    mins, secs = divmod(int(total_end_time - total_start_time), 60)
    hours, mins = divmod(mins, 60)
    print("\n" + "=" * 65)
    if hours > 0:
        print(f"PIPELINE COMPLETED IN: {hours}h {mins}m {secs}s")
    else:
        print(f"PIPELINE COMPLETED IN: {mins}m {secs}s")
    print("=" * 65)


if __name__ == "__main__":
    run_dynamic_grid_search()