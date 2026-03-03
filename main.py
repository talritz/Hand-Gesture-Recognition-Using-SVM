import os
import time
from datetime import datetime
import pandas as pd

# --- Intel CPU Acceleration ---
from sklearnex import patch_sklearn
patch_sklearn()
# ------------------------------

from data_loading import load_cleaned_ninapro_data
from feature_extraction import extract_all_features
from model_training import evaluate_all_kernels_ovo_test
import hyperparameters as hp

def format_duration(seconds):
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0: return f"{hours}h {mins}m {secs}s"
    elif mins > 0: return f"{mins}m {secs}s"
    else: return f"{secs}s"

def main():
    total_start_time = time.time()
    print("=" * 60)
    print(f"PAPER-ALIGNED VALIDATION EVALUATION STARTED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    paths_to_check = [
        r'C:\Users\Tal\OneDrive - Afeka College Of Engineering\הקבצים של Nadav Matza - פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'C:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets'
    ]

    base_path = next((p for p in paths_to_check if os.path.exists(p)), None)
    if not base_path:
        print("Error: Dataset path not found!")
        return

    # ---------------------------------------------------------
    # STRICT SPLIT: Train on 1-8, Validate on 9-12.
    # Subjects 13-20 remain locked in the vault!
    # ---------------------------------------------------------
    train_subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    val_subjects = [9, 10, 11, 12]

    print("\n--- PHASE 1: Loading Raw Data ---")
    df_train_raw = load_cleaned_ninapro_data(base_path, train_subjects, 'Train')
    df_val_raw = load_cleaned_ninapro_data(base_path, val_subjects, 'Validation')

    print("\n--- PHASE 2: Extracting Features (Paper-Aligned) ---")
    df_train_features = extract_all_features(df_train_raw)
    df_val_features = extract_all_features(df_val_raw)

    print("\n--- PHASE 3: Multi-Kernel Training & VALIDATION Set Evaluation (OvO) ---")
    results_df = evaluate_all_kernels_ovo_test(df_train_features, df_val_features)

    total_end_time = time.time()
    print("\n" + "=" * 60)
    print(f"PROJECT PIPELINE COMPLETED IN: {format_duration(total_end_time - total_start_time)}")
    print("=" * 60)

if __name__ == "__main__":
    main()