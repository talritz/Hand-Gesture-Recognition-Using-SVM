import os
import time
from datetime import datetime

# --- Intel CPU Acceleration (Must be called before other sklearn imports!) ---
from sklearnex import patch_sklearn
patch_sklearn()
# -------------------------------------------------------------------------

# Import functions from our custom modules
from data_loading import load_cleaned_ninapro_data
from feature_extraction import extract_all_features
from model_training import evaluate_svm_kernels


def format_duration(seconds):
    """
    Helper function to format time duration into a readable string (Hours, Minutes, Seconds).
    """
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours}h {mins}m {secs}s"
    elif mins > 0:
        return f"{mins}m {secs}s"
    else:
        return f"{secs}s"


def main():
    total_start_time = time.time()
    print("=" * 50)
    print(f"PIPELINE STARTED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50 + "\n")

    # ---------------------------------------------------------
    # 1. Dataset Path Configuration
    # ---------------------------------------------------------
    paths_to_check = [
        r'C:\Users\Tal\OneDrive - Afeka College Of Engineering\הקבצים של Nadav Matza - פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'C:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets'
    ]

    base_path = None
    for path in paths_to_check:
        if os.path.exists(path):
            base_path = path
            print(f"Dataset found at: {base_path}")
            break

    if base_path is None:
        print("Error: Dataset path not found! Please check your directories.")
        return

    # ---------------------------------------------------------
    # 2. Subject Splitting Configuration
    # ---------------------------------------------------------
    train_subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    val_subjects = [9, 10, 11, 12]
    # test_subjects  = [13, 14, 15, 16, 17, 18, 19, 20]  # Commented out for now

    # ---------------------------------------------------------
    # 3. Data Loading Pipeline
    # ---------------------------------------------------------
    print("\n--- PHASE 1: Loading Raw Data ---")
    phase1_start_time = time.time()
    print(f"[Start] Phase 1: {datetime.now().strftime('%H:%M:%S')}")

    df_train_raw = load_cleaned_ninapro_data(base_path, train_subjects, 'Train')
    df_val_raw = load_cleaned_ninapro_data(base_path, val_subjects, 'Validation')
    # df_test_raw  = load_cleaned_ninapro_data(base_path, test_subjects, 'Test') # Commented out

    if df_train_raw.empty or df_val_raw.empty:
        print("Failed to load required training/validation sets. Exiting.")
        return

    phase1_end_time = time.time()
    print(f"[End] Phase 1: {datetime.now().strftime('%H:%M:%S')}")
    print(f"[Duration] Phase 1: {format_duration(phase1_end_time - phase1_start_time)}")

    # ---------------------------------------------------------
    # 4. Feature Extraction Pipeline
    # ---------------------------------------------------------
    print("\n--- PHASE 2: Extracting Features ---")
    phase2_start_time = time.time()
    print(f"[Start] Phase 2: {datetime.now().strftime('%H:%M:%S')}")

    df_train_features = extract_all_features(df_train_raw)
    df_val_features = extract_all_features(df_val_raw)
    # df_test_features  = extract_all_features(df_test_raw) # Commented out

    phase2_end_time = time.time()
    print(f"[End] Phase 2: {datetime.now().strftime('%H:%M:%S')}")
    print(f"[Duration] Phase 2: {format_duration(phase2_end_time - phase2_start_time)}")

    # ---------------------------------------------------------
    # 5. Model Training and Evaluation Pipeline
    # ---------------------------------------------------------
    print("\n--- PHASE 3: Model Training & Evaluation ---")
    phase3_start_time = time.time()
    print(f"[Start] Phase 3: {datetime.now().strftime('%H:%M:%S')}")

    # Passing only Train and Validation sets
    results = evaluate_svm_kernels(df_train_features, df_val_features)

    phase3_end_time = time.time()
    print(f"[End] Phase 3: {datetime.now().strftime('%H:%M:%S')}")
    print(f"[Duration] Phase 3: {format_duration(phase3_end_time - phase3_start_time)}")

    # ---------------------------------------------------------
    # Pipeline Summary
    # ---------------------------------------------------------
    total_end_time = time.time()
    print("\n" + "=" * 50)
    print(f"PIPELINE COMPLETED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TOTAL RUNTIME: {format_duration(total_end_time - total_start_time)}")
    print("=" * 50)


if __name__ == "__main__":
    main()