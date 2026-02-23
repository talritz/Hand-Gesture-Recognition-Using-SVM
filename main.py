import os

# Import functions from our custom modules
from data_loading import load_cleaned_ninapro_data
from feature_extraction import extract_all_features
from model_training import evaluate_svm_kernels


def main():
    # ---------------------------------------------------------
    # 1. Dataset Path Configuration
    # ---------------------------------------------------------
    paths_to_check = [
        r'C:\Users\Tal\OneDrive - Afeka College Of Engineering\הקבצים של Nadav Matza - פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'C:\Users\Nadav\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets'
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
    # 2. Subject Splitting Configuration (Full Dataset)
    # ---------------------------------------------------------
    train_subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    val_subjects = [9, 10, 11, 12]
    test_subjects = [13, 14, 15, 16, 17, 18, 19, 20]

    # ---------------------------------------------------------
    # 3. Data Loading Pipeline
    # ---------------------------------------------------------
    print("\n--- PHASE 1: Loading Raw Data ---")
    df_train_raw = load_cleaned_ninapro_data(base_path, train_subjects, 'Train')
    df_val_raw = load_cleaned_ninapro_data(base_path, val_subjects, 'Validation')
    df_test_raw = load_cleaned_ninapro_data(base_path, test_subjects, 'Test')

    if df_train_raw.empty or df_val_raw.empty or df_test_raw.empty:
        print("Failed to load all required data sets. Exiting.")
        return

    # ---------------------------------------------------------
    # 4. Feature Extraction Pipeline
    # ---------------------------------------------------------
    print("\n--- PHASE 2: Extracting Features ---")
    df_train_features = extract_all_features(df_train_raw)
    df_val_features = extract_all_features(df_val_raw)
    df_test_features = extract_all_features(df_test_raw)

    # ---------------------------------------------------------
    # 5. Model Training and Evaluation Pipeline
    # ---------------------------------------------------------
    print("\n--- PHASE 3: Model Training & Evaluation ---")
    results = evaluate_svm_kernels(df_train_features, df_val_features, df_test_features)

    print("\nFull Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()