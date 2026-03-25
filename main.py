import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from data_loading import load_cleaned_ninapro_data
from feature_extraction import extract_all_features
from model_training import evaluate_all_kernels_ovo
from ROC_J.roc_optimizer import optimize_and_evaluate
import hyper_parameters as hp

from sklearnex import patch_sklearn

patch_sklearn()


def undersample_rest_class(df_features, rest_label=0):
    print(f"\n--- Applying Undersampling to Class {rest_label} ---")
    class_counts = df_features['Restimulus'].value_counts()
    active_classes = class_counts.drop(labels=[rest_label], errors='ignore')
    target_size = int(active_classes.mean())

    df_rest = df_features[df_features['Restimulus'] == rest_label]
    df_active = df_features[df_features['Restimulus'] != rest_label]

    if len(df_rest) > target_size:
        df_rest_sampled = df_rest.sample(n=target_size, random_state=42)
        print(f"-> Class {rest_label} reduced from {len(df_rest)} to {target_size} samples.")
    else:
        df_rest_sampled = df_rest

    df_balanced = pd.concat([df_rest_sampled, df_active]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_balanced


def main():
    paths_to_check = [
        r'C:\Users\Tal\OneDrive - Afeka College Of Engineering\הקבצים של Nadav Matza - פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'C:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets'
    ]
    base_path = next((p for p in paths_to_check if os.path.exists(p)), None)
    if not base_path:
        print("Error: Dataset path not found.")
        return

    # --- Suffix Generation ---
    file_suffix = ""
    if hp.USE_UNDERSAMPLING:
        file_suffix += "_Under_Sampling"
    if hp.USE_PCA:
        file_suffix += "_PCA"
    if hp.USE_YOUDENS_J:
        file_suffix += "_Youdens_J"

    train_subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    val_subjects = [9, 10, 11, 12]

    print("\n" + "=" * 50)
    print(" STEP 1: LOADING & FEATURE EXTRACTION")
    print("=" * 50)
    train_data_raw = load_cleaned_ninapro_data(base_path, train_subjects, 'Train')
    val_data_raw = load_cleaned_ninapro_data(base_path, val_subjects, 'Validation')
    df_train_features = extract_all_features(train_data_raw)
    df_val_features = extract_all_features(val_data_raw)

    print("\n" + "=" * 50)
    print(" STEP 2: DATA BALANCING & PREPROCESSING")
    print("=" * 50)

    if hp.USE_UNDERSAMPLING:
        df_train_features = undersample_rest_class(df_train_features)

    columns_to_drop = ['Restimulus', 'Subject', 'dataset_type']
    X_train = df_train_features.drop(columns=columns_to_drop)
    y_train = df_train_features['Restimulus']
    X_val = df_val_features.drop(columns=columns_to_drop)
    y_val = df_val_features['Restimulus']
    class_labels = sorted(y_train.unique())

    # Normalization (StandardScaler)
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # PCA Application
    if hp.USE_PCA:
        print(f"Applying PCA (Variance Threshold: {hp.PCA_VARIANCE_THRESHOLD})...")
        pca = PCA(n_components=hp.PCA_VARIANCE_THRESHOLD, random_state=42)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_val_scaled = pca.transform(X_val_scaled)
        print(f"-> Dimensionality reduced to {X_train_scaled.shape[1]} components.")

    print("\n" + "=" * 50)
    print(" STEP 3: MODEL TRAINING & BASELINE EVALUATION")
    print("=" * 50)
    baseline_df, trained_models = evaluate_all_kernels_ovo(
        X_train_scaled, y_train, X_val_scaled, y_val, class_labels, file_suffix, hp.NORMALIZE_CM
    )

    final_comparison_df = baseline_df

    if hp.USE_YOUDENS_J:
        print("\n" + "=" * 50)
        print(" STEP 4: ROC & YOUDEN'S J THRESHOLD OPTIMIZATION")
        print("=" * 50)
        optimized_results = []
        for kernel, model in trained_models.items():
            opt_metrics = optimize_and_evaluate(model, X_val_scaled, y_val, class_labels, kernel, file_suffix, hp.NORMALIZE_CM)
            optimized_results.append(opt_metrics)

        optimized_df = pd.DataFrame(optimized_results)
        final_comparison_df = pd.merge(baseline_df, optimized_df, on='Kernel')

    print("\n" + "=" * 80)
    print(f" FINAL RESULTS (Config: {file_suffix if file_suffix else 'Raw_Baseline'}) ")
    print("=" * 80)
    print(final_comparison_df.to_string(index=False))
    print("=" * 80)

    # Save the results to CSV
    csv_filename = f"Results_Summary{file_suffix}.csv"
    final_comparison_df.to_csv(csv_filename, index=False)
    print(f"\nSaved final results to: {csv_filename}")


if __name__ == "__main__":
    main()