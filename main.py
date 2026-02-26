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
from model_training import evaluate_svm_kernels


def main():
    print("=" * 60)
    print(f"PHASE 1 GRID SEARCH STARTED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    paths_to_check = [
        r'C:\Users\Tal\OneDrive - Afeka College Of Engineering\הקבצים של Nadav Matza - פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'C:\Users\Nadav\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets'
    ]

    base_path = next((p for p in paths_to_check if os.path.exists(p)), None)
    if not base_path:
        print("Error: Dataset path not found!")
        return

    train_subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    val_subjects = [9, 10, 11, 12]

    # --- Phase 1: Signal Processing Parameter Space ---
    margins = [1500, 1000]
    windows = [200, 400]
    steps = [50, 100]  # Note: 50 means double the amount of training windows!
    zc_thresholds = [1e-5, 1e-6, 1e-7]
    ssc_deltas = [1e-11, 1e-12, 1e-13]

    all_results = []
    total_runs = len(margins) * len(windows) * len(steps) * len(zc_thresholds) * len(ssc_deltas)
    current_run = 0

    # 1. Loop over Data Margins
    for margin in margins:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] >>> LOADING DATA (Margin: {margin}) <<<")
        df_train_raw = load_cleaned_ninapro_data(base_path, train_subjects, 'Train', margin_samples=margin)
        df_val_raw = load_cleaned_ninapro_data(base_path, val_subjects, 'Validation', margin_samples=margin)

        # 2. Loop over Window Sizes
        for window in windows:
            # 3. Loop over Step Sizes
            for step in steps:
                # 4. Loop over ZC Thresholds
                for zc in zc_thresholds:
                    # 5. Loop over SSC Deltas
                    for ssc in ssc_deltas:
                        current_run += 1
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] Run {current_run}/{total_runs} | Win:{window}, Step:{step}, ZC:{zc}, SSC:{ssc}")

                        # Extract features with specific thresholds
                        df_train_features = extract_all_features(df_train_raw, window_size=window, step_size=step,
                                                                 zc_thresh=zc, ssc_delta=ssc)
                        df_val_features = extract_all_features(df_val_raw, window_size=window, step_size=step,
                                                               zc_thresh=zc, ssc_delta=ssc)

                        # Train ONLY Linear SVM to test the feature quality
                        kernel_results = evaluate_svm_kernels(
                            df_train_features, df_val_features,
                            svm_c=1.0,
                            kernels_to_test=['linear']
                        )

                        # Save metrics
                        metrics = kernel_results['linear']
                        all_results.append({
                            'Margin': margin,
                            'Window': window,
                            'Step': step,
                            'ZC_Thresh': zc,
                            'SSC_Delta': ssc,
                            'Macro_F1 (%)': round(metrics['macro_f1'] * 100, 2),
                            'Balanced_Acc (%)': round(metrics['balanced_accuracy'] * 100, 2)
                        })

    # --- Print and Save Final Table ---
    print("\n" + "=" * 60)
    print("PHASE 1 GRID SEARCH COMPLETED!")

    results_df = pd.DataFrame(all_results)

    # Sort by the best Macro F1 score
    results_df = results_df.sort_values(by='Macro_F1 (%)', ascending=False).reset_index(drop=True)

    print("\nTOP 10 RESULTS (Sorted by Best Macro F1):")
    print(results_df.head(10).to_string())

    csv_filename = 'phase1_signal_processing_results.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"\nFull results saved to '{csv_filename}'")
    print("=" * 60)


if __name__ == "__main__":
    main()