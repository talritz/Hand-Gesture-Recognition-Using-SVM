import os
import sys
import time
import numpy as np
import pandas as pd
import itertools
import datetime

from data_loading import load_cleaned_ninapro_data
from feature_extraction import extract_all_features

class HiddenPrints:
    """Context manager to hide prints from external modules during bulk processing."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def main():
    print("=" * 60)
    print(" STAGE 1: OPTIMIZED FEATURE GENERATION")
    print("=" * 60)

    # Define potential paths for cross-computer compatibility
    paths_to_check = [
        r'C:\Users\Tal\OneDrive - Afeka College Of Engineering\הקבצים של Nadav Matza - פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'C:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets'
    ]

    base_path = next((path for path in paths_to_check if os.path.exists(path)), None)
    if not base_path:
        print("Error: Dataset path not found. Please verify your directory structure.")
        return

    # --- PARAMETER GRID ---
    margin_options = [600, 800, 1000]
    window_sizes = [400, 600, 800]
    step_percentages = [0.5]  # Fixed at 50% overlap
    zero_crossing_options = [1e-8, 1e-6, 1e-4]
    ssc_options = [1e-8, 1e-6, 1e-4]

    train_subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    val_subjects = [9, 10, 11, 12]

    output_dir = "Generated_Features_NPZ"
    os.makedirs(output_dir, exist_ok=True)
    start_global_time = time.time()

    total_combinations = len(margin_options) * len(window_sizes) * len(step_percentages) * len(
        zero_crossing_options) * len(ssc_options)
    current_iteration = 0

    for margin in margin_options:
        print(f"\n{'-' * 60}")
        print(f">>> LOADING RAW DATA FOR MARGIN: {margin}")
        print(f"{'-' * 60}")

        raw_train_data = load_cleaned_ninapro_data(base_path, train_subjects, 'Train_Group', margin)
        raw_val_data = load_cleaned_ninapro_data(base_path, val_subjects, 'Val_Group', margin)

        inner_combinations = list(itertools.product(window_sizes, step_percentages, zero_crossing_options, ssc_options))

        for window, step_pct, zc, ssc in inner_combinations:
            current_iteration += 1
            step = int(window * step_pct)

            base_filename = f"M{margin}_W{window}_S{step}_ZC{zc}_SSC{ssc}"
            train_path = os.path.join(output_dir, f"{base_filename}_TRAIN.npz")
            val_path = os.path.join(output_dir, f"{base_filename}_VAL.npz")

            status_message = f"[{current_iteration}/{total_combinations}] Processing {base_filename}..."

            # Skip if the combination has already been extracted (allows resuming)
            if os.path.exists(train_path) and os.path.exists(val_path):
                print(f"{status_message} (Skipping - Already Exists)".ljust(80), end='\r', flush=True)
                continue
            else:
                print(status_message.ljust(80), end='\r', flush=True)

            start_local_time = time.time()

            # Execute feature extraction silently to maintain console clarity
            with HiddenPrints():
                df_train = extract_all_features(raw_train_data, window, step, zc, ssc)
                df_val = extract_all_features(raw_val_data, window, step, zc, ssc)

            # Save the extracted features to compressed NumPy arrays
            for dataframe, filepath in [(df_train, train_path), (df_val, val_path)]:
                if not dataframe.empty:
                    columns_to_drop = ['Restimulus', 'Subject', 'dataset_type', 'Rerepetition', 'block_id']
                    existing_cols_to_drop = [col for col in columns_to_drop if col in dataframe.columns]

                    X = dataframe.drop(columns=existing_cols_to_drop).values
                    y = dataframe['Restimulus'].values
                    np.savez_compressed(filepath, X=X, y=y)

            elapsed_total = str(datetime.timedelta(seconds=int(time.time() - start_global_time)))
            log_entry = pd.DataFrame([{
                'File': base_filename,
                'Time_Seconds': round(time.time() - start_local_time, 2),
                'Elapsed_Total': elapsed_total
            }])
            log_entry.to_csv("Feature_Generation_Times.csv", mode='a',
                             header=not os.path.exists("Feature_Generation_Times.csv"), index=False)

    total_duration = str(datetime.timedelta(seconds=int(time.time() - start_global_time)))
    print(f"\n\nSTAGE 1 COMPLETE. Total Duration: {total_duration}")


if __name__ == "__main__":
    main()