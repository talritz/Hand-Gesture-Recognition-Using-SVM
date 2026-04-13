"""
===============================================================================
Data Distribution Analyzer (Standalone Utility)
===============================================================================
This script calculates the exact ratio and percentage of the 'Rest' class
compared to the active gestures in the training dataset.
It integrates directly with the established Ninapro data loading pipeline
to evaluate the true distribution of the extracted features.
"""

import os
import pandas as pd

# Custom Module Imports
from data_loading import load_cleaned_ninapro_data
from feature_extraction import extract_all_features
import hyper_parameters as hp

def calculate_rest_distribution(df_features, rest_label=0):
    """
    Analyzes the class distribution of the extracted EMG features and prints
    the mathematical metrics required for the final thesis report.
    """
    print("\n--- Calculating Data Distribution Metrics ---")

    # Calculate the total number of samples and the specific number of 'Rest' samples
    total_samples = len(df_features)
    rest_samples = len(df_features[df_features['Restimulus'] == rest_label])

    # 1. Calculate the percentage of the 'Rest' class out of the entire dataset
    rest_percentage = (rest_samples / total_samples) * 100

    # 2. Calculate the ratio of the 'Rest' class compared to the average of the active gestures
    class_counts = df_features['Restimulus'].value_counts()
    active_classes_avg = class_counts.drop(labels=[rest_label], errors='ignore').mean()
    ratio = rest_samples / active_classes_avg

    # Output the exact English sentences required for the report
    print("\n[+] Exact Text for the Final Report:")
    print("-" * 70)
    print(f"The 'Rest' state is exactly {ratio:.1f} times larger on average than the other gestures.")
    print(f"It constitutes approximately {rest_percentage:.1f}% of the entire dataset.")
    print("-" * 70)

    return ratio, rest_percentage


if __name__ == "__main__":
    # =========================================================================
    # Data Loading and Feature Extraction Pipeline
    # =========================================================================
    paths_to_check = [
        r'C:\Users\Tal\OneDrive - Afeka College Of Engineering\הקבצים של Nadav Matza - פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'C:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets'
    ]
    base_path = next((p for p in paths_to_check if os.path.exists(p)), None)

    if not base_path:
        print("Error: Dataset path not found. Please verify your directories.")
        exit()

    # We calculate the distribution specifically on the Training set (Subjects 1-8)
    train_subjects = [1, 2, 3, 4, 5, 6, 7, 8]

    # Temporarily set parameters to default 'linear' config for feature extraction
    # (The temporal parameters slightly alter the total number of windows,
    # but the distribution ratio remains consistent).
    default_cfg = hp.MODEL_DATA_CONFIG['linear']
    hp.MARGIN_SAMPLES = default_cfg['margin']
    hp.WINDOW_SIZE = default_cfg['window']
    hp.STEP_SIZE = default_cfg['step']
    hp.ZC_THRESH = default_cfg['zc']
    hp.SSC_DELTA = default_cfg['ssc']

    print("=" * 80)
    print(" INITIATING DATA DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # 1. Load Raw Data
    print("\n[+] Loading Train Dataset...")
    train_data_raw = load_cleaned_ninapro_data(base_path, train_subjects, 'Train')

    # 2. Extract Features
    print("\n[+] Extracting Features...")
    df_train_features = extract_all_features(train_data_raw)

    # 3. Calculate and Print the Distribution
    calculate_rest_distribution(df_train_features)