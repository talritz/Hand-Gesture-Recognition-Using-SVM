"""
===============================================================================
Hyperparameters Configuration File
===============================================================================
This module serves as the central configuration hub for the pipeline.
It stores the optimal parameters identified during the Grid Search phase,
ensuring reproducibility and isolating configuration from execution logic.
"""
import pandas as pd

# Optimal temporal parameters for signal windowing and feature extraction.
# These values were determined per kernel to maximize classification accuracy.
MODEL_DATA_CONFIG = {
    'linear': {
        'margin': 1000,
        'window': 800,
        'step': 400,
        'zc': 1e-08,
        'ssc': 0.0001
    },
    'poly': {
        'margin': 800,
        'window': 400,
        'step': 200,
        'zc': 1e-08,
        'ssc': 1e-08
    },
    'rbf': {
        'margin': 600,
        'window': 800,
        'step': 400,
        'zc': 1e-08,
        'ssc': 0.0001
    },
    'sigmoid': {
        'margin': 1000,
        'window': 800,
        'step': 400,
        'zc': 1e-08,
        'ssc': 1e-06
    }
}

# Optimal algorithmic parameters for the Support Vector Machine (SVM) models.
# 'class_weight': 'balanced' is applied globally to assist with class imbalances.
MODEL_PARAMS = {
    'linear':  {'C': 0.01,  'class_weight': 'balanced'},
    'rbf':     {'C': 1.0,   'gamma': 'auto',  'class_weight': 'balanced'},
    'poly':    {'C': 0.1,   'gamma': 0.1,     'degree': 3, 'class_weight': 'balanced'},
    'sigmoid': {'C': 100.0, 'gamma': 'scale', 'class_weight': 'balanced'}
}

# Global flags for controlling the execution of preprocessing and optimization modules.
USE_UNDERSAMPLING = False
USE_PCA = False
PCA_VARIANCE_THRESHOLD = 0.85
USE_YOUDENS_J = True
NORMALIZE_CM = True


def undersample_rest_class(df_features, rest_label=0):
    """
    Balances the dataset by undersampling the majority class (typically the 'Rest' state).
    The target size for the majority class is determined by calculating the mean
    sample size of all other active classes.
    """
    print(f"\n--- Applying Undersampling to Class {rest_label} ---")

    # Calculate the frequency of each class in the dataset.
    class_counts = df_features['Restimulus'].value_counts()

    # Exclude the rest label to compute the average size of the active classes.
    active_classes = class_counts.drop(labels=[rest_label], errors='ignore')
    target_size = int(active_classes.mean())

    # Separate the dataset into the majority class and the active classes.
    df_rest = df_features[df_features['Restimulus'] == rest_label]
    df_active = df_features[df_features['Restimulus'] != rest_label]

    # Sample the majority class down to the computed target size if it exceeds it.
    if len(df_rest) > target_size:
        df_rest_sampled = df_rest.sample(n=target_size, random_state=42)
        print(f"-> Class {rest_label} reduced to {target_size} samples.")
    else:
        df_rest_sampled = df_rest

    # Concatenate the balanced rest class with the active classes and shuffle the rows.
    return pd.concat([df_rest_sampled, df_active]).sample(frac=1, random_state=42).reset_index(drop=True)