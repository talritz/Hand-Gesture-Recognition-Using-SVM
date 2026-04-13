"""
===============================================================================
Variance & Robustness Experiment (Intra-Subject vs. Inter-Subject)
===============================================================================
Validates model generalization capabilities. Utilizes temporal division
(Repetitions) for training and evaluation instead of subject-based splitting,
assessing variance across multiple generalization scenarios.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Custom Module Imports
from data_loading import load_cleaned_ninapro_data
from feature_extraction import extract_all_features
from model_training import evaluate_single_kernel_ovo
import hyper_parameters as hp

# Enable Intel Scikit-learn framework optimizations.
from sklearnex import patch_sklearn
patch_sklearn()


def plot_variance_cm(y_true, y_pred, class_labels, kernel_name, experiment_name):
    """
    Renders and exports the Confusion Matrix specific to variance experimental data.
    """
    output_dir = "Variance_Experiment_Plots"
    os.makedirs(output_dir, exist_ok=True)

    # Apply global colormap scaling when normalizing data.
    if hp.NORMALIZE_CM:
        cm = confusion_matrix(y_true, y_pred, labels=class_labels, normalize='true') * 100
        val_format = '.1f'
        norm_txt = "(%)"
        scale_args = {'im_kw': {'vmin': 0, 'vmax': 100}}
    else:
        cm = confusion_matrix(y_true, y_pred, labels=class_labels)
        val_format = 'd'
        norm_txt = "(Counts)"
        scale_args = {}

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues, values_format=val_format, ax=plt.gca(), **scale_args)

    plt.title(f"Repetition Division: {experiment_name.replace('_', ' ')}\n{kernel_name.upper()} Kernel {norm_txt}",
              fontsize=12, fontweight='bold')

    clean_exp_name = experiment_name.replace(' ', '_').replace('+', 'and')
    filename = os.path.join(output_dir, f"CM_{clean_exp_name}_{kernel_name.upper()}.png")

    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    plt.close()
    print(f"    -> Saved CM Plot: {filename}")


def run_variance_experiment(base_path, subjects, experiment_name, kernel, model_params, data_cfg):
    """
    Orchestrates the entire temporal sequence testing process including
    temporal splitting, feature scaling, model training, and matrix generation.
    """
    print("\n" + "=" * 60)
    print(f" EXPERIMENT: {experiment_name.upper()} | KERNEL: {kernel.upper()} ")
    print(f" Subjects: {subjects}")
    print("=" * 60)

    # Reassign global temporal parameters specific to the target kernel.
    hp.MARGIN_SAMPLES = data_cfg['margin']
    hp.WINDOW_SIZE = data_cfg['window']
    hp.STEP_SIZE = data_cfg['step']
    hp.ZC_THRESH = data_cfg['zc']
    hp.SSC_DELTA = data_cfg['ssc']

    # Read raw datasets devoid of preprocessing.
    raw_data = load_cleaned_ninapro_data(base_path, subjects, 'Raw_Data')

    rep_col = next((c for c in ['Rerepetition', 'repetition', 'Repetition'] if c in raw_data.columns), None)
    if not rep_col:
        print("[!] Error: Could not find repetition column for splitting.")
        return None

    # Temporal Division: Allocate early repetitions for Training, later repetitions for Validation.
    print(f"Splitting raw data by {rep_col} (Train: Reps 1-4, Val: Reps 5-6)...")
    train_raw = raw_data[raw_data[rep_col].isin([1, 2, 3, 4])].copy()
    val_raw = raw_data[raw_data[rep_col].isin([5, 6])].copy()

    # Formulate features from the separated datasets.
    print("Extracting features based on defined kernel parameters...")
    df_train_features = extract_all_features(train_raw)
    df_val_features = extract_all_features(val_raw)

    if hp.USE_UNDERSAMPLING:
        df_train_features = hp.undersample_rest_class(df_train_features)

    cols_to_drop = ['Restimulus', 'Subject', 'dataset_type', rep_col]
    cols_to_drop_train = [c for c in cols_to_drop if c in df_train_features.columns]
    cols_to_drop_val = [c for c in cols_to_drop if c in df_val_features.columns]

    X_train = df_train_features.drop(columns=cols_to_drop_train)
    y_train = df_train_features['Restimulus']
    X_val = df_val_features.drop(columns=cols_to_drop_val)
    y_val = df_val_features['Restimulus']

    class_labels = sorted(y_train.unique())

    # Formulate feature scaling mapping explicitly on the training group.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train underlying model mapping and capture baseline characteristics.
    res_dict, trained_ovo, trained_ova, y_val_pred = evaluate_single_kernel_ovo(
        kernel, model_params, X_train_scaled, y_train, X_val_scaled, y_val, class_labels
    )

    plot_variance_cm(y_val, y_val_pred, class_labels, kernel, experiment_name)

    print(f"\n---> RESULTS FOR {experiment_name} ({kernel.upper()}):")
    print(f"     Macro F1: {res_dict['Macro_F1_Base (%)']}%")
    print(f"     Balanced Accuracy: {res_dict['Balanced_Acc_Base (%)']}%")
    print("-" * 60)

    res_dict['Experiment'] = experiment_name
    return res_dict


def main():
    # Resolve directory paths for execution environment robustness.
    paths_to_check = [
        r'C:\Users\Tal\OneDrive - Afeka College Of Engineering\הקבצים של Nadav Matza - פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'C:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets'
    ]
    base_path = next((p for p in paths_to_check if os.path.exists(p)), None)
    if not base_path:
        print("Error: Dataset path not found.")
        return

    all_results = []

    kernels_to_test = ['linear', 'rbf', 'poly', 'sigmoid']

    for kernel in kernels_to_test:
        if kernel not in hp.MODEL_PARAMS:
            continue

        model_params = hp.MODEL_PARAMS[kernel]
        data_cfg = hp.MODEL_DATA_CONFIG[kernel]

        # Scenario 1: Evaluation restricted to internal properties of Subject 1.
        res_intra = run_variance_experiment(base_path, [1], "Intra-Subject_Subj1", kernel, model_params, data_cfg)
        if res_intra: all_results.append(res_intra)

    if all_results:
        print("\n\n" + "=" * 60)
        print(" FINAL VARIANCE EXPERIMENT RESULTS ")
        print("=" * 60)

        results_df = pd.DataFrame(all_results)

        cols = ['Experiment', 'Kernel', 'Macro_F1_Base (%)', 'Balanced_Acc_Base (%)']
        final_df = results_df[cols]
        print(final_df.to_string(index=False))
        print("=" * 60)

        csv_filename = "Repetition_Division_Results.csv"
        final_df.to_csv(csv_filename, index=False)
        print(f"\n[+] Successfully saved experimental results to: {csv_filename}")
        print("[+] Confusion matrices saved in 'Variance_Experiment_Plots' directory.")


if __name__ == "__main__":
    main()