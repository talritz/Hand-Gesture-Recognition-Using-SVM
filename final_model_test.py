"""
===============================================================================
Final Test Set Evaluation Script
===============================================================================
Executes the final validation assessment in the pipeline sequence.
Maintains data sanctity by strictly mapping the underlying feature weights
to Subjects 1-8, evaluating performance entirely on the isolated testing set.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Custom Module Imports
from data_loading import load_cleaned_ninapro_data
from feature_extraction import extract_all_features
from model_training import evaluate_single_kernel_ovo
from roc_optimizer import optimize_and_evaluate_pairwise
from roc_optimizer import plot_subset_roc_for_report
import hyper_parameters as hp

# Enable Intel Scikit-learn framework optimizations.
from sklearnex import patch_sklearn
patch_sklearn()

def plot_cm_with_modules(y_true, y_pred, class_labels, kernel_name, stage, normalize_cm, output_dir):
    """
    Renders and exports the final evaluation Confusion Matrix.
    Applies fixed colormap boundaries for accurate visual comparisons.
    """
    if normalize_cm:
        cm = confusion_matrix(y_true, y_pred, labels=class_labels, normalize='true') * 100
        val_format = '.1f'
        norm_txt = "(%)"
        scale_args = {'im_kw': {'vmin': 0, 'vmax': 100}}
    else:
        cm = confusion_matrix(y_true, y_pred, labels=class_labels)
        val_format = 'd'
        norm_txt = "(Counts)"
        scale_args = {}

    active_modules = []
    if hp.USE_UNDERSAMPLING: active_modules.append("Undersampling")
    if hp.USE_PCA: active_modules.append("PCA")
    if stage == "Pairwise Opt": active_modules.append("Youden's J")

    modules_str = " + ".join(active_modules) if active_modules else "Standard Pipeline"

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues, values_format=val_format, ax=plt.gca(), **scale_args)

    plt.title(f"TEST SET: {kernel_name.upper()} Kernel - {stage} {norm_txt}\n[Modules: {modules_str}]",
              fontsize=12, fontweight='bold')

    filename = os.path.join(output_dir, f"CM_TEST_{stage.replace(' ', '')}_{kernel_name.upper()}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    plt.close()

    print(f"    -> Saved CM Plot: {filename}")


def main():
    # Resolve directory paths for execution environment robustness.
    paths_to_check = [
        r'C:\Users\Tal\OneDrive - Afeka College Of Engineering\הקבצים של Nadav Matza - פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'C:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets'
    ]
    base_path = next((p for p in paths_to_check if os.path.exists(p)), None)

    if not base_path:
        print("Error: Dataset path not found. Please verify your directories.")
        return

    # Final split definition enforcing model isolation boundaries.
    train_subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    test_subjects = [17, 18, 19, 20]

    # Map output locations dynamically via control flags.
    active_flags = []
    if hp.USE_YOUDENS_J: active_flags.append("J")
    if hp.USE_PCA: active_flags.append("PCA")
    if hp.USE_UNDERSAMPLING: active_flags.append("US")

    config_name = "_".join(active_flags) if active_flags else "Base"
    output_dir = f"Results_Final_TestSet_{config_name}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(f" FINAL TEST EVALUATION INITIALIZED | Active Configuration: {config_name}")
    print("=" * 80)

    all_results = []

    # Iteration mapping per valid kernel structure.
    for kernel, model_params in hp.MODEL_PARAMS.items():
        data_cfg = hp.MODEL_DATA_CONFIG[kernel]
        hp.MARGIN_SAMPLES = data_cfg['margin']
        hp.WINDOW_SIZE = data_cfg['window']
        hp.STEP_SIZE = data_cfg['step']
        hp.ZC_THRESH = data_cfg['zc']
        hp.SSC_DELTA = data_cfg['ssc']

        print("\n\n" + "*" * 70)
        print(f" STARTING TEST EXPERIMENTS FOR: {kernel.upper()} KERNEL")
        print("*" * 70)

        print("\n[+] LOADING DATA & EXTRACTING FEATURES (Train, Test)...")
        train_data_raw = load_cleaned_ninapro_data(base_path, train_subjects, 'Train')
        test_data_raw = load_cleaned_ninapro_data(base_path, test_subjects, 'Test')

        df_train_features = extract_all_features(train_data_raw)
        df_test_features = extract_all_features(test_data_raw)

        columns_to_drop = ['Restimulus', 'Subject', 'dataset_type']
        class_labels = sorted(df_train_features['Restimulus'].unique())

        if hp.USE_UNDERSAMPLING:
            df_train_features = hp.undersample_rest_class(df_train_features)

        X_train = df_train_features.drop(columns=columns_to_drop)
        y_train = df_train_features['Restimulus']
        X_test = df_test_features.drop(columns=columns_to_drop)
        y_test = df_test_features['Restimulus']

        # Enforce transformation metrics strictly onto the evaluation subjects.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if hp.USE_PCA:
            print(f"Applying PCA (Variance Threshold: {hp.PCA_VARIANCE_THRESHOLD})...")
            pca = PCA(n_components=hp.PCA_VARIANCE_THRESHOLD, random_state=42)
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_test_scaled = pca.transform(X_test_scaled)

        # Base classifier processing phase over evaluation test subjects.
        baseline_res, trained_ovo_model, trained_ova_model, y_test_pred = evaluate_single_kernel_ovo(
            kernel, model_params, X_train_scaled, y_train, X_test_scaled, y_test, class_labels
        )

        plot_cm_with_modules(y_test, y_test_pred, class_labels, kernel, "Baseline", hp.NORMALIZE_CM, output_dir)

        run_result = {'Kernel': kernel.upper(), 'Config': f"{config_name}_TestSet"}
        run_result.update(baseline_res)

        # Advanced evaluation optimization application mapping.
        if hp.USE_YOUDENS_J:
            opt_metrics, y_test_pred_custom = optimize_and_evaluate_pairwise(
                trained_ovo_model, trained_ova_model, X_test_scaled, y_test, class_labels, kernel
            )

            plot_cm_with_modules(y_test, y_test_pred_custom, class_labels, kernel, "Pairwise Opt", hp.NORMALIZE_CM,
                                 output_dir)
            run_result.update(opt_metrics)

            plot_subset_roc_for_report(
                trained_ovo_model, X_test_scaled, y_test, target_class=1, kernel_name=kernel, output_dir=output_dir
            )

        all_results.append(run_result)

    final_df = pd.DataFrame(all_results)
    cols = ['Kernel', 'Config'] + [c for c in final_df.columns if c not in ['Kernel', 'Config']]
    final_df = final_df[cols]
    final_df.dropna(axis=1, how='all', inplace=True)

    csv_path = os.path.join(output_dir, f"Summary_TestSet_{config_name}.csv")
    final_df.to_csv(csv_path, index=False)

    print("\n\n" + "=" * 100)
    print(f" FINAL TEST SET EXPERIMENT RESULTS: {config_name} ")
    print("=" * 100)
    print(final_df.to_string(index=False))
    print("=" * 100)

    print(f"\n[+] Saved complete Test Set run data to: {csv_path}")
    print(f"[+] Saved all Test Set Confusion Matrices into: {output_dir}/")


if __name__ == "__main__":
    main()