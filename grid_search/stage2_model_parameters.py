from sklearnex import patch_sklearn
patch_sklearn()

import os
import time
import warnings
import numpy as np
import pandas as pd
import itertools
import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.exceptions import ConvergenceWarning

def main():
    print("=" * 60)
    print(" STAGE 2: SVM GRID SEARCH (ALL KERNELS)")
    print("=" * 60)

    input_dir = "Generated_Features_NPZ"
    results_csv = "Final_GridSearch_Results.csv"

    # --- PARAMETER GRID ---
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    c_values = [100, 10, 1, 0.1, 0.01]
    class_weights = [None, 'balanced']
    gammas = ['scale', 'auto', 0.1, 1, 10, 100]
    degrees = [2, 3, 4, 5]

    train_files = [f for f in os.listdir(input_dir) if f.endswith('_TRAIN.npz')]
    num_files = len(train_files)
    start_global_time = time.time()

    # Total combinations per dataset: 10 Linear + 60 RBF + 240 Poly + 60 Sigmoid = 370
    total_svm_combinations = 370

    for file_index, train_file in enumerate(train_files, 1):
        val_file = train_file.replace('_TRAIN.npz', '_VAL.npz')

        data_train = np.load(os.path.join(input_dir, train_file))
        data_val = np.load(os.path.join(input_dir, val_file))

        X_full_train, y_full_train = data_train['X'], data_train['y']

        # =====================================================================
        # DATA SUBSAMPLING LOGIC (10% of Training Data)
        # =====================================================================
        # 1. Reproducibility: Fixing the random seed ensures that the exact same
        #    10% subset is chosen every time the code runs. This allows for fair
        #    and consistent comparisons across different SVM kernels and runs.
        np.random.seed(42)

        # 2. Calculate the target number of rows (10% of the total dataset).
        subset_size = int(len(X_full_train) * 0.1)

        # 3. Random Selection: Choose random row indices. 'replace=False' is
        #    critical here to guarantee we do not select the same row twice.
        indices = np.random.choice(len(X_full_train), size=subset_size, replace=False)

        # 4. Synchronization: We apply the exact same random indices to BOTH
        #    the feature matrix (X) and the label vector (y). This ensures
        #    the labels remain perfectly aligned with their corresponding features.
        X_train_subset = X_full_train[indices]
        y_train_subset = y_full_train[indices]
        # =====================================================================

        # Validation set remains at 100% to test the model's true generalization
        X_val_full, y_val_full = data_val['X'], data_val['y']

        # Scale based only on the training subset to prevent data leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_subset)
        X_val_scaled = scaler.transform(X_val_full)

        metadata = train_file.split('_')
        current_svm_iteration = 0

        print(f"\n>>> Starting SVM Grid for File {file_index}/{num_files}: {train_file}")

        for kernel, c_val, weight in itertools.product(kernels, c_values, class_weights):

            # Filter relevant parameters dynamically based on the current kernel type
            gamma_list = gammas if kernel in ['rbf', 'poly', 'sigmoid'] else [None]
            degree_list = degrees if kernel == 'poly' else [None]

            for gamma, degree in itertools.product(gamma_list, degree_list):
                current_svm_iteration += 1

                print(
                    f"File [{file_index}/{num_files}] | SVM [{current_svm_iteration}/{total_svm_combinations}] | {kernel.upper()} C={c_val} G={gamma} W={weight} D={degree}",
                    flush=True)

                # Limit iterations strictly for the Sigmoid kernel to prevent infinite execution times
                max_iterations = 2 if kernel == 'sigmoid' else -1

                base_svm = SVC(
                    kernel=kernel,
                    C=c_val,
                    gamma=gamma if gamma else 'scale',
                    cache_size=1000,
                    degree=degree if degree else 3,
                    class_weight=weight,
                    max_iter=max_iterations
                )

                clf = OneVsOneClassifier(base_svm)
                clf.fit(X_train_scaled, y_train_subset)

                y_pred = clf.predict(X_val_scaled)

                macro_f1 = f1_score(y_val_full, y_pred, average='macro', zero_division=0)
                balanced_acc = balanced_accuracy_score(y_val_full, y_pred)
                recalls = recall_score(y_val_full, y_pred, average=None, zero_division=0)
                support_vector_count = sum([len(estimator.support_) for estimator in clf.estimators_])

                results_df = pd.DataFrame([{
                    'Margin': metadata[0][1:],
                    'Window': metadata[1][1:],
                    'Kernel': kernel,
                    'C': c_val,
                    'Gamma': gamma,
                    'Degree': degree,
                    'Class_Weight': weight,
                    'Macro_F1': macro_f1,
                    'Balanced_Acc': balanced_acc,
                    'Class_0_Rec': recalls[0],
                    'SV_Count': support_vector_count,
                    'Elapsed': str(datetime.timedelta(seconds=int(time.time() - start_global_time)))
                }])

                results_df.to_csv(results_csv, mode='a', header=not os.path.exists(results_csv), index=False)

    total_duration = str(datetime.timedelta(seconds=int(time.time() - start_global_time)))
    print(f"\n\nSTAGE 2 COMPLETE. Total time: {total_duration}")


if __name__ == "__main__":
    main()