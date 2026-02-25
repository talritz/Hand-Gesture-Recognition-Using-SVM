from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score
import pandas as pd

import hyperparameters as hp

def evaluate_svm_kernels(df_train_features, df_val_features, svm_c=None, kernels_to_test=None):
    if svm_c is None: svm_c = hp.SVM_C
    # Default to testing only 'linear' for grid search speed, unless specified otherwise
    if kernels_to_test is None: kernels_to_test = ['linear']

    columns_to_drop = ['Restimulus', 'Subject', 'dataset_type']

    X_train = df_train_features.drop(columns=columns_to_drop)
    y_train = df_train_features['Restimulus']

    X_val = df_val_features.drop(columns=columns_to_drop)
    y_val = df_val_features['Restimulus']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    kernel_val_results = {}

    for kernel_name in kernels_to_test:
        if kernel_name == 'linear':
            svm_model = LinearSVC(
                random_state=42,
                dual=False,
                C=svm_c,
                max_iter=hp.SVM_MAX_ITER,
                class_weight=hp.SVM_CLASS_WEIGHT
            )
        else:
            svm_model = SVC(
                kernel=kernel_name,
                random_state=42,
                C=svm_c,
                gamma=hp.SVM_GAMMA,
                degree=hp.SVM_DEGREE,
                class_weight=hp.SVM_CLASS_WEIGHT
            )

        svm_model.fit(X_train_scaled, y_train)
        y_val_pred = svm_model.predict(X_val_scaled)

        _, _, macro_f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average='macro', zero_division=0)
        bal_acc = balanced_accuracy_score(y_val, y_val_pred)

        kernel_val_results[kernel_name] = {'macro_f1': macro_f1, 'balanced_accuracy': bal_acc}

    return kernel_val_results