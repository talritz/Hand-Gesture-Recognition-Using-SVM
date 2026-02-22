from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd


def evaluate_svm_kernels(df_train_features, df_val_features, df_test_features):
    """
    Normalizes the data, trains SVM models using different kernels,
    selects the best kernel using the validation set,
    and evaluates the final chosen model on the test set.
    """
    print("\nPreparing data for modeling...")

    columns_to_drop = ['Restimulus', 'Subject', 'dataset_type']

    # 1. Separate features (X) and labels (y) for all sets
    X_train = df_train_features.drop(columns=columns_to_drop)
    y_train = df_train_features['Restimulus']

    X_val = df_val_features.drop(columns=columns_to_drop)
    y_val = df_val_features['Restimulus']

    X_test = df_test_features.drop(columns=columns_to_drop)
    y_test = df_test_features['Restimulus']

    # 2. Data Normalization
    scaler = StandardScaler()

    # Fit ONLY on training data, transform all three
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 3. Setup the kernels competition
    kernels_to_test = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel_val_results = {}
    trained_models = {}

    print("\n--- Phase A: SVM Kernel Evaluation (on Validation Set) ---")
    print(f"Training on {len(X_train_scaled)} samples. This may take a while for non-linear kernels...")

    for kernel_name in kernels_to_test:
        print(f"Training SVM with '{kernel_name}' kernel...")

        # Optimization: Use LinearSVC for the linear kernel (drastically faster for large datasets)
        if kernel_name == 'linear':
            svm_model = LinearSVC(random_state=42, dual=False, max_iter=10000)
        else:
            svm_model = SVC(kernel=kernel_name, random_state=42)

        # Train the model
        svm_model.fit(X_train_scaled, y_train)

        # Predict on VALIDATION set to find the best kernel
        y_val_pred = svm_model.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, y_val_pred)

        kernel_val_results[kernel_name] = accuracy
        trained_models[kernel_name] = svm_model

        print(f"--> Validation Accuracy for '{kernel_name}': {accuracy * 100:.2f}%")

    # 4. Find the winning kernel based on Validation scores
    best_kernel = max(kernel_val_results, key=kernel_val_results.get)

    print("\n" + "=" * 45)
    print(f"Best Kernel Selected: '{best_kernel.upper()}'")
    print("=" * 45)

    # 5. Final Test Evaluation using the winning kernel
    print(f"\n--- Phase B: Final Test Evaluation (using '{best_kernel}' kernel) ---")

    best_model = trained_models[best_kernel]
    y_test_pred = best_model.predict(X_test_scaled)
    final_test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"FINAL UNBIASED TEST ACCURACY: {final_test_accuracy * 100:.2f}%")
    print("=" * 45)

    return final_test_accuracy