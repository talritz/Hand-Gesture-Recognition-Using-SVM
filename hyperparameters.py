# ==========================================
# Hyperparameters Configuration File
# ==========================================

# --- 1. Data Loading Parameters ---
# Safety margin around unwanted movements (in samples).
# At 2kHz sampling rate, 1500 samples = 0.75 seconds.
MARGIN_SAMPLES = 1500

# --- 2. Feature Extraction Parameters ---
# Window size for feature extraction (in samples).
# 200 samples at 2kHz = 100ms.
WINDOW_SIZE = 200

# Step size for the sliding window (in samples).
# 100 samples = 50ms overlap.
STEP_SIZE = 100

# Noise threshold for Zero Crossing (ZC)
ZC_THRESH = 1e-6

# Noise threshold for Slope Sign Change (SSC)
SSC_DELTA = 1e-12

# --- 3. SVM Model Parameters ---
# Regularization parameter. Higher C = tighter fit to training data (risk of overfitting).
# Lower C = smoother decision boundary. Default is 1.0.
SVM_C = 1.0

# Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
# Can be 'scale', 'auto', or a specific float value. Default is 'scale'.
SVM_GAMMA = 'scale'

# Degree of the polynomial kernel function ('poly'). Ignored by other kernels.
# Default is 3.
SVM_DEGREE = 3

# Class weight configuration to handle imbalanced data.
# 'balanced' automatically adjusts weights inversely proportional to class frequencies.
SVM_CLASS_WEIGHT = 'balanced'

# Maximum number of iterations for the LinearSVC model.
SVM_MAX_ITER = 10000