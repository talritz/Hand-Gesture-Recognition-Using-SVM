# ==========================================
# Hyperparameters Configuration File
# ==========================================

# --- 1. Data Loading Parameters ---
MARGIN_SAMPLES = 1500

# --- 2. Feature Extraction Parameters ---
WINDOW_SIZE = 400
STEP_SIZE = 200

# Thresholds for ZC and SSC features
ZC_THRESH = 1e-07
SSC_DELTA = 1e-12

# --- 3. SVM Model Parameters ---
SVM_C = 1.0
SVM_CLASS_WEIGHT = 'balanced'
SVM_MAX_ITER = 10000