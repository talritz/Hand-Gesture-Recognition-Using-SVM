# ==========================================
# Hyperparameters Configuration File
# Aligned with Gopal et al. (Sensors 2022)
# ==========================================

# --- 1. Data Loading Parameters ---
MARGIN_SAMPLES = 1500

# --- 2. Feature Extraction Parameters ---
# Window size of 400 samples (200ms at 2kHz)
WINDOW_SIZE = 400

# Step size of 200 samples gives a 50% overlap, as used in the study
STEP_SIZE = 200

# --- 3. SVM Model Parameters ---
SVM_C = 1.0
SVM_CLASS_WEIGHT = 'balanced'
SVM_MAX_ITER = 10000