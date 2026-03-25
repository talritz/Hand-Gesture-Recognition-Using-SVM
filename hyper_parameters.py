# ==========================================
# Hyperparameters Configuration File
# ==========================================

# --- 1. Data Loading Parameters ---
MARGIN_SAMPLES = 800

# --- 2. Feature Extraction Parameters ---
WINDOW_SIZE = 600
STEP_SIZE = 300
ZC_THRESH = 1e-06
SSC_DELTA = 1e-12

# --- 3. SVM Model Parameters ---
MODEL_PARAMS = {
    'linear': {'C': 1.5},
    'poly': {'C': 8, 'gamma': 'auto', 'degree': 3},
    'rbf': {'C': 0.1, 'gamma': 'scale'},
    'sigmoid': {'C': 0.05, 'gamma': 'scale'}
}

# --- 4. Pipeline Control Flags (The Control Panel) ---
USE_UNDERSAMPLING = True
USE_PCA = True
PCA_VARIANCE_THRESHOLD = 0.85
USE_YOUDENS_J = True
NORMALIZE_CM = True