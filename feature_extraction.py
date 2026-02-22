import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Hyperparameters Configuration
# ---------------------------------------------------------
WINDOW_SIZE = 200
STEP_SIZE = 100
ZC_THRESH = 1e-6  # Noise threshold for Zero Crossing
SSC_DELTA = 1e-12  # Noise threshold for Slope Sign Change


# ---------------------------------------------------------
# Individual Feature Calculation Functions
# ---------------------------------------------------------
def calc_rms(emg_signals):
    """Calculates Root Mean Square (RMS)"""
    return np.sqrt(np.mean(emg_signals ** 2, axis=0))


def calc_mav(emg_signals):
    """Calculates Mean Absolute Value (MAV)"""
    return np.mean(np.abs(emg_signals), axis=0)


def calc_wl(emg_signals):
    """Calculates Waveform Length (WL)"""
    return np.sum(np.abs(np.diff(emg_signals, axis=0)), axis=0)


def calc_zc(emg_signals, threshold):
    """Calculates Zero Crossing (ZC) with threshold"""
    return np.sum((emg_signals[:-1] * emg_signals[1:] < 0) &
                  (np.abs(emg_signals[:-1] - emg_signals[1:]) > threshold), axis=0)


def calc_ssc(emg_signals, threshold):
    """Calculates Slope Sign Change (SSC) with threshold"""
    diffs = np.diff(emg_signals, axis=0)
    ssc_products = -diffs[:-1] * diffs[1:]
    return np.sum(ssc_products >= threshold, axis=0)


# ---------------------------------------------------------
# Main Feature Extraction Function
# ---------------------------------------------------------
def extract_all_features(df):
    """
    Slides a window over the raw EMG data, extracts time-domain 
    features for each channel, and returns a new feature DataFrame.
    """
    if df.empty:
        return pd.DataFrame()

    features = []
    emg_cols = [col for col in df.columns if 'EMG' in col]
    dataset_type = df['dataset_type'].iloc[0]

    print(f"Extracting features from '{dataset_type}' dataset...")

    for start in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
        end = start + WINDOW_SIZE
        window_data = df.iloc[start:end]

        # Skip transition windows containing multiple gestures
        if window_data['Restimulus'].nunique() > 1:
            continue

        # Convert to numpy array for fast mathematical operations
        emg_signals = window_data[emg_cols].values

        # Calculate features using helper functions
        rms = calc_rms(emg_signals)
        mav = calc_mav(emg_signals)
        wl = calc_wl(emg_signals)
        zc = calc_zc(emg_signals, ZC_THRESH)
        ssc = calc_ssc(emg_signals, SSC_DELTA)

        # Organize labels and metadata for the current window
        row_info = {
            'Restimulus': window_data['Restimulus'].iloc[0],
            'Subject': window_data['Subject'].iloc[0],
            'dataset_type': dataset_type
        }

        # Assign features to their respective channels
        for i, col in enumerate(emg_cols):
            row_info[f'{col}_RMS'] = rms[i]
            row_info[f'{col}_MAV'] = mav[i]
            row_info[f'{col}_WL'] = wl[i]
            row_info[f'{col}_ZC'] = zc[i]
            row_info[f'{col}_SSC'] = ssc[i]

        features.append(row_info)

    final_features_df = pd.DataFrame(features)
    print(f"Extracted {len(final_features_df)} feature windows.")

    return final_features_df