import numpy as np
import pandas as pd
import hyperparameters as hp


# ---------------------------------------------------------
# Time-Domain Feature Functions (Based on Gopal et al.)
# ---------------------------------------------------------
def calc_rms(emg_signals):
    """ Equation (1): Root Mean Square """
    return np.sqrt(np.mean(emg_signals ** 2, axis=0))


def calc_var(emg_signals):
    """ Equation (2): Variance """
    return np.var(emg_signals, axis=0)


def calc_wl(emg_signals):
    """ Equation (4): Waveform Length """
    return np.sum(np.abs(np.diff(emg_signals, axis=0)), axis=0)


def calc_emav(emg_signals):
    """
    Equation (3): Enhanced Mean Absolute Value
    Applies p=0.75 to the middle 60% of the window, and p=0.50 to the edges.
    """
    N = emg_signals.shape[0]
    # Initialize p array with 0.50
    p = np.full((N, 1), 0.50)

    # Calculate indices for the 20% to 80% range
    start_idx = int(0.2 * N)
    end_idx = int(0.8 * N)

    # Set middle 60% to 0.75
    p[start_idx:end_idx + 1] = 0.75

    # Calculate EMAV
    return np.mean(np.abs(emg_signals) ** p, axis=0)


# ---------------------------------------------------------
# Main Feature Extraction Function
# ---------------------------------------------------------
def extract_all_features(df, window_size=None, step_size=None):
    if window_size is None: window_size = hp.WINDOW_SIZE
    if step_size is None: step_size = hp.STEP_SIZE

    if df.empty:
        return pd.DataFrame()

    features = []
    emg_cols = [col for col in df.columns if 'EMG' in col]
    dataset_type = df['dataset_type'].iloc[0]

    print(f"Extracting Paper-Aligned Features from '{dataset_type}' (Window: {window_size}, Step: {step_size})...")

    for start in range(0, len(df) - window_size, step_size):
        end = start + window_size
        window_data = df.iloc[start:end]

        if window_data['Restimulus'].nunique() > 1:
            continue

        emg_signals = window_data[emg_cols].values

        # Extract strictly the 4 features defined in the paper
        rms = calc_rms(emg_signals)
        var = calc_var(emg_signals)
        wl = calc_wl(emg_signals)
        emav = calc_emav(emg_signals)

        row_info = {
            'Restimulus': window_data['Restimulus'].iloc[0],
            'Subject': window_data['Subject'].iloc[0],
            'dataset_type': dataset_type
        }

        for i, col in enumerate(emg_cols):
            row_info[f'{col}_RMS'] = rms[i]
            row_info[f'{col}_VAR'] = var[i]
            row_info[f'{col}_WL'] = wl[i]
            row_info[f'{col}_EMAV'] = emav[i]

        features.append(row_info)

    return pd.DataFrame(features)