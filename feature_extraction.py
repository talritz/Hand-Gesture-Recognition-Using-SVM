import numpy as np
import pandas as pd

import hyperparameters as hp

def calc_rms(emg_signals):
    return np.sqrt(np.mean(emg_signals ** 2, axis=0))

def calc_mav(emg_signals):
    return np.mean(np.abs(emg_signals), axis=0)

def calc_wl(emg_signals):
    return np.sum(np.abs(np.diff(emg_signals, axis=0)), axis=0)

def calc_zc(emg_signals, threshold):
    return np.sum((emg_signals[:-1] * emg_signals[1:] < 0) &
                  (np.abs(emg_signals[:-1] - emg_signals[1:]) > threshold), axis=0)

def calc_ssc(emg_signals, threshold):
    diffs = np.diff(emg_signals, axis=0)
    ssc_products = -diffs[:-1] * diffs[1:]
    return np.sum(ssc_products >= threshold, axis=0)

def extract_all_features(df, window_size=None, step_size=None, zc_thresh=None, ssc_delta=None):
    if window_size is None: window_size = hp.WINDOW_SIZE
    if step_size is None: step_size = hp.STEP_SIZE
    if zc_thresh is None: zc_thresh = hp.ZC_THRESH
    if ssc_delta is None: ssc_delta = hp.SSC_DELTA

    if df.empty:
        return pd.DataFrame()

    features = []
    emg_cols = [col for col in df.columns if 'EMG' in col]
    dataset_type = df['dataset_type'].iloc[0]

    print(f"Extracting features from '{dataset_type}' (Window: {window_size})...")

    for start in range(0, len(df) - window_size, step_size):
        end = start + window_size
        window_data = df.iloc[start:end]

        if window_data['Restimulus'].nunique() > 1:
            continue

        emg_signals = window_data[emg_cols].values

        rms = calc_rms(emg_signals)
        mav = calc_mav(emg_signals)
        wl = calc_wl(emg_signals)
        zc = calc_zc(emg_signals, zc_thresh)
        ssc = calc_ssc(emg_signals, ssc_delta)

        row_info = {
            'Restimulus': window_data['Restimulus'].iloc[0],
            'Subject': window_data['Subject'].iloc[0],
            'dataset_type': dataset_type
        }

        for i, col in enumerate(emg_cols):
            row_info[f'{col}_RMS'] = rms[i]
            row_info[f'{col}_MAV'] = mav[i]
            row_info[f'{col}_WL'] = wl[i]
            row_info[f'{col}_ZC'] = zc[i]
            row_info[f'{col}_SSC'] = ssc[i]

        features.append(row_info)

    return pd.DataFrame(features)