"""
===============================================================================
Feature Extraction Module
===============================================================================
Computes time-domain features from raw EMG signals using a sliding window approach.
"""
import numpy as np
import pandas as pd
import hyper_parameters as hp


def calc_rms(emg_signals):
    """
    Computes the Root Mean Square (RMS).
    Represents the signal power and correlates with non-fatiguing muscle force.
    """
    return np.sqrt(np.mean(emg_signals ** 2, axis=0))


def calc_var(emg_signals):
    """
    Computes the Variance (VAR).
    Measures the dispersion of the signal around its mean, indicative of signal power.
    """
    return np.var(emg_signals, axis=0)


def calc_wl(emg_signals):
    """
    Computes the Waveform Length (WL).
    Captures amplitude, frequency, and duration information in a single metric.
    """
    return np.sum(np.abs(np.diff(emg_signals, axis=0)), axis=0)


def calc_emav(emg_signals):
    """
    Computes the Enhanced Mean Absolute Value (EMAV).
    Applies a piecewise weight function to emphasize the steady-state segment
    of the signal while suppressing initial and final transients within the window.
    """
    N = emg_signals.shape[0]
    p = np.full((N, 1), 0.50)

    # Assign higher weight (0.75) to the middle 60% of the window.
    start_idx = int(0.2 * N)
    end_idx = int(0.8 * N)
    p[start_idx:end_idx + 1] = 0.75

    return np.mean(np.abs(emg_signals) ** p, axis=0)


def calc_zc(emg_signals, threshold):
    """
    Computes Zero Crossings (ZC).
    Counts zero-crossings only if the amplitude difference exceeds a specified
    threshold to prevent voltage noise from artificially inflating the metric.
    """
    return np.sum((emg_signals[:-1] * emg_signals[1:] < 0) &
                  (np.abs(emg_signals[:-1] - emg_signals[1:]) > threshold), axis=0)


def calc_ssc(emg_signals, threshold):
    """
    Computes Slope Sign Changes (SSC).
    Counts local peaks and valleys, filtered by a threshold to mitigate noise.
    """
    diffs = np.diff(emg_signals, axis=0)
    ssc_products = -diffs[:-1] * diffs[1:]
    return np.sum(ssc_products >= threshold, axis=0)


def extract_all_features(df, window_size=None, step_size=None, zc_thresh=None, ssc_delta=None):
    """
    Iterates over the dataset using an overlapping sliding window.
    Calculates all defined features for each valid window.
    """
    # Fallback to hyperparameters if local arguments are omitted.
    if window_size is None: window_size = hp.WINDOW_SIZE
    if step_size is None: step_size = hp.STEP_SIZE
    if zc_thresh is None: zc_thresh = hp.ZC_THRESH
    if ssc_delta is None: ssc_delta = hp.SSC_DELTA

    if df.empty:
        return pd.DataFrame()

    features = []

    # Isolate columns corresponding to raw EMG channels.
    emg_cols = [col for col in df.columns if 'EMG' in col]
    dataset_type = df['dataset_type'].iloc[0]

    print(f"Extracting 6 Features (RMS, VAR, WL, EMAV, ZC, SSC) from '{dataset_type}'...")

    # Execute the sliding window traversal.
    for start in range(0, len(df) - window_size, step_size):
        end = start + window_size
        window_data = df.iloc[start:end]

        # Enforce label purity: Discard windows that overlap a transition between two classes.
        if window_data['Restimulus'].nunique() > 1:
            continue

        # Extract numerical data for vectorized calculations.
        emg_signals = window_data[emg_cols].values

        # Compute features simultaneously across all channels.
        rms = calc_rms(emg_signals)
        var = calc_var(emg_signals)
        wl = calc_wl(emg_signals)
        emav = calc_emav(emg_signals)
        zc = calc_zc(emg_signals, zc_thresh)
        ssc = calc_ssc(emg_signals, ssc_delta)

        # Retain identifying metadata for the current window.
        row_info = {
            'Restimulus': window_data['Restimulus'].iloc[0],
            'Subject': window_data['Subject'].iloc[0],
            'dataset_type': dataset_type
        }

        # Flatten the feature arrays into a single row dictionary.
        for i, col in enumerate(emg_cols):
            row_info[f'{col}_RMS'] = rms[i]
            row_info[f'{col}_VAR'] = var[i]
            row_info[f'{col}_WL'] = wl[i]
            row_info[f'{col}_EMAV'] = emav[i]
            row_info[f'{col}_ZC'] = zc[i]
            row_info[f'{col}_SSC'] = ssc[i]

        features.append(row_info)

    # Convert the assembled list of dictionaries to a DataFrame.
    return pd.DataFrame(features)