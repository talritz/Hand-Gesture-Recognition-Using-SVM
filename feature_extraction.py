#This file contains all the functions that take care of the feature extraction:
#Time domain features: RMS, MAV, SSC, ZC, WL
#Frequency domain features: MNF, MDF

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# הגדרות פרמטרים (Hyperparameters)
# ---------------------------------------------------------
WINDOW_SIZE = 200
STEP_SIZE = 100
ZC_THRESH = 0.001
SSC_DELTA = 0.0001


# ---------------------------------------------------------
# פונקציות חילוץ מאפיינים פרטניות
# ---------------------------------------------------------
def calc_rms(emg_data):
    return np.sqrt(np.mean(emg_data ** 2, axis=0))


def calc_mav(emg_data):
    return np.mean(np.abs(emg_data), axis=0)


def calc_wl(emg_data):
    return np.sum(np.abs(np.diff(emg_data, axis=0)), axis=0)


def calc_zc(emg_data, threshold):
    return np.sum((emg_data[:-1] * emg_data[1:] < 0) &
                  (np.abs(emg_data[:-1] - emg_data[1:]) > threshold), axis=0)


def calc_ssc(emg_data, threshold):
    diffs = np.diff(emg_data, axis=0)
    ssc_products = -diffs[:-1] * diffs[1:]
    return np.sum(ssc_products >= threshold, axis=0)


# ---------------------------------------------------------
# הפונקציה הראשית
# ---------------------------------------------------------
def extract_all_features(df):
    features = []
    emg_cols = [col for col in df.columns if 'EMG' in col]

    for start in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
        end = start + WINDOW_SIZE
        window_data = df.iloc[start:end]

        if window_data['Restimulus'].nunique() > 1:
            continue

        emg_signals = window_data[emg_cols].values

        # קריאה לפונקציות העזר
        rms = calc_rms(emg_signals)
        mav = calc_mav(emg_signals)
        wl = calc_wl(emg_signals)
        zc = calc_zc(emg_signals, ZC_THRESH)
        ssc = calc_ssc(emg_signals, SSC_DELTA)

        # ארגון התגיות והמידע של החלון
        row_info = {
            'Restimulus': window_data['Restimulus'].iloc[0],
            'Subject': window_data['Subject'].iloc[0],
            'dataset_type': window_data['dataset_type'].iloc[0]
        }

        # הוספת המאפיינים לכל ערוץ
        for i, col in enumerate(emg_cols):
            row_info[f'{col}_RMS'] = rms[i]
            row_info[f'{col}_MAV'] = mav[i]
            row_info[f'{col}_WL'] = wl[i]
            row_info[f'{col}_ZC'] = zc[i]
            row_info[f'{col}_SSC'] = ssc[i]

        features.append(row_info)

    return pd.DataFrame(features)