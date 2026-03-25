import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt


# ---------------------------------------------------------
# --- Signal Processing Functions ---
# ---------------------------------------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(data, lowcut=5.0, highcut=650.0, fs=2000.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y


# ---------------------------------------------------------
# --- Data Extraction Function ---
# ---------------------------------------------------------
def import_db2(folder_path, subject, rest_length_cap=999):
    """Function for extracting data from raw NinaiPro files for DB2.
    Modified:
    1. Bandpass filter (5-650Hz).
    2. IEEE Outlier Removal Method (Keep only where stimulus == restimulus).
    """
    fs = 2000

    # --- Load Exercise 1 ---
    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_E1_A1.mat')
    data = sio.loadmat(cur_path)
    emg = np.squeeze(np.array(data['emg']))
    rep = np.squeeze(np.array(data['rerepetition']))
    restimulus = np.squeeze(np.array(data['restimulus']))
    stimulus = np.squeeze(np.array(data['stimulus']))  # חילוץ ה-stimulus

    # --- Load Exercise 2 ---
    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_E2_A1.mat')
    data = sio.loadmat(cur_path)

    # איחוד האותות
    emg = np.vstack((emg, np.array(data['emg'])))
    rep = np.append(rep, np.squeeze(np.array(data['rerepetition'])))
    restimulus = np.append(restimulus, np.squeeze(np.array(data['restimulus'])))
    stimulus = np.append(stimulus, np.squeeze(np.array(data['stimulus'])))

    # ---------------------------------------------------------
    # 1. Preprocessing: Bandpass Filter
    # (חובה לעשות לפני החיתוך כדי לשמור על רציפות האות לפילטר)
    # ---------------------------------------------------------
    emg = apply_bandpass_filter(emg, lowcut=5.0, highcut=650.0, fs=fs, order=4)

    # ---------------------------------------------------------
    # 2. IEEE Outlier Detection & Removal (stimulus == restimulus)
    # ---------------------------------------------------------
    # יצירת מסיכה בוליאנית: True איפה שהם שווים, False איפה שלא
    valid_mask = (stimulus == restimulus)

    # סינון כל המשתנים בעזרת המסיכה (מחיקת זמני תגובה וטעויות)
    emg = emg[valid_mask]
    restimulus = restimulus[valid_mask]
    rep = rep[valid_mask]

    # מעכשיו אנחנו ממשיכים כרגיל, רק עם הנתונים הנקיים
    move = restimulus.astype('int8')

    move_regions = np.where(np.diff(move))[0]

    # Safety check in case of empty regions
    if len(move_regions) == 0:
        return {'emg': emg, 'rep': rep, 'move': move, 'rep_regions': np.array([]), 'nb_capped': 0}

    rep_regions = np.zeros((move_regions.shape[0],), dtype=int)
    nb_reps = int(round(move_regions.shape[0] / 2))
    last_end_idx = int(round(move_regions[0] / 2))
    nb_unique_reps = np.unique(rep).shape[0] - 1
    nb_capped = 0
    cur_rep = 1

    rep = np.zeros([rep.shape[0], ], dtype=np.int8)
    for i in range(nb_reps - 1):
        rep_regions[2 * i] = last_end_idx
        midpoint_idx = int(round((move_regions[2 * (i + 1) - 1] +
                                  move_regions[2 * (i + 1)]) / 2)) + 1

        trailing_rest_samps = midpoint_idx - move_regions[2 * (i + 1) - 1]
        if trailing_rest_samps <= rest_length_cap * fs:
            rep[last_end_idx:midpoint_idx] = cur_rep
            last_end_idx = midpoint_idx
            rep_regions[2 * i + 1] = midpoint_idx - 1
        else:
            rep_end_idx = (move_regions[2 * (i + 1) - 1] +
                           int(round(rest_length_cap * fs)))
            rep[last_end_idx:rep_end_idx] = cur_rep
            last_end_idx = ((move_regions[2 * (i + 1)] -
                             int(round(rest_length_cap * fs))))
            rep_regions[2 * i + 1] = rep_end_idx - 1
            nb_capped += 2

        cur_rep += 1
        if cur_rep > nb_unique_reps:
            cur_rep = 1

    end_idx = int(round((emg.shape[0] + move_regions[-1]) / 2))
    rep[last_end_idx:end_idx] = cur_rep
    rep_regions[-2] = last_end_idx
    rep_regions[-1] = end_idx - 1

    return {'emg': emg,
            'rep': rep,
            'move': move,
            'rep_regions': rep_regions,
            'nb_capped': nb_capped
            }