"""
===============================================================================
Data Loading and Signal Processing Module
===============================================================================
Handles the extraction of raw Ninapro DB2 .mat files, applies bandpass
filtering, implements IEEE outlier removal, and structures the data.
"""
import os
import pandas as pd
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
import hyper_parameters as hp


def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Computes the numerator (b) and denominator (a) polynomials of the IIR filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(data, lowcut=5.0, highcut=650.0, fs=2000.0, order=4):
    """
    Applies a zero-phase digital filter (filtfilt) to prevent phase distortion.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y


def _import_db2(folder_path, subject, rest_length_cap=999):
    """
    Internal helper function to extract and preprocess Ninapro DB2 data.
    """
    fs = 2000

    # Load data for Exercise 1.
    cur_path = os.path.normpath(os.path.join(folder_path, f'S{subject}_E1_A1.mat'))
    data = sio.loadmat(cur_path)
    emg = np.squeeze(np.array(data['emg']))
    rep = np.squeeze(np.array(data['rerepetition']))
    restimulus = np.squeeze(np.array(data['restimulus']))
    stimulus = np.squeeze(np.array(data['stimulus']))

    # Load data for Exercise 2.
    cur_path = os.path.normpath(os.path.join(folder_path, f'S{subject}_E2_A1.mat'))
    data = sio.loadmat(cur_path)

    # Vertically stack the EMG signals and append the label arrays.
    emg = np.vstack((emg, np.array(data['emg'])))
    rep = np.append(rep, np.squeeze(np.array(data['rerepetition'])))
    restimulus = np.append(restimulus, np.squeeze(np.array(data['restimulus'])))
    stimulus = np.append(stimulus, np.squeeze(np.array(data['stimulus'])))

    # Apply bandpass filtering prior to outlier removal to maintain signal continuity.
    emg = apply_bandpass_filter(emg, lowcut=5.0, highcut=650.0, fs=fs, order=4)

    # Implement IEEE outlier removal standard: retain only segments where the
    # intended stimulus matches the executed restimulus.
    valid_mask = (stimulus == restimulus)
    emg = emg[valid_mask]
    restimulus = restimulus[valid_mask]
    rep = rep[valid_mask]

    move = restimulus.astype('int8')

    # Identify indices where the movement class changes.
    move_regions = np.where(np.diff(move))[0]

    if len(move_regions) == 0:
        return {'emg': emg, 'rep': rep, 'move': move}

    # Perform internal repetition alignment based on the Ninapro protocol.
    rep_regions = np.zeros((move_regions.shape[0],), dtype=int)
    nb_reps = int(round(move_regions.shape[0] / 2))
    last_end_idx = int(round(move_regions[0] / 2))
    nb_unique_reps = np.unique(rep).shape[0] - 1
    cur_rep = 1

    rep = np.zeros([rep.shape[0], ], dtype=np.int8)
    for i in range(nb_reps - 1):
        rep_regions[2 * i] = last_end_idx
        midpoint_idx = int(round((move_regions[2 * (i + 1) - 1] + move_regions[2 * (i + 1)]) / 2)) + 1
        trailing_rest_samps = midpoint_idx - move_regions[2 * (i + 1) - 1]

        if trailing_rest_samps <= rest_length_cap * fs:
            rep[last_end_idx:midpoint_idx] = cur_rep
            last_end_idx = midpoint_idx
        else:
            rep_end_idx = (move_regions[2 * (i + 1) - 1] + int(round(rest_length_cap * fs)))
            rep[last_end_idx:rep_end_idx] = cur_rep
            last_end_idx = ((move_regions[2 * (i + 1)] - int(round(rest_length_cap * fs))))

        cur_rep += 1
        if cur_rep > nb_unique_reps:
            cur_rep = 1

    end_idx = int(round((emg.shape[0] + move_regions[-1]) / 2))
    rep[last_end_idx:end_idx] = cur_rep

    return {'emg': emg, 'rep': rep, 'move': move}


def load_cleaned_ninapro_data(base_path, subject_list, dataset_name='Dataset', margin_samples=None):
    """
    Iterates over the requested subjects, extracts raw data, isolates target classes,
    and trims transient states from the beginning and end of each movement block.
    """
    if margin_samples is None:
        margin_samples = hp.MARGIN_SAMPLES

    print(f"Loading {dataset_name} (Subjects: {subject_list})...")
    all_subject_data = []

    # Define the specific gesture classes evaluated in this study.
    TARGET_CLASSES = [0, 1, 5, 6, 7, 13, 14, 17, 31]

    for subj in subject_list:
        print(f"  -> Extracting Subject {subj}...")
        data_dict = _import_db2(base_path, subj)

        emg_data = data_dict['emg']
        move_data = data_dict['move'].flatten()
        rep_data = data_dict['rep'].flatten()

        # Construct a DataFrame for the current subject's channels.
        df = pd.DataFrame(emg_data, columns=[f'EMG_{i + 1}' for i in range(emg_data.shape[1])])
        df['Restimulus'] = move_data
        df['Rerepetition'] = rep_data
        df['Subject'] = subj

        # Filter out classes not included in the target set.
        df = df[df['Restimulus'].isin(TARGET_CLASSES)].copy()

        # Trim transient samples from the start and end of active movement blocks.
        if margin_samples > 0:
            # Assign a unique ID to each continuous block of the same movement.
            df['block_id'] = (df['Restimulus'] != df['Restimulus'].shift()).cumsum()

            def trim_transients(group):
                # Ensure the block is large enough to be trimmed without yielding an empty set.
                if len(group) > 2 * margin_samples:
                    return group.iloc[margin_samples:-margin_samples]
                return pd.DataFrame(columns=group.columns)

            # Apply the trimming function per block.
            df = df.groupby('block_id', group_keys=False).apply(trim_transients, include_groups=False)

            # Remove the temporary block identifier column.
            if 'block_id' in df.columns:
                df = df.drop('block_id', axis=1)

        all_subject_data.append(df)

    # Concatenate all subjects into a unified dataset.
    final_df = pd.concat(all_subject_data, ignore_index=True)
    final_df['dataset_type'] = dataset_name

    print(f"Finished loading {dataset_name}. Total valid samples extracted: {len(final_df)}")
    return final_df