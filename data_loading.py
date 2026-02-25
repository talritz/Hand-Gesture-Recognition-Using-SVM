import os
import scipy.io
import pandas as pd
import numpy as np
from scipy.ndimage import binary_dilation

import hyperparameters as hp

def load_cleaned_ninapro_data(base_path, subject_list, dataset_type, margin_samples=None):
    # Use global hyperparameter if no specific value is provided
    if margin_samples is None:
        margin_samples = hp.MARGIN_SAMPLES

    all_data_frames = []
    wanted_movements_map = {
        1: [1, 5, 6, 7, 13, 14, 17],
        2: [31]
    }

    print(f"Loading subjects {subject_list} for '{dataset_type}' dataset (Margin: {margin_samples})...")

    for subject in subject_list:
        for exercise_id, target_moves in wanted_movements_map.items():
            file_name = f'S{subject}_E{exercise_id}_A1.mat'
            file_path = os.path.join(base_path, file_name)

            if not os.path.exists(file_path):
                continue

            try:
                mat = scipy.io.loadmat(file_path)
                emg = mat['emg']
                restimulus = mat['restimulus'].flatten()

                if 'rerepetition' in mat:
                    rerepetition = mat['rerepetition'].flatten()
                else:
                    rerepetition = mat['repetition'].flatten()

                is_wanted = np.isin(restimulus, target_moves)
                is_rest = (restimulus == 0)
                is_bad_movement = ~(is_wanted | is_rest)

                expanded_bad_mask = binary_dilation(is_bad_movement, iterations=margin_samples)
                keep_mask = is_wanted | (is_rest & ~expanded_bad_mask)

                if np.sum(keep_mask) > 0:
                    df_temp = pd.DataFrame(emg[keep_mask], columns=[f'EMG_{i + 1}' for i in range(emg.shape[1])])
                    df_temp['Restimulus'] = restimulus[keep_mask]
                    df_temp['Rerepetition'] = rerepetition[keep_mask]
                    df_temp['Subject'] = subject
                    df_temp['Exercise'] = exercise_id
                    df_temp['dataset_type'] = dataset_type

                    df_temp = df_temp.astype({'Restimulus': 'int8', 'Rerepetition': 'int8',
                                              'Subject': 'int8', 'Exercise': 'int8'})
                    all_data_frames.append(df_temp)

            except Exception as e:
                print(f"Error loading {file_name}: {e}")

    if all_data_frames:
        final_df = pd.concat(all_data_frames, ignore_index=True)
        return final_df

    return pd.DataFrame()