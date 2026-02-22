import os
import scipy.io
import pandas as pd
import numpy as np
from scipy.ndimage import binary_dilation


def load_cleaned_ninapro_data(base_path, subject_list, dataset_type):
    """
    Loads Ninapro DB2 data for given subjects, filters out unwanted movements,
    applies safety margins using binary dilation, and returns a clean DataFrame.
    """
    all_data_frames = []

    # Map of wanted movements based on exercise ID
    wanted_movements_map = {
        1: [1, 5, 6, 7, 13, 14, 17],  # Exercise 1
        2: [31]  # Exercise 2
    }

    # Safety margin around unwanted movements (in samples)
    margin_samples = 1500

    print(f"Loading subjects {subject_list} for '{dataset_type}' dataset...")

    for subject in subject_list:
        for exercise_id, target_moves in wanted_movements_map.items():
            file_name = f'S{subject}_E{exercise_id}_A1.mat'
            file_path = os.path.join(base_path, file_name)

            if not os.path.exists(file_path):
                print(f"Warning: File {file_name} not found. Skipping.")
                continue

            try:
                mat = scipy.io.loadmat(file_path)

                # Extract signals and labels
                emg = mat['emg']
                restimulus = mat['restimulus'].flatten()

                # Handle different field names in older/newer mat versions
                if 'rerepetition' in mat:
                    rerepetition = mat['rerepetition'].flatten()
                else:
                    rerepetition = mat['repetition'].flatten()

                # --- Smart Cleaning Step ---
                is_wanted = np.isin(restimulus, target_moves)
                is_rest = (restimulus == 0)
                is_bad_movement = ~(is_wanted | is_rest)

                # Fast binary dilation to expand the "bad" areas
                expanded_bad_mask = binary_dilation(is_bad_movement, iterations=margin_samples)

                # Keep only wanted movements or clean rest periods
                keep_mask = is_wanted | (is_rest & ~expanded_bad_mask)

                # --- Create DataFrame for valid data ---
                if np.sum(keep_mask) > 0:
                    df_temp = pd.DataFrame(emg[keep_mask], columns=[f'EMG_{i + 1}' for i in range(emg.shape[1])])
                    df_temp['Restimulus'] = restimulus[keep_mask]
                    df_temp['Rerepetition'] = rerepetition[keep_mask]
                    df_temp['Subject'] = subject
                    df_temp['Exercise'] = exercise_id
                    df_temp['dataset_type'] = dataset_type

                    # Optimize memory usage
                    df_temp = df_temp.astype({'Restimulus': 'int8', 'Rerepetition': 'int8',
                                              'Subject': 'int8', 'Exercise': 'int8'})

                    all_data_frames.append(df_temp)

            except Exception as e:
                print(f"Error loading {file_name}: {e}")

    # Concatenate all subjects into a single DataFrame
    if all_data_frames:
        final_df = pd.concat(all_data_frames, ignore_index=True)
        print(f"Finished loading '{dataset_type}'. Total samples: {len(final_df)}")
        return final_df

    print(f"No data loaded for '{dataset_type}'.")
    return pd.DataFrame()