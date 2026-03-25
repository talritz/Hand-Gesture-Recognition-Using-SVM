import pandas as pd
import numpy as np
import hyper_parameters as hp
import nina_helper


def load_cleaned_ninapro_data(base_path, subject_list, dataset_name='Dataset', margin_samples=None):
    if margin_samples is None:
        margin_samples = hp.MARGIN_SAMPLES

    print(f"Loading {dataset_name} using nina_helper (Subjects: {subject_list})...")

    all_subject_data = []
    TARGET_CLASSES = [0, 1, 5, 6, 7, 13, 14, 17, 31]

    for subj in subject_list:
        print(f"  -> Extracting Subject {subj} via nina_helper...")

        data_dict = nina_helper.import_db2(base_path, subj)

        emg_data = data_dict['emg']
        move_data = data_dict['move'].flatten()
        rep_data = data_dict['rep'].flatten()

        df = pd.DataFrame(emg_data, columns=[f'EMG_{i + 1}' for i in range(emg_data.shape[1])])
        df['Restimulus'] = move_data
        df['Rerepetition'] = rep_data
        df['Subject'] = subj

        df = df[df['Restimulus'].isin(TARGET_CLASSES)].copy()

        if margin_samples > 0:
            df['block_id'] = (df['Restimulus'] != df['Restimulus'].shift()).cumsum()

            def trim_transients(group):
                if len(group) > 2 * margin_samples:
                    return group.iloc[margin_samples:-margin_samples]
                return pd.DataFrame(columns=group.columns)

            # include_groups=False automatically drops the 'block_id' column, preventing the KeyError
            df = df.groupby('block_id', group_keys=False).apply(trim_transients, include_groups=False)

        all_subject_data.append(df)

    final_df = pd.concat(all_subject_data, ignore_index=True)
    final_df['dataset_type'] = dataset_name

    print(f"Finished loading {dataset_name}. Total valid samples: {len(final_df)}")
    return final_df