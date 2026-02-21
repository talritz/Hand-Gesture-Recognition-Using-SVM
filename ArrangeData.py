import scipy.io
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import feature_extraction


# ---------------------------------------------------------
# פונקציות עזר
# ---------------------------------------------------------

def load_cleaned_ninapro_data(base_path, subject_list, dataset_type):
    """
    טוענת נתונים עבור רשימת נבדקים, מסננת תנועות לא רצויות ושוליים, ומוסיפה תגית סוג הסט.
    """
    all_data_frames = []

    wanted_movements_map = {
        1: [1, 5, 6, 7, 13, 14, 17],  # E1
        2: [31]  # E2
    }

    margin_samples = 1500

    print(f"Loading subjects {subject_list} for {dataset_type} dataset...")

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
                rerepetition = mat['rerepetition'].flatten() if 'rerepetition' in mat else mat['repetition'].flatten()

                # --- שלב הניקוי החכם ---
                is_wanted = np.isin(restimulus, target_moves)
                is_rest = (restimulus == 0)
                is_bad_movement = ~(is_wanted | is_rest)

                expanded_bad_mask = binary_dilation(is_bad_movement, iterations=margin_samples)

                keep_mask = is_wanted | (is_rest & ~expanded_bad_mask)

                # --- יצירת הטבלה רק עם הנתונים הטובים ---
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
        return pd.concat(all_data_frames, ignore_index=True)
    return pd.DataFrame()


# --- הגדרת נתיב דינמית ---
path1 = r'C:\Users\Tal\OneDrive - Afeka College Of Engineering\הקבצים של Nadav Matza - פרויקט גמר\עיבוד אותות אקראיים\data sets'
path2 = r'B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets'
path3 = r'C:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets'

if os.path.exists(path1):
    base_path = path1
elif os.path.exists(path2):
    base_path = path2
elif os.path.exists(path3):
    base_path = path3
else:
    print("Error: Dataset path not found!")
    base_path = None

# הגדרת 3 הקבוצות
train_subjects = [1, 2, 3, 4, 5, 6, 7, 8]
train_subjects = [7]
#val_subjects = [9, 10, 11, 12]
#test_subjects = [13, 14, 15, 16, 17, 18, 19, 20]

# טעינת נתוני האימון הגולמיים
df_train = load_cleaned_ninapro_data(base_path, train_subjects, 'Train')
# df_val = load_cleaned_ninapro_data(base_path, val_subjects, 'Val')
# df_test = load_cleaned_ninapro_data(base_path, test_subjects, 'Test')

# --- תצוגת נתונים גולמיים (דוגמה לנבדק אחד) ---
fs = 2000
index = train_subjects[0] # ניקח את הנבדק הראשון לצורך התצוגה
subject_data = df_train[df_train['Subject'] == index].reset_index(drop=True)
time_axis = subject_data.index / fs

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
fig.suptitle(f'EMG Signal for Subject {index} - Raw Data', fontsize=16)

ax1.plot(time_axis, subject_data['EMG_1'], linewidth=0.5)
ax1.set_title('Channel 1')
ax1.set_ylabel('Amplitude')
ax1.grid(True, linestyle='--', alpha=0.5)

ax2.plot(time_axis, subject_data['Restimulus'], linewidth=1.5, color='orange')
ax2.set_title('Restimulus (Gestures)')
ax2.set_ylabel('Class')
ax2.set_xlabel('Time [sec]')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

subject_to_test = [7]
df_sub7_raw = load_cleaned_ninapro_data(base_path, subject_to_test, 'Test_Features')

# מוודאים שנטענו נתונים לפני שממשיכים
if not df_sub7_raw.empty:
    print(f"Successfully loaded {len(df_sub7_raw)} raw samples for Subject 7.")

    # 2. חילוץ המאפיינים באמצעות הפונקציה הראשית שלנו
    df_sub7_features = feature_extraction.extract_all_features(df_sub7_raw)

    # 3. שמירה לקובץ CSV באותה תיקייה שבה רץ הקוד
    csv_filename = 'subject_7_features.csv'
    df_sub7_features.to_csv(csv_filename, index=False)

    print(f"Features extracted successfully!")
    print(f"Total feature windows (rows): {len(df_sub7_features)}")
    print(f"Total columns: {len(df_sub7_features.columns)}")
    print(f"Saved to: {csv_filename}")
else:
    print("No data loaded. Please check the dataset path.")