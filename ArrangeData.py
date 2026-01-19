import scipy.io
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def load_filtered_ninapro_data(base_path, num_subjects=1):
    """
    טוען נתונים מ-Ninapro DB2 עם המיפוי המתוקן:
    Exercise B -> קובץ E1
    Exercise C -> קובץ E2
    """
    all_data_frames = []

    # --- המיפוי המעודכן ---
    # המפתח (Key) הוא מספר הקובץ ב-Ninapro (ה-E בפועל)
    # הערך (Value) הוא רשימת ה-Restimulus IDs שביקשת
    wanted_movements = {
        1: [0, 1, 5, 6, 7, 13, 14, 17],  # היה ExB, כעת מוגדר כ-E1
        2: [ 0, 31]  # היה ExC, כעת מוגדר כ-E2
    }

    print(f"מתחיל בטעינת נתונים עבור {num_subjects} נבדקים...")
    print(f"מחוות נבחרות: {wanted_movements}")

    for subject in range(1, num_subjects + 1):
        # רצים על התרגילים שהגדרנו (1 ו-2)
        for exercise_file_id, movements in wanted_movements.items():

            # בניית שם הקובץ עם ה-ID הנכון
            # לדוגמה: S1_E1_A1.mat או S1_E2_A1.mat
            file_name = f'S{subject}_E{exercise_file_id}_A1.mat'
            file_path = os.path.join(base_path, file_name)

            if not os.path.exists(file_path):
                print(f"קובץ לא נמצא: {file_name} - מדלג.")
                continue

            try:
                mat_data = scipy.io.loadmat(file_path)

                # בדיקת Rerepetition
                if 'rerepetition' in mat_data:
                    rerepetition = mat_data['rerepetition']
                else:
                    # גיבוי למקרה שאין (קורה לפעמים בגרסאות ישנות)
                    rerepetition = mat_data['repetition']
                    print('rerepetition col doesnt exist')

                emg_data = mat_data['emg']
                restimulus = mat_data['restimulus']
                stimulus=mat_data['stimulus']

                # יצירת DataFrame
                # אנו יוצרים עמודות EMG_1 עד EMG_12
                df_temp = pd.DataFrame(emg_data, columns=[f'EMG_{i + 1}' for i in range(emg_data.shape[1])])

                df_temp['Restimulus'] = restimulus
                df_temp['Stimulus'] = stimulus
                df_temp['Rerepetition'] = rerepetition
                df_temp['Subject'] = subject
                df_temp['Exercise'] = exercise_file_id  # שומרים את ה-E האמיתי (1 או 2)


                # --- סינון לפי המחוות שביקשת ---
                df_filtered = df_temp[df_temp['Stimulus'].isin(movements)].copy()

                if not df_filtered.empty:
                    # המרה ל-int8 לחיסכון בזיכרון
                    cols_to_int = ['Restimulus','Stimulus', 'Rerepetition', 'Subject', 'Exercise']
                    df_filtered[cols_to_int] = df_filtered[cols_to_int].astype('int8')

                    all_data_frames.append(df_filtered)

            except Exception as e:
                print(f"שגיאה בטעינת {file_name}: {e}")

    # איחוד סופי
    if all_data_frames:
        full_df = pd.concat(all_data_frames, ignore_index=True)

        # סידור עמודות סופי
        cols_order = ['Subject', 'Exercise','Stimulus', 'Restimulus', 'Rerepetition'] + [col for col in full_df.columns if
                                                                              'EMG' in col]
        full_df = full_df[cols_order]

        print(f"\nהתהליך הסתיים. סך הכל דגימות: {len(full_df)}")
        return full_df
    else:
        print("לא נטענו נתונים.")
        return pd.DataFrame()



#def load_cleaned_ninapro_data(base_path, num_subjects=1):
    all_data_frames = []

    # הגדרת התנועות הרצויות (ללא 0 בשלב זה, נוסיף אותו בלוגיקה)
    wanted_movements_map = {
        1: [1, 5, 6, 7, 13, 14, 17],  # E1
        2: [31]  # E2
    }

    # כמה דגימות למחוק מסביב לכל תנועה לא רצויה?
    # בתדר 2000Hz, שניה אחת = 2000 דגימות. נלך על טווח ביטחון נדיב.
    margin_samples = 1500

    print(f"Loading {num_subjects} subjects with safety margins...")

    for subject in range(1, num_subjects + 1):
        for exercise_id, target_moves in wanted_movements_map.items():
            file_name = f'S{subject}_E{exercise_id}_A1.mat'
            file_path = os.path.join(base_path, file_name)

            if not os.path.exists(file_path):
                continue

            try:
                mat = scipy.io.loadmat(file_path)

                # חילוץ נתונים
                emg = mat['emg']
                stimulus = mat['stimulus'].flatten()
                restimulus = mat['restimulus'].flatten()  # משטיחים למערך חד מימדי
                rerepetition = mat['rerepetition'].flatten() if 'rerepetition' in mat else mat['repetition'].flatten()

                # --- שלב הניקוי החכם ---

                # 1. זיהוי איזה דגימות הן "תנועה רצויה"
                # (משתמשים ב-NumPy כי זה מהיר יותר מ-Pandas בשלב הזה)
                is_wanted = np.isin(restimulus, target_moves)

                # 2. זיהוי איזה דגימות הן "מנוחה" (Rest)
                is_rest = (restimulus == 0)

                # 3. כל השאר = תנועות לא רצויות (Bad Movements)
                # אלו התנועות שאנחנו רוצים למחוק + השוליים שלהן
                is_bad_movement = ~(is_wanted | is_rest)

                # 4. הרחבת ה"אזור הרע" (Dilation)
                # אנו יוצרים מסכה שמרחיבה את ה-True של התנועות הרעות ימינה ושמאלה
                # משתמשים ב-convolve כדי "למרוח" את האזור האסור
                window_size = margin_samples * 2 + 1
                kernel = np.ones(window_size, dtype=bool)
                # המרה ל-int לצורך הקונבולוציה
                bad_mask_int = is_bad_movement.astype(int)
                # קונבולוציה: כל מקום שהיה בו 1, ימרח לרוחב החלון
                expanded_bad_mask = np.convolve(bad_mask_int, kernel, mode='same') > 0

                # 5. המסכה הסופית לשמירה:
                # נשמור אם זה: (תנועה רצויה) או (מנוחה שלא קרובה לתנועה רעה)
                keep_mask = is_wanted | (is_rest & ~expanded_bad_mask)

                # --- יצירת הטבלה רק עם הנתונים הטובים ---
                # ניקח רק את האינדקסים ששרדו את הסינון
                if np.sum(keep_mask) > 0:
                    df_temp = pd.DataFrame(emg[keep_mask], columns=[f'EMG_{i + 1}' for i in range(emg.shape[1])])
                    df_temp['Stimulus'] = stimulus
                    df_temp['Restimulus'] = restimulus[keep_mask]
                    df_temp['Rerepetition'] = rerepetition[keep_mask]
                    df_temp['Subject'] = subject
                    df_temp['Exercise'] = exercise_id

                    # אופטימיזציה
                    df_temp = df_temp.astype({'Stimulus': 'int8','Restimulus': 'int8', 'Rerepetition': 'int8',
                                              'Subject': 'int8', 'Exercise': 'int8'})

                    all_data_frames.append(df_temp)

            except Exception as e:
                print(f"Error loading {file_name}: {e}")

    if all_data_frames:
        return pd.concat(all_data_frames, ignore_index=True)
    return pd.DataFrame()


def load_cleaned_ninapro_data(base_path, num_subjects=1):
    all_data_frames = []

    # הגדרת התנועות הרצויות (ללא 0 בשלב זה, נוסיף אותו בלוגיקה)
    wanted_movements_map = {
        1: [1, 5, 6, 7, 13, 14, 17],  # E1
        2: [31]  # E2
    }

    # כמה דגימות למחוק מסביב לכל תנועה לא רצויה?
    # בתדר 2000Hz, שניה אחת = 2000 דגימות. נלך על טווח ביטחון נדיב.
    margin_samples = 1500

    print(f"Loading {num_subjects} subjects with safety margins...")

    for subject in range(1, num_subjects + 1):
        for exercise_id, target_moves in wanted_movements_map.items():
            file_name = f'S{subject}_E{exercise_id}_A1.mat'
            file_path = os.path.join(base_path, file_name)

            if not os.path.exists(file_path):
                continue

            try:
                mat = scipy.io.loadmat(file_path)

                # חילוץ נתונים
                emg = mat['emg']
                restimulus = mat['restimulus'].flatten()  # משטיחים למערך חד מימדי
                rerepetition = mat['rerepetition'].flatten() if 'rerepetition' in mat else mat['repetition'].flatten()

                # --- שלב הניקוי החכם ---

                # 1. זיהוי איזה דגימות הן "תנועה רצויה"
                # (משתמשים ב-NumPy כי זה מהיר יותר מ-Pandas בשלב הזה)
                is_wanted = np.isin(restimulus, target_moves)

                # 2. זיהוי איזה דגימות הן "מנוחה" (Rest)
                is_rest = (restimulus == 0)

                # 3. כל השאר = תנועות לא רצויות (Bad Movements)
                # אלו התנועות שאנחנו רוצים למחוק + השוליים שלהן
                is_bad_movement = ~(is_wanted | is_rest)

                # 4. הרחבת ה"אזור הרע" (Dilation)
                # אנו יוצרים מסכה שמרחיבה את ה-True של התנועות הרעות ימינה ושמאלה
                # משתמשים ב-convolve כדי "למרוח" את האזור האסור
                window_size = margin_samples * 2 + 1
                kernel = np.ones(window_size, dtype=bool)
                # המרה ל-int לצורך הקונבולוציה
                bad_mask_int = is_bad_movement.astype(int)
                # קונבולוציה: כל מקום שהיה בו 1, ימרח לרוחב החלון
                expanded_bad_mask = np.convolve(bad_mask_int, kernel, mode='same') > 0

                # 5. המסכה הסופית לשמירה:
                # נשמור אם זה: (תנועה רצויה) או (מנוחה שלא קרובה לתנועה רעה)
                keep_mask = is_wanted | (is_rest & ~expanded_bad_mask)

                # --- יצירת הטבלה רק עם הנתונים הטובים ---
                # ניקח רק את האינדקסים ששרדו את הסינון
                if np.sum(keep_mask) > 0:
                    df_temp = pd.DataFrame(emg[keep_mask], columns=[f'EMG_{i + 1}' for i in range(emg.shape[1])])
                    df_temp['Restimulus'] = restimulus[keep_mask]
                    df_temp['Rerepetition'] = rerepetition[keep_mask]
                    df_temp['Subject'] = subject
                    df_temp['Exercise'] = exercise_id

                    # אופטימיזציה
                    df_temp = df_temp.astype({'Restimulus': 'int8', 'Rerepetition': 'int8',
                                              'Subject': 'int8', 'Exercise': 'int8'})

                    all_data_frames.append(df_temp)

            except Exception as e:
                print(f"Error loading {file_name}: {e}")

    if all_data_frames:
        return pd.concat(all_data_frames, ignore_index=True)
    return pd.DataFrame()

base_path = r'C:\Users\Tal\OneDrive - Afeka College Of Engineering\הקבצים של Nadav Matza - פרויקט גמר\עיבוד אותות אקראיים\data sets'

df=load_cleaned_ninapro_data(base_path)

fs = 2000  # תדר דגימה (Hz)
subject_id = 1  # הנבדק שרוצים להציג

# 1. סינון הנתונים עבור הנבדק הספציפי
# (אנו עושים reset_index כדי שהאינדקס יתחיל מ-0 לצורך חישוב הזמן)
subject_data = df[df['Subject'] == subject_id].reset_index(drop=True)

# 2. יצירת ציר זמן (בשניות)
# הנוסחה: אינדקס הדגימה / תדר הדגימה
time_axis = subject_data.index / fs

# 3. הגדרת הגרף (Figure)
# ניצור מערך של 4x3 גרפים (סה"כ 12)
fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
fig.suptitle(f'EMG Signals for Subject {subject_id} - Raw Data', fontsize=16)

# שיטוח מערך הצירים לרשימה אחת (כדי שיהיה נוח לרוץ בלולאה)
axes_flat = axes.flatten()

# 4. לולאה לציור כל ערוץ
for i in range(1):
    col_name = f'EMG_{i + 1}'
    col_name2 =f'Restimulus'
    #col_name3 = f'Stimulus'
    ax = axes_flat[i]
    ax2=axes_flat[i+1]
   # ax3 = axes_flat[i + 2]

    # ציור האות
    # אנו לוקחים חתיכה קטנה (למשל ה-2000 דגימות הראשונות) כדי שהגרף יהיה ברור,
    # או את כל הדאטה אם רוצים (פשוט מחק את .iloc[:2000])
    ax.plot(time_axis, subject_data[col_name], linewidth=0.5)
    ax2.plot(time_axis,subject_data[col_name2],linewidth=0.5)
   # ax3.plot(time_axis,subject_data[col_name3],linewidth=0.5)

    # עיצוב
    ax.set_title(f'Channel {i + 1}')
    ax.grid(True, linestyle='--', alpha=0.5)

    ax2.set_title(f'Restimulus')
    ax2.grid(True, linestyle='--', alpha=0.5)

   # ax3.set_title(f'Stimulus')
  # ax3.grid(True, linestyle='--', alpha=0.5)

    # תוויות צירים (רק בגרפים החיצוניים למניעת עומס)
    if i >= 9:  # שורה תחתונה
        ax.set_xlabel('Time [sec]')
    if i % 3 == 0:  # עמודה שמאלית
        ax.set_ylabel('Amplitude')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # סידור המרווחים
plt.show()

