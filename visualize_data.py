import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import hyperparameters as hp
from data_loading import load_cleaned_ninapro_data


def main():
    print("=" * 50)
    print("Starting EMG Data Visualization for Report...")
    print("=" * 50)

    paths_to_check = [
        r'C:\Users\Tal\OneDrive - Afeka College Of Engineering\הקבצים של Nadav Matza - פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'B:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets',
        r'C:\OneDrive - Afeka College Of Engineering\פרויקט גמר\עיבוד אותות אקראיים\data sets'
    ]

    base_path = next((p for p in paths_to_check if os.path.exists(p)), None)
    if not base_path:
        print("Error: Dataset path not found!")
        return

    # אנחנו טוענים את נבדק 1, אבל שים לב - עם margin_samples=0 !
    # המטרה היא לראות את האות הגולמי הטהור לפני שחתכנו אותו, כדי להבין למה אנחנו חותכים.
    print("Loading Subject 1 raw data (without margins) to visualize transients...")
    df = load_cleaned_ninapro_data(base_path, [1], dataset_name='Visualization', margin_samples=0)

    # ניצור עמודת בלוקים כדי למצוא תנועה רציפה
    df['block_id'] = (df['Restimulus'] != df['Restimulus'].shift()).cumsum()

    # נמצא את הבלוק הראשון שבו יש תנועה (לא מנוחה)
    move_blocks = df[df['Restimulus'] != 0]['block_id'].unique()
    target_block = move_blocks[0]  # ניקח את התנועה הראשונה הממשית
    target_class = df[df['block_id'] == target_block]['Restimulus'].iloc[0]

    block_data = df[df['block_id'] == target_block]
    true_start = block_data.index[0]
    true_end = block_data.index[-1]

    # ניקח חלון זמן שכולל 1.5 שניות לפני התנועה (מנוחה) ו-1.5 שניות אחרי התנועה
    padding = int(1.5 * 2000)  # 2000Hz = 1.5 seconds
    plot_start = true_start - padding
    plot_end = true_end + padding

    plot_df = df.iloc[plot_start:plot_end].copy()

    # ציר זמן בשניות (חלקי 2000 Hz)
    time_axis = np.arange(len(plot_df)) / 2000.0

    # זמני ההתחלה והסיום של התנועה מתוך ציר הזמן המקומי שלנו
    move_start_time = padding / 2000.0
    move_end_time = (padding + (true_end - true_start)) / 2000.0

    # הגדרת המרווחים שאנחנו זורקים לפח (Margin) - 1500 דגימות = 0.75 שניות
    margin_sec = hp.MARGIN_SAMPLES / 2000.0

    # --- תחילת שרטוט הגרף לדוח ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # 1. פלוט של אותות ה-EMG (ניקח 3 ערוצים מתוך ה-10 כדי לא לייצר עומס ויזואלי)
    ax1.plot(time_axis, plot_df['EMG_1'], label='Channel 1', color='royalblue', linewidth=1, alpha=0.8)
    ax1.plot(time_axis, plot_df['EMG_2'], label='Channel 2', color='darkorange', linewidth=1, alpha=0.8)
    ax1.plot(time_axis, plot_df['EMG_3'], label='Channel 3', color='forestgreen', linewidth=1, alpha=0.8)

    # הדגשת האזורים שאנחנו זורקים (Transients)
    ax1.axvspan(move_start_time, move_start_time + margin_sec, color='red', alpha=0.2,
                label='Transient (Discarded Margin)')
    ax1.axvspan(move_end_time - margin_sec, move_end_time, color='red', alpha=0.2)

    # הדגשת האזור היציב שאנחנו באמת מאמנים עליו (Steady-State)
    ax1.axvspan(move_start_time + margin_sec, move_end_time - margin_sec, color='limegreen', alpha=0.2,
                label='Steady-State (Training Region)')

    ax1.set_title(f"EMG Raw Signals & Segmentation Overlay (Subject 1, Gesture Class: {target_class})", fontsize=14,
                  fontweight='bold')
    ax1.set_ylabel("Amplitude (mV)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')

    # 2. פלוט של התווית (Label)
    ax2.plot(time_axis, plot_df['Restimulus'], color='black', linewidth=2, label='Gesture Command (Restimulus)')
    ax2.fill_between(time_axis, plot_df['Restimulus'], color='gray', alpha=0.3)
    ax2.set_ylabel("Class Label", fontsize=12)
    ax2.set_xlabel("Time (Seconds)", fontsize=12)
    ax2.set_yticks([0, target_class])
    ax2.set_yticklabels(['Rest (0)', f'Gesture ({target_class})'])
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    # שמירת הגרף כתמונה
    filename = "EMG_Segmentation_Visualization.png"
    plt.savefig(filename, dpi=300)
    print(f"\nVisualization saved successfully as: {filename}")
    plt.show()


if __name__ == "__main__":
    main()