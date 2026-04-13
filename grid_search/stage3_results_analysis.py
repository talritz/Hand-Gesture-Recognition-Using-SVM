import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def extract_parameters_from_filename(filename):
    """Helper function to cleanly extract ZC and SSC from a filename using Regex."""
    if not filename or filename == "Unknown":
        return "Unknown", "Unknown"

    zc_match = re.search(r'ZC([0-9eE\.\-]+)', filename)
    ssc_match = re.search(r'SSC([0-9eE\.\-]+)', filename)

    zc_val = zc_match.group(1) if zc_match else "Unknown"
    ssc_val = ssc_match.group(1) if ssc_match else "Unknown"

    return zc_val, ssc_val


def analyze_and_export(results_csv="GridSearch_Results.csv", features_dir="Generated_Features_NPZ"):
    """
    Consolidated analysis pipeline:
    1. Reconstructs the full dataset with ZC & SSC parameters.
    2. Generates performance visualizations (Boxplots, Heatmaps).
    3. Exports the full enriched dataset to the analysis folder.
    4. Extracts and exports the Top 5 models per kernel to the analysis folder.
    """
    print("=" * 80)
    print(" STAGE 3: NINAPRO DB2 - FINAL ANALYSIS & DATA EXPORT")
    print("=" * 80)

    if not os.path.exists(results_csv):
        print(f"[!] Error: Could not find {results_csv}. Please run Stage 2 first.")
        return

    # Load the raw results from Stage 2
    raw_df = pd.read_csv(results_csv)

    # --- ALL FINAL OUTPUTS WILL GO HERE ---
    output_dir = "grid_search_analysis"
    os.makedirs(output_dir, exist_ok=True)

    try:
        all_train_files = [f for f in os.listdir(features_dir) if f.endswith('_TRAIN.npz')]
    except FileNotFoundError:
        print(f"[!] Error: Could not find directory '{features_dir}' to recover ZC/SSC parameters.")
        return

    print(f"[*] Total Models Evaluated: {len(raw_df)}")
    print("[*] Reconstructing dataset with ZC and SSC parameters... This might take a few seconds.")

    # ---------------------------------------------------------
    # PART 1: RECONSTRUCT THE FULL DATASET (Add ZC, SSC, and Filename)
    # ---------------------------------------------------------
    enriched_rows = []

    grouped_by_file = raw_df.groupby(['Margin', 'Window'])

    for (margin, window), group in grouped_by_file:
        target_prefix = f"M{margin}_W{window}_"
        matching_files = [f for f in all_train_files if f.startswith(target_prefix)]

        num_combos_per_file = 370

        for idx_in_group, (original_index, row) in enumerate(group.iterrows()):
            file_index = idx_in_group // num_combos_per_file

            winning_file = matching_files[file_index] if file_index < len(matching_files) else "Unknown"
            zc_val, ssc_val = extract_parameters_from_filename(winning_file)

            row_dict = row.to_dict()
            row_dict['Zero_Crossing_Thresh'] = zc_val
            row_dict['Slope_Sign_Change_Delta'] = ssc_val
            row_dict['Source_File'] = winning_file

            enriched_rows.append(row_dict)

    enriched_df = pd.DataFrame(enriched_rows)

    cols = ['Kernel', 'Macro_F1', 'Balanced_Acc', 'Class_0_Rec', 'Margin', 'Window',
            'Zero_Crossing_Thresh', 'Slope_Sign_Change_Delta', 'C', 'Gamma', 'Degree',
            'Class_Weight', 'SV_Count', 'Source_File', 'Elapsed']

    final_cols = [c for c in cols if c in enriched_df.columns]
    enriched_df = enriched_df[final_cols]

    # --- EXPORT 1: THE FULL ENRICHED DATASET ---
    full_export_path = os.path.join(output_dir, "GridSearch_Full_Results.csv")
    enriched_df.to_csv(full_export_path, index=False)
    print(f"[+] Full Enriched Dataset saved to: {full_export_path}")

    # ---------------------------------------------------------
    # PART 2: TOP 5 MODELS PER KERNEL (PRINT & EXPORT)
    # ---------------------------------------------------------
    print("\n" + "-" * 80)
    print(" TOP 5 COMBINATIONS PER SVM KERNEL")
    print("-" * 80)

    top_per_kernel_list = []
    display_cols = ['Margin', 'Window', 'Zero_Crossing_Thresh', 'Slope_Sign_Change_Delta',
                    'C', 'Gamma', 'Degree', 'Class_Weight', 'Macro_F1', 'Balanced_Acc']

    for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
        top_5 = enriched_df[enriched_df['Kernel'] == kernel].sort_values(by='Balanced_Acc', ascending=False).head(5)

        if not top_5.empty:
            print(f"\n>>> KERNEL: {kernel.upper()}")
            print(top_5[display_cols].to_string(index=False))
            top_per_kernel_list.append(top_5)

    if top_per_kernel_list:
        combined_top = pd.concat(top_per_kernel_list)

        # --- EXPORT 2: THE TOP 5 DATASET ---
        top_5_csv_path = os.path.join(output_dir, "Top5_Per_Kernel.csv")
        combined_top.to_csv(top_5_csv_path, index=False)
        print(f"\n[+] Detailed Top 5 report saved to: {top_5_csv_path}")

    # ---------------------------------------------------------
    # PART 3: DATA VISUALIZATIONS
    # ---------------------------------------------------------
    print("\n[*] Generating Visualizations...")
    sns.set_theme(style="whitegrid")
    kernel_palette = {"linear": "#2ecc71", "rbf": "#3498db", "poly": "#9b59b6", "sigmoid": "#e74c3c"}

    # VIZ 1: Macro F1 Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Kernel', y='Macro_F1', hue='Kernel', data=enriched_df, palette=kernel_palette,
                legend=False, showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "8"})
    plt.title('Macro F1 Performance by SVM Kernel', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_dir, '1_Macro_F1_Comparison.png'), dpi=300)
    plt.close()

    # VIZ 2: Balanced Accuracy Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Kernel', y='Balanced_Acc', hue='Kernel', data=enriched_df, palette=kernel_palette,
                legend=False, showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "8"})
    plt.title('Balanced Accuracy by SVM Kernel', fontsize=14, fontweight='bold')
    plt.ylabel('Balanced Accuracy', fontsize=12)
    plt.xlabel('Kernel Type', fontsize=12)
    plt.savefig(os.path.join(output_dir, '2_Balanced_Acc_Comparison.png'), dpi=300)
    plt.close()

    # VIZ 3: Window Size vs Margin Heatmap (Based on Max F1)
    heatmap_data = enriched_df.groupby(['Window', 'Margin'])['Macro_F1'].max().unstack()
    plt.figure(figsize=(9, 7))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Max Macro F1'})
    plt.title('Performance Heatmap: Finding the Sweet Spot (Max F1)', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_dir, '3_Sweet_Spot_Heatmap.png'), dpi=300)
    plt.close()

    print(f"[+] Visualizations and CSV reports saved to the '{output_dir}' directory.")
    print("\n" + "=" * 80)
    print(" ALL OPERATIONS COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    analyze_and_export()