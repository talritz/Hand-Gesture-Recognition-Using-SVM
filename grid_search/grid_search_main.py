import os
import time
import datetime
import stage1_data_parameters as stage1
import stage2_model_parameters as stage2

def main():
    global_start_time = time.time()

    print("=" * 60)
    print("      GRID SEARCH")
    print(f"     Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("      Strategy: Train (Subjects 1-8) | Validation (Subjects 9-12)")
    print("=" * 60)

    # --- PHASE 1: Feature Generation ---
    print("\n[STEP 1/2] Launching Feature Generation Stage...")
    try:
        stage1.main()
    except Exception as error:
        print(f"\nCRITICAL ERROR in Stage 1: {error}")
        return

    # --- PHASE 2: Model Training ---
    print("\n[STEP 2/2] Launching Model Training Stage...")
    try:
        stage2.main()
    except Exception as error:
        print(f"\nCRITICAL ERROR in Stage 2: {error}")
        return

    total_duration = str(datetime.timedelta(seconds=int(time.time() - global_start_time)))

    print("\n" + "=" * 60)
    print(f"      EXPERIMENT COMPLETE! Total Duration: {total_duration}")
    print("=" * 60)

if __name__ == "__main__":
    main()