# first 10 mins only
import os
import pandas as pd
from anp_data_loader import ANPDataLoader
from anp_feat_extractor import ANPFeatExtractor

def main():
    # 1. PATH SETTING
    BASE_PATH = "/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/ANPHY-Sleep"
    DETAILS_CSV = os.path.join(BASE_PATH, "Details information for healthy subjects.csv")
    RESULTS_DIR = "/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/results"
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # 2. DATA LOADING
    data_loader = ANPDataLoader(BASE_PATH)
    details = data_loader.load_details_csv(DETAILS_CSV)
    print("Details CSV head:")
    print(details.head())
    
    # Container for features from all subjects/epochs/electrodes.
    results_list = []
    
    # Process all 29 subjects (EPCTL01 to EPCTL29)
    for i in range(1, 30):
        subj = f"EPCTL{str(i).zfill(2)}"
        subject_folder = os.path.join(BASE_PATH, subj)
        edf_file = os.path.join(subject_folder, f"{subj}.edf")
        print(f"Processing {subj} (first 10 minutes using 30-sec epochs)...")
        
        # Load EDF metadata to get sampling rate and electrode labels
        edf_meta = data_loader.load_edf_metadata(edf_file)
        fs = edf_meta["sample_rates"][0]
        labels = edf_meta["labels"]
        
        # Initialize feature extractor for this subject
        feat_extractor = ANPFeatExtractor(fs)
        
        # Process the first 10 minutes: 20 epochs of 30 seconds each.
        num_epochs = 20  # 10 minutes = 600 seconds / 30 s per epoch
        for epoch_index in range(num_epochs):
            start_time = epoch_index * 30  # in seconds
            duration = 30  # 30-second epoch
            try:
                epoch_data, fs_epoch = data_loader.load_edf_epoch(edf_file, start_time, duration)
            except Exception as e:
                print(f"Error loading epoch starting at {start_time}s for {subj}: {e}")
                continue
            
            # Extract features for each electrode (returns shape: [n_channels, 4])
            features = feat_extractor.extract_features(epoch_data)
            
            # Save one row per electrode for this epoch.
            for ch_idx, feat in enumerate(features):
                result = {
                    "Subject": subj,
                    "Epoch": epoch_index,
                    "StartTime": start_time,
                    "Electrode": ch_idx,
                    "Electrode_Label": labels[ch_idx] if ch_idx < len(labels) else f"Ch{ch_idx}",
                    "Delta": feat[0],
                    "Theta": feat[1],
                    "Alpha": feat[2],
                    "Beta": feat[3]
                }
                results_list.append(result)
    
    # Create a DataFrame from the results and save to CSV.
    results_df = pd.DataFrame(results_list)
    output_file = os.path.join(RESULTS_DIR, "pds_feats_10min.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Saved features for {len(results_df)} electrode epochs to {output_file}")

if __name__ == "__main__":
    main()
