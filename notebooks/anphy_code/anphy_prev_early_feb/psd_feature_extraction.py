import os
import pandas as pd
import numpy as np
from config.config import DataConfig
from anp_data_loader import ANPDataLoader
from anp_feat_extractor import ANPFeatExtractor

def main():
    # Initialize data loader
    data_loader = ANPDataLoader(DataConfig.BASE_PATH)
    
    # Load details CSV
    details = data_loader.load_details_csv(DataConfig.DETAILS_CSV)
    print("Details CSV head:")
    print(details.head())
    
    # Container for features from all subjects/epochs/electrodes
    results_list = []
    
    # Process all subjects using config
    for subject_id in DataConfig.SUBJECTS:
        paths = DataConfig.get_subject_paths(subject_id)
        print(f"Processing {subject_id} (first 10 minutes using 30-sec epochs)...")
        
        try:
            # Load EDF metadata to get sampling rate and electrode labels
            edf_meta = data_loader.load_edf_metadata(paths['edf'])
            fs = edf_meta["sample_rates"][0]
            labels = edf_meta["labels"]
            
            # Initialize feature extractor for this subject
            feat_extractor = ANPFeatExtractor(fs)
            
            # Process the first 10 minutes: 20 epochs of 30 seconds each
            num_epochs = 20  # 10 minutes = 600 seconds / 30 s per epoch
            for epoch_index in range(num_epochs):
                start_time = epoch_index * 30  # in seconds
                duration = 30  # 30-second epoch
                try:
                    epoch_data, fs_epoch = data_loader.load_edf_epoch(paths['edf'], start_time, duration)
                except Exception as e:
                    print(f"Error loading epoch starting at {start_time}s for {subject_id}: {e}")
                    continue
                
                # Extract features for each electrode (returns shape: [n_channels, 4])
                features = feat_extractor.extract_features(epoch_data)
                
                # Save one row per electrode for this epoch
                for ch_idx, feat in enumerate(features):
                    result = {
                        "Subject": subject_id,
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
                    
        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
            continue
    
    # Create a DataFrame from the results and save to CSV
    results_df = pd.DataFrame(results_list)
    output_file = os.path.join(DataConfig.RESULTS_DIR, "psd_feats_10min.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Saved features for {len(results_df)} electrode epochs to {output_file}")

if __name__ == "__main__":
    main()