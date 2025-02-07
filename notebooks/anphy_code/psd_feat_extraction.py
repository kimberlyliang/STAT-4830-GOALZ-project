# first 10 minutes for all
import os
import pandas as pd
from anp_data_loader import ANPDataLoader
from anp_feat_extractor import ANPFeatExtractor
from config.config import DataConfig

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
        print(f"Processing {subject_id} (first 10 minutes)...")
        
        try:
            # Load EDF metadata to get sampling rate and electrode labels
            edf_meta = data_loader.load_edf_metadata(paths['edf'])
            fs = edf_meta["sample_rates"][0]
            labels = edf_meta["labels"]
            
            # Initialize feature extractor for this subject
            feat_extractor = ANPFeatExtractor(fs)
            
            # Process the first 10 minutes: 10 epochs of 60 seconds each
            for epoch_index in range(10):
                start_time = epoch_index * 60  # in seconds
                duration = 60  # 60-second epoch
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
    output_file = os.path.join(DataConfig.RESULTS_DIR, "anphy_psd_features.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Saved features for {len(results_df)} electrode epochs to {output_file}")

if __name__ == "__main__":
    main()


# first 5 subjects only
# import os
# import pandas as pd
# from anp_data_loader import ANPDataLoader
# from anp_feat_extractor import ANPFeatExtractor

# def main():
#     # 1. PATH SETTING
#     BASE_PATH = "/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/ANPHY-Sleep"
#     ARTIFACT_FOLDER = os.path.join(BASE_PATH, "Artifact matrix")
#     DETAILS_CSV = os.path.join(BASE_PATH, "Details information for healthy subjects.csv")
#     RESULTS_DIR = "/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/results"
#     if not os.path.exists(RESULTS_DIR):
#         os.makedirs(RESULTS_DIR)
    
#     # 2. DATA LOADING
#     data_loader = ANPDataLoader(BASE_PATH)
#     details = data_loader.load_details_csv(DETAILS_CSV)
#     print("Details CSV head:")
#     print(details.head())
    
#     # Container for features from all subjects/epochs/electrodes.
#     results_list = []
    
#     # Temporary: Process only the first 3 subjects (EPCTL01 to EPCTL03)
#     for i in range(1, 4):
#         subj = f"EPCTL{str(i).zfill(2)}"
#         subject_folder = os.path.join(BASE_PATH, subj)
#         edf_file = os.path.join(subject_folder, f"{subj}.edf")
#         print(f"Processing {subj}...")
        
#         # Load EDF metadata to get sampling rate and electrode labels
#         edf_meta = data_loader.load_edf_metadata(edf_file)
#         fs = edf_meta["sample_rates"][0]
#         labels = edf_meta["labels"]
        
#         # Load sleep stage annotations (columns: stage, start_time, duration)
#         annotations = data_loader.load_annotations(subject_folder)
#         if annotations is None:
#             print(f"No annotations for {subj}. Skipping.")
#             continue
        
#         # Initialize feature extractor for this subject
#         feat_extractor = ANPFeatExtractor(fs)
        
#         # Process each epoch defined in the annotations.
#         for row in annotations:
#             stage = row[0]
#             try:
#                 start_time = float(row[1])
#                 duration = float(row[2])
#             except ValueError:
#                 continue
            
#             try:
#                 epoch_data, fs_epoch = data_loader.load_edf_epoch(edf_file, start_time, duration)
#             except Exception as e:
#                 print(f"Error loading epoch for {subj} at {start_time}s: {e}")
#                 continue
            
#             # Extract features for each channel (electrode)
#             features = feat_extractor.extract_features(epoch_data)  # shape: (n_channels, 4)
#             # Save a row per electrode.
#             for ch_idx, feat in enumerate(features):
#                 result = {
#                     "Subject": subj,
#                     "Stage": stage,
#                     "StartTime": start_time,
#                     "Duration": duration,
#                     "Electrode": ch_idx,
#                     "Electrode_Label": labels[ch_idx] if ch_idx < len(labels) else f"Ch{ch_idx}",
#                     "Delta": feat[0],
#                     "Theta": feat[1],
#                     "Alpha": feat[2],
#                     "Beta": feat[3]
#                 }
#                 results_list.append(result)
    
#     # Create a DataFrame from results and save to CSV.
#     results_df = pd.DataFrame(results_list)
#     output_file = os.path.join(RESULTS_DIR, "anphy_features.csv")
#     results_df.to_csv(output_file, index=False)
#     print(f"Saved features for {len(results_df)} electrode epochs to {output_file}")

# if __name__ == "__main__":
#     main()

# import os
# import pandas as pd
# from anp_data_loader import ANPDataLoader
# from anp_feat_extractor import ANPFeatExtractor

# def main():
#     # 1. PATH SETTING
#     BASE_PATH = "/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/ANPHY-Sleep"
#     ARTIFACT_FOLDER = os.path.join(BASE_PATH, "Artifact matrix")
#     DETAILS_CSV = os.path.join(BASE_PATH, "Details information for healthy subjects.csv")
#     RESULTS_DIR = "/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/results"
#     if not os.path.exists(RESULTS_DIR):
#         os.makedirs(RESULTS_DIR)
    
#     # 2. DATA LOADING
#     data_loader = ANPDataLoader(BASE_PATH)
#     details = data_loader.load_details_csv(DETAILS_CSV)
#     print("Details CSV head:")
#     print(details.head())
    
#     # Container for features from all subjects/epochs/electrodes.
#     results_list = []
    
#     # Loop over all subject folders (EPCTL01 to EPCTL29)
#     for i in range(1, 30):
#         subj = f"EPCTL{str(i).zfill(2)}"
#         subject_folder = os.path.join(BASE_PATH, subj)
#         edf_file = os.path.join(subject_folder, f"{subj}.edf")
#         print(f"Processing {subj}...")
        
#         # Load EDF metadata to get sampling rate and electrode labels
#         edf_meta = data_loader.load_edf_metadata(edf_file)
#         fs = edf_meta["sample_rates"][0]
#         labels = edf_meta["labels"]
        
#         # Load sleep stage annotations (columns: stage, start_time, duration)
#         annotations = data_loader.load_annotations(subject_folder)
#         if annotations is None:
#             print(f"No annotations for {subj}. Skipping.")
#             continue
        
#         # Initialize feature extractor for this subject
#         feat_extractor = ANPFeatExtractor(fs)
        
#         # Process each epoch defined in the annotations.
#         for row in annotations:
#             stage = row[0]
#             try:
#                 start_time = float(row[1])
#                 duration = float(row[2])
#             except ValueError:
#                 continue
            
#             try:
#                 epoch_data, fs_epoch = data_loader.load_edf_epoch(edf_file, start_time, duration)
#             except Exception as e:
#                 print(f"Error loading epoch for {subj} at {start_time}s: {e}")
#                 continue
            
#             # Extract features for each channel (electrode)
#             features = feat_extractor.extract_features(epoch_data)  # shape: (n_channels, 4)
#             # Save a row per electrode.
#             for ch_idx, feat in enumerate(features):
#                 result = {
#                     "Subject": subj,
#                     "Stage": stage,
#                     "StartTime": start_time,
#                     "Duration": duration,
#                     "Electrode": ch_idx,
#                     "Electrode_Label": labels[ch_idx] if ch_idx < len(labels) else f"Ch{ch_idx}",
#                     "Delta": feat[0],
#                     "Theta": feat[1],
#                     "Alpha": feat[2],
#                     "Beta": feat[3]
#                 }
#                 results_list.append(result)
    
#     # Create a DataFrame from results and save to CSV.
#     results_df = pd.DataFrame(results_list)
#     output_file = os.path.join(RESULTS_DIR, "anphy_features.csv")
#     results_df.to_csv(output_file, index=False)
#     print(f"Saved features for {len(results_df)} electrode epochs to {output_file}")

# if __name__ == "__main__":
#     main()
