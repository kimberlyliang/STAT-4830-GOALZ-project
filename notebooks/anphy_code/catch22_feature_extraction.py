# first 10 mins only
import os
import pandas as pd
import pycatch22 as catch22  
from anp_data_loader import ANPDataLoader
from config.config import DataConfig

def main():
    data_loader = ANPDataLoader(DataConfig.BASE_PATH)
    results_list = []
    
    for subject_id in DataConfig.SUBJECTS:
        paths = DataConfig.get_subject_paths(subject_id)
        print(f"Processing subject {subject_id} (first 10 minutes using 30-sec epochs)...")
        
        try:
            # Load EDF metadata: get sampling rate, channel labels, duration, and number of channels
            edf_meta = data_loader.load_edf_metadata(paths['edf'])
            fs = edf_meta["sample_rates"][0]
            labels = edf_meta["labels"]
            duration = edf_meta["duration"]
            n_channels = edf_meta["n_channels"]
            
            # Restrict to first 10 minutes: 10 minutes = 600 sec, so 20 epochs of 30 sec each.
            num_epochs = min(20, int(duration // 30))
            print(f"  Duration: {duration:.1f} sec, Processing Epochs (30 sec each): {num_epochs}")
            
            # Loop over the first 20 epochs (or fewer if duration < 600 sec)
            for epoch_index in range(num_epochs):
                start_time = epoch_index * 30
                try:
                    epoch_data, fs_epoch = data_loader.load_edf_epoch(paths['edf'], start_time, 30)
                except Exception as e:
                    print(f"Error loading epoch {epoch_index} (start {start_time}s) for {subject_id}: {e}")
                    continue
                
                # Compute catch22 features on each electrode (channel)
                for ch_idx in range(n_channels):
                    # Extract 1D time series for this electrode in the current epoch
                    ts = epoch_data[ch_idx, :]
                    # pycatch22 expects a list (or tuple) as input
                    ts_list = ts.tolist()
                    result_dict = catch22.catch22_all(ts_list)
                    feat_names = result_dict['names']
                    feat_values = result_dict['values']
                    
                    # Create a row for this electrode and epoch with catch22 features
                    row = {
                        "Subject": subject_id,
                        "Epoch": epoch_index,
                        "StartTime": start_time,
                        "Electrode": ch_idx,
                        "Electrode_Label": labels[ch_idx] if ch_idx < len(labels) else f"Ch{ch_idx}"
                    }
                    for name, value in zip(feat_names, feat_values):
                        row[f"catch22_{name}"] = value
                    results_list.append(row)
                    
        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
            continue
    
    # Create DataFrame and save to CSV in results directory
    results_df = pd.DataFrame(results_list)
    output_file = os.path.join(DataConfig.RESULTS_DIR, "catch22_anphy_features_10min.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Saved catch22 features for {len(results_df)} electrode epochs to {output_file}")

if __name__ == "__main__":
    main()

# full duration
# import os
# import pandas as pd
# import pycatch22 as catch22  
# from anp_data_loader import ANPDataLoader

# def main():
#     # 1. PATH SETTING
#     BASE_PATH = "/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/ANPHY-Sleep"
#     RESULTS_DIR = "/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/results"
#     if not os.path.exists(RESULTS_DIR):
#         os.makedirs(RESULTS_DIR)
    
#     # 2. INITIALIZE DATA LOADER
#     data_loader = ANPDataLoader(BASE_PATH)
#     results_list = []
    
#     # Process all 29 subjects (EPCTL01 to EPCTL29)
#     for i in range(1, 30):
#         subj = f"EPCTL{str(i).zfill(2)}"
#         subject_folder = os.path.join(BASE_PATH, subj)
#         edf_file = os.path.join(subject_folder, f"{subj}.edf")
#         print(f"Processing subject {subj} (entire recording, sampled every 30 seconds)...")
        
#         # Load EDF metadata: get sampling rate, channel labels, duration, and number of channels
#         edf_meta = data_loader.load_edf_metadata(edf_file)
#         fs = edf_meta["sample_rates"][0]
#         labels = edf_meta["labels"]
#         duration = edf_meta["duration"]
#         n_channels = edf_meta["n_channels"]
#         num_epochs = int(duration // 30)
#         print(f"  Duration: {duration:.1f} sec, Epochs (30 sec each): {num_epochs}")
        
#         # Loop over all 30-second epochs in the recording
#         for epoch_index in range(num_epochs):
#             start_time = epoch_index * 30
#             try:
#                 epoch_data, fs_epoch = data_loader.load_edf_epoch(edf_file, start_time, 30)
#             except Exception as e:
#                 print(f"Error loading epoch {epoch_index} (start {start_time}s) for {subj}: {e}")
#                 continue
            
#             # Compute catch22 features on each electrode (channel)
#             for ch_idx in range(n_channels):
#                 # Extract 1D time series for this electrode in the current epoch
#                 ts = epoch_data[ch_idx, :]
#                 # pycatch22 expects a list (or tuple) as input
#                 ts_list = ts.tolist()
#                 result_dict = catch22.catch22_all(ts_list)
#                 feat_names = result_dict['names']
#                 feat_values = result_dict['values']
                
#                 # Create a row for this electrode and epoch with catch22 features
#                 row = {
#                     "Subject": subj,
#                     "Epoch": epoch_index,
#                     "StartTime": start_time,
#                     "Electrode": ch_idx,
#                     "Electrode_Label": labels[ch_idx] if ch_idx < len(labels) else f"Ch{ch_idx}"
#                 }
#                 for name, value in zip(feat_names, feat_values):
#                     row[f"catch22_{name}"] = value
#                 results_list.append(row)
    
#     # 3. SAVE THE RESULTS TO CSV
#     results_df = pd.DataFrame(results_list)
#     output_file = os.path.join(RESULTS_DIR, "catch22_anphy_features_10min.csv")
#     results_df.to_csv(output_file, index=False)
#     print(f"Saved catch22 features for {len(results_df)} electrode epochs to {output_file}")

# if __name__ == "__main__":
#     main()
