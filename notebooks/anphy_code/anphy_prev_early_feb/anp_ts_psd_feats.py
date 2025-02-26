#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# Import our data loader and feature extractors.
from anp_data_loader import ANPDataLoader
from anp_feat_extractor import ANPFeatExtractor
from anp_catch22_extractor import ANPCatch22Extractor  # assumed to exist

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter on each channel of the data.
    
    Args:
        data (np.ndarray): array of shape (n_channels, n_samples)
        lowcut (float): low frequency cutoff.
        highcut (float): high frequency cutoff.
        fs (float): sampling frequency
        order (int): order of the Butterworth filter (default=4)
        
    Returns:
        np.ndarray: filtered data of the same shape as input.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i] = filtfilt(b, a, data[i])
    return filtered_data

def main():
    # --- PATH SETTING ---
    BASE_PATH = "/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/ANPHY-Sleep"
    DETAILS_CSV = os.path.join(BASE_PATH, "Details information for healthy subjects.csv")
    RESULTS_DIR = "/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/results"
    # Designated subfolder for these time series feature extraction results.
    DESIGNATED_SUBFOLDER = os.path.join(RESULTS_DIR, "anp_ts_feats")
    if not os.path.exists(DESIGNATED_SUBFOLDER):
        os.makedirs(DESIGNATED_SUBFOLDER)
    
    # --- INITIAL DATA LOADING ---
    data_loader = ANPDataLoader(BASE_PATH)
    details = data_loader.load_details_csv(DETAILS_CSV)
    print("Details CSV head:")
    print(details.head())
    
    results_list = []

    # --- FEATURE EXTRACTION PARAMETERS ---
    # Filter parameters for frequency domain (relative band power) extraction.
    lowcut = 0.3
    highcut = 45.0

    # Define epoch parameters: use a 30-second epoch.
    epoch_duration = 30  # seconds
    # Sample an epoch every 15 minutes.
    step_interval = 15 * 60  # 900 seconds

    # Process all 29 subjects (EPCTL01 to EPCTL29)
    for i in range(1, 30):
        subj = f"EPCTL{str(i).zfill(2)}"
        subject_folder = os.path.join(BASE_PATH, subj)
        edf_file = os.path.join(subject_folder, f"{subj}.edf")
        print(f"Processing {subj} ...")
        
        # Load EDF metadata to get sampling rate, channel labels, and recording duration.
        try:
            edf_meta = data_loader.load_edf_metadata(edf_file)
        except Exception as e:
            print(f"Error loading metadata for {subj}: {e}")
            continue

        fs = edf_meta["sample_rates"][0]
        labels = edf_meta["labels"]
        recording_duration = edf_meta["duration"]

        # Initialize feature extractors.
        psd_extractor = ANPFeatExtractor(fs)
        catch22_extractor = ANPCatch22Extractor()  # Assumes an extractor interface

        # Iterate over the recording in steps of 15 minutes.
        epoch_counter = 0
        current_time = 0
        while current_time + epoch_duration <= recording_duration:
            try:
                # Load a 30-second epoch starting at current_time.
                epoch_data, _ = data_loader.load_edf_epoch(edf_file, current_time, epoch_duration)
            except Exception as e:
                print(f"Error loading epoch at {current_time}s for {subj}: {e}")
                current_time += step_interval
                continue

            # For PSD feature extraction, apply bandpass filtering.
            filtered_epoch = apply_bandpass_filter(epoch_data, lowcut, highcut, fs)
            
            # Extract PSD (relative band power) features using filtered data.
            psd_features = psd_extractor.extract_features(filtered_epoch)
            # Extract catch22 time series features on raw epoch data.
            catch22_features = catch22_extractor.extract_features(epoch_data)
            
            # Both feature extractors are assumed to return one feature vector per channel.
            n_channels = epoch_data.shape[0]
            for ch_idx in range(n_channels):
                row = {
                    "Subject": subj,
                    "Epoch": epoch_counter,
                    "StartTime": current_time,
                    "Electrode": ch_idx,
                    "Electrode_Label": labels[ch_idx] if ch_idx < len(labels) else f"Ch{ch_idx}"
                }
                # Add PSD features (assumed order: Delta, Theta, Alpha, Beta).
                row["Delta"] = psd_features[ch_idx][0]
                row["Theta"] = psd_features[ch_idx][1]
                row["Alpha"] = psd_features[ch_idx][2]
                row["Beta"] = psd_features[ch_idx][3]
                
                # Add catch22 features. Assuming catch22_features[ch_idx] returns a list of 22 features.
                for feat_idx, feat_val in enumerate(catch22_features[ch_idx]):
                    row[f"catch22_{feat_idx}"] = feat_val
                
                results_list.append(row)
            
            epoch_counter += 1
            current_time += step_interval

    # Save the results to a CSV file in the designated results subfolder.
    output_file = os.path.join(DESIGNATED_SUBFOLDER, "anp_ts_psd_feats.csv")
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(output_file, index=False)
    print(f"Saved feature extraction results for {len(results_list)} electrode epochs to {output_file}")

if __name__ == "__main__":
    main() 