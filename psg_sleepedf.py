#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import time
from scipy.signal import welch
import mne

# Configuration
PROJECT_ROOT = "/Users/kimberly/Documents/STAT4830/STAT-4830-GOALZ-project"
BASE_DIR = os.path.join(PROJECT_ROOT, "data/sleep-edf-database-expanded-1.0.0")
SUBFOLDERS = ['sleep-cassette', 'sleep-telemetry'] # Subfolders with raw data
# Output directory for the PSD features
FEATURES_DIR = os.path.join(PROJECT_ROOT, 'features_psd_sleep_edf')
os.makedirs(FEATURES_DIR, exist_ok=True)

# Preprocessing & Feature Extraction Parameters
TARGET_SFREQ = 100.0 # Renamed FS for clarity, value remains 100.0 Hz
FS = TARGET_SFREQ    # Keep FS for existing functions using it
USE_MULTIPLE_CHANNELS = True # Keep consistent with preprocessing script assumption
if USE_MULTIPLE_CHANNELS:
    # Ensure these channel names exactly match those in the EDF files
    CHANNELS_TO_LOAD = ["EEG Fpz-Cz", "EOG horizontal"] 
else:
    CHANNELS_TO_LOAD = ["EEG Fpz-Cz"]
LOW_FREQ = 0.5
HIGH_FREQ = 30.0
NOTCH_FREQ = 60.0 # Hz, set to 50.0 for Europe if needed
EPOCH_LENGTH = 30.0 # seconds

# Sleep stage mapping from preprocessing script
ANNOTATION_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3, # Map stage 4 to stage 3
    "Sleep stage R": 4
}

def get_true_subject_id(filename):
    """Extract true subject ID ignoring the night number. 
       Input filename should be the base recording ID (e.g., SC4001E0)
    """
    # Logic based on typical Sleep-EDF naming
    if filename.startswith('SC4') and len(filename) >= 5:
        return filename[:5]  # SC4xx
    elif filename.startswith('ST7') and len(filename) >= 5:
        return filename[:5]  # ST7xx
    else:
        return filename[:6] # Fallback, adjust length as needed

def find_hypnogram(psg_file):
    """Find the corresponding hypnogram file for a PSG file."""
    # Assumes Hypnogram file is in the same directory as PSG file
    # Tries to match based on the subject ID part (first 6 chars)
    basename = os.path.basename(psg_file)
    if len(basename) < 6:
        print(f"Warning: PSG filename {basename} too short to extract ID for hypnogram search.")
        return None
    subject_id_part = basename[:6] # e.g., SC4001, ST7011
    
    dir_path = os.path.dirname(psg_file)
    # Pattern like: .../SC4001*Hypnogram.edf
    pattern = os.path.join(dir_path, f"{subject_id_part}*Hypnogram.edf")
    hyp_files = glob.glob(pattern)
    
    if len(hyp_files) == 1:
        return hyp_files[0]
    elif len(hyp_files) > 1:
        # If multiple hypnograms match the base ID (e.g., different nights?), try to refine
        # Try matching the full recording ID before the suffix if possible
        rec_id_part = basename.split('-')[0] # e.g., SC4001E0
        refined_pattern = os.path.join(dir_path, f"{rec_id_part}*Hypnogram.edf")
        refined_hyp_files = glob.glob(refined_pattern)
        if len(refined_hyp_files) == 1:
             print(f"Warning: Found multiple hypnograms for {subject_id_part}, using specific match: {refined_hyp_files[0]}")
             return refined_hyp_files[0]
        else:
            # If still ambiguous, default to the first one found with the base ID
            print(f"Warning: Found multiple hypnograms for {subject_id_part} ({len(hyp_files)}), using first: {hyp_files[0]}")
            return hyp_files[0]
    else:
        print(f"Warning: No hypnogram found matching pattern {pattern}")
        return None

def bandpower_welch(signal, fs, band):
    """Computes power in a frequency band using Welch's method."""
    nperseg = min(len(signal), fs) # Use fs samples (1 sec segment)
    if nperseg == 0: return 0.0
    try:
        f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
    except ValueError:
        return 0.0
    idx = np.logical_and(f >= band[0], f <= band[1])
    if not np.any(idx): return 0.0
    return np.sum(Pxx[idx])

def extract_psd_features(data_chunk, fs=FS):
    """Extracts PSD bandpower features from one epoch for all channels."""
    n_channels, _ = data_chunk.shape
    channel_prefixes = ['eeg', 'eog'] # Assumes order matches CHANNELS_TO_LOAD
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta":  (12, 30),
        # "gamma": (30, 50), # Gamma limited by TARGET_SFREQ=100Hz (Nyquist=50Hz)
    }
    features = {}
    for i in range(n_channels):
        if i >= len(channel_prefixes): break
        prefix = channel_prefixes[i]
        channel_data = data_chunk[i, :]
        for bname, (lowf, highf) in bands.items():
            effective_highf = min(highf, fs / 2.0)
            if effective_highf <= lowf: continue
            power = bandpower_welch(channel_data, fs, (lowf, effective_highf))
            features[f"{prefix}_{bname}"] = power
    return features

def process_file(psg_path): # Input is now the raw PSG file path
    """Preprocess one raw PSG file, extract PSD features, and save."""
    start_time = time.time()
    rec_id_base = os.path.basename(psg_path).split('-')[0] # e.g., SC4001E0
    print(f"Starting processing for: {rec_id_base}")
    
    hyp_path = find_hypnogram(psg_path)
    if not hyp_path:
        print(f"  Skipping {rec_id_base}: Hypnogram not found.")
        return None, None

    try:
        # 1. Preprocessing (adapted from sleepedf_prepro.ipynb)
        # Load raw data with specified channels
        # Use try-except for MNE loading as channels might differ slightly
        try:
            raw = mne.io.read_raw_edf(psg_path, include=CHANNELS_TO_LOAD, preload=True, verbose=False)
        except ValueError as e:
             # Try loading without EOG if the specific EOG channel wasn't found
             print(f"  Warning: Could not load all channels for {rec_id_base} ({e}). Trying EEG only.")
             try:
                  raw = mne.io.read_raw_edf(psg_path, include=[CHANNELS_TO_LOAD[0]], preload=True, verbose=False)
                  # If successful, update channels list for this file
                  effective_channels = [CHANNELS_TO_LOAD[0]]
             except Exception as e_inner:
                  print(f"  Skipping {rec_id_base}: Failed to load even primary EEG channel ({e_inner}).")
                  return None, None
        else:
             effective_channels = CHANNELS_TO_LOAD

        # Resample if necessary
        if raw.info['sfreq'] != TARGET_SFREQ:
            print(f"  Resampling {rec_id_base} from {raw.info['sfreq']} Hz to {TARGET_SFREQ} Hz")
            raw.resample(TARGET_SFREQ, npad="auto", verbose=False)
        
        # Pick types based on effective channels loaded
        picks = mne.pick_channels(raw.ch_names, include=effective_channels)
        if len(picks) == 0: 
            print(f"  Skipping {rec_id_base}: No valid channels selected after loading.")
            return None, None
        
        # Apply filters
        raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ, picks=picks, verbose=False)
        
        # Load annotations and create epochs
        try:
            ann = mne.read_annotations(hyp_path)
            raw.set_annotations(ann, emit_warning=False)
            # Filter out annotations not in our map before creating events
            valid_event_ids = {k:v for k,v in ANNOTATION_MAP.items() if k in ann.description}
            if not valid_event_ids:
                print(f"  Skipping {rec_id_base}: No valid annotations found in {hyp_path} matching ANNOTATION_MAP.")
                return None, None
            events, _ = mne.events_from_annotations(raw, event_id=valid_event_ids, chunk_duration=EPOCH_LENGTH, verbose=False)
        except Exception as e_ann:
             print(f"  Skipping {rec_id_base}: Error reading or processing annotations ({e_ann}).")
             return None, None
            
        if len(events) == 0:
             print(f"  Skipping {rec_id_base}: No epochs could be created from annotations.")
             return None, None
            
        tmin = 0.0; tmax = EPOCH_LENGTH - 1/raw.info['sfreq']
        epochs_mne = mne.Epochs(raw, events=events, event_id=valid_event_ids, tmin=tmin, tmax=tmax,
                                baseline=None, preload=True, verbose=False)
        
        # Clear raw object to save memory before feature extraction
        del raw
        
        epochs_data = epochs_mne.get_data() # Shape (n_epochs, n_channels, n_samples)
        labels = epochs_mne.events[:, -1] # Get the mapped label (0-4)
        del epochs_mne # Free memory
        
        n_epochs, n_data_channels, n_samples = epochs_data.shape

        # Standardize each channel across all its epochs
        for ch in range(n_data_channels):
            # Calculate mean and std across all epochs for this channel
            ch_data_flat = epochs_data[:, ch, :].flatten()
            m = np.mean(ch_data_flat)
            s = np.std(ch_data_flat)
            if s == 0: s = 1.0 # Avoid division by zero for flat signals
            epochs_data[:, ch, :] = (epochs_data[:, ch, :] - m) / s

        # 2. Feature Extraction
        all_epoch_features = []
        for i in range(n_epochs):
            epoch_data_single = epochs_data[i] # Shape (n_channels, n_samples)
            # Pass the actual number of channels present in the data
            features = extract_psd_features(epoch_data_single, fs=FS)
            all_epoch_features.append(features)

        all_df = pd.DataFrame(all_epoch_features)
        if all_df.empty:
             print(f"  Skipping {rec_id_base}: No features extracted.")
             return None, None
            
        all_df['label'] = labels

        # 3. Saving Results
        subj_id = get_true_subject_id(rec_id_base)
        csv_path = os.path.join(FEATURES_DIR, f"{rec_id_base}_psd.csv")
        all_df.to_csv(csv_path, index=False)

        if 'label' in all_df.columns:
            feature_values = all_df.drop(columns=['label']).values.astype('float32')
            feature_names = list(all_df.drop(columns=['label']).columns)
        else:
             print(f"  Warning: 'label' column missing in final DataFrame for {rec_id_base}.")
             feature_values = all_df.values.astype('float32')
             feature_names = list(all_df.columns)
            
        npz_path = os.path.join(FEATURES_DIR, f"{rec_id_base}_psd.npz")
        np.savez_compressed(
            npz_path,
            features=feature_values,
            labels=labels.astype('int8'),
            feature_names=feature_names,
            subject_id=subj_id,
            recording_id=rec_id_base
        )

        elapsed = time.time() - start_time
        print(f"  Successfully processed PSD for {rec_id_base} (subj {subj_id}): {n_epochs} epochs in {elapsed:.1f}s")
        return subj_id, rec_id_base

    except MemoryError:
        print(f"  Skipping {rec_id_base}: MemoryError during processing. File might be too large or system memory limited.")
        return None, None
    except Exception as e:
        print(f"  Error processing {rec_id_base} ({psg_path}): {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed traceback during debugging
        return None, None

def main():
    # Find all raw PSG files
    all_psg_files = []
    print(f"Searching for raw PSG files in {BASE_DIR} subfolders: {SUBFOLDERS}")
    for sf in SUBFOLDERS:
        folder_path = os.path.join(BASE_DIR, sf)
        # Use case-insensitive globbing for *PSG.edf
        # This finds files ending in -PSG.edf or -psg.edf
        psg_in_folder = glob.glob(os.path.join(folder_path, "*[Pp][Ss][Gg].edf"))
        print(f"  Found {len(psg_in_folder)} PSG files in {folder_path}")
        all_psg_files.extend(psg_in_folder)

    all_psg_files = sorted(list(set(all_psg_files))) # Remove duplicates and sort
    print(f"Found {len(all_psg_files)} total raw PSG files to process.")

    if not all_psg_files:
        print("No raw PSG files found. Please check BASE_DIR and SUBFOLDERS configuration.")
        return

    # Group by subject
    subj_map = {}
    for psg_path in all_psg_files:
        rec_base = os.path.basename(psg_path).split('-')[0] # e.g., SC4001E0
        subj = get_true_subject_id(rec_base)
        subj_map.setdefault(subj, []).append(psg_path)

    print(f"Data covers {len(subj_map)} unique subjects.")
    
    # Configure parallel processing
    import multiprocessing
    n_jobs = max(1, multiprocessing.cpu_count() - 1) 
    print(f"Starting parallel processing using {n_jobs} jobs...")

    # Parallelize the processing of each raw file
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_file)(f) # Pass the raw PSG path
        for files in subj_map.values() for f in files # Flatten the list
    )
    
    # Filter out None results from failed files
    valid_results = [r for r in results if r is not None and r[0] is not None]

    if not valid_results:
        print("No files were processed successfully.")
        return
        
    # Create mapping from valid results
    mapping_list = []
    processed_recordings = set()
    for subj_id, rec_id in valid_results:
         if rec_id not in processed_recordings:
              mapping_list.append({'subject_id': subj_id, 'recording_id': rec_id})
              processed_recordings.add(rec_id)

    if mapping_list:
        map_df = pd.DataFrame(mapping_list)
        map_df = map_df[['subject_id', 'recording_id']] # Ensure column order
        map_df.to_csv(os.path.join(FEATURES_DIR, 'subject_recording_mapping_psd.csv'), index=False)
        print(f"Saved subject-recording mapping to {FEATURES_DIR}")
    else:
        print("No successful results to create mapping file.")

    print("Combined preprocessing and PSD feature extraction complete.")

if __name__ == "__main__":
    # Add basic check for MNE
    try: 
        import mne
    except ImportError:
        print("MNE Python not found. Please install it: pip install mne")
        exit()
    main()
