#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import time
from scipy.signal import welch, iirnotch, filtfilt # Added iirnotch, filtfilt
import mne
import xml.etree.ElementTree as ET # For XML parsing

# ==============================================
# Configuration for MESA Dataset
# ==============================================
PROJECT_ROOT = "/Users/kimberly/Documents/STAT4830/STAT-4830-GOALZ-project"
BASE_DIR = os.path.join(PROJECT_ROOT, "data/MESA")

# --- Input Data Paths ---
# Directory containing the subfolders (mesa_200_subset_edfs, mesa_200_subset_edfs 2, ...)
EDF_PARENT_DIR = os.path.join(BASE_DIR, 'edfs') 
# List of the actual subfolder names containing EDFs
EDF_SUBDIR_NAMES = ['mesa_200_subset_edfs'] + [f'mesa_200_subset_edfs {i}' for i in range(2, 19)]
# Directory containing the XML annotation files
ANNOTATION_DIR = os.path.join(BASE_DIR, 'mesa_200_annotations-events-nssr')

# --- Output Data Path ---
FEATURES_DIR = os.path.join(PROJECT_ROOT, 'features_psd_mesa')
os.makedirs(FEATURES_DIR, exist_ok=True)

# --- Preprocessing Parameters (from MESA_preopro.ipynb) ---
# Ensure these match the channels available and desired in MESA EDFs
# Common MESA EEG: "EEG 1", "EEG 2", "EEG 3", EOG: "EOG L", "EOG R"
# Choose the primary ones you want to process (e.g., EEG 1 for Fpz-Cz equivalent, EOG L/R)
CHANNELS_TO_LOAD = ["EEG1", "EOG-L", "EOG-R"] 
TARGET_SFREQ = 100.0 # MESA typically higher sampling rate
LOW_FREQ = 0.5
HIGH_FREQ = 30.0
NOTCH_FREQ_GUESS = 60.0 # Hz, adaptive notch will refine this
EPOCH_LENGTH = 30.0 # seconds
FS = TARGET_SFREQ # Sampling frequency for feature extraction

# --- Sleep Stage Mapping (from MESA_preopro.ipynb) ---
# Based on MESA XML annotation values
ANNOTATION_MAP = {
    "0": 0,  # Wake
    "1": 1,  # N1
    "2": 2,  # N2
    "3": 3,  # N3
    "5": 4   # REM
}
# Create reverse map for potential use, mapping ID back to stage string
STAGE_ID_TO_NAME = {v: k for k, v in ANNOTATION_MAP.items()} 

# ==============================================
# Helper Functions
# ==============================================

def get_mesa_recording_id(filepath):
    """Extracts recording ID (e.g., mesa-sleep-0001) from path."""
    basename = os.path.basename(filepath)
    # Assumes format like 'mesa-sleep-XXXX.edf'
    if basename.startswith('mesa-sleep-') and basename.lower().endswith('.edf'):
        return basename[:-4] # Remove .edf
    else:
        # Fallback or warning if format unexpected
        print(f"Warning: Unexpected filename format for ID extraction: {basename}")
        return basename.replace('.edf', '').replace('.EDF', '') # Basic fallback

def find_annotation_xml(edf_file):
    """Find the corresponding -nsrr.xml annotation file."""
    rec_id = get_mesa_recording_id(edf_file)
    xml_filename = f"{rec_id}-nsrr.xml"
    xml_path = os.path.join(ANNOTATION_DIR, xml_filename)
    if os.path.exists(xml_path):
        return xml_path
    else:
        print(f"Warning: Annotation XML not found at {xml_path}")
        return None

def parse_xml_annotations(xml_path):
    """Parse the MESA XML file to extract sleep stage annotations."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        stages = []
        for event in root.findall(".//ScoredEvent"):
            event_type_elem = event.find("EventType")
            if event_type_elem is not None and event_type_elem.text == "Stages|Stages":
                concept_elem = event.find("EventConcept")
                start_elem = event.find("Start")
                duration_elem = event.find("Duration")
                
                if all([concept_elem is not None, concept_elem.text, 
                        start_elem is not None, start_elem.text, 
                        duration_elem is not None, duration_elem.text]):
                        
                    concept = concept_elem.text.strip().split("|")[-1] 
                    if concept in ANNOTATION_MAP:
                        stage_id = ANNOTATION_MAP[concept]
                        start = float(start_elem.text)
                        duration = float(duration_elem.text)
                        
                        # Create events for each 30-second epoch within the duration
                        num_epochs = int(round(duration / EPOCH_LENGTH))
                        for i in range(num_epochs):
                            epoch_start = start + i * EPOCH_LENGTH
                            # Append (start_time, duration, stage_id)
                            stages.append((epoch_start, EPOCH_LENGTH, stage_id))
                    # else: print(f"Debug: Skipping unknown concept '{concept}' in {xml_path}") # Optional debug
        return stages
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_path}: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error reading annotations from {xml_path}: {e}")
        return []

def auto_notch_filter_mne(raw, picks, center_guess=60.0, search_range=1.0, quality=30.0):
    """Applies an adaptive notch filter to selected channels of an MNE Raw object."""
    fs = raw.info['sfreq']
    data = raw.get_data(picks=picks)
    filtered_data = np.zeros_like(data)
    
    for i in range(data.shape[0]): # Iterate through selected channels
        signal = data[i, :]
        # Estimate exact notch frequency from PSD
        freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), int(4*fs))) # Use longer segment for better freq resolution
        idx_range = np.where((freqs >= center_guess - search_range) & (freqs <= center_guess + search_range))[0]
        
        if len(idx_range) > 0:
            true_freq = freqs[idx_range[np.argmax(psd[idx_range])]]
            # Apply notch filter using scipy.signal functions
            w0 = true_freq / (0.5 * fs)
            if w0 >= 1.0: # Avoid error if true_freq is at or above Nyquist
                 print(f"Warning: Estimated notch freq {true_freq} >= Nyquist. Skipping notch for channel {i}.")
                 filtered_data[i,:] = signal
                 continue
            b, a = iirnotch(w0, quality)
            filtered_data[i,:] = filtfilt(b, a, signal)
            # print(f"  Applied notch at {true_freq:.2f} Hz for channel {i}") # Optional debug
        else:
            # print(f"  No significant peak near {center_guess} Hz found for channel {i}. Skipping notch.") # Optional debug
            filtered_data[i,:] = signal # No filtering if peak not found
            
    # Update the data in the MNE Raw object
    raw._data[picks] = filtered_data
    return raw # Return modified Raw object

def bandpower_welch(signal, fs, band):
    """Computes power in a frequency band using Welch's method."""
    nperseg = min(len(signal), int(fs)) # Use 1-second window for Welch
    if nperseg == 0: return 0.0
    try:
        f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
    except ValueError: return 0.0
    idx = np.logical_and(f >= band[0], f <= band[1])
    if not np.any(idx): return 0.0
    return np.sum(Pxx[idx])

def extract_psd_features(data_chunk, fs=FS):
    """Extracts PSD bandpower features from one epoch for all channels."""
    n_channels, _ = data_chunk.shape
    # Adjust prefixes based on actual channels loaded/processed if needed
    # Assuming CHANNELS_TO_LOAD order: EEG 1, EOG L, EOG R -> eeg, eogL, eogR
    channel_prefixes = ['eeg', 'eogL', 'eogR'] 
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta":  (12, 30),
        "gamma": (30, 50),
    }
    features = {}
    for i in range(n_channels):
        if i >= len(channel_prefixes): 
             prefix = f'ch{i}' # Generic prefix if more channels than expected
        else:
             prefix = channel_prefixes[i]
        
        channel_data = data_chunk[i, :]
        for bname, (lowf, highf) in bands.items():
            effective_highf = min(highf, fs / 2.0) # Nyquist limit
            if effective_highf <= lowf: continue
            power = bandpower_welch(channel_data, fs, (lowf, effective_highf))
            features[f"{prefix}_{bname}"] = power
    return features

# ==============================================
# Main Processing Function
# ==============================================

def process_mesa_file(edf_path):
    """Preprocess one raw MESA EDF file, extract PSD features, and save."""
    start_time = time.time()
    rec_id = get_mesa_recording_id(edf_path)
    print(f"Starting processing for: {rec_id}")
    
    xml_path = find_annotation_xml(edf_path)
    if not xml_path:
        print(f"  Skipping {rec_id}: Annotation XML not found.")
        return None, None

    # --- Check if output already exists --- 
    npz_output_path = os.path.join(FEATURES_DIR, f"{rec_id}_psd.npz")
    if os.path.exists(npz_output_path):
        print(f"  Skipping {rec_id}: Output file already exists.")
        return None, None # Return None to indicate skipped

    try:
        # 1. Preprocessing 
        # Load raw data - handle potential missing channels gracefully
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            # Select only the desired channels AFTER loading all
            raw.pick_channels(CHANNELS_TO_LOAD, ordered=True) 
        except ValueError as e:
            print(f"  Skipping {rec_id}: Could not pick all required channels ({CHANNELS_TO_LOAD}). Error: {e}")
            return None, None
        except Exception as e:
            print(f"  Skipping {rec_id}: Error loading EDF file {edf_path}. Error: {e}")
            return None, None

        # Resample if necessary
        if raw.info['sfreq'] != TARGET_SFREQ:
            print(f"  Resampling {rec_id} from {raw.info['sfreq']} Hz to {TARGET_SFREQ} Hz")
            try:
                raw.resample(TARGET_SFREQ, npad="auto", verbose=False)
            except Exception as e_resample:
                print(f"  Skipping {rec_id}: Error during resampling: {e_resample}")
                return None, None

        # Filters (Apply bandpass first)
        picks = mne.pick_types(raw.info, eeg=True, eog=True) # Pick EEG/EOG after channel selection
        if len(picks) == 0: 
             print(f"  Skipping {rec_id}: No EEG/EOG channels found after picking {CHANNELS_TO_LOAD}.")
             return None, None
             
        raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ, picks=picks, verbose=False)
        # Apply adaptive notch filter
        raw = auto_notch_filter_mne(raw, picks=picks, center_guess=NOTCH_FREQ_GUESS)
        
        # Load annotations and create epochs
        stages = parse_xml_annotations(xml_path)
        if not stages:
            print(f"  Skipping {rec_id}: No valid stage annotations parsed from {xml_path}.")
            return None, None
            
        # Convert stage tuples to MNE-compatible events array
        events = []
        for start, duration, stage_id in stages:
             sample = int(start * raw.info['sfreq'])
             # MNE event format: [sample, previous_event_id (0), event_id]
             events.append([sample, 0, stage_id])
        events = np.array(events)

        if events.shape[0] == 0:
            print(f"  Skipping {rec_id}: No MNE events created from annotations.")
            return None, None
        
        # Create MNE Epochs object
        # Map stage IDs to event IDs MNE understands
        event_id_map = {str(name): val for val, name in STAGE_ID_TO_NAME.items() if val in events[:,2]} # Only map stages present
        tmin = 0.0
        tmax = EPOCH_LENGTH - 1.0 / raw.info['sfreq'] # Duration of epoch
        try:
             epochs_mne = mne.Epochs(raw, events=events, event_id=event_id_map, 
                                     tmin=tmin, tmax=tmax, baseline=None, 
                                     preload=True, verbose=False, on_missing='warn')
        except Exception as e_epoch:
             print(f"  Skipping {rec_id}: Error creating MNE Epochs object: {e_epoch}")
             return None, None

        if len(epochs_mne) == 0:
             print(f"  Skipping {rec_id}: MNE Epochs object is empty after creation.")
             return None, None
             
        # Clear raw object to save memory 
        del raw 
        
        epochs_data = epochs_mne.get_data() # Shape (n_epochs, n_channels, n_samples)
        labels = epochs_mne.events[:, -1] # Get the mapped label (0-4)
        channel_names_used = epochs_mne.ch_names # Actual channels in the epoch object
        del epochs_mne # Free memory
        
        n_epochs, n_data_channels, n_samples = epochs_data.shape

        # Standardize each channel across all its epochs
        for ch in range(n_data_channels):
            ch_data_flat = epochs_data[:, ch, :].flatten()
            m = np.mean(ch_data_flat)
            s = np.std(ch_data_flat)
            if s == 0: s = 1.0 
            epochs_data[:, ch, :] = (epochs_data[:, ch, :] - m) / s

        # 2. Feature Extraction
        print(f"  Extracting PSD features for {n_epochs} epochs...")
        all_epoch_features = []
        for i in range(n_epochs):
            features = extract_psd_features(epochs_data[i], fs=FS)
            all_epoch_features.append(features)

        all_df = pd.DataFrame(all_epoch_features)
        if all_df.empty:
             print(f"  Skipping {rec_id}: No features extracted.")
             return None, None
             
        all_df['label'] = labels

        # 3. Saving Results
        # Note: MESA doesn't have distinct subject/recording IDs in the same way Sleep-EDF does
        # We use the recording ID as the primary identifier.
        subject_id = rec_id # Treat recording ID as subject ID for consistency if needed
        
        csv_path = os.path.join(FEATURES_DIR, f"{rec_id}_psd.csv")
        all_df.to_csv(csv_path, index=False)

        if 'label' in all_df.columns:
            feature_values = all_df.drop(columns=['label']).values.astype('float32')
            feature_names = list(all_df.drop(columns=['label']).columns)
        else:
             feature_values = all_df.values.astype('float32')
             feature_names = list(all_df.columns)
            
        npz_path = os.path.join(FEATURES_DIR, f"{rec_id}_psd.npz")
        np.savez_compressed(
            npz_path,
            features=feature_values,
            labels=labels.astype('int8'),
            feature_names=feature_names,
            subject_id=subject_id, # Using rec_id here
            recording_id=rec_id
        )

        elapsed = time.time() - start_time
        print(f"  Successfully processed PSD for {rec_id}: {n_epochs} epochs, {len(feature_names)} features in {elapsed:.1f}s")
        return subject_id, rec_id # Return IDs for mapping file

    except MemoryError:
        print(f"  Skipping {rec_id}: MemoryError during processing.")
        return None, None
    except Exception as e:
        print(f"  Error processing {rec_id} ({edf_path}): {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed traceback
        return None, None

# ==============================================
# Main Execution Logic
# ==============================================

def main():
    # Find all raw EDF files in the specified subdirectories
    all_edf_files = []
    print(f"Searching for raw MESA EDF files in subfolders under: {EDF_PARENT_DIR}")
    for subdir_name in EDF_SUBDIR_NAMES:
        dir_path = os.path.join(EDF_PARENT_DIR, subdir_name)
        if os.path.isdir(dir_path):
            # Find *.edf (case-insensitive)
            edf_files_in_subdir = glob.glob(os.path.join(dir_path, '*.edf')) + \
                                  glob.glob(os.path.join(dir_path, '*.EDF'))
            if edf_files_in_subdir:
                print(f"  Found {len(edf_files_in_subdir)} EDF files in {dir_path}")
                all_edf_files.extend(edf_files_in_subdir)
        else:
            print(f"Warning: EDF subdirectory not found - {dir_path}")

    all_edf_files = sorted(list(set(all_edf_files))) 
    print(f"Found {len(all_edf_files)} total raw MESA EDF files to process.")

    if not all_edf_files:
        print("No raw MESA EDF files found. Please check paths and subdirectory names.")
        return

    # Configure parallel processing
    import multiprocessing
    n_jobs = max(1, multiprocessing.cpu_count() - 1) 
    print(f"Starting parallel processing using {n_jobs} jobs... Check {FEATURES_DIR} for outputs.")

    # Parallelize the processing of each raw MESA file
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_mesa_file)(f) 
        for f in all_edf_files
    )
    
    # Filter out None results from skipped or failed files
    valid_results = [r for r in results if r is not None and r[0] is not None]

    if not valid_results:
        print("No files were processed successfully.")
        return
        
    # Create mapping from valid results (subject_id is same as recording_id here)
    mapping_list = []
    processed_recordings = set()
    for subj_id, rec_id in valid_results:
         if rec_id not in processed_recordings:
              # For MESA, subject_id and recording_id are the same in our context
              mapping_list.append({'subject_id': rec_id, 'recording_id': rec_id})
              processed_recordings.add(rec_id)

    if mapping_list:
        map_df = pd.DataFrame(mapping_list)
        map_df = map_df[['subject_id', 'recording_id']] # Ensure column order
        map_filename = 'subject_recording_mapping_mesa_psd.csv'
        map_df.to_csv(os.path.join(FEATURES_DIR, map_filename), index=False)
        print(f"Saved subject-recording mapping to {os.path.join(FEATURES_DIR, map_filename)}")
    else:
        print("No successful results to create mapping file.")

    print("MESA PSD feature extraction complete.")

if __name__ == "__main__":
    # Add basic check for MNE
    try: 
        import mne
    except ImportError:
        print("MNE Python not found. Please install it: pip install mne")
        exit()
    main() 