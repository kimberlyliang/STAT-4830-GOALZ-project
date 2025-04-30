#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import glob
import numpy as np
import mne
import xml.etree.ElementTree as ET
from scipy.signal import butter, filtfilt, iirnotch, welch
from joblib import Parallel, delayed


# In[8]:


# --- Dynamically Determine Project Root and Set Paths ---
# Get the absolute path of the directory containing this script (notebooks/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the project root directory (parent of notebooks/)
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Define paths relative to the project root
BASE_DIR = os.path.join(project_root, 'data', 'MESA')
EDF_DIR = os.path.join(BASE_DIR, 'edfs')
ANNOTATION_DIR = os.path.join(BASE_DIR, 'mesa_200_annotations-events-nssr')
# Define OUTPUT_DIR relative to project root as well (adjust if needed)
OUTPUT_DIR = os.path.join(project_root, 'data', 'new_processed_mesa')
# --- End Path Configuration ---

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# In[4]:


print(f"DEBUG: Using Base Directory: {BASE_DIR}") # Debug print
print(f"DEBUG: Using EDF Directory: {EDF_DIR}") # Debug print
print(f"DEBUG: Using Annotation Directory: {ANNOTATION_DIR}") # Debug print

# --- Locate all EDF files across multiple subdirectories AND the main EDF dir ---
# List of directory names containing EDFs under BASE_DIR
edf_subdir_names = ['mesa_200_subset_edfs'] + [f'mesa_200_subset_edfs {i}' for i in range(2, 19)]
all_edf_files = [] # This list will be used by main()

# First, get files directly in EDF_DIR
print(f"DEBUG: Searching in {EDF_DIR} for *.edf / *.EDF") # Debug print
edf_files_in_main_dir = glob.glob(os.path.join(EDF_DIR, '*.edf')) + \
                          glob.glob(os.path.join(EDF_DIR, '*.EDF'))
print(f"DEBUG: Found {len(edf_files_in_main_dir)} files directly in {EDF_DIR}") # Debug print
all_edf_files.extend(edf_files_in_main_dir)

# Then, get files from subdirectories
for subdir_name in edf_subdir_names:
    dir_path = os.path.join(EDF_DIR, subdir_name)
    print(f"DEBUG: Checking directory {dir_path}") # Debug print
    if os.path.isdir(dir_path):
        print(f"DEBUG: Searching in {dir_path} for *.edf / *.EDF") # Debug print
        edf_files_in_subdir = glob.glob(os.path.join(dir_path, '*.edf')) + \
                              glob.glob(os.path.join(dir_path, '*.EDF'))
        print(f"DEBUG: Found {len(edf_files_in_subdir)} files in {dir_path}") # Debug print
        all_edf_files.extend(edf_files_in_subdir)
    else:
        print(f"DEBUG: Directory not found: {dir_path}") # Keep this debug print
        pass

# Remove duplicates if any file exists in multiple lists and sort
all_edf_files = sorted(list(set(all_edf_files)))

print(f"Found {len(all_edf_files)} EDF files to process.") # Use this list
# --- End of EDF file location ---


# In[9]:


# List of already-processed PSG recording IDs (without .edf)
PROCESSED_ALREADY = [
]


# In[10]:


# EEG 1 is the same as EEG Fz-Cz
# EEG 2 is the same as EEG Cz-Oz
# EEG 3 is the same as EEG C4 M1
# Kim's source: https://sleepdata.org/datasets/mesa/pages/equipment/montage-and-sampling-rate-information.md 

CHANNELS_TO_LOAD = ["EEG1", "EOG-L", "EOG-R"]  # Adjust based on available channels
TARGET_SFREQ = 256.0
LOW_FREQ = 0.5
HIGH_FREQ = 30.0
EPOCH_LENGTH = 30.0
SEQ_LENGTH = 20
SEQ_STRIDE = 10

# Sleep stage mapping based on NSRR annotations
ANNOTATION_MAP = {
    "0": 0,  # Wake
    "1": 1,  # N1
    "2": 2,  # N2
    "3": 3,  # N3
    "5": 4   # REM
}

def auto_notch_filter(signal, fs, center_guess=60.0, search_range=1.0, quality=30.0):
    freqs, psd = welch(signal, fs=fs, nperseg=2048)
    idx = np.where((freqs >= center_guess - search_range) & (freqs <= center_guess + search_range))[0]
    if len(idx) == 0:
        return signal
    true_freq = freqs[idx[np.argmax(psd[idx])]]
    w0 = true_freq / (0.5 * fs)
    b, a = iirnotch(w0, quality)
    return filtfilt(b, a, signal)

def parse_xml_annotations(xml_path):
    """Parse the XML file to extract sleep stage annotations."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    stages = []
    for event in root.findall(".//ScoredEvent"):
        if event.find("EventType").text == "Stages|Stages":
            concept = event.find("EventConcept").text.strip().split("|")[-1]  # Get the last part of the stage name
            if concept not in ANNOTATION_MAP:
                continue
            stage_id = ANNOTATION_MAP[concept]
            start = float(event.find("Start").text)
            duration = float(event.find("Duration").text)

            # Split duration into multiple 30-second epochs
            num_epochs = int(duration // 30)
            for i in range(num_epochs):
                epoch_start = start + i * 30
                stages.append((epoch_start, 30.0, stage_id))
    
    return stages

def process_record(psg_path, xml_path, channels, target_sfreq, low_freq, high_freq, epoch_length):
    """Process a single PSG file with its corresponding XML annotations."""
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)

    # --- Select only the desired channels ---
    try:
        # Check if all desired channels exist before trying to pick
        missing_channels = list(set(channels) - set(raw.ch_names))
        if missing_channels:
            raise ValueError(f"Channels {missing_channels} not found in {psg_path}")
        raw.pick_channels(channels, ordered=True)
    except ValueError as e:
        # Reraise the error with more context or handle as needed
        raise ValueError(f"Error picking channels for {psg_path}: {e}") from e
    # --- End channel selection ---

    if raw.info['sfreq'] != target_sfreq:
        raw.resample(target_sfreq, npad="auto", verbose=False)

    # --- Filter the selected channels ---
    raw.filter(l_freq=low_freq, h_freq=high_freq, picks=None, verbose=False) # picks=None applies to all channels in raw
    # --- End filtering ---


    # Apply notch to each picked channel individually
    for ch_idx in range(len(raw.ch_names)):
        signal = raw.get_data(picks=[ch_idx])[0]
        filtered = auto_notch_filter(signal, fs=raw.info['sfreq'], center_guess=60.0)
        raw._data[ch_idx] = filtered

    annotations = parse_xml_annotations(xml_path)

    # Convert annotations to MNE-compatible events
    events = []
    for start, duration, stage_id in annotations:
        sample = int(start * raw.info['sfreq'])
        events.append([sample, 0, stage_id])

    events = np.array(events)
    tmin = 0.0
    tmax = epoch_length - 1 / raw.info['sfreq']
    # Epochs will now only contain the selected channels
    epochs = mne.Epochs(raw, events=events, event_id=ANNOTATION_MAP, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False)
    data = epochs.get_data()
    labels = epochs.events[:, -1]

    # Standardize each channel
    for ch in range(data.shape[1]):
        m = np.mean(data[:, ch, :])
        s = np.std(data[:, ch, :]) if np.std(data[:, ch, :]) != 0 else 1.0
        data[:, ch, :] = (data[:, ch, :] - m) / s
    # Return the processed data, labels, and the names of the channels ACTUALLY processed
    return data, labels, raw.ch_names

def create_sequences(data, labels, seq_length, seq_stride):
    """Create sequences of epochs."""
    n_epochs = data.shape[0]
    sequences, seq_labels = [], []
    for start in range(0, n_epochs - seq_length + 1, seq_stride):
        sequences.append(data[start:start+seq_length])
        seq_labels.append(labels[start:start+seq_length])
    return np.array(sequences), np.array(seq_labels)

def process_and_save(psg_file, output_dir, channels):
    """Process and save data for a single PSG file."""
    rec_id = os.path.basename(psg_file).replace(".edf", "")
    
    if rec_id in PROCESSED_ALREADY:
        print(f"Skipping {rec_id}: already processed.")
        return

    xml_file = os.path.join(ANNOTATION_DIR, rec_id + "-nsrr.xml")
    if not os.path.exists(xml_file):
        print(f"Annotation file not found for {psg_file}, skipping.")
        return
    try:
        data, labels, ch_names = process_record(psg_file, xml_file, CHANNELS_TO_LOAD,
                                                TARGET_SFREQ, LOW_FREQ, HIGH_FREQ, EPOCH_LENGTH)
        
        if data.shape[0] == 0 or labels.shape[0] == 0:
            print(f"No valid epochs for {psg_file}, skipping.")
            return

    except Exception as e:
        print(f"Error processing {psg_file}: {e}")
        return

    try:
        np.savez_compressed(os.path.join(output_dir, f"{rec_id}_epochs.npz"),
                            data=data.astype('float32'), labels=labels.astype('int8'))
        sequences, seq_labels = create_sequences(data, labels, SEQ_LENGTH, SEQ_STRIDE)
        np.savez_compressed(os.path.join(output_dir, f"{rec_id}_sequences.npz"),
                            sequences=sequences.astype('float32'), seq_labels=seq_labels.astype('int8'))
        print(f"Processed {rec_id}: epochs {data.shape[0]}, sequences {sequences.shape[0]}, channels: {ch_names}")
    except Exception as e:
        print(f"Error saving {psg_file}: {e}")
        return


# In[11]:


def main():
    # The comprehensive file search happens earlier and populates `all_edf_files`
    # We no longer need to search here.
    # psg_files = glob.glob(os.path.join(EDF_DIR, "*.edf")) # REMOVED
    # print(f"Found {len(psg_files)} PSG files.") # REMOVED
    
    # Use the list populated earlier
    if not all_edf_files:
        print("No EDF files found by the initial search. Exiting.")
        return
        
    print(f"\nStarting parallel processing for {len(all_edf_files)} files...")
    Parallel(n_jobs=2)(delayed(process_and_save)(f, OUTPUT_DIR, CHANNELS_TO_LOAD) for f in all_edf_files)

if __name__ == '__main__':
    main() 