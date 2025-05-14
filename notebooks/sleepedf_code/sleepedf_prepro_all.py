import os, glob, numpy as np, mne, pandas as pd
from joblib import Parallel, delayed
import mne

 # Example using absolute path (adjust to your actual path)
PROJECT_ROOT = "/content/drive/MyDrive/Stat483-GOALZ/4830_project/sleepedf_data"
BASE_DIR = os.path.join(PROJECT_ROOT, "sleep-edf-database-expanded-1.0.0")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'completed_psd_sleep_edf')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# USE_MULTIPLE_CHANNELS = True
# if USE_MULTIPLE_CHANNELS:
#     CHANNELS_TO_LOAD = ["EEG Fpz-Cz", "EOG horizontal"]
# else:
#     CHANNELS_TO_LOAD = ["EEG Fpz-Cz"]

TARGET_SFREQ = 50
LOW_FREQ = 0.5
HIGH_FREQ = 30.0
EPOCH_LENGTH = 30.0
SEQ_LENGTH = 20
SEQ_STRIDE = 10

ANNOTATION_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4
}

def process_record(psg_path, hyp_path, target_sfreq, low_freq, high_freq, epoch_length):
    # Load all channels
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)

    # Pick only EEG, EOG, and EMG channels (exclude respiration, rectal temp, event marker, etc.)
    picks = mne.pick_types(raw.info, eeg=True, eog=True, emg=True)

    # Filter relevant channels
    raw.filter(l_freq=low_freq, h_freq=high_freq, picks=picks, verbose=False)

    # Resample if needed
    if raw.info['sfreq'] != target_sfreq:
        raw.resample(target_sfreq, npad="auto", verbose=False)

    # Apply annotations from hypnogram
    ann = mne.read_annotations(hyp_path)
    raw.set_annotations(ann, emit_warning=False)

    # Extract epochs using annotation-based events
    events, _ = mne.events_from_annotations(raw, event_id=ANNOTATION_MAP, chunk_duration=epoch_length)
    tmin = 0.0
    tmax = epoch_length - 1 / raw.info['sfreq']
    epochs = mne.Epochs(raw, events=events, event_id=ANNOTATION_MAP, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False)

    data = epochs.get_data()          # shape: (n_epochs, n_channels, n_samples)
    labels = epochs.events[:, -1]     # integer labels
    ch_names = [raw.ch_names[i] for i in picks]

    # Z-score normalize each channel
    for ch in range(data.shape[1]):
        m = np.mean(data[:, ch, :])
        s = np.std(data[:, ch, :]) if np.std(data[:, ch, :]) != 0 else 1.0
        data[:, ch, :] = (data[:, ch, :] - m) / s

    return data, labels, ch_names

def create_sequences(data, labels, seq_length, seq_stride):
    n_epochs = data.shape[0]
    sequences, seq_labels = [], []
    for start in range(0, n_epochs - seq_length + 1, seq_stride):
        sequences.append(data[start:start+seq_length])
        seq_labels.append(labels[start:start+seq_length])
    return np.array(sequences), np.array(seq_labels)

def find_hypnogram(psg_file):
    # Extract subject ID (assumes first 6 characters, e.g., "SC4001")
    subject_id = os.path.basename(psg_file)[:6]
    dir_path = os.path.dirname(psg_file)
    pattern = os.path.join(dir_path, f"{subject_id}*Hypnogram.edf")
    hyp_files = glob.glob(pattern)
    if len(hyp_files) == 1:
        return hyp_files[0]
    elif len(hyp_files) > 1:
        return hyp_files[0]  # or choose based on additional rules if needed
    else:
        return None

def process_and_save(psg_file, output_dir):
    hyp_file = find_hypnogram(psg_file)
    if not hyp_file:
        print(f"Hypnogram not found for {psg_file}, skipping.")
        return
    try:
        data, labels, ch_names = process_record(psg_file, hyp_file,
                                                TARGET_SFREQ, LOW_FREQ, HIGH_FREQ, EPOCH_LENGTH)
    except Exception as e:
        print(f"Error processing {psg_file}: {e}")
        return

    rec_id = os.path.basename(psg_file).replace('-PSG.edf', '')

    # Save epoch-level data with channel names
    np.savez_compressed(os.path.join(output_dir, f"{rec_id}_epochs.npz"),
                        data=data.astype('float32'),
                        labels=labels.astype('int8'),
                        ch_names=np.array(ch_names))

    # Save sequence-level data
    sequences, seq_labels = create_sequences(data, labels, SEQ_LENGTH, SEQ_STRIDE)
    np.savez_compressed(os.path.join(output_dir, f"{rec_id}_sequences.npz"),
                        sequences=sequences.astype('float32'),
                        seq_labels=seq_labels.astype('int8'))

    print(f"Processed {rec_id}: epochs {data.shape[0]}, sequences {sequences.shape[0]}, channels: {ch_names}")

def main():
    psg_files = glob.glob(os.path.join(BASE_DIR, '**', '*-PSG.edf'), recursive=True)
    print(f"Found {len(psg_files)} PSG files.")
    Parallel(n_jobs=2)(delayed(process_and_save)(f, OUTPUT_DIR) for f in psg_files)

if __name__ == '__main__':
    main()

import numpy as np
import os

# Path to the specific file
fpath = '/content/drive/MyDrive/Stat483-GOALZ/4830_project/sleepedf_data/completed_psd_sleep_edf/ST7111J0_epochs.npz'

# Check if file exists
if not os.path.exists(fpath):
    print(f"File does not exist: {fpath}")
else:
    try:
        data = np.load(fpath, allow_pickle=True)
        ch_count = data['data'].shape[1]
        print(f"File loaded successfully with {ch_count} channels.")

        if 'ch_names' in data:
            ch_names = data['ch_names']
            print("📡 Channel names:")
            for i, ch in enumerate(ch_names):
                print(f"  {i}: {ch}")
        else:
            print("Channel names not found in file.")
            print("Showing channel indices instead:")
            for i in range(ch_count):
                print(f"  Channel {i}")
    except Exception as e:
        print(f"Failed to load {fpath}: {e}")

# Path to your output directory where *_epochs.npz files are saved
OUTPUT_DIR = '/content/drive/MyDrive/Stat483-GOALZ/4830_project/sleepedf_data/complete_psd_sleep_edf'

# Loop through all *_epochs.npz files and check channel count
channel_counts = []
for fname in os.listdir(OUTPUT_DIR):
    if fname.endswith('_epochs.npz'):
        fpath = os.path.join(OUTPUT_DIR, fname)
        data = np.load(fpath)
        ch_count = data['data'].shape[1]
        channel_counts.append((fname, ch_count))

# Print results
for fname, count in channel_counts:
    print(f"{fname}: {count} channels")

# Optionally assert that all files have the same number of channels
unique_counts = set(count for _, count in channel_counts)
if len(unique_counts) == 1:
    print(f"✅ All files have {unique_counts.pop()} channels.")
else:
    print("⚠️ Channel count mismatch across files:")
    for fname, count in channel_counts:
        print(f"  {fname}: {count}")

OUTPUT_DIR = '../../data/psd_sleep_edf'
rec_id = 'ST7121J0'

# Load epoch-level data
epoch_file = os.path.join(OUTPUT_DIR, f"{rec_id}_epochs.npz")
epoch_data = np.load(epoch_file)
data = epoch_data['data']      # shape: (n_epochs, n_channels, n_samples)
labels = epoch_data['labels']  # shape: (n_epochs,)

print(f"Epoch-level data shape: {data.shape}")
print(f"Epoch-level labels shape: {labels.shape}")
print("First 10 labels:", labels[:100])

# Load sequence-level data
seq_file = os.path.join(OUTPUT_DIR, f"{rec_id}_sequences.npz")
seq_data = np.load(seq_file)
sequences = seq_data['sequences']       # shape: (n_sequences, SEQ_LENGTH, n_channels, n_samples)
seq_labels = seq_data['seq_labels']     # shape: (n_sequences, SEQ_LENGTH)

print(f"Sequence-level data shape: {sequences.shape}")
print(f"Sequence-level labels shape: {seq_labels.shape}")
print("First sequence labels:", seq_labels[0])

hyp_path = '/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4121EC-Hypnogram.edf'
ann = mne.read_annotations(hyp_path)
print("Annotation descriptions:", ann.description)