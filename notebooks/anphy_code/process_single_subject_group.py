#!/usr/bin/env python
import os, sys, glob, pickle, numpy as np, pandas as pd, pyedflib
from scipy.signal import butter, filtfilt, resample_poly, iirnotch
from config.config import DataConfig

def notch_filter(data, fs, freq=60, Q=30):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, data)

def lowpass_filter(data, fs, cutoff=90, order=4):
    nyq = fs / 2.0
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def downsample_window(data, fs, target_fs=200):
    return resample_poly(data, target_fs, fs)

def main(subject, group_index):
    data_dir = DataConfig.BASE_PATH
    subj_path = os.path.join(data_dir, subject)
    # Check for CSV and EDF files.
    csv_files = glob.glob(os.path.join(subj_path, "*.csv"))
    edf_files = glob.glob(os.path.join(subj_path, "*.edf"))
    if not csv_files or not edf_files:
        print(f"Missing CSV or EDF for {subject}")
        sys.exit(1)

    df = pd.read_csv(csv_files[0], index_col=0)
    # Extract epochs for all 5 classes.
    df_W = df[df["stage"]=="W"]
    df_N1 = df[df["stage"]=="N1"]
    df_N2 = df[df["stage"]=="N2"]
    df_N3 = df[df["stage"]=="N3"]
    df_R  = df[df["stage"]=="R"]

    n_per_stage = 1  # If k=2, then we sample 1 epoch per stage.
    if (len(df_W) < n_per_stage or len(df_N1) < n_per_stage or
        len(df_N2) < n_per_stage or len(df_N3) < n_per_stage or len(df_R) < n_per_stage):
        print(f"Not enough epochs for one stage in {subject}")
        sys.exit(1)

    # Fixed random_state for reproducibility (adjust seed if desired).
    sample_W  = df_W.sample(n=n_per_stage, random_state=0)["time_index"].values
    sample_N1 = df_N1.sample(n=n_per_stage, random_state=0)["time_index"].values
    sample_N2 = df_N2.sample(n=n_per_stage, random_state=0)["time_index"].values
    sample_N3 = df_N3.sample(n=n_per_stage, random_state=0)["time_index"].values
    sample_R  = df_R.sample(n=n_per_stage, random_state=0)["time_index"].values

    reader = pyedflib.EdfReader(edf_files[0])
    fs = reader.getSampleFrequency(0)  # e.g., 1000 Hz
    total_samples = reader.getNSamples()[0]
    n_channels = reader.signals_in_file
    signals = [reader.readSignal(i) for i in range(n_channels)]
    channel_labels = reader.getSignalLabels()
    reader.close()

    # Split 93 electrodes into 10 groups.
    electrode_indices = np.arange(93)
    electrode_groups = np.array_split(electrode_indices, 10)
    current_group = electrode_groups[group_index]

    window_sec = 2   # 2-second window (±1 sec)
    half_window_sec = window_sec / 2

    subject_windows = {}
    for i in current_group:
        if i >= len(channel_labels):
            continue
        ch_label = channel_labels[i]
        for stage, sample_times in zip(["W", "N1", "N2", "N3", "R"],
                                       [sample_W, sample_N1, sample_N2, sample_N3, sample_R]):
            for j, t in enumerate(sample_times):
                start = int((t - half_window_sec) * fs)
                end = int((t + half_window_sec) * fs)
                if start < 0 or end > total_samples:
                    continue
                win = signals[i][start:end]
                win = notch_filter(win, fs, freq=60, Q=30)
                win = lowpass_filter(win, fs, cutoff=90, order=4)
                ds_win = downsample_window(win, fs, target_fs=200)
                key = f"{subject.lower()}_{stage.lower()}_win_{j+1}_{ch_label.replace('-', '').replace(' ', '')}"
                subject_windows[key] = {"window": ds_win, "time_index": t}

    out_path = os.path.join(subj_path, f"{subject.lower()}_all_stages_group{group_index+1}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(subject_windows, f)
    print(f"Saved {len(subject_windows)} windows for {subject} (electrode group {group_index+1}) at {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_single_subject_group.py <SUBJECT_FOLDER> <GROUP_INDEX>")
        sys.exit(1)
    subject = sys.argv[1]
    group_index = int(sys.argv[2])
    try:
        main(subject, group_index)
    except Exception as e:
        print(f"Error processing {subject} group {group_index}: {e}")
        sys.exit(1)

# def notch_filter(data, fs, freq=60, Q=30):
#     b, a = iirnotch(freq, Q, fs)
#     return filtfilt(b, a, data)

# def lowpass_filter(data, fs, cutoff=90, order=4):
#     nyq = fs/2.0
#     normal_cutoff = cutoff/nyq
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     return filtfilt(b, a, data)

# def downsample_window(data, fs, target_fs=200):
#     return resample_poly(data, target_fs, fs)

# def main(subject, group_index):
#     data_dir = DataConfig.BASE_PATH
#     subj_path = os.path.join(data_dir, subject)
#     # Check for CSV and EDF files.
#     csv_files = glob.glob(os.path.join(subj_path, "*.csv"))
#     edf_files = glob.glob(os.path.join(subj_path, "*.edf"))
#     if not csv_files or not edf_files:
#         print(f"Missing CSV or EDF for {subject}")
#         sys.exit(1)

#     df = pd.read_csv(csv_files[0], index_col=0)
#     # Extract epochs for all 5 classes.
#     df_W = df[df["stage"]=="W"]
#     df_N1 = df[df["stage"]=="N1"]
#     df_N2 = df[df["stage"]=="N2"]
#     df_N3 = df[df["stage"]=="N3"]
#     df_R  = df[df["stage"]=="R"]

#     n_per_stage = 1  # If k=2, then we sample 1 epoch per stage.
#     if (len(df_W) < n_per_stage or len(df_N1) < n_per_stage or
#         len(df_N2) < n_per_stage or len(df_N3) < n_per_stage or len(df_R) < n_per_stage):
#         print(f"Not enough epochs for one stage in {subject}")
#         sys.exit(1)

#     # fixed random_state (could be varied per batch).
#     sample_W  = df_W.sample(n=n_per_stage, random_state=0)["time_index"].values
#     sample_N1 = df_N1.sample(n=n_per_stage, random_state=0)["time_index"].values
#     sample_N2 = df_N2.sample(n=n_per_stage, random_state=0)["time_index"].values
#     sample_N3 = df_N3.sample(n=n_per_stage, random_state=0)["time_index"].values
#     sample_R  = df_R.sample(n=n_per_stage, random_state=0)["time_index"].values

#     reader = pyedflib.EdfReader(edf_files[0])
#     fs = reader.getSampleFrequency(0)  # 1000 Hz
#     total_samples = reader.getNSamples()[0]
#     n_channels = reader.signals_in_file
#     signals = [reader.readSignal(i) for i in range(n_channels)]
#     channel_labels = reader.getSignalLabels()
#     reader.close()

#     # Split 93 electrodes into 10 groups.
#     electrode_indices = np.arange(93)
#     electrode_groups = np.array_split(electrode_indices, 10)
#     current_group = electrode_groups[group_index]

#     window_sec = 2   # 2-second window (±1 sec)
#     half_window_sec = window_sec / 2

#     subject_windows = {}
#     for i in current_group:
#         if i >= len(channel_labels):
#             continue
#         ch_label = channel_labels[i]
#         for stage, sample_times in zip(["W", "N1", "N2", "N3", "R"],
#                                        [sample_W, sample_N1, sample_N2, sample_N3, sample_R]):
#             for j, t in enumerate(sample_times):
#                 start = int((t - half_window_sec) * fs)
#                 end = int((t + half_window_sec) * fs)
#                 if start < 0 or end > total_samples:
#                     continue
#                 win = signals[i][start:end]
#                 win = notch_filter(win, fs, freq=60, Q=30)
#                 win = lowpass_filter(win, fs, cutoff=90, order=4)
#                 ds_win = downsample_window(win, fs, target_fs=200)
#                 key = f"{subject.lower()}_{stage.lower()}_win_{j+1}_{ch_label.replace('-', '').replace(' ', '')}"
#                 # Save processed window and its center time.
#                 subject_windows[key] = {"window": ds_win, "time_index": t}

#     out_path = os.path.join(subj_path, f"{subject.lower()}_all_stages_group{group_index+1}.pkl")
#     with open(out_path, "wb") as f:
#         pickle.dump(subject_windows, f)
#     print(f"Saved {len(subject_windows)} windows for {subject} (electrode group {group_index+1}) at {out_path}")

# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("Usage: python process_single_subject_group.py <SUBJECT_FOLDER> <GROUP_INDEX>")
#         sys.exit(1)
#     subject = sys.argv[1]
#     group_index = int(sys.argv[2])
#     main(subject, group_index)
