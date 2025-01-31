import os
import pyedflib
import numpy as np
import pandas as pd
import h5py

class ANPDataLoader:
    def __init__(self, base_path):
        self.base_path = base_path

    def load_edf_metadata(self, edf_path):
        with pyedflib.EdfReader(edf_path) as f:
            n_channels = f.signals_in_file
            labels = f.getSignalLabels()
            sample_rates = [f.getSampleFrequency(i) for i in range(n_channels)]
            duration = f.file_duration
        return {"n_channels": n_channels, "labels": labels, "sample_rates": sample_rates, "duration": duration}

    def load_edf_signals(self, edf_path):
        with pyedflib.EdfReader(edf_path) as f:
            n_channels = f.signals_in_file
            n_samples = f.getNSamples()[0]  # assumes same length for all channels
            signals = np.zeros((n_channels, n_samples))
            for i in range(n_channels):
                signals[i, :] = f.readSignal(i)
        return signals

    def load_edf_epoch(self, edf_path, start_time, duration):
        # Load only the required segment (epoch) from the EDF file.
        with pyedflib.EdfReader(edf_path) as f:
            fs = f.getSampleFrequency(0)
            start_idx = int(start_time * fs)
            n_samples = int(duration * fs)
            n_channels = f.signals_in_file
            epoch_signals = np.zeros((n_channels, n_samples))
            for i in range(n_channels):
                epoch_signals[i, :] = f.readSignal(i, start=start_idx, n=n_samples)
        return epoch_signals, fs

    def load_artifact_matrices(self, artifact_folder):
        artifact_files = [f for f in os.listdir(artifact_folder) if f.endswith(".mat")]
        artifact_data = {}
        for file_name in artifact_files:
            file_path = os.path.join(artifact_folder, file_name)
            try:
                with h5py.File(file_path, 'r') as mat_data:
                    artifact_data[file_name] = {key: np.array(mat_data[key]) for key in mat_data.keys()}
            except Exception as e:
                print("Error loading", file_name, ":", e)
        return artifact_data

    def load_annotations(self, subject_folder):
        # Assumes one .txt file per subject with columns: stage, start_time, duration.
        txt_files = [f for f in os.listdir(subject_folder) if f.endswith(".txt")]
        if txt_files:
            txt_path = os.path.join(subject_folder, txt_files[0])
            data = np.loadtxt(txt_path, dtype=str)
            return data
        else:
            return None

    def load_details_csv(self, csv_path):
        return pd.read_csv(csv_path)
