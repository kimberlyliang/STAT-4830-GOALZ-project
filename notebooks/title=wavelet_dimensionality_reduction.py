#!/usr/bin/env python3
"""
Wavelet-based Dimensionality Reduction for EEG Data

This script loads an EEG dataset (in EDF, npy or npz format),
computes wavelet energy features for each epoch, and then performs
dimensionality reduction with PCA.

Usage:
    python wavelet_dimensionality_reduction.py --input <input_path> --output <output_path> [other options]

Notes:
- For sleep staging (where you want to capture details up to 40 Hz reliably),
  we now sample at 500 Hz. (This matches the data loaded by extract_anphy_sleep_data.)
- This script uses PyWavelets (pywt) for the discrete wavelet transform and
  scikit-learn's PCA for dimensionality reduction.
- When using EDF files, the entire recording is segmented into non-overlapping epochs,
  where each epoch has a duration specified by --duration (in seconds).
"""

import numpy as np
import pywt
from sklearn.decomposition import PCA
import argparse
import os
from scipy.signal import resample

def extract_wavelet_energy(epoch_data, wavelet='db4', level=4):
    """
    Compute the discrete wavelet transform of a 1D EEG signal and extract energy features.

    Parameters:
        epoch_data (np.ndarray): 1D array containing an EEG epoch.
        wavelet (str): Wavelet type (default 'db4').
        level (int): Decomposition level (default 4).

    Returns:
        energy_features (list): Energy of the approximation and detail coefficients.
    """
    coeffs = pywt.wavedec(epoch_data, wavelet=wavelet, level=level)
    energy_features = [np.sum(np.square(c)) for c in coeffs]
    return energy_features

def process_epochs(eeg_data, wavelet='db4', level=4):
    """
    Extract wavelet energy features from EEG epochs.

    Assumes input eeg_data is either:
      - 2D: shape (n_epochs, n_samples) for single-channel data, or
      - 3D: shape (n_epochs, n_channels, n_samples) for multi-channel data.

    Returns:
        features (np.ndarray): Feature matrix of shape (n_epochs, feature_dim)
    """
    n_epochs = eeg_data.shape[0]
    if eeg_data.ndim == 3:
        n_channels = eeg_data.shape[1]
        features = []
        for i in range(n_epochs):
            epoch_features = []
            for ch in range(n_channels):
                feats = extract_wavelet_energy(eeg_data[i, ch, :], wavelet=wavelet, level=level)
                epoch_features.extend(feats)
            features.append(epoch_features)
        features = np.array(features)
    elif eeg_data.ndim == 2:
        # Single channel: (n_epochs, n_samples)
        features = np.array([extract_wavelet_energy(epoch, wavelet, level) for epoch in eeg_data])
    else:
        raise ValueError("EEG data must be 2D or 3D")
    return features

def reduce_features(features, n_components=0.95):
    """
    Reduce the dimensionality of the features matrix using PCA.

    Parameters:
        features (np.ndarray): Feature matrix shape (n_samples, n_features).
        n_components (float or int): If float (between 0 and 1), retain enough components to explain that fraction of variance.

    Returns:
        reduced_features (np.ndarray): The PCA-transformed feature matrix.
        pca_model (PCA): The fitted PCA object.
    """
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features, pca

def resample_eeg_data(eeg_data, original_fs, target_fs=500):
    """
    Resample EEG data to the target sampling rate.

    Parameters:
        eeg_data (np.ndarray): EEG data array, either 2D (n_epochs, n_samples) or 3D (n_epochs, n_channels, n_samples).
        original_fs (float): The original sampling rate.
        target_fs (float): The target sampling rate (default: 500).

    Returns:
        np.ndarray: The resampled EEG data.
    """
    if eeg_data.ndim == 2:
        n_epochs, n_samples = eeg_data.shape
        new_n_samples = int(n_samples * target_fs / original_fs)
        resampled = np.array([resample(eeg_data[i, :], new_n_samples) for i in range(n_epochs)])
    elif eeg_data.ndim == 3:
        n_epochs, n_channels, n_samples = eeg_data.shape
        new_n_samples = int(n_samples * target_fs / original_fs)
        resampled = np.empty((n_epochs, n_channels, new_n_samples))
        for i in range(n_epochs):
            for ch in range(n_channels):
                resampled[i, ch, :] = resample(eeg_data[i, ch, :], new_n_samples)
    else:
        raise ValueError("EEG data must be either a 2D or 3D array")
    return resampled

def main():
    parser = argparse.ArgumentParser(description="Wavelet-based Dimensionality Reduction for EEG Data")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the input EEG data file/folder (EDF, npy, or npz).")
    parser.add_argument("--output", type=str, required=True,
                        help="Path for saving the reduced features (npy format).")
    parser.add_argument("--wavelet", type=str, default="db4",
                        help="Wavelet type to use (default: 'db4').")
    parser.add_argument("--level", type=int, default=4,
                        help="Decomposition level (default: 4).")
    parser.add_argument("--n_components", type=float, default=0.95,
                        help="Number of PCA components (if float, fraction of variance; default 0.95).")
    parser.add_argument("--original_fs", type=float, default=None,
                        help="Original sampling rate of EEG data. If provided and different from 500, data will be resampled to 500 Hz.")
    parser.add_argument("--start_time", type=float, default=0.0,
                        help="Start time (in seconds) for EDF epoch extraction (used only in single epoch mode).")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Duration (in seconds) for each epoch segment (EDF only).")
    args = parser.parse_args()

    # Convert input/output paths to absolute paths for consistency.
    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output)

    # New behavior: if args.input is a directory, loop over each subject subfolder.
    if os.path.isdir(args.input):
        subject_folders = [d for d in os.listdir(args.input)
                           if os.path.isdir(os.path.join(args.input, d))]
        for subject in sorted(subject_folders):
            subject_folder = os.path.join(args.input, subject)
            # Look for an EDF file inside the subject folder.
            edf_files = [f for f in os.listdir(subject_folder) if f.lower().endswith(".edf")]
            if not edf_files:
                print(f"[WARNING] No EDF file found in {subject_folder}. Skipping...")
                continue
            edf_path = os.path.join(subject_folder, edf_files[0])
            print(f"Processing subject {subject}: {edf_path}")
            from anp_data_loader import ANPDataLoader
            loader = ANPDataLoader(subject_folder)
            # Load the full signal from the EDF and its metadata.
            full_signals = loader.load_edf_signals(edf_path)
            metadata = loader.load_edf_metadata(edf_path)
            fs = metadata["sample_rates"][0]
            # Use provided original_fs or the one from metadata.
            current_fs = fs if args.original_fs is None else args.original_fs
            # Segment the full signal into epochs.
            epoch_duration = args.duration  # in seconds
            samples_per_epoch = int(epoch_duration * current_fs)
            total_samples = full_signals.shape[1]
            n_epochs = total_samples // samples_per_epoch
            if n_epochs == 0:
                print(f"[WARNING] The recording for subject {subject} is shorter than one epoch. Skipping...")
                continue
            epochs = []
            for i in range(n_epochs):
                start_idx = i * samples_per_epoch
                end_idx = start_idx + samples_per_epoch
                epoch = full_signals[:, start_idx:end_idx]  # shape: (n_channels, samples_per_epoch)
                epochs.append(epoch)
            # eeg_data now has shape (n_epochs, n_channels, samples_per_epoch)
            eeg_data = np.array(epochs)

            # Resample data to target sampling rate of 500 Hz if necessary.
            target_fs = 500
            if current_fs != target_fs:
                print(f"Resampling subject {subject} from {current_fs} Hz to {target_fs} Hz...")
                eeg_data = resample_eeg_data(eeg_data, current_fs, target_fs)
                print(f"Data after resampling for {subject} has shape: {eeg_data.shape}")
            else:
                print(f"Data assumed to be sampled at {target_fs} Hz for subject {subject}.")

            # Extract wavelet energy features.
            features = process_epochs(eeg_data, wavelet=args.wavelet, level=args.level)
            print(f"Extracted wavelet energy features for {subject} with shape: {features.shape}")

            # Perform PCA-based dimensionality reduction.
            reduced_features, pca_model = reduce_features(features, n_components=args.n_components)
            print(f"Reduced feature shape for {subject}: {reduced_features.shape}")

            # Save the output in the specified output directory.
            os.makedirs(args.output, exist_ok=True)
            out_file = os.path.join(args.output, f"{subject}_reduced_features.npy")
            np.save(out_file, reduced_features)
            print(f"Saved reduced features for subject {subject} to {out_file}")
        return  # End processing when in directory mode.
    else:
        # Single file mode: Process the entire signal and segment it into epochs.
        if args.input.endswith(".edf"):
            from anp_data_loader import ANPDataLoader
            loader = ANPDataLoader(os.path.dirname(args.input))
            full_signals = loader.load_edf_signals(args.input)
            metadata = loader.load_edf_metadata(args.input)
            fs = metadata["sample_rates"][0]
            current_fs = fs if args.original_fs is None else args.original_fs
            epoch_duration = args.duration
            samples_per_epoch = int(epoch_duration * current_fs)
            total_samples = full_signals.shape[1]
            n_epochs = total_samples // samples_per_epoch
            if n_epochs == 0:
                raise ValueError("Recording is shorter than one epoch.")
            epochs = []
            for i in range(n_epochs):
                start_idx = i * samples_per_epoch
                end_idx = start_idx + samples_per_epoch
                epoch = full_signals[:, start_idx:end_idx]
                epochs.append(epoch)
            eeg_data = np.array(epochs)
        elif args.input.endswith(".npz"):
            data = np.load(args.input)
            if 'data' in data:
                eeg_data = data['data']
            else:
                eeg_data = data[data.files[0]]
        elif args.input.endswith(".npy"):
            eeg_data = np.load(args.input)
        else:
            raise ValueError("Unsupported file format. Use edf, npy, or npz.")

    print(f"Loaded EEG data with shape: {eeg_data.shape}")

    # Resample data to target sampling rate of 500 Hz if necessary.
    target_fs = 500
    if args.original_fs is not None and args.original_fs != target_fs:
        print(f"Resampling data from {args.original_fs} Hz to {target_fs} Hz...")
        eeg_data = resample_eeg_data(eeg_data, args.original_fs, target_fs)
        print(f"Data after resampling has shape: {eeg_data.shape}")
    else:
        print(f"Data assumed to be sampled at {target_fs} Hz.")

    # Extract wavelet energy features (from data sampled at 500 Hz).
    features = process_epochs(eeg_data, wavelet=args.wavelet, level=args.level)
    print(f"Extracted wavelet energy features with shape: {features.shape}")

    # Perform PCA-based dimensionality reduction.
    reduced_features, pca_model = reduce_features(features, n_components=args.n_components)
    print(f"Reduced feature shape: {reduced_features.shape}")

    # Save reduced features.
    np.save(args.output, reduced_features)
    print(f"Saved reduced features to {args.output}")

if __name__ == "__main__":
    main() 