#!/usr/bin/env python
import os
import h5py
import numpy as np
import pandas as pd
from collections import defaultdict

def load_artifact_matrices(artifact_folder):
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

def inspect_artifact_structure(artifact_data):
    for file_name, data_dict in artifact_data.items():
        print(f"Artifact file: {file_name}")
        for key, value in data_dict.items():
            print(f"  Key: {key} - shape: {value.shape} - dtype: {value.dtype}")
        print("\n")

def analyze_artifact_matrix(matrix, good_threshold=0.9):
    if matrix.shape[0] > matrix.shape[1]:
        matrix = matrix.T

    n_channels, n_epochs = matrix.shape
    channel_stats = []
    for ch in range(n_channels):
        num_good = np.sum(matrix[ch] == 1)
        fraction_good = num_good / n_epochs
        quality = "good" if fraction_good >= good_threshold else "poor"
        channel_stats.append((ch, fraction_good, quality))
    
    bad_epoch_indices = [ep for ep in range(n_epochs) if np.any(matrix[:, ep] != 1)]
    
    return channel_stats, bad_epoch_indices

def main():
    ARTIFACT_FOLDER = '/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/ANPHY-Sleep/artifact_matrix'
    artifact_data = load_artifact_matrices(ARTIFACT_FOLDER)
    
    print("Inspecting artifact matrix structure...\n")
    inspect_artifact_structure(artifact_data)
    
    summary_rows = []
    subject_bad_electrodes = defaultdict(list)
    subjects_with_all_bad_electrodes = 0  # Counter for subjects with all bad electrodes
    bad_subjects = []  # List to store (index, filename) of subjects with all bad electrodes

    for idx, (file_name, data_dict) in enumerate(artifact_data.items()):
        if "artifact_matrix" in data_dict:
            matrix = data_dict["artifact_matrix"]
        else:
            matrix = list(data_dict.values())[0]
        
        channel_stats, bad_epoch_indices = analyze_artifact_matrix(matrix, good_threshold=0.90)
        
        print(f"Results for {file_name}:")
        print("Channel stats (channel_index, fraction_artifact_free, quality):")
        for stat in channel_stats:
            print(stat)
        print(f"Total epochs: {matrix.shape[1]}, Number of bad epochs: {len(bad_epoch_indices)}")
        print("-" * 40)
        
        num_poor_channels = 0
        for ch, frac, quality in channel_stats:
            summary_rows.append({
                "File": file_name,
                "Channel": ch,
                "FractionArtifactFree": frac,
                "Quality": quality,
                "TotalEpochs": matrix.shape[1],
                "BadEpochs": len(bad_epoch_indices)
            })
            if quality == "poor":
                subject_bad_electrodes[file_name].append(ch)
                num_poor_channels += 1
        
        # Check if all electrodes for this subject are flagged as "poor"
        if num_poor_channels == len(channel_stats):
            subjects_with_all_bad_electrodes += 1
            bad_subjects.append((idx, file_name))  # Store index and filename
    
    summary_df = pd.DataFrame(summary_rows)
    output_csv = os.path.join(ARTIFACT_FOLDER, "artifact_summary.csv")
    summary_df.to_csv(output_csv, index=False)
    print(f"Artifact summary saved to {output_csv}\n")
    
    print("\nSummary of Poor Electrodes Per Subject:")
    for subject_file, bad_channels in subject_bad_electrodes.items():
        bad_channels_sorted = sorted(bad_channels)
        print(f"Subject {subject_file}: Total Poor Electrodes = {len(bad_channels_sorted)}; Indices: {bad_channels_sorted}")
    
    # Print count and indices of subjects with all bad electrodes
    print(f"\nNumber of subjects with all electrodes flagged as 'poor': {subjects_with_all_bad_electrodes}")
    if subjects_with_all_bad_electrodes > 0:
        print("Indices and filenames of these subjects:")
        for idx, filename in bad_subjects:
            print(f"  Index: {idx}, Filename: {filename}")

if __name__ == "__main__":
    main()
