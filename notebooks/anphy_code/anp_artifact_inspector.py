import numpy as np
import os
import re

def extract_subject_number(filename):
    """
    Extract numeric subject ID from filename.
    Assumes filename contains a substring like 'EPCTLXX' (e.g. EPCTL01, EPCTL14)
    """
    m = re.search(r'EPCTL(\d+)', filename, re.IGNORECASE)
    return int(m.group(1)) if m else float('inf')

def analyze_artifact_matrix(matrix, good_threshold=0.97):
    """
    Given an artifact matrix (assumed to be binary: 1 indicates artifact-free and
    0 indicates an artifact), computes, for each channel, the fraction of epochs that are
    artifact-free, and flags channels as 'poor' if this fraction is below the specified good_threshold.

    Also flags epochs as 'bad' if any channel shows an artifact (i.e. if any channel is 0).

    Args:
        matrix (np.ndarray): Assumed shape (n_channels, n_epochs) or (n_epochs, n_channels).
        good_threshold (float): Minimum required fraction of artifact-free epochs.

    Returns:
        channel_stats (list): List of tuples (channel_index, fraction_artifact_free, quality_label)
        bad_epoch_indices (list): List of epoch indices flagged as bad.
    """
    # Verify matrix structure. In our dataset, the number of channels is far lower than the number of epochs.
    # If the matrix shape has more rows than columns, it's likely transposed; so we transpose it.
    if matrix.shape[0] > matrix.shape[1]:
         matrix = matrix.T

    n_channels, n_epochs = matrix.shape

    # Compute per-channel artifact-free fraction and quality.
    channel_stats = []
    for ch in range(n_channels):
         good_count = np.sum(matrix[ch] == 1)
         fraction_good = good_count / n_epochs
         quality = "good" if fraction_good >= good_threshold else "poor"
         channel_stats.append((ch, fraction_good, quality))

    # Epoch-level analysis: flag an epoch as "bad" if any channel is not artifact free.
    bad_epoch_indices = []
    for ep in range(n_epochs):
         if np.any(matrix[:, ep] == 0):
              bad_epoch_indices.append(ep)

    return channel_stats, bad_epoch_indices

def main():
    ARTIFACT_FOLDER = '/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/ANPHY-Sleep/artifact_matrix'
    artifact_data = load_artifact_matrices(ARTIFACT_FOLDER)
    
    # Iterate over subjects in sorted order by filename.
    summary_rows = []
    for file_name in sorted(artifact_data.keys(), key=extract_subject_number):
        data_dict = artifact_data[file_name]
        if "artifact_matrix" in data_dict:
            matrix = data_dict["artifact_matrix"]
        else:
            matrix = list(data_dict.values())[0]
        
        channel_stats, bad_epoch_indices = analyze_artifact_matrix(matrix, good_threshold=0.97)
        
        # Print out the total number of electrodes for this subject.
        print(f"Subject {file_name} has {len(channel_stats)} electrodes.")
        
        print(f"Results for {file_name}:")
        print("Channel stats (channel_index, fraction_artifact_free, quality):")
        for stat in channel_stats:
            print(stat)
        print(f"Total epochs: {matrix.shape[1]}, Number of bad epochs: {len(bad_epoch_indices)}")
        print("-" * 40)
        
        for ch, frac, quality in channel_stats:
            summary_rows.append({
                "File": file_name,
                "Channel": ch,
                "FractionArtifactFree": frac,
                "Quality": quality,
                "TotalEpochs": matrix.shape[1],
                "BadEpochs": len(bad_epoch_indices)
            })

def load_artifact_matrices(artifact_folder):
    artifact_files = sorted([f for f in os.listdir(artifact_folder) if f.endswith(".mat")], key=extract_subject_number) 