# Choose the primary ones you want to process (e.g., EEG 1 for Fpz-Cz equivalent, EOG L/R)
# Use the exact names found in the MESA EDF files
CHANNELS_TO_LOAD = ["EEG1", "EOG-L", "EOG-R"] 
TARGET_SFREQ = 256.0 # MESA typically higher sampling rate
LOW_FREQ = 0.5

def extract_psd_features(data_chunk, fs=FS):
    """Extracts PSD bandpower features from one epoch for all channels."""
    n_channels, _ = data_chunk.shape
    # Adjust prefixes based on the actual channels in CHANNELS_TO_LOAD
    # Assuming order matches CHANNELS_TO_LOAD: EEG1, EOG-L, EOG-R
    channel_prefixes = ['eeg1', 'eogL', 'eogR'] 
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
        "gamma": (30, 50)
    }
    for i in range(n_channels):
        # Ensure we don't go out of bounds if fewer prefixes than channels
        if i >= len(channel_prefixes): 
            prefix = f'ch{i}' # Generic prefix if more channels than expected
        else:
            prefix = channel_prefixes[i]
        
        channel_data = data_chunk[i, :]
        for bname, (lowf, highf) in bands.items():
            # ... existing code ...
            # ... existing code ...