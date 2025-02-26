import os
import numpy as np

def main():
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
            # Load the full signal from the EDF.
            full_signals = loader.load_edf_signals(edf_path)
            metadata = loader.load_edf_metadata(edf_path)
            fs = metadata["sample_rates"][0]
            # Determine the sampling rate to use.
            current_fs = fs if args.original_fs is None else args.original_fs
            # Segment the full signal into epochs
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
                epoch = full_signals[:, start_idx:end_idx]  # (n_channels, samples_per_epoch)
                epochs.append(epoch)
            # eeg_data now has shape (n_epochs, n_channels, samples_per_epoch)
            eeg_data = np.array(epochs)
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