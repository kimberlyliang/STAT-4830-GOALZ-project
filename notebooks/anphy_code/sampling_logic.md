process_single_subject_group.py

This script takes two command-line arguments: the subject folder name (e.g. EPCTL01) and the electrode group index (0 through 9).
It loads the CSV and EDF files for that subject, samples one time index per class for each stage (W, N1, N2, N3, R), extracts a 2-second window around that time (using ±1 sec), applies a 60 Hz notch filter (Q=30), a fourth-order lowpass filter with cutoff 90 Hz, then downsamples the window from the original rate (e.g. 1000 Hz) to 200 Hz (yielding 400 samples per window).
It saves the results in a pickle file named with the subject and the group number (e.g. "epctl01_all_stages_group1.pkl"). Each key in the dictionary encodes the subject, stage, window number, and electrode label, and the value is a dictionary containing the processed window and its center time index.

process_all.py

This driver script loops over all 29 subjects and over all 10 electrode groups.
For each subject and each group, it calls process_single_subject_group.py in a separate process using subprocess.run().
Running each combination in its own process ensures that memory is released between runs (equivalent to a kernel restart) and prevents cumulative memory buildup that might crash your notebook kernel.