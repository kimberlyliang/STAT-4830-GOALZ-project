import os
import mne
import glob

# Define the extracted data path
extract_dir = "/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/ANPHY-Sleep"

# Dictionary to store loaded data
data_dict = {}

# Get a sorted list of EPCTLxx folders
folders = sorted(os.listdir(extract_dir))

# Iterate over each folder, ensuring the order is correct
for folder in folders:
    folder_path = os.path.join(extract_dir, folder)

    if os.path.isdir(folder_path):  # Check if it is a directory
        print(f"Processing folder: {folder}")

        # Find the EDF and TXT files
        edf_files = glob.glob(os.path.join(folder_path, "*.edf"))
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

        if not edf_files and not txt_files:
            print(f"Warning: No .edf or .txt files found in {folder}. Skipping this folder.")
            continue  # Skip folders without relevant files

        # Load EDF file
        for edf_file in edf_files:
            var_name = f"{folder}_edf"
            try:
                raw = mne.io.read_raw_edf(edf_file, preload=True)
                data_dict[var_name] = raw
                print(f"Loaded EDF: {var_name} ({raw.info})")
            except Exception as e:
                print(f"Error loading EDF {edf_file}: {e}")

        # Load TXT file
        for txt_file in txt_files:
            var_name = f"{folder}_txt"
            try:
                with open(txt_file, 'r', encoding="utf-8") as f:
                    data_dict[var_name] = f.readlines()
                print(f"Loaded TXT: {var_name} (Lines: {len(data_dict[var_name])})")
            except Exception as e:
                print(f"Error loading TXT {txt_file}: {e}")

print("\nData loading complete. Access variables from `data_dict`.")

# Example: Print available variables
if data_dict:
    print("\nAvailable Variables:")
    for key in sorted(data_dict.keys()):  # Show all loaded files in order
        print(f"{key}: {type(data_dict[key])}")
else:
    print("\nWarning: No data loaded. Check the extracted directory structure.")
