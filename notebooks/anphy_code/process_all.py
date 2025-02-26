from config.config import DataConfig
import os, subprocess

data_dir = DataConfig.BASE_PATH
subject_dirs = sorted([d for d in os.listdir(data_dir)
                       if d.lower().startswith("epctl") and os.path.isdir(os.path.join(data_dir, d))])

for subject in subject_dirs:
    print(f"Processing subject: {subject}")
    for group_index in range(10):
        cmd = f"python process_single_subject_group.py {subject} {group_index}"
        print(f"  Running command: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {subject} group {group_index}: {e}")


# for subject in subject_dirs:
#     print(f"Processing subject: {subject}")
#     for group_index in range(10):
#         cmd = f"python process_single_subject_group.py {subject} {group_index}"
#         print(f"  Running command: {cmd}")
#         try:
#             # launches a separate Python process for each batch
#             subprocess.run(cmd, shell=True, check=True)
#         except subprocess.CalledProcessError as e:
#             print(f"Error processing {subject} group {group_index}: {e}")

