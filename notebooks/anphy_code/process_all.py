from config.config import DataConfig
import os, subprocess
data_dir = "/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/ANPHY-Sleep_data"
subject_dirs = sorted([d for d in os.listdir(data_dir)
                       if d.lower().startswith("epctl") and os.path.isdir(os.path.join(data_dir, d))])

# goes over all subjects
for subject in subject_dirs:
    print(f"Processing subject: {subject}")
    # goes over all 10 electrode groups (group indices 0 to 9)
    for group_index in range(10):
        cmd = f"python process_single_subject_group.py {subject} {group_index}"
        print(f"  Running command: {cmd}")
        # launches a separate Python process for each batch
        subprocess.run(cmd, shell=True)
