anp_artifact_inspector.py

This script is used to inspect the artifact matrix and summarize the quality of the data for each subject.


For each electrode, we compute the fraction of epochs that are artifact‑free. If this fraction is below the threshold (0.97 by default), the electrode is flagged as "poor".