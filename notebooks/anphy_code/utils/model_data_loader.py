import mne
import pandas as pd
import numpy as np
from gluonts.dataset.pandas import PandasDataset

def load_eeg_data(edf_file, label_file):
    """Load EEG data and labels"""
    # Load EEG data
    raw = mne.io.read_raw_edf(edf_file, preload=True)
    
    # Get data and times
    data = raw.get_data()
    times = raw.times
    
    # Create DataFrame
    df = pd.DataFrame(data.T, index=pd.to_datetime(times, unit='s'))
    df.columns = raw.ch_names
    
    # Load and merge labels
    labels = pd.read_csv(label_file)
    df['sleep_stage'] = labels['stage'].values
    
    return df

def prepare_dataset(df):
    """Convert DataFrame to GluonTS dataset"""
    dataset_dict = {
        "target": df['sleep_stage'].values,
        "start": df.index[0],
        "feat_dynamic_real": df.drop('sleep_stage', axis=1).values.T
    }
    return PandasDataset(dataset_dict) 