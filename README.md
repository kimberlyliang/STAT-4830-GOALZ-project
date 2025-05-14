# Optimizing EEG-Based Deep Learning Models for N1 Sleep Onset Detection



## Team Members

Joshua George, Kimberly Liang, Tereza Okalova, Stefan Zaharia



## High-Level Summary

Sleep onset detection, particularly the accurate identification of N1 sleep (the first stage of non-REM sleep), plays a critical role in sleep disorder diagnosis and clinical evaluation. Yet, it remains one of the most elusive stages for both manual and automated systems to classify due to its transitional nature. N1 represents a fleeting phase, occupying only about 5\% of sleep and exhibiting significant overlap in features with both wakefulness and deeper sleep stages.



In this project, we propose an end-to-end deep learning framework, combining convolutional and transformer-based modules specifically tailored to enhance N1 classification accuracy from single-channel EEG recordings. Our approach leverages both spectral features and temporal dynamics of EEG signals to better distinguish the subtle transition from wakefulness to light sleep.



## Repository Structure

```

├── _development_history/         # Archive of development process, drafts, and LLM logs

├── docs/

│   ├── figures/                  # Figures used in visualizations (confusion_matrix.png, training_curves.png)

│   └── Final_Presentation_Slides.pdf  # Final presentation slides

├── notebooks/                    # Jupyter notebooks and Python scripts for analysis and model development

│   ├── mesa_code/                # Code for processing and analyzing MESA sleep dataset

│   ├── sleepedf_code/            # Code for processing and analyzing SleepEDF dataset

│   ├── anphy_code/               # Code for processing and analyzing Anphy dataset

│   └── *.ipynb, *.py             # Various notebooks for model training, analysis, and visualization

├── src/                          # Source code for the project

│   ├── mesa_psd_extraction.py    # Power Spectral Density extraction for MESA dataset

│   ├── psg_sleepedf.py           # Processing utilities for SleepEDF dataset

│   ├── catch22_feature_extraction.py # Feature extraction using catch22 method

│   ├── cnn_tfm_fcl_n1.py         # CNN-Transformer model for N1 sleep detection

│   ├── hybrid_sleep_transformer.py # Hybrid transformer model for sleep stage classification

│   ├── hybrid_corr_sampling.py   # Correlation-based sampling for balanced training

│   ├── psd_feature_extraction.py # Power Spectral Density feature extraction utilities

│   ├── prepro_sleepedf_all_eld.py # Preprocessing for SleepEDF elderly dataset

│   └── other utility scripts     # Additional preprocessing and analysis utilities

├── .gitignore                    # Git ignore file for data, cache, etc.

├── README.md                     # This file

├── TODO.md                       # Todo list for the project

└── requirements.txt              # Project dependencies

```



## Setup Instructions

To set up the environment for running this project:



1. Python version: 3.9+ recommended

2. Create a virtual environment (optional but recommended):

   ```

   python -m venv venv

   source venv/bin/activate  # On Windows: venv\Scripts\activate

   ```

3. Install dependencies:

   ```

   pip install -r requirements.txt

   ```

4. Data preparation: Our models are trained on three sleep EEG datasets:

   - SleepEDF: Available from [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/)

   - MESA: Available from [NSRR](https://sleepdata.org/datasets/mesa)

   - Anphy: Used with permission (not publicly available)



## Running the Code

Our project implements a pipeline for sleep stage classification with special focus on N1 detection:



1. For data preprocessing and feature extraction:

   ```

   # For SleepEDF preprocessing

   python src/prepro_sleepedf_all_eld.py

   

   # For feature extraction

   python src/psd_feature_extraction.py

   python src/catch22_feature_extraction.py

   ```

   

2. For model training and evaluation:

   ```

   # Primary transformer model analysis

   jupyter notebook notebooks/tfm_w_analytics.ipynb

   

   # For running the hybrid model

   python src/hybrid_corr_sampling.py

   ```



3. For dataset-specific analysis:

   ```

   # MESA dataset

   jupyter notebook notebooks/mesa_code/mesa_data_exploration.ipynb

   

   # SleepEDF dataset

   jupyter notebook notebooks/sleepedf_code/sleepedf_data_xplor.ipynb

   ```



4. Main model implementation:

   ```python

   from src.cnn_tfm_fcl_n1 import SleepTransformer

   from src.hybrid_sleep_transformer import HybridSleepTransformer

   # Create and train the model - see notebooks for detailed examples

   ```



## Executable Demo Link

[Interactive Demo on Google Colab](https://colab.research.google.com/drive/1ivTWX9GGnq9TX-M4ZLtmF7WxXxyPNAkZ?usp=sharing)



## Final Report

Final report details our approaches, implementation, and results. It is located in the root directory.



## Key Findings

Our approach demonstrates improved accuracy in detecting the elusive N1 sleep stage, with the following key results:



- Improved N1 detection precision compared to baseline methods by utilizing a hybrid CNN-Transformer architecture

- Successfully differentiated N1 from wake states with higher accuracy through better feature extraction

- Enhanced model robustness through correlation-based sampling techniques that address class imbalance

- Demonstrated the effectiveness of combining spectral features (PSD) with temporal dynamics (CNN-Transformer)



## Acknowledgments

We would like to thank the providers of the MESA, SleepEDF, and Anphy datasets for making their data available for research. We also acknowledge the guidance and support of the STAT 4830 course instructors.