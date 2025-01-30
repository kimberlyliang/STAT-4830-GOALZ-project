Understanding the Problem

We are optimizing the classification of sleep onset and different sleep stages, particularly focusing on having the possibility to distinguish wakefulness in both daytime and nighttime sleep patterns. Previous methods have struggled with accuracy in this domain, so we aim to improve upon existing models.

Why This Matters

Accurate sleep classification has implications for diagnosing sleep disorders, improving wearable sleep tracking, and advancing neuroscience research on circadian rhythms.

Existing models often fail to generalize well, particularly for daytime sleep, making it difficult to apply these models to shift workers, individuals with irregular sleep patterns, or those with sleep disorders.

Mathematical Formulation

Objective: Maximize classification accuracy for sleep stages given EEG/PSG input

where  represents EEG signals,  the corresponding sleep stage, and  the model parameters.

Algorithm/Approach Choice & Justification

Baseline: Convolutional Neural Network (CNN) for time-series EEG data

Alternative Approaches: Transformers (for capturing long-range dependencies) and Graph Neural Networks (GNNs, if inter-channel relationships are meaningful)

Justification: CNNs have shown success in sleep classification, but they may not fully capture sequential dependencies. Exploring self-attention mechanisms in transformers or spatial relationships in GNNs could yield better results. We're using data from ANPHY-Sleep to start, but we hope to use MESA in the future (just waiting on a two week period for approval to access the dataset).

PyTorch Implementation Strategy

Step 1: Implement a baseline CNN using PyTorch

Step 2: Train on a subset of the Dreem Open Dataset for rapid validation

Step 3: Experiment with modifications (e.g., attention layers, deeper architectures)

Step 4: Compare against prior benchmarks

Initial Results

Successfully loaded and preprocessed EEG data (I hope aha let's fill this in more)

Baseline CNN implemented and trained on a small subset

Achieved reasonable accuracy on binary sleep-wake classification

Basic Performance Metrics

Baseline CNN accuracy: X% (to be updated based on experiments)

F1-score for sleep onset classification: Y%

Compute time per epoch: Z seconds


# to be filled out
Current limitations
Resource usage measurements
Unexpected challenges
Next Steps (1/2 page)
Immediate improvements needed
Technical challenges to address
Questions you need help with
Alternative approaches to try
What you've learned so far