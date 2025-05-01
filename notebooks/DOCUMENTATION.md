# Documentation: Sleep Stage Classification using Transformer Model (`tfm_w_new_patient_split_balanced.py`)

This document provides a detailed explanation of the Python script `tfm_w_new_patient_split_balanced.py`, which trains and evaluates a Transformer-based model for classifying sleep stages using EEG and EOG data from the Sleep-EDF dataset.

## Table of Contents

1.  [Overview](#overview)
2.  [Prerequisites](#prerequisites)
3.  [Script Breakdown](#script-breakdown)
    *   [Imports](#imports)
    *   [Configuration and Constants](#configuration-and-constants)
    *   [Dataset Class (`SleepSequenceDataset`)](#dataset-class-sleepsequencedataset)
    *   [Balanced Sampler (`create_balanced_sampler`)](#balanced-sampler-create_balanced_sampler)
    *   [Epoch Encoder CNN (`EpochEncoder`)](#epoch-encoder-cnn-epochencoder)
    *   [Transformer Model (`SleepTransformer`)](#transformer-model-sleeptransformer)
    *   [Training Function (`train_epoch`)](#training-function-train_epoch)
    *   [Evaluation Function (`eval_epoch`)](#evaluation-function-eval_epoch)
    *   [Plotting Functions](#plotting-functions)
    *   [Metrics Calculation (`compute_metrics`)](#metrics-calculation-compute_metrics)
    *   [Patient Splitting (`split_patients`)](#patient-splitting-split_patients)
    *   [Main Execution (`main`)](#main-execution-main)
    *   [Script Entry Point](#script-entry-point)
4.  [How to Run](#how-to-run)
5.  [Output](#output)

## Overview

The script aims to classify 30-second epochs of sleep into one of five stages (Wake, N1, N2, N3, REM) based on EEG and EOG signals. It uses preprocessed data where sleep recordings are divided into sequences of consecutive epochs.

The core components are:
*   A **CNN-based Epoch Encoder** to extract features from individual 30-second epochs (EEG and EOG channels).
*   A **Transformer Encoder** to model the temporal dependencies between consecutive epochs within a sequence.
*   A **patient-based split** for training and testing to ensure the model generalizes to unseen subjects.
*   **Weighted sampling and loss** to handle the inherent class imbalance in sleep data.

## Prerequisites

*   **Python 3.x**
*   **PyTorch**: For building and training the neural network.
*   **NumPy**: For numerical operations.
*   **Scikit-learn**: For evaluation metrics (confusion matrix, accuracy, F1-score, Kappa, classification report, ROC curve) and class weight calculation.
*   **Matplotlib & Seaborn**: For plotting results (loss/accuracy curves, confusion matrix, ROC curves).
*   **tqdm**: For progress bars during training.
*   **Preprocessed Data**: The script expects preprocessed `.npz` files located in the `DATA_DIR`. Each file should contain sequences of epochs and corresponding labels for a single night recording. The preprocessing script (not included here) should generate files named like `<PatientID><NightID>_sequences.npz` containing 'sequences' (shape: `(num_sequences, seq_length, channels, time_points)`) and 'seq_labels' (shape: `(num_sequences, seq_length)`).

## Script Breakdown

### Imports

```python
import os, glob, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import itertools
import seaborn as sns
from tqdm.notebook import tqdm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    cohen_kappa_score,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
```

*   `os`: Used for interacting with the operating system, specifically for creating directories (`os.makedirs`) and joining paths (`os.path.join`).
*   `glob`: Used for finding files matching a specific pattern (finding all `_sequences.npz` files).
*   `numpy` (as `np`): Fundamental package for numerical computation in Python. Used for array manipulation.
*   `torch`, `torch.nn` (as `nn`), `torch.optim` (as `optim`): Core PyTorch libraries for tensor operations, defining neural network layers, and optimization algorithms.
*   `torch.utils.data.Dataset`, `DataLoader`, `random_split`, `WeightedRandomSampler`: PyTorch utilities for creating custom datasets, loading data in batches, splitting datasets, and handling imbalanced data.
*   `matplotlib.pyplot` (as `plt`): Library for creating static, animated, and interactive visualizations.
*   `sklearn.metrics`: Module from Scikit-learn containing tools for evaluating model performance (confusion matrix, accuracy, F1 score, Kappa score, classification report, ROC analysis).
*   `sklearn.utils.class_weight.compute_class_weight`: Function to calculate weights to balance classes.
*   `itertools`: Not explicitly used in the final version shown but often useful for iteration tasks.
*   `seaborn` (as `sns`): Data visualization library based on Matplotlib, used here for plotting the confusion matrix heatmap.
*   `tqdm.notebook`: Provides progress bars, specifically formatted for Jupyter notebooks (though it works in standard scripts too).
*   `sklearn.preprocessing.label_binarize`: Function to convert multi-class labels into a binary format suitable for ROC curve calculation.

### Configuration and Constants

```python
DATA_DIR = '/Users/kimberly/Documents/STAT4830/STAT-4830-GOALZ-project/processed_sleepedf'
RESULT_DIR = '/Users/kimberly/Documents/STAT4830/STAT-4830-GOALZ-project/results'
os.makedirs(RESULT_DIR, exist_ok=True)

print("Files in directory:", os.listdir(DATA_DIR))

# Model parameters
BATCH_SIZE = 16
NUM_EPOCHS = 20

# does this matter?? # Yes, learning rate is crucial for training stability and convergence.
LEARNING_RATE = 1e-3
TRAIN_VAL_SPLIT = 0.8 # Note: This is used for patient splitting, not a standard validation split within training data.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())

EMBEDDING_DIM = 128   # Dimension of the CNN encoder output per epoch
NUM_CLASSES = 5       # 0: Wake, 1: N1, 2: N2, 3: N3, 4: REM
NUM_TRANSFORMER_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.1
SEQ_LENGTH = 20       # Number of epochs per sequence (must match preprocessing)
```

*   `DATA_DIR`: Path to the directory containing the preprocessed `.npz` sequence files. **Ensure this path is correct for your system.**
*   `RESULT_DIR`: Path to the directory where output files (plots, model checkpoint, metrics) will be saved. **Ensure this path is correct and writable.**
*   `os.makedirs(RESULT_DIR, exist_ok=True)`: Creates the results directory if it doesn't already exist. `exist_ok=True` prevents an error if the directory is already present.
*   `print("Files in directory:", os.listdir(DATA_DIR))`: Lists the files in the data directory to help verify the path is correct.
*   `BATCH_SIZE`: Number of sequences processed in each training iteration.
*   `NUM_EPOCHS`: Total number of times the entire training dataset is passed through the model.
*   `LEARNING_RATE`: Step size for the optimizer (Adam) during weight updates.
*   `TRAIN_VAL_SPLIT`: Ratio used to split the *patients* into training and testing sets (e.g., 0.8 means 80% of patients for training, 20% for testing).
*   `DEVICE`: Automatically selects the GPU ("cuda") if available, otherwise falls back to the CPU ("cpu"). Training is significantly faster on a GPU.
*   `print("CUDA available:", torch.cuda.is_available())`: Checks and prints whether a CUDA-enabled GPU is detected by PyTorch.
*   `EMBEDDING_DIM`: The dimensionality of the feature vector produced by the `EpochEncoder` for each 30-second epoch. This is also the input dimension for the Transformer layers.
*   `NUM_CLASSES`: The number of sleep stages to classify (5: Wake, N1, N2, N3, REM).
*   `NUM_TRANSFORMER_LAYERS`: The number of stacked Transformer encoder layers.
*   `NUM_HEADS`: The number of attention heads in the multi-head self-attention mechanism within each Transformer layer. `EMBEDDING_DIM` must be divisible by `NUM_HEADS`.
*   `DROPOUT`: Dropout probability used in the EpochEncoder and Transformer layers to prevent overfitting.
*   `SEQ_LENGTH`: The number of consecutive 30-second epochs grouped into a single sequence. This **must match** the sequence length used during data preprocessing.

### Dataset Class (`SleepSequenceDataset`)

```python
class SleepSequenceDataset(Dataset):
    def __init__(self, data_dir, patient_ids=None):
        # ... (implementation) ...
    def __len__(self):
        # ... (implementation) ...
    def __getitem__(self, idx):
        # ... (implementation) ...
```

*   **Purpose**: This class defines how to load and access the sleep sequence data. It inherits from `torch.utils.data.Dataset`.
*   `__init__(self, data_dir, patient_ids=None)`:
    *   Initializes the dataset.
    *   `data_dir`: The directory containing the `.npz` files.
    *   `patient_ids`: An optional list of patient IDs (e.g., `['SC4001', 'SC4012']`). If provided, only data from these patients will be loaded. This is used to create separate train/test datasets based on the patient split.
    *   Uses `glob.glob` to find all `*_sequences.npz` files in `data_dir`.
    *   Filters these files based on `patient_ids` if provided.
    *   Iterates through the selected files:
        *   Loads the `'sequences'` and `'seq_labels'` arrays from each `.npz` file using `np.load`.
        *   Appends them to `sequences_list` and `labels_list`.
        *   Extracts and stores the patient ID (first 6 characters of the filename) in `self.patient_ids`.
    *   Concatenates all loaded sequences and labels into single NumPy arrays (`self.sequences`, `self.seq_labels`).
    *   Converts the NumPy arrays to PyTorch tensors (`.float()` for sequences, `.long()` for labels).
    *   Prints summary statistics (number of files, unique patients, total sequences, class distribution).
*   `__len__(self)`: Returns the total number of sequences in the dataset. Required by PyTorch `Dataset`.
*   `__getitem__(self, idx)`: Returns the sequence and its corresponding labels at the given index `idx`. Required by PyTorch `Dataset` and used by the `DataLoader`.

### Balanced Sampler (`create_balanced_sampler`)

```python
def create_balanced_sampler(dataset):
    # ... (implementation) ...
```

*   **Purpose**: To address class imbalance (common in sleep data, where N2 is often dominant and N1/N3 are rarer) during training. It creates a sampler that draws samples with probabilities inversely proportional to their class frequencies.
*   **Steps**:
    1.  Extracts the first label of each sequence (`dataset.seq_labels[:, 0]`). *Assumption*: Sequences are relatively homogeneous or the first label is representative enough for balancing purposes.
    2.  Calculates class weights using `sklearn.utils.class_weight.compute_class_weight` with `class_weight='balanced'`. This gives higher weights to under-represented classes.
    3.  Creates `sample_weights`: An array where each element corresponds to a sequence in the dataset, and its value is the weight associated with the class of that sequence's first label.
    4.  Initializes `torch.utils.data.WeightedRandomSampler`:
        *   `weights=sample_weights`: Provides the weights for each sample.
        *   `num_samples=len(dataset)`: The sampler will draw a total number of samples equal to the original dataset size in each epoch.
        *   `replacement=True`: Allows samples to be drawn multiple times, necessary for oversampling minority classes.
    5.  Returns the `sampler` object (to be used with the training `DataLoader`) and the calculated `class_weights` as a PyTorch tensor (to be used in the loss function).

### Epoch Encoder CNN (`EpochEncoder`)

```python
class EpochEncoder(nn.Module):
    def __init__(self, embedding_dim):
        # ... (implementation) ...
    def forward(self, x):
        # ... (implementation) ...
```

*   **Purpose**: Encodes a single 30-second sleep epoch (containing EEG and EOG channels) into a fixed-size feature vector (`embedding_dim`). It uses a Convolutional Neural Network (CNN).
*   `__init__(self, embedding_dim)`:
    *   Defines the CNN layers:
        *   `conv1`: 1D convolution (input channels=2, output channels=16, kernel=5, padding=2).
        *   `conv2`: 1D convolution (input=16, output=32, kernel=3, padding=1).
        *   `conv3`: 1D convolution (input=32, output=64, kernel=3, padding=1).
        *   `pool`: Max pooling layer (kernel=2), reducing the temporal dimension by half after each conv layer.
        *   `dropout`: Dropout layer (p=0.1).
    *   `self.fc`: The final fully connected layer is **not** initialized here. Its input size depends on the output shape of the convolutional/pooling layers, which is calculated dynamically in the first `forward` pass.
    *   `self.embedding_dim`: Stores the desired output dimension.
*   `forward(self, x)`:
    *   Input `x` shape: `(batch_size, seq_len, channels, time_points)`.
    *   `batch_size, seq_len, channels, time_points = x.shape`: Extracts dimensions.
    *   `x = x.view(batch_size * seq_len, channels, time_points)`: Reshapes the input to treat each epoch within the sequence independently for the CNN processing. New shape: `(batch_size * seq_len, channels, time_points)`.
    *   Passes `x` through `conv1 -> relu -> pool -> conv2 -> relu -> pool -> conv3 -> relu -> pool`.
    *   **Dynamic FC Layer Initialization**:
        *   `if self.fc is None:`: Checks if the fully connected layer has been initialized.
        *   `self.calculate_fc_input_dim = x.shape[1] * x.shape[2]`: Calculates the flattened size of the output from the conv/pool layers.
        *   `self.fc = nn.Linear(...)`: Initializes the `nn.Linear` layer with the calculated input dimension and the target `self.embedding_dim`. `.to(x.device)` ensures the layer is on the same device (CPU/GPU) as the input tensor.
        *   Prints the calculated input dimension (useful for debugging).
    *   `x = x.view(batch_size * seq_len, -1)`: Flattens the output of the conv layers before the fully connected layer.
    *   `x = self.dropout(torch.relu(self.fc(x)))`: Passes through the fully connected layer, applies ReLU activation, and then dropout.
    *   `x = x.view(batch_size, seq_len, -1)`: Reshapes the output back into sequence format `(batch_size, seq_len, embedding_dim)`.
    *   Returns the sequence of epoch embeddings.

### Transformer Model (`SleepTransformer`)

```python
class SleepTransformer(nn.Module):
    def __init__(self, embedding_dim, num_classes, num_layers, num_heads, dropout, seq_length):
        # ... (implementation) ...
    def forward(self, x):
        # ... (implementation) ...
```

*   **Purpose**: The main model that takes a sequence of epoch data, uses the `EpochEncoder` to get embeddings, adds positional information, processes the sequence with a Transformer Encoder, and classifies each epoch in the sequence.
*   `__init__(...)`:
    *   `self.epoch_encoder = EpochEncoder(embedding_dim)`: Instantiates the CNN encoder.
    *   `self.pos_encoder = nn.Parameter(torch.randn(1, seq_length, embedding_dim))`: Creates learnable positional encodings. These are added to the epoch embeddings to give the Transformer information about the position of each epoch within the sequence. Shape: `(1, seq_length, embedding_dim)`. `nn.Parameter` ensures these values are treated as model parameters and updated during training.
    *   `encoder_layer = nn.TransformerEncoderLayer(...)`: Defines a single Transformer encoder layer using PyTorch's built-in module.
        *   `d_model=embedding_dim`: The input/output dimension.
        *   `nhead=num_heads`: Number of attention heads.
        *   `dim_feedforward=4*embedding_dim`: Dimension of the feedforward network within the layer (a common setting).
        *   `dropout=dropout`: Dropout rate.
        *   `batch_first=True`: Specifies that the batch dimension comes first in the input tensor shape `(batch, seq, feature)`.
    *   `self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)`: Stacks multiple `encoder_layer` instances.
    *   `self.fc_out = nn.Linear(embedding_dim, num_classes)`: The final linear layer that maps the Transformer output for each epoch to the scores for each class (logits).
*   `forward(self, x)`:
    *   Input `x` shape: `(batch_size, seq_len, channels, time_points)`.
    *   `x = self.epoch_encoder(x)`: Gets epoch embeddings. Output shape: `(batch_size, seq_len, embedding_dim)`.
    *   `x = x + self.pos_encoder`: Adds positional encodings (broadcast across the batch dimension).
    *   `x = self.transformer_encoder(x)`: Processes the sequence of embeddings through the Transformer layers. Output shape remains `(batch_size, seq_len, embedding_dim)`.
    *   `x = self.fc_out(x)`: Classifies each epoch in the sequence. Output shape: `(batch_size, seq_len, num_classes)`. These are the raw logits.
    *   Returns the logits for each epoch in the sequence.

### Training Function (`train_epoch`)

```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    # ... (implementation) ...
```

*   **Purpose**: Performs one full pass over the training data (one epoch).
*   **Steps**:
    1.  `model.train()`: Sets the model to training mode (enables dropout, batch normalization updates, etc.).
    2.  Initializes `running_loss`, `all_preds`, `all_labels`.
    3.  Iterates through the `dataloader` (which provides batches of `seq`, `labels`):
        *   Moves `seq` and `labels` to the specified `device` (GPU/CPU).
        *   `optimizer.zero_grad()`: Clears gradients from the previous iteration.
        *   `logits = model(seq)`: Performs the forward pass to get model predictions (logits). Shape: `(batch_size, seq_len, num_classes)`.
        *   `loss = criterion(logits.view(-1, NUM_CLASSES), labels.view(-1))`: Calculates the loss.
            *   `logits.view(-1, NUM_CLASSES)` reshapes logits to `(batch_size * seq_len, num_classes)`.
            *   `labels.view(-1)` reshapes labels to `(batch_size * seq_len)`.
            *   `criterion` (CrossEntropyLoss) computes the loss between the predicted logits and the true labels for all epochs in the batch. If class weights were provided during criterion initialization, they are applied here.
        *   `loss.backward()`: Computes gradients of the loss with respect to model parameters.
        *   `optimizer.step()`: Updates model parameters based on the computed gradients and the learning rate.
        *   Updates `running_loss`.
        *   `preds = torch.argmax(logits, dim=-1)`: Gets the predicted class index (0-4) for each epoch by finding the index with the highest logit.
        *   Appends predictions and labels (moved back to CPU and converted to NumPy) to `all_preds` and `all_labels`.
    4.  `epoch_loss = running_loss / len(dataloader.dataset)`: Calculates the average loss over the entire epoch.
    5.  Concatenates and flattens all predictions and labels.
    6.  `acc = accuracy_score(all_labels, all_preds)`: Calculates the overall accuracy for the epoch.
    7.  Returns the average epoch loss and accuracy.

### Evaluation Function (`eval_epoch`)

```python
def eval_epoch(model, dataloader, criterion, device):
    # ... (implementation) ...
```

*   **Purpose**: Evaluates the model performance on a given dataset (typically the test set).
*   **Steps**:
    1.  `model.eval()`: Sets the model to evaluation mode (disables dropout, uses running statistics for batch normalization, etc.).
    2.  Initializes `running_loss`, `all_preds`, `all_labels`, `all_probs`.
    3.  `with torch.no_grad():`: Disables gradient calculation, which is unnecessary during evaluation and saves memory/computation.
    4.  Iterates through the `dataloader`:
        *   Moves `seq` and `labels` to the `device`.
        *   `logits = model(seq)`: Performs the forward pass.
        *   `loss = criterion(...)`: Calculates the loss (useful for monitoring test loss, but not used for backpropagation).
        *   Updates `running_loss`.
        *   `probs = torch.softmax(logits, dim=-1)`: Calculates class probabilities from logits using the softmax function.
        *   `preds = torch.argmax(logits, dim=-1)`: Gets predicted class indices.
        *   Appends predictions, labels, and probabilities (moved to CPU and converted to NumPy) to respective lists.
    5.  Calculates average `epoch_loss`.
    6.  Concatenates and flattens predictions and labels.
    7.  Concatenates probabilities and reshapes them to `(total_epochs, num_classes)`.
    8.  Calculates overall `acc`uracy.
    9.  Returns the average epoch loss, accuracy, all predictions, all true labels, and all predicted probabilities.

### Plotting Functions

*   `plot_curves(train_losses, test_losses, train_accs, test_accs)`:
    *   Takes lists of training/test losses and accuracies per epoch.
    *   Uses Matplotlib to create two subplots: one for loss curves, one for accuracy curves.
    *   Saves the plot to `<RESULT_DIR>/training_curves.png`.
*   `plot_confusion(y_true, y_pred)`:
    *   Takes true labels and predicted labels.
    *   Computes the confusion matrix using `sklearn.metrics.confusion_matrix`.
    *   Uses Seaborn's `heatmap` to visualize the confusion matrix.
    *   Saves the plot to `<RESULT_DIR>/confusion_matrix.png`.
*   `plot_roc_curves(y_true, y_prob, class_names)`:
    *   Takes true labels, predicted probabilities (for each class), and class names.
    *   Uses `sklearn.preprocessing.label_binarize` to convert true labels into a one-hot-like format.
    *   For each class:
        *   Calculates the False Positive Rate (FPR) and True Positive Rate (TPR) using `sklearn.metrics.roc_curve`.
        *   Calculates the Area Under the Curve (AUC) using `sklearn.metrics.auc`.
        *   Plots the ROC curve for that class using Matplotlib.
    *   Plots a diagonal dashed line representing random guessing.
    *   Saves the plot to `<RESULT_DIR>/roc_curves.png`.
*   `plot_sample(sample, label)`:
    *   Takes a single sample sequence (e.g., `(seq_len, channels, time_points)`) and its label.
    *   Plots the EEG (channel 0) and EOG (channel 1) signals for the *first epoch* in the sequence.
    *   Displays the plot using `plt.show()`. Used for visually inspecting the input data.

### Metrics Calculation (`compute_metrics`)

```python
def compute_metrics(y_true, y_pred, class_names):
    # ... (implementation) ...
```

*   **Purpose**: Calculates various performance metrics beyond simple accuracy.
*   **Arguments**: True labels (`y_true`), predicted labels (`y_pred`), and a list of `class_names`.
*   **Returns**: A dictionary containing:
    *   `'Accuracy'`: Overall accuracy (`accuracy_score`).
    *   `'F1_macro'`: Macro-averaged F1 score (unweighted mean of F1 scores for each class) (`f1_score(..., average='macro')`).
    *   `'F1_per_class'`: F1 score for each individual class (`f1_score(..., average=None)`).
    *   `'Kappa'`: Cohen's Kappa score, measuring inter-rater agreement (useful for imbalanced datasets) (`cohen_kappa_score`).
    *   `'Classification_Report'`: A text report including precision, recall, and F1 score for each class, plus overall accuracy and averages (`classification_report`). `zero_division=0` prevents warnings if a class has no predicted samples.

### Patient Splitting (`split_patients`)

```python
def split_patients(data_dir, train_ratio=0.8, random_state=42):
    # ... (implementation) ...
```

*   **Purpose**: Divides the available patients into distinct training and testing sets. This is crucial to prevent data leakage and ensure the model is evaluated on data from subjects it has never seen during training.
*   **Steps**:
    1.  Finds all `*_sequences.npz` files in `data_dir`.
    2.  Extracts unique patient IDs (assumed to be the first 6 characters of the filenames).
    3.  Sets the random seed (`np.random.seed`) for reproducible splits.
    4.  Shuffles the list of patient IDs randomly.
    5.  Calculates the number of training patients based on `train_ratio`.
    6.  Splits the shuffled list into `train_ids` and `test_ids`.
    7.  Prints the number of patients and the sorted lists of IDs in each set.
    8.  Returns the `train_ids` and `test_ids` lists.

### Main Execution (`main`)

```python
def main():
    # ... (implementation) ...
```

*   **Purpose**: Orchestrates the entire process of training and evaluation.
*   **Steps**:
    1.  `train_patients, test_patients = split_patients(...)`: Splits patients into train/test sets.
    2.  `train_dataset = SleepSequenceDataset(..., patient_ids=train_patients)`: Creates the training dataset containing only data from training patients.
    3.  `test_dataset = SleepSequenceDataset(..., patient_ids=test_patients)`: Creates the test dataset containing only data from test patients.
    4.  `plot_sample(...)`: Displays a sample from the training data for visual inspection.
    5.  `input("Press Enter to continue...")`: Pauses execution, allowing the user to view the sample plot before starting the potentially long training process.
    6.  `train_sampler, class_weights = create_balanced_sampler(train_dataset)`: Creates the weighted random sampler for the training data and gets class weights.
    7.  `train_loader = DataLoader(..., sampler=train_sampler)`: Creates the DataLoader for training. It uses the `train_sampler` to draw balanced batches. `shuffle` must be `False` or `None` when using a sampler.
    8.  `test_loader = DataLoader(..., shuffle=False)`: Creates the DataLoader for testing. `shuffle=False` ensures consistent evaluation order.
    9.  `model = SleepTransformer(...)`: Initializes the `SleepTransformer` model with the defined parameters.
    10. `model.to(DEVICE)`: Moves the model parameters to the selected device (GPU/CPU).
    11. `class_weights = class_weights.to(DEVICE)`: Moves the calculated class weights tensor to the device.
    12. `criterion = nn.CrossEntropyLoss(weight=class_weights)`: Initializes the loss function. Using `weight=class_weights` makes the loss function penalize errors on minority classes more heavily, complementing the balanced sampler.
    13. `optimizer = optim.Adam(...)`: Initializes the Adam optimizer.
    14. Prints the class weights being used.
    15. Initializes lists (`train_losses`, `test_losses`, etc.) to store metrics per epoch and variables (`best_n1_f1`, `best_model_state`) to track the best model found so far. The criterion for "best" is the highest F1 score for the N1 class on the test set.
    16. **Training Loop**: Iterates from `epoch = 1` to `NUM_EPOCHS`.
        *   Calls `train_epoch` to train the model for one epoch.
        *   Calls `eval_epoch` to evaluate the model on the test set.
        *   Appends the returned losses and accuracies to their respective lists.
        *   Calls `compute_metrics` to get detailed performance metrics on the test set for the current epoch.
        *   Extracts the F1 score for the N1 class (`metrics_dict['F1_per_class'][1]`).
        *   **Save Best Model**: If the current N1 F1 score is better than `best_n1_f1`, update `best_n1_f1` and save the current model's state (`model.state_dict()`) to `best_model_state`.
        *   Prints epoch summary (losses, accuracies, current and best N1 F1).
        *   Prints detailed metrics every 5 epochs.
    17. `model.load_state_dict(best_model_state)`: Loads the parameters of the model that achieved the best N1 F1 score during training.
    18. Performs a final evaluation using the best model state by calling `eval_epoch` again.
    19. Calls `plot_curves`, `plot_confusion`, and `plot_roc_curves` using the collected epoch data and the final evaluation results from the best model.
    20. Saves the best model's state dictionary and other relevant information (patient splits, class weights, final metrics, best N1 F1 score) into a single file: `<RESULT_DIR>/sleep_transformer_model_balanced.pth` using `torch.save`.

### Script Entry Point

```python
if __name__ == '__main__':
    main()
```

*   This is a standard Python construct.
*   It ensures that the `main()` function is called only when the script is executed directly (not when it's imported as a module into another script).

## How to Run

1.  **Install Prerequisites**: Make sure you have Python and all the libraries listed in [Prerequisites](#prerequisites) installed (`pip install torch numpy scikit-learn matplotlib seaborn tqdm`). You might need a specific PyTorch version depending on your CUDA setup if using a GPU.
2.  **Prepare Data**: Ensure your preprocessed data (`*_sequences.npz` files) is located in the directory specified by `DATA_DIR`. Each file should contain `'sequences'` and `'seq_labels'`. The sequences should have the shape `(num_sequences, SEQ_LENGTH, channels, time_points)`.
3.  **Configure Paths**: Modify `DATA_DIR` and `RESULT_DIR` at the beginning of the script to point to your data and desired output locations.
4.  **Adjust Parameters (Optional)**: You can change `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, model architecture parameters (`EMBEDDING_DIM`, `NUM_TRANSFORMER_LAYERS`, `NUM_HEADS`, `DROPOUT`), or `SEQ_LENGTH` if needed (ensure `SEQ_LENGTH` matches your preprocessed data).
5.  **Execute Script**: Run the script from your terminal:
    ```bash
    python notebooks/tfm_w_new_patient_split_balanced.py
    ```
6.  **Interact (if needed)**: The script will print dataset information, show a sample plot, and then pause, waiting for you to press Enter before starting training.

## Output

The script will:

1.  Print information about the loaded data (number of files, patients, sequences, class distribution).
2.  Print whether CUDA (GPU) is available.
3.  Print the dynamically calculated input dimension for the `EpochEncoder`'s fully connected layer.
4.  Display a plot of a sample EEG/EOG epoch from the training set.
5.  Wait for user input (Press Enter).
6.  Print the class weights being used for the loss function.
7.  Show a progress bar for each training epoch using `tqdm`.
8.  Print training/test loss, accuracy, and N1 F1 score after each epoch.
9.  Print detailed metrics every 5 epochs.
10. After training, save the following files to the `RESULT_DIR`:
    *   `training_curves.png`: Plot of training and testing loss and accuracy over epochs.
    *   `confusion_matrix.png`: Confusion matrix based on the final evaluation of the best model on the test set.
    *   `roc_curves.png`: ROC curves for each class based on the final evaluation.
    *   `sleep_transformer_model_balanced.pth`: A PyTorch save file containing a dictionary with:
        *   `'model_state_dict'`: The state dictionary of the best performing model (based on N1 F1 score).
        *   `'train_patients'`: List of patient IDs used for training.
        *   `'test_patients'`: List of patient IDs used for testing.
        *   `'class_weights'`: The class weights used in the loss function.
        *   `'final_metrics'`: The detailed metrics dictionary from the final evaluation.
        *   `'best_n1_f1'`: The best N1 F1 score achieved on the test set during training.