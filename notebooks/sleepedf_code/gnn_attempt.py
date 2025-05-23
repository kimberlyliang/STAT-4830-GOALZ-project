# -*- coding: utf-8 -*-
"""GNN_attempt

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1T0W0H0RjqReAb6kUNEhFxMgJzLddMoSB
"""

# Complete Sleep Stage Classification with Graph Neural Networks
# Improved version with lazy loading and verbose progress reporting

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
import glob
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import KFold
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import time
import psutil
import gc
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION AND HYPERPARAMETERS
# ============================================

class Config:
    # Paths
    BASE_DIR = Path("/content/drive/MyDrive/4830_project/sleepedf_data")
    PROCESSED_DATA_DIR = BASE_DIR / "completed_psd_sleep_edf"
    RESULTS_DIR = BASE_DIR / "gnn_model_results"

    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 2e-4
    TRAIN_RATIO = 0.8
    SEQ_LENGTH = 20
    SEQ_STRIDE = 5
    SEED = 42
    NUM_FOLDS = 5

    # Model parameters
    NODE_FEAT_DIM = 64
    GNN_HIDDEN_DIM = 128
    LSTM_HIDDEN_DIM = 256
    GNN_DROPOUT = 0.3
    NUM_GNN_LAYERS = 2
    NUM_LSTM_LAYERS = 2
    NUM_CLASSES = 5

    # Dataset parameters
    LAZY_LOAD = True  # Use lazy loading by default
    NUM_WORKERS = 0   # Number of dataloader workers

# Create directories
def setup_directories(config: Config):
    """Create necessary directories for saving results"""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ["plots", "models", "metrics"]:
        (config.RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {config.RESULTS_DIR}")

# Set random seeds
def set_seeds(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================
# DATA LOADING AND DATASET
# ============================================

def get_true_subject_id(filename: str) -> str:
    """Extract true subject ID from filename"""
    basename = Path(filename).stem
    if basename.startswith(("SC4", "ST7")):
        return basename[:5]
    return basename[:6]

def group_by_true_subjects(data_dir: Path) -> Dict[str, List[str]]:
    """Group recordings by true subject IDs"""
    print(f"Grouping recordings by subject ID in {data_dir}")
    files = glob.glob(str(data_dir / "*_sequences.npz"))
    subj_map = {}
    for f in files:
        rid = Path(f).stem.split("_")[0]
        subj = get_true_subject_id(rid)
        subj_map.setdefault(subj, []).append(rid)
    return subj_map

def split_true_subjects(data_dir: Path, train_ratio: float = 0.8,
                       random_state: int = 42) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Split subjects into train and test sets"""
    print("Splitting subjects into train and test sets...")
    subj_map = group_by_true_subjects(data_dir)
    subs = list(subj_map.keys())

    np.random.seed(random_state)
    np.random.shuffle(subs)

    n_train = int(len(subs) * train_ratio)
    train_subs = subs[:n_train]
    test_subs = subs[n_train:]

    train_ids = [rid for s in train_subs for rid in subj_map.get(s, [])]
    test_ids = [rid for s in test_subs for rid in subj_map.get(s, [])]

    print(f"Total subjects: {len(subs)}. Train: {len(train_subs)}, Test: {len(test_subs)}")
    print(f"Total recordings: {sum(len(v) for v in subj_map.values())}. Train IDs: {len(train_ids)}, Test IDs: {len(test_ids)}")

    return train_ids, test_ids, train_subs, test_subs

class SleepDatasetLazy(Dataset):
    """Sleep EEG Dataset with lazy loading and verbose progress reporting"""

    def __init__(self, raw_dir: Path, recording_ids: Optional[List[str]] = None,
                 seq_length: int = 20, lazy_load: bool = True):
        self.raw_dir = raw_dir
        self.seq_length = seq_length
        self.lazy_load = lazy_load

        # Get process for memory monitoring
        self.process = psutil.Process(os.getpid())

        # Get all sequence files
        print(f"Searching for sequence files in: {raw_dir}")
        start_time = time.time()
        all_raw_files = glob.glob(str(raw_dir / "*_sequences.npz"))
        print(f"Found {len(all_raw_files)} total files in {time.time() - start_time:.2f} seconds")

        # Filter files based on recording IDs
        valid_files = []
        if recording_ids is not None:
            print(f"Filtering files for {len(recording_ids)} recording IDs...")
            target_ids_set = set(recording_ids)
            for p in all_raw_files:
                rid = Path(p).stem.replace("_sequences", "")
                if rid in target_ids_set:
                    valid_files.append(p)
        else:
            valid_files = all_raw_files

        print(f"Found {len(valid_files)} matching sequence files.")
        self.file_paths = valid_files

        # Initialize attributes
        self.num_channels = None
        self.time_points = None

        if self.lazy_load:
            # Only load metadata for lazy loading
            print("Using lazy loading - loading metadata only...")
            self._load_metadata()
        else:
            # Load all data into memory
            print("Loading all data into memory...")
            self._load_all_data()

    def _load_metadata(self):
        """Load only metadata from the first file"""
        if not self.file_paths:
            raise ValueError("No valid files found!")

        print("Loading metadata from first file...")
        with np.load(self.file_paths[0]) as data:
            if "sequences" not in data or "seq_labels" not in data:
                raise ValueError(f"First file missing required keys: {self.file_paths[0]}")

            first_seq = data["sequences"]
            self.num_channels = first_seq.shape[2]
            self.time_points = first_seq.shape[3]

        print(f"Inferred {self.num_channels} channels and {self.time_points} time points per epoch.")

        # Count total sequences and create mapping
        self.sequence_to_file = []
        total_sequences = 0

        print("Counting sequences in all files...")
        start_time = time.time()

        for i, file_path in enumerate(self.file_paths):
            try:
                with np.load(file_path) as data:
                    n_sequences = data["sequences"].shape[0]
                    for j in range(n_sequences):
                        self.sequence_to_file.append((file_path, j))
                    total_sequences += n_sequences

                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    memory_mb = self.process.memory_info().rss / 1024 / 1024
                    print(f"  Processed {i+1}/{len(self.file_paths)} files, "
                          f"Total sequences: {total_sequences}, "
                          f"Time: {elapsed:.1f}s, Memory: {memory_mb:.1f} MB")

            except Exception as e:
                print(f"Error counting sequences in {file_path}: {e}")

        print(f"Metadata loading complete. Total sequences: {len(self.sequence_to_file)}")

        # Create edge index
        self.edge_index = self._create_edge_index()

        # Estimate class distribution (sample-based for speed)
        self._estimate_class_distribution()

    def _estimate_class_distribution(self, sample_size: int = 1000):
        """Estimate class distribution by sampling"""
        print(f"Estimating class distribution from {sample_size} samples...")

        if len(self.sequence_to_file) <= sample_size:
            sample_indices = list(range(len(self.sequence_to_file)))
        else:
            sample_indices = np.random.choice(len(self.sequence_to_file),
                                            sample_size, replace=False)

        all_labels = []
        for idx in sample_indices:
            file_path, seq_idx = self.sequence_to_file[idx]
            with np.load(file_path) as data:
                labels = data["seq_labels"][seq_idx]
                all_labels.extend(labels)

        all_labels = np.array(all_labels)
        unique, counts = np.unique(all_labels, return_counts=True)
        total = len(all_labels)

        print("Estimated class distribution:")
        classes = ['W', 'N1', 'N2', 'N3', 'REM']
        self.class_counts = np.zeros(5)
        for cls, count in zip(unique, counts):
            percentage = (count / total) * 100
            print(f"  {classes[cls]}: {count} ({percentage:.2f}%)")
            self.class_counts[cls] = count

    def _create_edge_index(self):
        """Create edge index for fully connected graph"""
        edge_list = []
        for src in range(self.num_channels):
            for tgt in range(self.num_channels):
                if src != tgt:
                    edge_list.append([src, tgt])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        print(f"Created fully connected graph edge_index for {self.num_channels} channels.")
        return edge_index

    def __len__(self):
        return len(self.sequence_to_file)

    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        file_path, seq_idx = self.sequence_to_file[idx]

        with np.load(file_path) as data:
            sequence = data["sequences"][seq_idx].astype(np.float32)
            label = data["seq_labels"][seq_idx].astype(np.int64)

        return {
            'x': torch.from_numpy(sequence),  # (seq_length, num_channels, time_points)
            'y': torch.from_numpy(label),     # (seq_length,)
            'edge_index': self.edge_index.clone()
        }

def create_weighted_sampler_lazy(dataset: SleepDatasetLazy) -> WeightedRandomSampler:
    """Create weighted random sampler for balanced training (lazy version)"""
    print("Creating weighted sampler...")

    # Use estimated class distribution
    class_weights = 1.0 / (dataset.class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()

    # Sample a subset to estimate sequence weights
    sample_size = min(1000, len(dataset))
    sample_indices = np.random.choice(len(dataset), sample_size, replace=False)

    sample_weights = []
    for idx in sample_indices:
        _, labels = dataset[idx]['y'].numpy(), dataset[idx]['y'].numpy()
        seq_weight = sum(class_weights[label] for label in labels) / len(labels)
        sample_weights.append(seq_weight)

    # Estimate weights for all sequences
    mean_weight = np.mean(sample_weights)
    all_weights = [mean_weight] * len(dataset)

    return WeightedRandomSampler(
        weights=all_weights,
        num_samples=len(all_weights),
        replacement=True
    )

# ============================================
# MODEL ARCHITECTURE (Same as before)
# ============================================

class SleepStageGNN(nn.Module):
    """Sleep Stage Classification model using CNN + GCN + LSTM"""

    def __init__(self, config: Config, num_channels: int, time_points: int):
        super().__init__()
        self.config = config
        self.num_channels = num_channels
        self.time_points = time_points

        # 1. Channel feature extractor (1D CNN)
        self.channel_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(3, 3),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(3, 3),
            nn.Conv1d(32, config.NODE_FEAT_DIM, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(config.NODE_FEAT_DIM),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 2. GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(config.NODE_FEAT_DIM, config.GNN_HIDDEN_DIM))
        for _ in range(config.NUM_GNN_LAYERS - 1):
            self.gcn_layers.append(GCNConv(config.GNN_HIDDEN_DIM, config.GNN_HIDDEN_DIM))

        self.gnn_dropout = nn.Dropout(config.GNN_DROPOUT)
        self.gnn_ln = nn.LayerNorm(config.GNN_HIDDEN_DIM)

        # 3. LSTM
        self.lstm = nn.LSTM(
            config.GNN_HIDDEN_DIM,
            config.LSTM_HIDDEN_DIM,
            batch_first=True,
            bidirectional=True,
            num_layers=config.NUM_LSTM_LAYERS,
            dropout=config.GNN_DROPOUT if config.NUM_LSTM_LAYERS > 1 else 0
        )
        self.lstm_ln = nn.LayerNorm(config.LSTM_HIDDEN_DIM * 2)

        # 4. Classifier
        self.classifier = nn.Linear(config.LSTM_HIDDEN_DIM * 2, config.NUM_CLASSES)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv1d, GCNConv)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def extract_channel_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from each channel using CNN"""
        x = x.unsqueeze(1)  # (num_channels, 1, time_points)
        features = self.channel_cnn(x)
        return features.squeeze(-1)  # (num_channels, feat_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the entire model"""
        x = batch['x']  # (batch_size, seq_length, num_channels, time_points)
        edge_index = batch['edge_index'][0].to(x.device)

        batch_size, seq_length = x.size(0), x.size(1)

        # Process each sequence
        lstm_inputs = []

        for b in range(batch_size):
            sequence_embeddings = []

            for t in range(seq_length):
                # Get epoch data
                epoch_data = x[b, t]  # (num_channels, time_points)

                # Extract features from each channel
                node_features = self.extract_channel_features(epoch_data)

                # Apply GCN layers
                h = node_features
                for i, gcn in enumerate(self.gcn_layers):
                    h = gcn(h, edge_index)
                    if i < len(self.gcn_layers) - 1:
                        h = F.relu(h)
                        h = self.gnn_dropout(h)

                h = self.gnn_ln(h)

                # Global graph pooling
                graph_embedding = h.mean(dim=0)
                sequence_embeddings.append(graph_embedding)

            sequence_tensor = torch.stack(sequence_embeddings)
            lstm_inputs.append(sequence_tensor)

        # Batch all sequences for LSTM
        lstm_input = torch.stack(lstm_inputs)

        # Process through LSTM
        lstm_out, _ = self.lstm(lstm_input)

        # Layer normalization
        lstm_out_reshaped = lstm_out.reshape(-1, lstm_out.size(-1))
        lstm_out_norm = self.lstm_ln(lstm_out_reshaped)
        lstm_out = lstm_out_norm.view(batch_size, seq_length, -1)

        # Classification
        output = self.classifier(lstm_out)

        return output

# ============================================
# LOSS FUNCTION AND TRAINING
# ============================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha_general: float = 0.25, alpha_n1: float = 0.9,
                 gamma: float = 2.5):
        super().__init__()
        self.alpha_general = alpha_general
        self.alpha_n1 = alpha_n1
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, num_classes = inputs.shape

        logits = inputs.view(-1, num_classes)
        targets_flat = targets.view(-1)

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs).clamp(min=1e-7, max=1-1e-7)

        nll_loss = F.nll_loss(log_probs, targets_flat, reduction='none')
        pt = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

        n1_mask = (targets_flat == 1).float()
        n3_mask = (targets_flat == 3).float()
        rem_mask = (targets_flat == 4).float()

        alpha = self.alpha_general * (1 - n1_mask - n3_mask - rem_mask) + \
                self.alpha_n1 * n1_mask + \
                0.5 * n3_mask + \
                0.45 * rem_mask

        focal_loss = alpha * ((1 - pt) ** self.gamma) * nll_loss

        return focal_loss.mean()

def train_epoch(model: nn.Module, dataloader: DataLoader,
                optimizer: optim.Optimizer, criterion: nn.Module,
                device: torch.device, scheduler=None, epoch: int = 0) -> Tuple[float, float]:
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        batch_start = time.time()

        # Move data to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch['y'])

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(dim=-1)
        correct += (predicted == batch['y']).sum().item()
        total += batch['y'].numel()

        # Print progress
        if (batch_idx + 1) % 10 == 0:
            batch_time = time.time() - batch_start
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']

            print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}: "
                  f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%, "
                  f"LR: {current_lr:.6f}, Batch Time: {batch_time:.2f}s, "
                  f"Total Time: {elapsed:.1f}s")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def evaluate(model: nn.Module, dataloader: DataLoader,
             criterion: nn.Module, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(batch)
            loss = criterion(outputs, batch['y'])

            running_loss += loss.item()
            _, predicted = outputs.max(dim=-1)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(batch['y'].cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Evaluating... {batch_idx+1}/{len(dataloader)} batches")

    # Flatten predictions and labels
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    avg_loss = running_loss / len(dataloader)
    accuracy = (all_preds == all_labels).mean()

    return avg_loss, accuracy, all_preds, all_labels

# ============================================
# MAIN TRAINING FUNCTION
# ============================================

def main():
    """Main training function"""
    config = Config()
    setup_directories(config)
    set_seeds(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Split data
    print("\n=== Splitting data ===")
    train_ids, test_ids, train_subs, test_subs = split_true_subjects(
        config.PROCESSED_DATA_DIR, config.TRAIN_RATIO, config.SEED
    )

    # Create datasets with lazy loading
    print("\n=== Creating training dataset ===")
    train_dataset = SleepDatasetLazy(
        config.PROCESSED_DATA_DIR,
        recording_ids=train_ids,
        lazy_load=config.LAZY_LOAD
    )

    print("\n=== Creating testing dataset ===")
    test_dataset = SleepDatasetLazy(
        config.PROCESSED_DATA_DIR,
        recording_ids=test_ids,
        lazy_load=config.LAZY_LOAD
    )

    # Create dataloaders
    print("\n=== Creating dataloaders ===")
    train_sampler = create_weighted_sampler_lazy(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # Initialize model
    print("\n=== Initializing model ===")
    model = SleepStageGNN(
        config,
        train_dataset.num_channels,
        train_dataset.time_points
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss function and optimizer
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LEARNING_RATE, epochs=config.NUM_EPOCHS,
        steps_per_epoch=len(train_loader), pct_start=0.3
    )

    # Training loop
    print("\n=== Starting training ===")
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")

        # Train
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler, epoch
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        print(f"\nValidating epoch {epoch+1}...")
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, test_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Epoch Time: {epoch_time:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = config.RESULTS_DIR / "models/best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': vars(config)
            }, model_path)
            print(f"  New best model saved with accuracy: {best_val_acc:.4f}")

        # Save training curves periodically
        if (epoch + 1) % 5 == 0:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(train_accs, label='Train Acc')
            plt.plot(val_accs, label='Val Acc')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(config.RESULTS_DIR / f"plots/training_curves_epoch{epoch+1}.png")
            plt.close()

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Garbage collection
        gc.collect()

    # Final evaluation
    print("\n=== Final Evaluation ===")
    checkpoint = torch.load(config.RESULTS_DIR / "models/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    # Generate final report
    classes = ['W', 'N1', 'N2', 'N3', 'REM']
    report = classification_report(test_labels, test_preds, target_names=classes)
    cm = confusion_matrix(test_labels, test_preds)
    kappa = cohen_kappa_score(test_labels, test_preds)

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print("\nClassification Report:")
    print(report)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "plots/final_confusion_matrix.png")
    plt.close()

    # Save final results
    results = {
        'test_accuracy': test_acc,
        'test_kappa': kappa,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'config': vars(config),
        'training_history': {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs
        }
    }

    results_path = config.RESULTS_DIR / "metrics/final_results.npy"
    np.save(results_path, results)

    print(f"\nTraining complete! Results saved to {config.RESULTS_DIR}")

if __name__ == "__main__":
    main()