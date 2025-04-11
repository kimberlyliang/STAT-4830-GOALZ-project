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

DATA_DIR = '/Users/kimberly/Documents/STAT4830/STAT-4830-GOALZ-project/processed_sleepedf'
RESULT_DIR = '/Users/kimberly/Documents/STAT4830/STAT-4830-GOALZ-project/results'
os.makedirs(RESULT_DIR, exist_ok=True)

print("Files in directory:", os.listdir(DATA_DIR))

# Model parameters
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
TRAIN_VAL_SPLIT = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())

EMBEDDING_DIM = 128   # Dimension of the CNN encoder output per epoch
NUM_CLASSES = 5       # 0: Wake, 1: N1, 2: N2, 3: N3, 4: REM
NUM_TRANSFORMER_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.1
SEQ_LENGTH = 20       # Number of epochs per sequence (must match preprocessing)

class SleepSequenceDataset(Dataset):
    def __init__(self, data_dir, patient_ids=None):
        """
        Args:
            data_dir: Directory containing the NPZ files
            patient_ids: List of patient IDs to include (e.g., ['SC4252', 'SC4231'])
        """
        # Get all sequence files
        all_files = glob.glob(os.path.join(data_dir, '*_sequences.npz'))
        
        # Filter by patient IDs if specified
        if patient_ids is not None:
            self.files = [f for f in all_files if any(pid in os.path.basename(f) for pid in patient_ids)]
        else:
            self.files = all_files
            
        sequences_list = []
        labels_list = []
        self.patient_ids = []
        
        for f in self.files:
            loaded = np.load(f)
            sequences_list.append(loaded['sequences'])
            labels_list.append(loaded['seq_labels'])
            self.patient_ids.append(os.path.basename(f)[:6])  # Extract patient ID (e.g., 'SC4252')
            
        self.sequences = np.concatenate(sequences_list, axis=0)
        self.seq_labels = np.concatenate(labels_list, axis=0)
        self.sequences = torch.from_numpy(self.sequences).float()
        self.seq_labels = torch.from_numpy(self.seq_labels).long()
        
        print(f"Loaded {len(self.files)} files from {len(set(self.patient_ids))} unique patients")
        print(f"Total sequences: {len(self.sequences)}")
        
        # Print class distribution
        unique, counts = np.unique(self.seq_labels.numpy().flatten(), return_counts=True)
        print("\nClass distribution:")
        for label, count in zip(unique, counts):
            print(f"Class {label} ({'W N1 N2 N3 REM'.split()[label]}): {count} samples")

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        return self.sequences[idx], self.seq_labels[idx]

def create_balanced_sampler(dataset):
    """
    Create a weighted sampler to balance class distributions
    """
    # Get all labels (we'll use the first label of each sequence since sequences are contiguous)
    all_labels = dataset.seq_labels[:, 0].numpy()  # Only take first label of each sequence
    
    # Compute class weights
    classes = np.unique(all_labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=all_labels
    )
    
    # Create sample weights
    sample_weights = np.array([class_weights[label] for label in all_labels])
    
    # Create sampler with length equal to dataset
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),  # Use actual dataset length
        replacement=True
    )
    
    return sampler, torch.FloatTensor(class_weights)

class EpochEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # Let's calculate the correct dimension
        self.calculate_fc_input_dim = None  # Will be set in forward pass
        self.fc = None  # Will be initialized in first forward pass
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, channels, time_points)
        batch_size, seq_len, channels, time_points = x.shape
        
        # Process each sequence element independently
        x = x.view(batch_size * seq_len, channels, time_points)
        
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        # Initialize fc layer if not done yet
        if self.fc is None:
            self.calculate_fc_input_dim = x.shape[1] * x.shape[2]
            self.fc = nn.Linear(self.calculate_fc_input_dim, self.embedding_dim).to(x.device)
            print(f"Initialized fc layer with input dim: {self.calculate_fc_input_dim}")
        
        x = x.view(batch_size * seq_len, -1)  # Flatten
        x = self.dropout(torch.relu(self.fc(x)))
        
        # Reshape back to sequence form
        x = x.view(batch_size, seq_len, -1)
        return x

class SleepTransformer(nn.Module):
    def __init__(self, embedding_dim, num_classes, num_layers, num_heads, dropout, seq_length):
        super().__init__()
        
        self.epoch_encoder = EpochEncoder(embedding_dim)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_length, embedding_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4*embedding_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc_out = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, channels, time_points)
        
        # Get embeddings for each epoch
        x = self.epoch_encoder(x)
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Get predictions for each time step
        x = self.fc_out(x)
        
        return x

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for seq, labels in dataloader:
        seq, labels = seq.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(seq)
        loss = criterion(logits.view(-1, NUM_CLASSES), labels.view(-1))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * seq.size(0)
        preds = torch.argmax(logits, dim=-1)
        all_preds.append(preds.cpu().detach().numpy())
        all_labels.append(labels.cpu().detach().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, acc

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for seq, labels in dataloader:
            seq, labels = seq.to(device), labels.to(device)
            logits = model(seq)
            loss = criterion(logits.view(-1, NUM_CLASSES), labels.view(-1))
            
            running_loss += loss.item() * seq.size(0)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    all_probs = np.concatenate(all_probs).reshape(-1, NUM_CLASSES)
    acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, acc, all_preds, all_labels, all_probs

def plot_curves(train_losses, test_losses, train_accs, test_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss Curves')
    
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(test_accs, label='Test Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'training_curves.png'))
    plt.close()

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(RESULT_DIR, 'confusion_matrix.png'))
    plt.close()

def compute_metrics(y_true, y_pred, class_names):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1_macro': f1_score(y_true, y_pred, average='macro'),
        'F1_per_class': f1_score(y_true, y_pred, average=None, zero_division=0),
        'Kappa': cohen_kappa_score(y_true, y_pred),
        'Classification_Report': classification_report(
            y_true, 
            y_pred, 
            target_names=class_names,
            zero_division=0
        )
    }

def plot_roc_curves(y_true, y_prob, class_names):
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    
    plt.figure(figsize=(10, 8))
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig(os.path.join(RESULT_DIR, 'roc_curves.png'))
    plt.close()

def split_patients(data_dir, train_ratio=0.8, random_state=42):
    """
    Split patients into train and test sets
    """
    # Get all sequence files
    all_files = glob.glob(os.path.join(data_dir, '*_sequences.npz'))
    
    # Extract unique patient IDs
    patient_ids = list(set([os.path.basename(f)[:6] for f in all_files]))
    
    # Random split of patients
    np.random.seed(random_state)
    np.random.shuffle(patient_ids)
    
    n_train = int(len(patient_ids) * train_ratio)
    train_ids = patient_ids[:n_train]
    test_ids = patient_ids[n_train:]
    
    print(f"Train patients ({len(train_ids)}): {sorted(train_ids)}")
    print(f"Test patients ({len(test_ids)}): {sorted(test_ids)}")
    
    return train_ids, test_ids

def main():
    # Split patients into train and test sets
    train_patients, test_patients = split_patients(DATA_DIR, train_ratio=0.8)
    
    # Create datasets with specific patients
    train_dataset = SleepSequenceDataset(DATA_DIR, patient_ids=train_patients)
    test_dataset = SleepSequenceDataset(DATA_DIR, patient_ids=test_patients)
    
    # Create balanced sampler and get class weights
    train_sampler, class_weights = create_balanced_sampler(train_dataset)
    
    # Create data loaders (use sampler for training)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    
    # Initialize model
    model = SleepTransformer(EMBEDDING_DIM, NUM_CLASSES, NUM_TRANSFORMER_LAYERS, 
                            NUM_HEADS, DROPOUT, SEQ_LENGTH)
    model.to(DEVICE)
    
    # Use weighted cross-entropy loss
    class_weights = class_weights.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Print class weights being used
    print("\nClass weights:")
    for i, w in enumerate(class_weights.cpu().numpy()):
        print(f"Class {i} ({'W N1 N2 N3 REM'.split()[i]}): {w:.3f}")
    
    # Training loop
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    best_n1_f1 = 0
    best_model_state = None
    
    for epoch in tqdm(range(1, NUM_EPOCHS+1), desc="Training Epochs"):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Test
        test_loss, test_acc, test_preds, test_labels, test_probs = eval_epoch(
            model, test_loader, criterion, DEVICE
        )
        
        # Store metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # Compute metrics
        metrics_dict = compute_metrics(test_labels, test_preds, 
                                    class_names=["W","N1","N2","N3","REM"])
        n1_f1 = metrics_dict['F1_per_class'][1]
        
        # Save best model based on N1 F1 score
        if n1_f1 > best_n1_f1:
            best_n1_f1 = n1_f1
            best_model_state = model.state_dict()
        
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}:")
        print(f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")
        print(f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")
        print(f"N1 F1={n1_f1:.4f} (Best: {best_n1_f1:.4f})")
        
        # Full metrics every 5 epochs
        if epoch % 5 == 0:
            print("\nDetailed Metrics:")
            print(metrics_dict)
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    test_loss, test_acc, test_preds, test_labels, test_probs = eval_epoch(
        model, test_loader, criterion, DEVICE
    )
    
    # Plot results
    plot_curves(train_losses, test_losses, train_accs, test_accs)
    plot_confusion(test_labels, test_preds)
    plot_roc_curves(test_labels, test_probs, class_names=["W","N1","N2","N3","REM"])
    
    # Save the model with additional information
    torch.save({
        'model_state_dict': best_model_state,
        'train_patients': train_patients,
        'test_patients': test_patients,
        'class_weights': class_weights.cpu(),
        'final_metrics': compute_metrics(test_labels, test_preds, 
                                       class_names=["W","N1","N2","N3","REM"]),
        'best_n1_f1': best_n1_f1
    }, os.path.join(RESULT_DIR, 'sleep_transformer_model_balanced.pth'))

if __name__ == '__main__':
    main()
