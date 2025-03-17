#!/usr/bin/env python3
import os, glob, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools

# My absolute paths - needs config
DATA_DIR = '/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/processed_sleepedf'
RESULT_DIR = '/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/results/sleepedf_res'
os.makedirs(RESULT_DIR, exist_ok=True)

# Training parameters
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
TRAIN_VAL_SPLIT = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
EMBEDDING_DIM = 128   # Dimension of the CNN encoder output per epoch
NUM_CLASSES = 5       # 0: Wake, 1: N1, 2: N2, 3: N3, 4: REM
NUM_TRANSFORMER_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.1
SEQ_LENGTH = 20       # Number of epochs per sequence (must match preprocessing)

class SleepSequenceDataset(Dataset):
    def __init__(self, data_dir):
        # Find all NPZ files ending with _sequences.npz
        self.files = glob.glob(os.path.join(data_dir, '*_sequences.npz'))
        sequences_list = []
        labels_list = []
        for f in self.files:
            loaded = np.load(f)
            sequences_list.append(loaded['sequences'])  # shape: (n_seq, 20, 2, 3000)
            labels_list.append(loaded['seq_labels'])      # shape: (n_seq, 20)
        self.sequences = np.concatenate(sequences_list, axis=0)
        self.seq_labels = np.concatenate(labels_list, axis=0)
        self.sequences = torch.from_numpy(self.sequences).float()
        self.seq_labels = torch.from_numpy(self.seq_labels).long()
    
    def __len__(self):
        return self.sequences.shape[0]
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.seq_labels[idx]

class EpochEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(EpochEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=50, stride=6),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        # The output dimension is computed approximately based on the input size 3000
        self.fc = nn.Linear(32 * (((3000 - 50) // 6 + 1) // 8 // 4), embedding_dim)
    
    def forward(self, x):
        # x: (batch, 2, 3000)
        out = self.net(x)
        out = out.view(out.size(0), -1)
        embedding = self.fc(out)
        return embedding

class SleepTransformer(nn.Module):
    def __init__(self, embedding_dim, num_classes, num_layers, num_heads, dropout, seq_length):
        super(SleepTransformer, self).__init__()
        self.encoder = EpochEncoder(embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.seq_length = seq_length

    def forward(self, x):
        # x: (batch, seq_length, 2, 3000)
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(-1, x.size(2), x.size(3))  # (batch*seq_length, 2, 3000)
        epoch_emb = self.encoder(x)           # (batch*seq_length, embedding_dim)
        epoch_emb = epoch_emb.view(batch_size, seq_len, -1)
        epoch_emb = epoch_emb.transpose(0, 1)   # (seq_length, batch, embedding_dim)
        transformer_out = self.transformer(epoch_emb)
        transformer_out = transformer_out.transpose(0, 1)  # (batch, seq_length, embedding_dim)
        logits = self.classifier(transformer_out)  # (batch, seq_length, num_classes)
        return logits

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for seq, labels in dataloader:
        seq, labels = seq.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(seq)  # (batch, seq_length, num_classes)
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
    with torch.no_grad():
        for seq, labels in dataloader:
            seq, labels = seq.to(device), labels.to(device)
            logits = model(seq)
            loss = criterion(logits.view(-1, NUM_CLASSES), labels.view(-1))
            running_loss += loss.item() * seq.size(0)
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, acc, all_preds, all_labels

def plot_curves(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_confusion(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    classes = ['W', 'N1', 'N2', 'N3', 'R']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

from sklearn.metrics import confusion_matrix, accuracy_score

# ----- Main Training Pipeline -----
def main():
    # Load dataset from processed sequence NPZ files
    dataset = SleepSequenceDataset(DATA_DIR)
    total_len = len(dataset)
    train_len = int(TRAIN_VAL_SPLIT * total_len)
    val_len = total_len - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SleepTransformer(EMBEDDING_DIM, NUM_CLASSES, NUM_TRANSFORMER_LAYERS, NUM_HEADS, DROPOUT, SEQ_LENGTH)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(1, NUM_EPOCHS+1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_preds, val_labels = eval_epoch(model, val_loader, criterion, DEVICE)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Epoch {epoch}/{NUM_EPOCHS}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    plot_curves(train_losses, val_losses, train_accs, val_accs)
    _, _, val_preds, val_labels = eval_epoch(model, val_loader, criterion, DEVICE)
    plot_confusion(val_labels, val_preds)
    
    # Save the model checkpoint in RESULT_DIR
    torch.save(model.state_dict(), os.path.join(RESULT_DIR, 'sleep_transformer_model.pth'))
    
if __name__ == '__main__':
    main()
