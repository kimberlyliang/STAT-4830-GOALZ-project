import os
import sys
import glob
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import shap
except ImportError:
    print("SHAP library not found. Please install it (`pip install shap`) to enable SHAP plots.")
    shap = None

# -------------------------
# ==== Initialization ====
# -------------------------

from pathlib import Path
# print(os.environ["HOME"])
# BASE = Path("/content/mydrive/MyDrive/4830_project/sleepedf_data")
# print(f"Using base directory: {BASE!s}")
# BASE = Path("/home1/k/kimliang/sleep/sleep_staging/data")
BASE = Path("/Users/kimberly/Documents/STAT4830/STAT-4830-GOALZ-project/data")

PROCESSED_DATA_DIR = BASE/"processed_mesa"
CATCH22_DATA_DIR   = BASE/"c22_processed_mesa"
PSD_DATA_DIR       = BASE/"features_psd_mesa"
RESULTS_DIR        = BASE/"mesa_hybrid_model_results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
for sub in ["plots", "models", "metrics"]:
    (RESULTS_DIR/sub).mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {RESULTS_DIR/sub}")

print("Checking directory access:")
# print(f"  {PROCESSED_DATA_DIR!s} exists: {PROCESSED_DATA_DIR.exists()}")
# print(f"  {CATCH22_DATA_DIR!s} exists: {CATCH22_DATA_DIR.exists()}")

# if not PROCESSED_DATA_DIR.exists() or not CATCH22_DATA_DIR.exists():
#     print("ERROR: Data directories not found. Please re-run preprocessing & Catch22 steps.")
#     sys.exit(1)

# for sub in ["plots","models","metrics"]:
#     (RESULTS_DIR/sub).mkdir(parents=True, exist_ok=True)

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameters
# BATCH_SIZE    = 32
# NUM_EPOCHS    = 35
# LEARNING_RATE = 1e-5
# TRAIN_RATIO   = 0.8
# SEQ_LENGTH    = 20
# SEQ_STRIDE    = 10
BATCH_SIZE    = 32
NUM_EPOCHS    = 50
LEARNING_RATE = 2e-4
TRAIN_RATIO   = 0.8
SEQ_LENGTH    = 30  # for better temporal context
SEQ_STRIDE    = 5   # for denser temporal sampling

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------
# ==== Utilities  =========
# -------------------------
def get_true_subject_id(filename):
    """Extracts the base subject ID (e.g., mesa-sleep-XXXX) from a filename."""
    basename = Path(filename).stem
    # MESA filenames look like 'mesa-sleep-XXXX_...' or 'mesa-sleep-XXXX'
    parts = basename.split('_')
    if len(parts) > 0 and parts[0].startswith("mesa-sleep-"):
        return parts[0]
    else:
        # Fallback or error handling if needed
        print(f"Warning: Could not extract MESA subject ID from {filename}")
        return basename # Or raise an error

def group_by_true_subjects(data_dir):
    files = glob.glob(str(data_dir/"*_sequences.npz"))
    subj_map = {}
    for f in files:
        rid = Path(f).stem.split("_")[0]
        subj = get_true_subject_id(rid)
        subj_map.setdefault(subj, []).append(rid)
    return subj_map

def split_true_subjects(data_dir, train_ratio=TRAIN_RATIO, random_state=SEED):
    subj_map = group_by_true_subjects(data_dir)
    subs = list(subj_map.keys())
    np.random.seed(random_state)
    np.random.shuffle(subs)
    n_train = int(len(subs)*train_ratio)
    train_subs = subs[:n_train]
    test_subs  = subs[n_train:]
    train_ids = [rid for s in train_subs for rid in subj_map[s]]
    test_ids  = [rid for s in test_subs  for rid in subj_map[s]]
    return train_ids, test_ids, train_subs, test_subs

# -------------------------
# ==== Dataset    =========
# -------------------------
class HybridSleepDataset(Dataset):
    def __init__(self, raw_dir, c22_dir, psd_dir, recording_ids=None):
        all_raw = glob.glob(str(raw_dir/"*_sequences.npz"))
        all_c22 = glob.glob(str(c22_dir  /"*_c22.csv"))
        all_psd = glob.glob(str(psd_dir  /"*_psd.npz"))
        if recording_ids is not None:
            all_raw = [p for p in all_raw if any(rid in p for rid in recording_ids)]
            all_c22 = [p for p in all_c22 if any(rid in p for rid in recording_ids)]
            all_psd = [p for p in all_psd if any(rid in p for rid in recording_ids)]
        self.raw_map = {Path(p).stem.split("_")[0]: p for p in all_raw}
        self.c22_map = {Path(p).stem.split("_")[0]: p for p in all_c22}
        self.psd_map = {Path(p).stem.split("_")[0]: p for p in all_psd}
        common = sorted(set(self.raw_map) & set(self.c22_map) & set(self.psd_map))
        if not common:
            raise ValueError("No overlapping recordings between raw & Catch22!")
        self.recording_ids = common

        seq_list, c22_list, psd_list, lbl_list = [], [], [], []
        for rid in common:
            # load raw
            data = np.load(self.raw_map[rid])
            seqs, labels = data["sequences"], data["seq_labels"]
            # load catch22
            df = pd.read_csv(self.c22_map[rid])
            feats_c22 = df.drop(columns=["label"]).values
            # psd features
            arr = np.load(self.psd_map[rid])
            feats_psd = arr["psd"] if "psd" in arr else arr[arr.files[0]]
            # if mismatch, pad/truncate
            n_seq = seqs.shape[0]
            expected = n_seq * SEQ_LENGTH
            def align(feats):
                if feats.shape[0] != expected:
                    feat_dim = feats.shape[1]
                    newf = np.zeros((n_seq, SEQ_LENGTH, feat_dim), dtype=np.float32)
                    for i in range(n_seq):
                        start = i * SEQ_STRIDE
                        end   = start + SEQ_LENGTH
                        if end <= feats.shape[0]:
                            newf[i] = feats[start:end]
                        else:
                            avail = feats.shape[0] - start
                            if avail>0:
                                newf[i,:avail] = feats[start:]
                                newf[i,avail:] = feats[start+avail-1]
                            else:
                                newf[i] = newf[i-1]
                    return newf
                else:
                    return feats.reshape(n_seq, SEQ_LENGTH, -1).astype(np.float32)

            feats_c22 = align(feats_c22)
            feats_psd = align(feats_psd)

            seq_list.append(seqs.astype(np.float32))
            c22_list.append(feats_c22)
            psd_list.append(feats_psd)
            lbl_list.append(labels.astype(np.int64))

        self.sequences   = torch.from_numpy(np.concatenate(seq_list,axis=0))
        self.c22_feats   = torch.from_numpy(np.concatenate(c22_list,axis=0))
        self.psd_feats  = torch.from_numpy(np.concatenate(psd_list, axis=0))
        self.seq_labels  = torch.from_numpy(np.concatenate(lbl_list,axis=0))
        # self.raw    = torch.from_numpy(np.stack(seqs,   axis=0))  # (N_segments, S, 2, T)
        # # self.c22_feats = torch.from_numpy(np.concatenate(c22_list, axis=0))
        # self.c22    = torch.from_numpy(np.stack(c22_list,   axis=0))  # (N_segments, S, feat_dim)
        # self.labels = torch.from_numpy(np.stack(labels, axis=0))  # (N_segments, S)


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (self.sequences[idx],
                self.c22_feats[idx],
                self.psd_feats[idx],
                self.seq_labels[idx])

# -------------------------
# ==== Model      =========
# -------------------------
class EpochEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # More filters and smaller kernels for finer feature detection
        # Adjust input channels from 2 to 3 for MESA (EEG, EOG-L, EOG-R)
        self.conv1 = nn.Conv1d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Add a fourth convolutional layer for more depth
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)

        # Feature refinement with attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(128, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 128, kernel_size=1),
            nn.Sigmoid()
        )

        # Use adaptive pooling to handle any input size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(48)

        # Fixed output size after adaptive pooling
        self.fc = nn.Linear(128 * 48, embedding_dim)
        self.ln = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.2)

        # Debug prints
        # print(f"EpochEncoder initialized with embedding_dim={embedding_dim}")

    def forward(self, x):
        B, S, C, T = x.shape
        # print(f"EpochEncoder input shape: B={B}, S={S}, C={C}, T={T}")

        x = x.view(B*S, C, T)
        # print(f"Reshaped to: {x.shape}")

        # First conv block with BN and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        # print(f"After first conv block: {x.shape}")

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        # print(f"After second conv block: {x.shape}")

        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        # print(f"After third conv block: {x.shape}")

        # Fourth conv block without pooling for finer features
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        # print(f"After fourth conv block: {x.shape}")

        # Apply channel attention
        attn = self.channel_attn(x)
        x = x * attn
        # print(f"After attention: {x.shape}")

        # Use adaptive pooling to get fixed output size
        x = self.adaptive_pool(x)
        # print(f"After adaptive pooling: {x.shape}")

        # Flatten and project
        x = x.view(B*S, -1)
        # print(f"After flattening: {x.shape}")

        x = self.dropout(F.relu(self.fc(x)))
        x = self.ln(x)
        # print(f"After FC and LN: {x.shape}")

        output = x.view(B, S, -1)
        # print(f"EpochEncoder final output shape: {output.shape}")

        return output

class C22Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=64):
        super().__init__()
        # print(f"Initializing C22Encoder with input_dim={input_dim}, embedding_dim={embedding_dim}")

        # Initial normalization
        self.ln0 = nn.LayerNorm(input_dim)

        # Simplified architecture - just use a standard MLP
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, embedding_dim)
        self.ln3 = nn.LayerNorm(embedding_dim)

        # Increased dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        B, S, D = x.shape
        # print(f"C22Encoder input shape: B={B}, S={S}, D={D}")

        x_flat = x.view(B*S, D)
        # print(f"x_flat shape: {x_flat.shape}")

        # Main encoder path
        norm_x = self.ln0(x_flat)
        x1 = self.dropout(F.relu(self.ln1(self.fc1(norm_x))))
        x2 = self.dropout(F.relu(self.ln2(self.fc2(x1))))
        x3 = self.dropout(F.relu(self.ln3(self.fc3(x2))))

        # print(f"C22Encoder output shape before reshape: {x3.shape}")
        output = x3.view(B, S, -1)
        # print(f"C22Encoder final output shape: {output.shape}")

        return output

class PSDEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=64):
        super().__init__()
        self.ln0 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        B,S,D = x.shape
        x = x.view(B*S, D)
        x = self.dropout(F.relu(self.ln1(self.fc1(self.ln0(x)))))
        x = self.dropout(F.relu(self.ln2(self.fc2(x))))
        return x.view(B, S, -1)

class HybridSleepTransformerPrevious(nn.Module):
    def __init__(self, c22_dim, psd_dim, raw_emb=128, c22_emb=64, psd_emb=64,
                 num_classes=5, num_layers=3, num_heads=8, dropout=0.2, seq_length=30):
        super().__init__()
        self.seq_length = seq_length
        self.epoch_enc = EpochEncoder(raw_emb)
        self.c22_enc   = C22Encoder(c22_dim, c22_emb)
        self.psd_enc   = PSDEncoder(psd_dim, psd_emb)
        # compute combined dim
        self.combined_dim = raw_emb + c22_emb + psd_emb

        # fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim),
            nn.LayerNorm(self.combined_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.combined_dim, self.combined_dim)
        )
        self.ln_fusion = nn.LayerNorm(self.combined_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_length, self.combined_dim))
        self.class_tokens = nn.Parameter(torch.randn(1, num_classes, self.combined_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.combined_dim,
            nhead=num_heads,
            dim_feedforward=8*self.combined_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # auxiliary classifiers
        self.aux_raw_classifier = nn.Sequential(
            nn.Linear(raw_emb,128),nn.LayerNorm(128),nn.ReLU(),nn.Dropout(dropout),nn.Linear(128,num_classes)
        )
        self.aux_c22_classifier = nn.Sequential(
            nn.Linear(c22_emb,128),nn.LayerNorm(128),nn.ReLU(),nn.Dropout(dropout),nn.Linear(128,num_classes)
        )
        self.aux_psd_classifier = nn.Sequential(
            nn.Linear(psd_emb,128),nn.LayerNorm(128),nn.ReLU(),nn.Dropout(dropout),nn.Linear(128,num_classes)
        )

        # shared and class-specific heads
        self.fc_shared = nn.Linear(self.combined_dim,256)
        self.ln_shared = nn.LayerNorm(256)
        self.dropout = nn.Dropout(dropout)
        self.fc_classes = nn.ModuleList([nn.Linear(256,1) for _ in range(num_classes)])

        # N1 detector
        self.n1_detector = nn.Sequential(nn.Linear(self.combined_dim,128),nn.LayerNorm(128),nn.ReLU())
        self.n1_lstm = nn.LSTM(128,64,batch_first=True,bidirectional=True)
        self.n1_attn = nn.MultiheadAttention(128,4,batch_first=True)
        self.n1_output = nn.Linear(128,1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias,0)

    def forward(self, raw, c22, psd):
        r = self.epoch_enc(raw)
        c = self.c22_enc(c22)
        p = self.psd_enc(psd)
        # align sequence lengths
        B, Sr, _ = r.shape
        _, Sc, _ = c.shape
        _, Sp, _ = p.shape
        S_min = min(Sr, Sc, Sp)
        r, c, p = r[:, :S_min, :], c[:, :S_min, :], p[:, :S_min, :]

        # auxiliary preds
        aux_r = self.aux_raw_classifier(r)
        aux_c = self.aux_c22_classifier(c)
        aux_p = self.aux_psd_classifier(p)

        # fuse
        x = torch.cat([r, c, p], dim=2)
        B,S,D = x.shape
        x_flat = x.view(B*S, D)
        x_fused = F.relu(self.ln_fusion(self.fusion(x_flat)))
        x = x_fused.view(B,S,D)
        x = x + self.pos_encoder[:, :S, :]
        class_tokens = self.class_tokens.expand(B, -1, -1)
        x = torch.cat([x, class_tokens], dim=1)
        out = self.transformer(x)
        seq_out, class_out = out[:, :S, :], out[:, S:, :]

        # N1 path
        n1_feat = self.n1_detector(seq_out)
        n1_lstm, _ = self.n1_lstm(n1_feat)
        n1_attn, _ = self.n1_attn(n1_lstm, n1_lstm, n1_lstm)
        n1_score = self.n1_output(n1_attn)

        shared = self.dropout(F.relu(self.ln_shared(self.fc_shared(seq_out))))
        class_outputs = []
        for i, fc in enumerate(self.fc_classes):
            score = fc(shared)
            if i == 1:
                score = score + n1_score
            class_outputs.append(score)
        output = torch.cat(class_outputs, dim=2)

        if self.training:
            return output, aux_r, aux_c, aux_p
        else:
            return output


#THIS ONE NOW HAS A ONE VS ALL CLASSIFIER
class HybridSleepTransformer(nn.Module):
    def __init__(self, c22_dim, psd_dim, raw_emb=128, c22_emb=64, psd_emb=64,
                 num_classes=5, num_layers=3, num_heads=8, dropout=0.2, seq_length=30):
        super().__init__()
        self.seq_length = seq_length
        # existing encoders
        self.epoch_enc = EpochEncoder(raw_emb)
        self.c22_enc   = C22Encoder(c22_dim, c22_emb)
        self.psd_enc   = PSDEncoder(psd_dim, psd_emb)
        self.combined_dim = raw_emb + c22_emb + psd_emb

        # fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim),
            nn.LayerNorm(self.combined_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.combined_dim, self.combined_dim)
        )
        self.ln_fusion = nn.LayerNorm(self.combined_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_length, self.combined_dim))
        self.class_tokens = nn.Parameter(torch.randn(1, num_classes, self.combined_dim))

        # transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.combined_dim,
            nhead=num_heads,
            dim_feedforward=8*self.combined_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # main multi-class heads
        self.fc_shared = nn.Linear(self.combined_dim,256)
        self.ln_shared = nn.LayerNorm(256)
        self.dropout = nn.Dropout(dropout)
        self.fc_classes = nn.ModuleList([nn.Linear(256,1) for _ in range(num_classes)])

        # one-vs-rest head for N1 (class index 1)
        self.n1_ovr_head = nn.Sequential(
            nn.Linear(self.combined_dim,128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

        # initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, raw, c22, psd):
        # encode inputs
        r = self.epoch_enc(raw)
        c = self.c22_enc(c22)
        p = self.psd_enc(psd)
        # align sequence lengths
        B, Sr, _ = r.shape
        S = min(r.size(1), c.size(1), p.size(1))
        r, c, p = r[:, :S, :], c[:, :S, :], p[:, :S, :]

        # fuse modalities
        x = torch.cat([r, c, p], dim=2)
        B, S, D = x.shape
        x_flat = x.view(B*S, D)
        x_fused = F.relu(self.ln_fusion(self.fusion(x_flat)))
        x = x_fused.view(B, S, D) + self.pos_encoder[:, :S, :]

        # transformer
        class_tokens = self.class_tokens.expand(B, -1, -1)
        x_trans = self.transformer(torch.cat([x, class_tokens], dim=1))
        seq_out = x_trans[:, :S, :]
        class_out = x_trans[:, S:, :]

        # multi-class output
        shared = self.dropout(F.relu(self.ln_shared(self.fc_shared(seq_out))))
        logits = torch.cat([fc(shared) for fc in self.fc_classes], dim=2)

        # one-vs-rest N1 output
        # binary probability for N1 vs all for each time step
        n1_ovr_prob = self.n1_ovr_head(seq_out).squeeze(-1)

        if self.training:
            # during training, return both outputs for computing losses
            return logits, n1_ovr_prob
        else:
            # inference: return both multi-class predictions and N1 probabilities
            preds = logits.argmax(dim=-1)
            return preds, n1_ovr_prob

# ------------------------- 
# ==== Plotting Utils ====
# -------------------------

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]

def plot_train_val_curves(history, fold_num, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation loss')
    plt.title(f'Fold {fold_num} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training accuracy')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation accuracy')
    plt.title(f'Fold {fold_num} Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / f"fold{fold_num}_train_val_curves.png")
    plt.close()

def plot_confusion_matrix(cm, fold_num, save_dir, title_suffix=""):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Fold {fold_num} Confusion Matrix {title_suffix}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_dir / f"fold{fold_num}_confusion_matrix{title_suffix.replace(' ','_')}.png")
    plt.close()

def plot_roc_curves(labels, probs, fold_num, save_dir):
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(labels == i, probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Fold {fold_num} Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_dir / f"fold{fold_num}_roc_curves.png")
    plt.close()

def plot_precision_recall_curves(labels, probs, fold_num, save_dir):
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(CLASS_NAMES):
        precision, recall, _ = precision_recall_curve(labels == i, probs[:, i])
        avg_prec = average_precision_score(labels == i, probs[:, i])
        plt.plot(recall, precision, label=f'{class_name} (AP = {avg_prec:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Fold {fold_num} Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(save_dir / f"fold{fold_num}_precision_recall_curves.png")
    plt.close()

def plot_calibration_curve(labels, probs, fold_num, save_dir):
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for i, class_name in enumerate(CLASS_NAMES):
        prob_true, prob_pred = calibration_curve(labels == i, probs[:, i], n_bins=10, strategy='uniform')
        plt.plot(prob_pred, prob_true, "s-", label=f'{class_name}')

    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted probability")
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")
    plt.title(f'Fold {fold_num} Calibration Curve')
    plt.grid(True)
    plt.savefig(save_dir / f"fold{fold_num}_calibration_curve.png")
    plt.close()

def plot_spectrograms(raw_data, labels, n_examples=5, fold_num=0, save_dir=None):
    # raw_data shape: (N_seq, S, C, T) -> select first sequence in batch, first few epochs
    if raw_data.shape[0] == 0:
      print("Cannot plot spectrograms, no raw data provided.")
      return

    seq_idx = 0 # Plot from the first sequence
    epochs_to_plot = min(n_examples, raw_data.shape[1])
    fs = 100 # Assuming 100 Hz sampling rate from typical preprocessing
    nperseg = 64 # Segment length for FFT

    fig, axes = plt.subplots(epochs_to_plot, raw_data.shape[2], figsize=(15, 3 * epochs_to_plot), squeeze=False)
    fig.suptitle(f"Fold {fold_num} Example Spectrograms (First {epochs_to_plot} epochs)", fontsize=16)

    for epoch_idx in range(epochs_to_plot):
        epoch_label = CLASS_NAMES[labels[seq_idx, epoch_idx]]
        for channel_idx in range(raw_data.shape[2]): # Iterate through channels (EEG, EOG-L, EOG-R)
            ax = axes[epoch_idx, channel_idx]
            signal = raw_data[seq_idx, epoch_idx, channel_idx, :].cpu().numpy()

            # Use matplotlib's specgram
            Pxx, freqs, bins, im = ax.specgram(signal, NFFT=nperseg, Fs=fs, noverlap=nperseg//2, cmap='viridis')
            ax.set_title(f"Epoch {epoch_idx+1} (True: {epoch_label}) - Ch {channel_idx+1}")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Frequency [Hz]")
            # Add colorbar
            # fig.colorbar(im, ax=ax).set_label('Intensity [dB]') # Can clutter plot

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    if save_dir:
        plt.savefig(save_dir / f"fold{fold_num}_example_spectrograms.png")
    else:
        plt.show()
    plt.close()

def run_shap_analysis(model, loader, fold_num, save_dir, n_background=50, n_test=20):
    if shap is None:
        print("SHAP library not installed. Skipping SHAP analysis.")
        return

    print(f"Running SHAP analysis for Fold {fold_num}...")
    model.eval()

    # Get background data (e.g., from training loader)
    background_raw, background_c22, background_psd = [], [], []
    count = 0
    bg_loader = DataLoader(loader.dataset, batch_size=n_background, shuffle=True)
    try:
      for raw, c22, psd, _ in bg_loader:
          background_raw.append(raw.to(device))
          background_c22.append(c22.to(device))
          background_psd.append(psd.to(device))
          break # Take first batch as background
    except Exception as e:
        print(f"Error getting background data: {e}. Skipping SHAP.")
        return

    if not background_raw:
        print("Could not get background data. Skipping SHAP.")
        return

    background_raw = torch.cat(background_raw, dim=0)
    background_c22 = torch.cat(background_c22, dim=0)
    background_psd = torch.cat(background_psd, dim=0)

    # Get test data subset
    test_raw, test_c22, test_psd, test_labels = [], [], [], []
    test_loader_subset = DataLoader(loader.dataset, batch_size=n_test, shuffle=True) # Use test loader's dataset
    try:
        for raw, c22, psd, lbl in test_loader_subset:
            test_raw.append(raw.to(device))
            test_c22.append(c22.to(device))
            test_psd.append(psd.to(device))
            test_labels.append(lbl)
            break # Take first batch
    except Exception as e:
        print(f"Error getting test data subset: {e}. Skipping SHAP.")
        return

    if not test_raw:
        print("Could not get test data subset. Skipping SHAP.")
        return

    test_raw = torch.cat(test_raw, dim=0)
    test_c22 = torch.cat(test_c22, dim=0)
    test_psd = torch.cat(test_psd, dim=0)
    # test_labels = torch.cat(test_labels, dim=0)

    # Define a wrapper function for the model, as SHAP expects a function
    # that takes numpy arrays or tensors and returns numpy arrays.
    # This needs careful handling based on SHAP explainer type.

    # Using DeepExplainer (requires background data)
    # Note: SHAP interaction with complex inputs (raw + features) can be tricky.
    # We might need to explain parts separately or simplify the input.
    # Explaining based on the fused representation before the final classifier might be more feasible.

    # --- Attempting explanation on combined input (might be slow/memory intensive) ---
    try:
        # Need a model wrapper that takes *only* tensors SHAP can handle directly.
        # Let's try explaining the final classification layers based on the Transformer output.

        # Create a forward hook to capture transformer output
        transformer_output = None
        def hook_fn(module, input, output):
            nonlocal transformer_output
            # Output from transformer: seq_out = output[:, :S, :]
            transformer_output = output[0][:, :model.seq_length, :].detach() # Get sequence output

        hook = model.transformer.register_forward_hook(hook_fn)

        # Run background data through model up to transformer
        _ = model(background_raw, background_c22, background_psd)
        background_transformer_output = transformer_output

        # Run test data through model up to transformer
        _ = model(test_raw, test_c22, test_psd)
        test_transformer_output = transformer_output
        hook.remove() # Remove hook

        if background_transformer_output is None or test_transformer_output is None:
             raise ValueError("Could not capture transformer output.")

        # Define a sub-model from transformer output to logits
        class PostTransformerModel(nn.Module):
            def __init__(self, main_model):
                super().__init__()
                self.dropout = main_model.dropout
                self.ln_shared = main_model.ln_shared
                self.fc_shared = main_model.fc_shared
                self.fc_classes = main_model.fc_classes
                self.n1_ovr_head = main_model.n1_ovr_head # Assuming this exists

            def forward(self, seq_out):
                 shared = self.dropout(F.relu(self.ln_shared(self.fc_shared(seq_out))))
                 logits = torch.cat([fc(shared) for fc in self.fc_classes], dim=2)
                 n1_ovr_prob = self.n1_ovr_head(seq_out).squeeze(-1)
                 # SHAP DeepExplainer typically needs a single output tensor (e.g., logits)
                 return logits

        sub_model = PostTransformerModel(model).to(device)
        sub_model.eval()

        explainer = shap.DeepExplainer(sub_model, background_transformer_output)
        shap_values = explainer.shap_values(test_transformer_output)

        # Generate SHAP summary plot (feature importance)
        # shap_values is a list (one per class), each entry (n_test, seq_len, feature_dim)
        # We need to average over sequence length or flatten
        # Flattening approach:
        shap_values_flat = [val.reshape(test_transformer_output.shape[0], -1) for val in shap_values]
        test_transformer_output_flat = test_transformer_output.reshape(test_transformer_output.shape[0], -1)

        plt.figure()
        # Plot for each class
        shap.summary_plot(shap_values_flat, test_transformer_output_flat, plot_type="bar", class_names=CLASS_NAMES, show=False)
        plt.title(f"Fold {fold_num} SHAP Feature Importance (Transformer Output)")
        plt.tight_layout()
        plt.savefig(save_dir / f"fold{fold_num}_shap_summary_bar.png")
        plt.close()

        # You can add other SHAP plots like force plots for individual samples if needed

    except Exception as e:
        print(f"Error during SHAP analysis for fold {fold_num}: {e}")
        import traceback
        traceback.print_exc()

    print(f"SHAP analysis for Fold {fold_num} finished.")

# -------------------------
# ==== Loss       =========
# -------------------------
def focal_loss(inputs, targets, alpha_general=0.25, alpha_n1=0.9, gamma=2.5):
    """
    Enhanced Focal Loss with stronger focus on N1 class.
    - inputs: logits of shape (B, S, C)
    - targets: ground‑truth indices of shape (B, S)
    """
    # Flatten to (N, C) and (N,)
    B, S, C = inputs.shape
    logits = inputs.view(-1, C)      # (N, C)
    tgt    = targets.view(-1)        # (N,)

    # Log‑softmax + probabilities
    logp = F.log_softmax(logits, dim=1)         # (N, C)
    p    = torch.exp(logp).clamp(min=1e-7)      # (N, C)

    # Cross‑entropy per sample
    ce   = F.nll_loss(logp, tgt, reduction='none')  # (N,)

    # p_t = probability of the true class for each sample
    pt   = p.gather(1, tgt.unsqueeze(1)).squeeze(1)  # (N,)

    # Enhanced alpha weighting for difficult classes
    # N1 gets highest weight, but also boost N3 and REM moderately
    n1_mask = (tgt == 1).float()
    n3_mask = (tgt == 3).float()
    rem_mask = (tgt == 4).float()

    # Apply different alpha weights to different classes
    alpha = alpha_general * (1 - n1_mask - n3_mask - rem_mask) + \
            alpha_n1 * n1_mask + \
            0.5 * n3_mask + \
            0.45 * rem_mask

    # Enhanced focal term with higher gamma
    loss = alpha * ((1 - pt) ** gamma) * ce   # (N,)

    return loss.mean()

# -------------------------
# ==== Training Loop ======
# -------------------------
def mixup_batch(raw, c22, psd, labels, alpha=0.2):
    """Apply mixup augmentation to a batch."""
    lam = np.random.beta(alpha, alpha) if alpha>0 else 1
    batch_size = raw.size(0)
    idx = torch.randperm(batch_size).to(raw.device)
    mixed_raw = lam * raw + (1 - lam) * raw[idx]
    mixed_c22 = lam * c22 + (1 - lam) * c22[idx]
    mixed_psd = lam * psd + (1 - lam) * psd[idx]
    return mixed_raw, mixed_c22, mixed_psd, labels, labels[idx], lam

def train_epoch(model, loader, optimizer, scheduler=None, mixup_alpha=0.2):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for raw, c22, psd, labels in loader:
        raw = torch.nan_to_num(raw, nan=0.0, posinf=1e5, neginf=-1e5).to(device)
        c22 = torch.nan_to_num(c22, nan=0.0, posinf=1e5, neginf=-1e5).to(device)
        psd = torch.nan_to_num(psd, nan=0.0, posinf=1e5, neginf=-1e5).to(device)
        labels = labels.to(device)

        if np.random.rand() < 0.5:
            mixed_raw, mixed_c22, mixed_psd, lbl_a, lbl_b, lam = mixup_batch(raw, c22, psd, labels, alpha=mixup_alpha)
            use_mixup = True
        else:
            mixed_raw, mixed_c22, mixed_psd = raw, c22, psd
            lbl_a = labels
            use_mixup = False

        optimizer.zero_grad()

        # Forward pass with auxiliary outputs during training
        outputs = model(mixed_raw, mixed_c22, mixed_psd)

        outputs = model(mixed_raw, mixed_c22, mixed_psd)
        main_preds, aux_r, aux_c, aux_p = outputs

        main_loss = focal_loss(main_preds, lbl_a)
        if use_mixup:
            loss = main_loss \
                + 0.3*(lam*focal_loss(aux_r,lbl_a)+(1-lam)*focal_loss(aux_r,lbl_b)) \
                + 0.3*(lam*focal_loss(aux_c,lbl_a)+(1-lam)*focal_loss(aux_c,lbl_b)) \
                + 0.3*(lam*focal_loss(aux_p,lbl_a)+(1-lam)*focal_loss(aux_p,lbl_b))
        else:
            loss = main_loss + 0.3*(focal_loss(aux_r,labels)+focal_loss(aux_c,labels)+focal_loss(aux_p,labels))

        # Check for numerical stability
        if not torch.isfinite(loss):
            # print("Skipping batch: non-finite loss")
            continue

        # Backward pass and optimization
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Increased from 0.5
        optimizer.step()

        # Update learning rate if using OneCycleLR or similar
        if scheduler is not None and isinstance(scheduler, (
            torch.optim.lr_scheduler.OneCycleLR,
            torch.optim.lr_scheduler.CyclicLR
        )):
            scheduler.step()

        # Calculate running statistics
        running_loss += loss.item() * raw.size(0)

        # Calculate accuracy (only for non-mixup batches)
        if not use_mixup:
            preds = main_preds.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()


    avg_loss = running_loss / len(loader.dataset)
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy

def smooth_predictions(predictions, window_size=5):
    """Apply temporal smoothing to predictions with special handling for N1 class.

    Args:
        predictions: 1D array of class predictions
        window_size: Size of smoothing window (should be odd)

    Returns:
        1D array of smoothed predictions
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure window size is odd

    smoothed = np.copy(predictions)
    padded = np.pad(predictions, (window_size//2, window_size//2), mode='edge')

    for i in range(len(smoothed)):
        window = padded[i:i+window_size]

        # Special handling for N1 class (index 1)
        if 1 in window:  # If N1 is in the window
            n1_count = np.sum(window == 1)
            # If N1 appears multiple times or is in the center, preserve it
            if n1_count > 1 or window[window_size//2] == 1:
                smoothed[i] = 1
            # Otherwise use consensus voting
            else:
                counts = np.bincount(window, minlength=5)
                # Exclude N1 from voting if it's just a single occurrence
                if counts[1] == 1:
                    counts[1] = 0
                smoothed[i] = np.argmax(counts)
        else:
            # For non-N1 windows, use standard mode
            smoothed[i] = np.argmax(np.bincount(window, minlength=5))

    # Additional rule: prevent isolated W (0) or REM (4) classes
    for i in range(1, len(smoothed)-1):
        if (smoothed[i] == 0 or smoothed[i] == 4) and smoothed[i-1] == smoothed[i+1] and smoothed[i] != smoothed[i-1]:
            smoothed[i] = smoothed[i-1]

    return smoothed

def eval_epoch(model, loader, apply_smoothing=True):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    all_raw_preds = []  # Store pre-smoothing predictions

    with torch.no_grad():
        for raw, c22, psd, labels in loader:
            raw = torch.nan_to_num(raw,0.0,1e5,-1e5).to(device)
            c22 = torch.nan_to_num(c22,0.0,1e5,-1e5).to(device)
            psd = torch.nan_to_num(psd,0.0,1e5,-1e5).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(raw, c22, psd)

            # Handle outputs based on model training mode
            if isinstance(outputs, tuple):
                # Assuming the first element is logits, second is N1 OvR prob
                logits = outputs[0]
                # We might use n1_ovr_prob later if needed
            else:
                logits = outputs

            if not torch.isfinite(logits).all():
                print("Warning: non-finite logits in evaluation")
                continue

            # Calculate loss using logits
            loss = focal_loss(logits, labels)
            running_loss += loss.item() * raw.size(0)

            # Get predictions (argmax from logits)
            preds = logits.argmax(dim=-1)

            # Calculate probabilities for metrics/plots
            probs = torch.softmax(logits, dim=-1)

            # Store predictions, labels, and probabilities (flattened)
            all_raw_preds.append(preds.cpu().numpy().ravel())
            all_labels.append(labels.cpu().numpy().ravel())
            all_probs.append(probs.cpu().numpy().reshape(-1, logits.shape[-1])) # Reshape probs correctly

    # Return early if no valid predictions
    if not all_raw_preds:
        return float('nan'), 0.0, 0.0, [], [], [] # Added empty list for probs

    # Concatenate all collected data
    all_raw_preds = np.concatenate(all_raw_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)

    # Calculate raw accuracy
    raw_acc = (all_raw_preds == all_labels).mean()

    # Apply temporal smoothing if requested
    if apply_smoothing:
        all_smoothed_preds = smooth_predictions(all_raw_preds)
        smoothed_acc = (all_smoothed_preds == all_labels).mean()
        final_preds = all_smoothed_preds
    else:
        smoothed_acc = raw_acc
        final_preds = all_raw_preds

    # No need to calculate class_accs here, classification_report does it
    # class_accs = []
    # ...

    return running_loss/len(loader.dataset), raw_acc, smoothed_acc, final_preds, all_labels, all_probs

torch.autograd.set_detect_anomaly(True)

# Hyperparameters (updated)
BATCH_SIZE = 32
NUM_EPOCHS = 50  # Increased from 35
LEARNING_RATE = 2e-4  # Increased from 1e-5
SEQ_LENGTH = 30  # Increased from 20
SEQ_STRIDE = 5   # Decreased from 10 for more temporal context

# prepare CV splits
subj_map = group_by_true_subjects(PROCESSED_DATA_DIR)
subjects = list(subj_map.keys())
np.random.seed(SEED)
np.random.shuffle(subjects)
folds = np.array_split(subjects, 5)

# Track metrics for all folds
fold_results = {
    'accuracy': [],
    'f1_n1': [],
    'f1_n2': [],
    'f1_n3': [],
    'f1_rem': [],
    'f1_wake': []
}
# Store epoch histories for all folds
all_fold_histories = []

# Full per-class results for all folds
all_fold_confusion = np.zeros((5, 5))  # 5 classes x 5 classes
all_fold_f1 = np.zeros((5, 5))  # 5 folds x 5 classes

for k in range(5):
    print(f"\n=== Starting Fold {k+1}/5 ===")
    test_subs = folds[k]
    train_subs = [s for i, f in enumerate(folds) if i!=k for s in f]
    train_ids = [rid for s in train_subs for rid in subj_map[s]]
    test_ids = [rid for s in test_subs for rid in subj_map[s]]

    # Create datasets with updated sequence length
    train_ds = HybridSleepDataset(
        PROCESSED_DATA_DIR,
        CATCH22_DATA_DIR,
        PSD_DATA_DIR,
        recording_ids=train_ids
    )
    test_ds = HybridSleepDataset(
        PROCESSED_DATA_DIR,
        CATCH22_DATA_DIR,
        PSD_DATA_DIR,
        recording_ids=test_ids
    )

    # Enhanced weighted sampler with more accurate class distribution

    # new start
    # Flatten all sequence labels to get true distribution
    flat_labels = train_ds.seq_labels.reshape(-1).numpy()
    class_counts = np.bincount(flat_labels, minlength=5)

    # Print class distribution
    print("Class distribution in training set (all epochs):")
    classes = ["Wake", "N1", "N2", "N3", "REM"]
    for i, cls in enumerate(classes):
        print(f"  {cls}: {class_counts[i]} samples ({class_counts[i]/len(flat_labels)*100:.1f}%)")

    # Calculate weights based on the true distribution
    class_weights = 1.0 / np.sqrt(class_counts + 1e-6)
    # Extra boost for N1 class
    class_weights[1] *= 1.5

    # For sampling, we need weights per sequence
    # Using the mean of the class weights for all epochs in each sequence
    sequence_weights = np.zeros(len(train_ds))
    for i in range(len(train_ds)):
        seq_labels = train_ds.seq_labels[i].numpy()
        seq_weights = class_weights[seq_labels]
        sequence_weights[i] = np.mean(seq_weights)

    sampler = WeightedRandomSampler(sequence_weights, len(sequence_weights), replacement=True)
    # new end

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Create model with updated architecture
    c22_dim = train_ds.c22_feats.shape[-1]
    psd_dim = train_ds.psd_feats.shape[-1]
    model = HybridSleepTransformer(
        c22_dim=c22_dim,
        psd_dim=psd_dim,
        raw_emb=128,
        c22_emb=64,
        psd_emb=64,
        num_classes=5,
        num_layers=3,
        num_heads=8,
        dropout=0.2,
        seq_length=SEQ_LENGTH
    ).to(device)

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Use OneCycleLR for better convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # Warm up for 30% of training
        div_factor=25,  # Initial LR is max_lr/25
        final_div_factor=1000  # End with LR 1000x smaller than max
    )

    # Early stopping setup
    patience = 7
    patience_counter = 0
    best_f1_n1 = 0  # Track best N1 F1 score specifically
    best_loss = float('inf')
    best_epoch = 0

    # Training loop
    fold_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler)

        # Evaluate with smoothing
        val_loss, raw_acc, smoothed_acc, val_preds, val_labels, val_probs = eval_epoch(model, test_loader, apply_smoothing=True)

        # Store metrics for plotting curves
        fold_history['train_loss'].append(train_loss)
        fold_history['train_acc'].append(train_acc) # Note: train_acc might be 0 if all batches used mixup
        fold_history['val_loss'].append(val_loss)
        fold_history['val_acc'].append(smoothed_acc) # Use smoothed accuracy for consistency

        # Calculate per-class metrics
        report = classification_report(val_labels, val_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        f1_scores = [report[c]['f1-score'] for c in ["W","N1","N2","N3","REM"]]

        # Print detailed metrics
        print(f"Fold{k+1} E{epoch+1}/{NUM_EPOCHS}:")
        print(f"  Loss: train={train_loss:.4f}, val={val_loss:.4f}")
        print(f"  Acc : raw={raw_acc:.4f}, smoothed={smoothed_acc:.4f}")
        print(f"  F1  : W={f1_scores[0]:.4f}, N1={f1_scores[1]:.4f}, N2={f1_scores[2]:.4f}, N3={f1_scores[3]:.4f}, REM={f1_scores[4]:.4f}")

        # Adjust learning rate for ReduceLROnPlateau
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) and np.isfinite(val_loss):
            scheduler.step(val_loss)

        # Check if this is the best model focused on N1 performance
        current_f1_n1 = report['N1']['f1-score']
        if current_f1_n1 > best_f1_n1:
            best_f1_n1 = current_f1_n1
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) else None,
                'val_loss': val_loss,
                'smoothed_acc': smoothed_acc,
                'f1_scores': f1_scores,
            }, RESULTS_DIR/f"models/best_fold{k+1}.pth")

            print(f"  → New best model saved (F1-N1: {current_f1_n1:.4f})")
        else:
            patience_counter += 1
            print(f"  → No improvement for {patience_counter}/{patience} epochs. Best F1-N1: {best_f1_n1:.4f}")

        # Early stopping
        if patience_counter >= patience:
            print(f"  → Early stopping triggered after {epoch+1} epochs")
            break

    # Final evaluation on best model
    print(f"\nLoading best model from epoch {best_epoch+1} for final evaluation and plotting.")
    try:
      checkpoint = torch.load(RESULTS_DIR/f"models/best_fold{k+1}.pth")
      model.load_state_dict(checkpoint['model_state_dict'])
    except FileNotFoundError:
      print(f"Warning: Checkpoint file not found for fold {k+1}. Using model state from last epoch.")

    # Final evaluation with detailed metrics
    _, _, final_acc, final_preds, final_labels, final_probs = eval_epoch(model, test_loader, apply_smoothing=True)
    final_report = classification_report(final_labels, final_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    final_cm = confusion_matrix(final_labels, final_preds)

    # Print final results
    print(f"\n==> Fold{k+1} final results:")
    print(f"  Accuracy: {final_acc:.4f}")
    print("  Per-class F1 scores:")
    for i, cls in enumerate(["W","N1","N2","N3","REM"]):
        print(f"    {cls}: {final_report[cls]['f1-score']:.4f}")
        all_fold_f1[k, i] = final_report[cls]['f1-score']

    print("\n  Confusion Matrix:")
    print("    W    N1    N2    N3    REM")
    for i in range(5):
        row_str = "  " + " ".join(f"{final_cm[i,j]:5d}" for j in range(5))
        print(f"{['W','N1','N2','N3','REM'][i]}: {row_str}")

    # Accumulate confusion matrix
    all_fold_confusion += final_cm

    # Store results for this fold
    fold_results['accuracy'].append(final_acc)
    fold_results['f1_wake'].append(final_report['W']['f1-score'])
    fold_results['f1_n1'].append(final_report['N1']['f1-score'])
    fold_results['f1_n2'].append(final_report['N2']['f1-score'])
    fold_results['f1_n3'].append(final_report['N3']['f1-score'])
    fold_results['f1_rem'].append(final_report['REM']['f1-score'])

    # Save detailed results for this fold
    np.savez(
        RESULTS_DIR/f"metrics/fold{k+1}_results.npz",
        predictions=final_preds,
        labels=final_labels,
        confusion_matrix=final_cm,
        report=final_report
    )

    # Plot example spectrograms (using a sample from test loader)
    try:
        raw_sample, _, _, lbl_sample = next(iter(test_loader))
        plot_spectrograms(raw_sample, lbl_sample, fold_num=k+1, save_dir=RESULTS_DIR/"plots")
    except Exception as e:
        print(f"Could not plot spectrograms for fold {k+1}: {e}")

    # Run SHAP analysis
    run_shap_analysis(model, test_loader, k+1, RESULTS_DIR/"plots")

    # Store epoch history for this fold
    all_fold_histories.append(fold_history)

# Print overall cross-validation summary
print("\n=== Cross-Validation Summary ===")
print("Fold accuracies:", fold_results['accuracy'])
print(f"Mean accuracy: {np.mean(fold_results['accuracy']):.4f} ± {np.std(fold_results['accuracy']):.4f}")

print("\nPer-class F1 scores across folds:")
for cls, key in zip(["W","N1","N2","N3","REM"],
                    ['f1_wake', 'f1_n1', 'f1_n2', 'f1_n3', 'f1_rem']):
    print(f"  {cls}: {np.mean(fold_results[key]):.4f} \u00B1 {np.std(fold_results[key]):.4f}")

# Print overall confusion matrix
plot_confusion_matrix(all_fold_confusion, fold_num="Overall", save_dir=RESULTS_DIR/"plots")
print("\nOverall Confusion Matrix:")
print("      W      N1      N2      N3      REM")
for i in range(5):
    row_str = "  " + " ".join(f"{all_fold_confusion[i,j]:7.0f}" for j in range(5))
    print(f"{['W','N1','N2','N3','REM'][i]}: {row_str}")

# Calculate per-class metrics from overall confusion matrix
precision = np.zeros(5)
recall = np.zeros(5)
f1 = np.zeros(5)

for i in range(5):
    # Precision: TP / (TP + FP)
    precision[i] = all_fold_confusion[i, i] / np.sum(all_fold_confusion[:, i]) if np.sum(all_fold_confusion[:, i]) > 0 else 0
    # Recall: TP / (TP + FN)
    recall[i] = all_fold_confusion[i, i] / np.sum(all_fold_confusion[i, :]) if np.sum(all_fold_confusion[i, :]) > 0 else 0
    # F1: 2 * precision * recall / (precision + recall)
    f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

print("\nOverall Per-class Metrics:")
for i, cls in enumerate(["W","N1","N2","N3","REM"]):
    print(f"  {cls}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")

np.savez(
    RESULTS_DIR/"metrics/cv_summary.npz",
    fold_accuracies=fold_results['accuracy'],
    fold_f1_scores=all_fold_f1,
    overall_confusion=all_fold_confusion,
    overall_precision=precision,
    overall_recall=recall,
    overall_f1=f1
)

# Plot average train/val curves (optional)
if all_fold_histories:
    avg_train_loss = np.mean([h['train_loss'] for h in all_fold_histories], axis=0)
    avg_val_loss = np.mean([h['val_loss'] for h in all_fold_histories], axis=0)
    avg_train_acc = np.mean([h['train_acc'] for h in all_fold_histories], axis=0)
    avg_val_acc = np.mean([h['val_acc'] for h in all_fold_histories], axis=0)
    epochs = range(1, len(avg_train_loss) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, avg_train_loss, 'bo-', label='Avg Training loss')
    plt.plot(epochs, avg_val_loss, 'ro-', label='Avg Validation loss')
    plt.title('Average Training and Validation Loss Across Folds')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, avg_train_acc, 'bo-', label='Avg Training accuracy')
    plt.plot(epochs, avg_val_acc, 'ro-', label='Avg Validation accuracy')
    plt.title('Average Training and Validation Accuracy Across Folds')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plots/overall_avg_train_val_curves.png")
    plt.close()

print("\nTraining completed. Results and plots saved to:", RESULTS_DIR)
