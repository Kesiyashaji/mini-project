"""
ASL Sequence Model Trainer

Trains a PyTorch sequence model (LSTM or Transformer) on landmark clip data
collected by collect_seq.py, then exports the model to ONNX for browser inference.

Usage:
    python train_seq.py                  # Train with default LSTM
    python train_seq.py --model transformer  # Train with Transformer

The trained model is exported to:
    web-app/public/models/model.onnx
"""

import argparse
import json
import math
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- Paths ---
DATASET_PATH = Path(__file__).parent.parent / "data" / "sequence_dataset" / "sequences.json"
MODEL_OUTPUT = Path(__file__).parent.parent.parent / "web-app" / "public" / "models" / "model.onnx"
METADATA_OUTPUT = Path(__file__).parent.parent.parent / "web-app" / "public" / "models" / "model_meta.json"


# ============================================================
#  Dataset
# ============================================================

class SignSequenceDataset(Dataset):
    """PyTorch dataset for sign language landmark sequences."""

    def __init__(self, sequences, labels):
        # sequences: list of [seq_length, n_features] arrays
        self.sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ============================================================
#  Models
# ============================================================

class LSTMClassifier(nn.Module):
    """Bidirectional LSTM for sequence classification.

    Architecture:
        Input (T, 63) -> BiLSTM (128 hidden) -> Dropout -> FC (64) -> FC (n_classes)

    We take the LAST hidden state from both directions and concatenate them.
    """

    def __init__(self, n_features=63, hidden_size=128, n_layers=2, n_classes=5, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 64)  # *2 for bidirectional
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        lstm_out, (h_n, _) = self.lstm(x)
        # h_n: (n_layers * 2, batch, hidden_size)
        # Take the last layer's forward and backward hidden states
        h_forward = h_n[-2]  # (batch, hidden_size)
        h_backward = h_n[-1]  # (batch, hidden_size)
        hidden = torch.cat([h_forward, h_backward], dim=1)  # (batch, hidden_size * 2)
        out = self.dropout(hidden)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerClassifier(nn.Module):
    """Transformer Encoder for sequence classification.

    Architecture:
        Input (T, 63) -> Linear projection (128) -> PosEnc -> TransformerEncoder (4 layers)
        -> Mean Pool -> FC (64) -> FC (n_classes)
    """

    def __init__(self, n_features=63, d_model=128, n_heads=4, n_layers=4, n_classes=5, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)       # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        # Global average pooling over the sequence dimension
        x = x.mean(dim=1)            # (batch, d_model)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================
#  Training
# ============================================================

def load_dataset():
    """Load the sequence dataset from JSON."""
    print(f"Loading dataset from {DATASET_PATH}")
    with open(DATASET_PATH, "r") as f:
        data = json.load(f)

    sign_names = data["sign_names"]
    seq_length = data["seq_length"]
    n_features = data["n_features"]

    sequences = []
    labels = []
    for clip in data["clips"]:
        sequences.append(clip["frames"])
        labels.append(clip["label"])

    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print(f"  Loaded {len(labels)} clips")
    print(f"  Shape: ({len(labels)}, {seq_length}, {n_features})")
    print(f"  Signs: {sign_names}")
    print(f"  Classes: {len(set(labels))}")

    return sequences, labels, sign_names, seq_length, n_features


def train(model_type="lstm", epochs=100, lr=1e-3, batch_size=16):
    """Train the sequence model."""
    sequences, labels, sign_names, seq_length, n_features = load_dataset()
    n_classes = len(sign_names)

    if len(set(labels)) < 2:
        print("ERROR: Need at least 2 classes. Collect data for more signs.")
        return

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_ds = SignSequenceDataset(X_train, y_train)
    test_ds = SignSequenceDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Create model
    if model_type == "transformer":
        print("\n=== Training Transformer Classifier ===")
        model = TransformerClassifier(n_features=n_features, n_classes=n_classes)
    else:
        print("\n=== Training BiLSTM Classifier ===")
        model = LSTMClassifier(n_features=n_features, n_classes=n_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Training loop
    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    outputs = model(batch_x)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(batch_y.numpy())

            acc = accuracy_score(all_labels, all_preds)
            print(f"  Epoch {epoch+1:3d}/{epochs}  |  Loss: {avg_loss:.4f}  |  Acc: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_state = model.state_dict().copy()

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    print(f"\n=== Final Evaluation (Best Acc: {best_acc:.4f}) ===")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    present_labels = sorted(set(all_labels))
    target = [sign_names[i] for i in present_labels]
    print(classification_report(all_labels, all_preds, target_names=target))

    # Export to ONNX
    export_to_onnx(model, seq_length, n_features, device)

    # Save metadata for the web app
    save_metadata(sign_names, seq_length, n_features, model_type)

    return model


def export_to_onnx(model, seq_length, n_features, device):
    """Export trained PyTorch model to ONNX."""
    print("\n=== Exporting to ONNX ===")
    model.eval()

    # Dummy input: (batch=1, seq_length, n_features)
    dummy_input = torch.randn(1, seq_length, n_features).to(device)

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(MODEL_OUTPUT),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
    )

    print(f"  ONNX model saved to {MODEL_OUTPUT}")
    print("  Restart your web app to use the new model.")


def save_metadata(sign_names, seq_length, n_features, model_type):
    """Save model metadata so the web app knows how to use the model."""
    meta = {
        "model_type": model_type,
        "sign_names": sign_names,
        "seq_length": seq_length,
        "n_features": n_features,
        "mode": "sequence",
    }
    METADATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_OUTPUT, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to {METADATA_OUTPUT}")


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sequence model for ASL recognition")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "transformer"],
                        help="Model architecture: lstm or transformer (default: lstm)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")

    args = parser.parse_args()

    print("=" * 50)
    print("  ASL Sign Language - Sequence Model Trainer")
    print("=" * 50)

    if not DATASET_PATH.exists():
        print(f"\nNo dataset found at {DATASET_PATH}")
        print("Run collect_seq.py first to collect data.")
        exit(1)

    train(
        model_type=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
