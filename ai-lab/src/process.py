import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from pathlib import Path

# ==========================================
# 1. Configuration Settings
# ==========================================

MODEL_EXPORT_DIR = r"c:\Users\harsh\mini-project\web-app\public\models"
os.makedirs(MODEL_EXPORT_DIR, exist_ok=True)

SEQ_LENGTH = 16
N_FEATURES = 63  # 21 landmarks * 3 coords (x,y,z)

# The dataset will be compiled based on these signs. 
# You can customize this list!
SIGNS = ["hello", "thanks", "iloveyou", "yes", "no", "help"]  
NUM_SAMPLES_PER_SIGN = 30 # Feel free to increase this for better accuracy

DATA_SAVE_PATH = "sign_data.npy"
LABELS_SAVE_PATH = "sign_labels.npy"

# ==========================================
# 2. Perception & Preprocessing
# ==========================================

def get_hand_landmarker():
    """Create and return a MediaPipe hand landmarker."""
    model_path = Path(__file__).parent.parent / "hand_landmarker.task"
    if not model_path.exists():
        print("Downloading hand landmarker model...")
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, str(model_path))
        print("Download complete.")

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_landmarks_on_frame(frame, landmarks):
    """Draw hand landmarks and connections on the frame."""
    h, w = frame.shape[:2]
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    for c1, c2 in connections:
        p1 = (int(landmarks[c1].x * w), int(landmarks[c1].y * h))
        p2 = (int(landmarks[c2].x * w), int(landmarks[c2].y * h))
        cv2.line(frame, p1, p2, (0, 255, 0), 2)


def normalize_landmarks(landmarks):
    """
    Apply wrist-relative normalization and scale by max distance 
    for translation and scale invariance.
    Subtracts the wrist coordinates (Index 0) from all other 20 landmarks.
    """
    wrist = landmarks[0]
    
    relative_landmarks = []
    # Translate to be wrist-relative
    for lm in landmarks:
        relative_landmarks.append({
            'x': lm.x - wrist.x,
            'y': lm.y - wrist.y,
            'z': lm.z - wrist.z
        })
        
    # Find max distance to normalize scales (matches frontend normalizer)
    max_dist = 0.0
    for lm in relative_landmarks:
        dist = math.sqrt(lm['x']**2 + lm['y']**2 + lm['z']**2)
        if dist > max_dist:
            max_dist = dist
            
    if max_dist < 1e-6:
        max_dist = 1.0
        
    # Flatten into 1D array of 63 floats
    flattened = []
    for lm in relative_landmarks:
        flattened.extend([lm['x'] / max_dist, lm['y'] / max_dist, lm['z'] / max_dist])
        
    return flattened

def collect_data(num_samples_per_sign=30):
    """
    Capture video from webcam, extract hand landmarks, normalize them 
    and store fixed-length sequences.
    """
    hand_landmarker = get_hand_landmarker()
    cap = cv2.VideoCapture(0)
    
    all_sequences = []
    all_labels = []
    frame_timestamp_ms = 0
    
    print("\n=== ASL Sequence Data Collector ===")
    print(f"Collect {num_samples_per_sign} samples per sign.")
    print("SPACE = capture, N = next sign, Q = finish\n")
    
    label_idx = 0
    while cap.isOpened() and label_idx < len(SIGNS):
        sign = SIGNS[label_idx]
        sequence_count = 0
        
        while sequence_count < num_samples_per_sign:
            buffer = []
            
            # Sequence collection loop
            recording = False
            while len(buffer) < SEQ_LENGTH:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame = cv2.flip(frame, 1) # Mirror display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                frame_timestamp_ms += 33
                res = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                
                detected = False
                # If a hand is found, grab the landmarks
                if res.hand_landmarks and len(res.hand_landmarks) > 0:
                    detected = True
                    hand_landmarks = res.hand_landmarks[0]
                    draw_landmarks_on_frame(frame, hand_landmarks)
                    
                    if recording:
                        features = normalize_landmarks(hand_landmarks)
                        buffer.append(features)
                elif recording:
                    # If hand is lost while recording, reset the buffer and stop recording
                    print("  Hand lost! Restarting clip...")
                    buffer = []
                    recording = False
                        
                # UI Overlay
                cv2.rectangle(frame, (0, 0), (640, 80), (0, 0, 0), -1)
                cv2.putText(frame, f"Sign: {sign}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                status_parts = []
                if recording:
                    status_parts.append("** RECORDING **")
                status_parts.append("HAND DETECTED" if detected else "No hand")
                status_parts.append(f"Seq: {sequence_count+1}/{num_samples_per_sign}")
                status_parts.append(f"Sign {label_idx+1}/{len(SIGNS)}")
                
                color = (0, 0, 255) if recording else ((0, 255, 0) if detected else (100, 100, 255))
                cv2.putText(frame, "  |  ".join(status_parts), (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                
                # Progress bar for current sequence
                rec_progress = len(buffer) / SEQ_LENGTH
                cv2.rectangle(frame, (10, 72), (10 + int(620 * rec_progress), 78), (0, 0, 255) if recording else (0, 255, 0), -1)
                
                # Recording indicator border
                if recording:
                    cv2.rectangle(frame, (0, 0), (639, 479), (0, 0, 255), 4)

                cv2.imshow("Data Collection - SPACE=record, N=next, Q=quit", frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Capture
                    if not recording:
                        if detected:
                            recording = True
                            buffer = []
                            print(f"  ● Recording clip {sequence_count + 1} for '{sign}'...")
                        else:
                            print("  No hand detected! Position your hand clearly.")
                    else:
                        recording = False
                        buffer = []
                        print(f"  ✗ Recording cancelled.")
                        
                elif key == ord('n'):  # Next letter
                    recording = False
                    buffer = []
                    label_idx += 1
                    sequence_count = num_samples_per_sign # Force break inner loop
                    if label_idx < len(SIGNS):
                        print(f"\n-> Skipped to: '{SIGNS[label_idx]}'")
                    break
                    
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    hand_landmarker.close()
                    return None, None
            
            # Sequence complete
            if len(buffer) == SEQ_LENGTH:
                all_sequences.append(buffer)
                all_labels.append(label_idx)
                sequence_count += 1
                print(f"  ✓ Clip {sequence_count}/{num_samples_per_sign} saved for '{sign}'")
                
                if sequence_count >= num_samples_per_sign:
                    label_idx += 1
                    if label_idx < len(SIGNS):
                        print(f"\n-> Next sign: '{SIGNS[label_idx]}'")
                
    cap.release()
    cv2.destroyAllWindows()
    
    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    np.save(DATA_SAVE_PATH, X)
    np.save(LABELS_SAVE_PATH, y)
    print(f"\n[SUCCESS] Dataset Saved: Features {X.shape}, Labels {y.shape}")
    
    return X, y

# ==========================================
# 3. Model Architecture (Tiny GRU)
# ==========================================

class TinyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.2):
        super(TinyGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Input shape: [batch_size, seq_length=16, input_size=63]
        out, _ = self.gru(x)
        
        # We only care about the GRU output at the final time step
        last_out = out[:, -1, :]
        last_out = self.dropout(last_out)
        
        # Logits output: [batch_size, num_classes] 
        # (CrossEntropyLoss expects raw logits)
        out = self.fc(last_out)
        return out

# ==========================================
# 4. Training Loop & ONNX Export
# ==========================================

def train_and_export(X, y):
    num_classes = len(SIGNS)
    
    # 1. Setup PyTorch Dataset
    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(y).long()
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 2. Initialize model, optimizer, and loss criterion
    model = TinyGRU(input_size=N_FEATURES, hidden_size=64, num_classes=num_classes, num_layers=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    print("\n[INFO] Starting Model Training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        acc = 100 * correct / total
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(dataloader):.4f} - Accuracy: {acc:.2f}%")
            
    # Save standard PyTorch model weighing
    torch_model_path = "tiny_gru_model.pth"
    torch.save(model.state_dict(), torch_model_path)
    print(f"\n[SUCCESS] PyTorch model saved temporarily to: {torch_model_path}")
    
    # 3. Export to Web-Browser Ready ONNX
    print("\n[INFO] Exporting Model to ONNX format...")
    model.eval()
    
    # Create proxy dummy input: (Batch Size 1, 16 frames, 63 features)
    dummy_input = torch.randn(1, SEQ_LENGTH, N_FEATURES)
    
    onnx_path = os.path.join(MODEL_EXPORT_DIR, "model.onnx")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        # Allow dynamic batch sizes if batch prediction is needed
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"[SUCCESS] ONNX runtime model seamlessly exported to: {onnx_path}")
    
    # 4. Save metadata dictionary for the web app
    metadata = {
        "mode": "sequence",
        "sign_names": SIGNS,
        "seq_length": SEQ_LENGTH,
        "n_features": N_FEATURES,
        "model_type": "tiny_gru"
    }
    
    meta_path = os.path.join(MODEL_EXPORT_DIR, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[SUCCESS] Web interface metadata exported to: {meta_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Signstream End-to-End Pipeline")
    parser.add_argument("--collect", action="store_true", help="Force run data collection step")
    args = parser.parse_args()
    
    if args.collect or not (os.path.exists(DATA_SAVE_PATH) and os.path.exists(LABELS_SAVE_PATH)):
        print("Dataset not found locally or --collect flag passed. Initializing camera...")
        X, y = collect_data(num_samples_per_sign=NUM_SAMPLES_PER_SIGN)
    else:
        print("Loading previously collected dataset...")
        X = np.load(DATA_SAVE_PATH)
        y = np.load(LABELS_SAVE_PATH)
        
    if X is not None and y is not None:
        train_and_export(X, y)
    else:
        print("Pipeline aborted or no data was gathered.")
