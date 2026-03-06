"""
ASL Sequence Data Collector

This script collects SEQUENCES of hand landmarks for dynamic sign recognition.
Instead of capturing a single frame per sign, it records a clip of N frames,
capturing the movement over time.

Usage:
    python collect_seq.py

Controls:
    R      - Start/stop recording a clip for the current sign
    N      - Skip to the next sign
    Q      - Finish collecting and save dataset
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Configuration ---
SEQ_LENGTH = 30             # Number of frames per clip (~1 second at 30fps)
CLIPS_PER_SIGN = 20        # Number of clips to record per sign
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sequence_dataset"

# Signs to collect — start with a small set of dynamic words
# You can expand this list as you add more signs
SIGNS = [
    "hello",
    "thank_you",
    "yes",
    "no",
    "i_love_you",
]


def normalize_landmarks(landmarks) -> list:
    """Normalize 21 landmarks relative to wrist (index 0), return 63 floats.

    Also applies scale normalization by dividing by the max distance
    from the wrist to make the features invariant to hand distance from camera.
    """
    wrist = landmarks[0]
    relative = []
    for lm in landmarks:
        relative.append((lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z))

    # Scale normalization: divide by max distance from wrist
    max_dist = max(
        (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** 0.5
        for r in relative[1:]  # skip wrist itself (0,0,0)
    )
    if max_dist < 1e-6:
        max_dist = 1.0  # avoid division by zero

    result = []
    for r in relative:
        result.extend([r[0] / max_dist, r[1] / max_dist, r[2] / max_dist])

    return result


def get_hand_landmarker():
    """Create and return a MediaPipe hand landmarker."""
    model_path = Path(__file__).parent / "hand_landmarker.task"
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


def collect_sequences():
    """Collect sequence data from webcam."""
    hand_landmarker = get_hand_landmarker()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    all_clips = []  # List of (sign_index, [seq_length x 63])
    sign_idx = 0
    clip_count = 0
    frame_timestamp_ms = 0

    # Recording state
    recording = False
    current_clip_frames = []

    print("\n" + "=" * 50)
    print("  ASL Sequence Data Collector")
    print("=" * 50)
    print(f"\nCollect {CLIPS_PER_SIGN} clips of {SEQ_LENGTH} frames each per sign.")
    print("R = start/stop recording a clip, N = next sign, Q = finish\n")

    while cap.isOpened() and sign_idx < len(SIGNS):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_timestamp_ms += 33
        results = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        detected = False
        current_landmarks = None
        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            detected = True
            current_landmarks = results.hand_landmarks[0]
            draw_landmarks_on_frame(frame, current_landmarks)

        # If recording, accumulate frames
        if recording and detected and current_landmarks:
            normalized = normalize_landmarks(current_landmarks)
            current_clip_frames.append(normalized)

            # Show recording progress
            rec_progress = len(current_clip_frames) / SEQ_LENGTH
            cv2.rectangle(frame, (10, 440), (10 + int(620 * rec_progress), 450), (0, 0, 255), -1)

            # Auto-stop when we reach SEQ_LENGTH frames
            if len(current_clip_frames) >= SEQ_LENGTH:
                all_clips.append((sign_idx, current_clip_frames))
                clip_count += 1
                recording = False
                current_clip_frames = []
                print(f"  ✓ Clip {clip_count}/{CLIPS_PER_SIGN} saved for '{SIGNS[sign_idx]}'")

                if clip_count >= CLIPS_PER_SIGN:
                    sign_idx += 1
                    clip_count = 0
                    if sign_idx < len(SIGNS):
                        print(f"\n-> Next sign: '{SIGNS[sign_idx]}'")

        # --- UI Overlay ---
        sign_name = SIGNS[sign_idx] if sign_idx < len(SIGNS) else "DONE"
        cv2.rectangle(frame, (0, 0), (640, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Sign: {sign_name}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        status_parts = []
        if recording:
            status_parts.append("** RECORDING **")
        status_parts.append("HAND OK" if detected else "No hand")
        status_parts.append(f"Clips: {clip_count}/{CLIPS_PER_SIGN}")
        status_parts.append(f"Sign {sign_idx + 1}/{len(SIGNS)}")
        status_text = "  |  ".join(status_parts)

        color = (0, 0, 255) if recording else ((0, 255, 0) if detected else (100, 100, 255))
        cv2.putText(frame, status_text, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Recording indicator border
        if recording:
            cv2.rectangle(frame, (0, 0), (639, 479), (0, 0, 255), 4)

        cv2.imshow("ASL Sequence Collector - R=record, N=next, Q=quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            if not recording:
                recording = True
                current_clip_frames = []
                print(f"  ● Recording clip for '{sign_name}'...")
            else:
                # Cancel current recording
                recording = False
                current_clip_frames = []
                print(f"  ✗ Recording cancelled.")

        elif key == ord('n'):
            recording = False
            current_clip_frames = []
            sign_idx += 1
            clip_count = 0
            if sign_idx < len(SIGNS):
                print(f"\n-> Skipped to: '{SIGNS[sign_idx]}'")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()

    return all_clips


def save_dataset(all_clips):
    """Save collected clips to disk as JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / "sequences.json"

    dataset = {
        "sign_names": SIGNS,
        "seq_length": SEQ_LENGTH,
        "n_features": 63,
        "clips": [
            {"label": clip[0], "frames": clip[1]}
            for clip in all_clips
        ],
    }

    with open(filepath, "w") as f:
        json.dump(dataset, f)

    print(f"\nSaved {len(all_clips)} clips to {filepath}")
    print(f"  Signs: {SIGNS}")
    print(f"  Seq length: {SEQ_LENGTH} frames")
    print(f"  Features per frame: 63")
    return filepath


if __name__ == "__main__":
    print("=" * 50)
    print("  ASL Sign Language - Sequence Data Collector")
    print("=" * 50)

    clips = collect_sequences()
    if len(clips) > 0:
        save_dataset(clips)
        print("\nDone! Now run train_seq.py to train the sequence model.")
    else:
        print("\nNo clips collected!")
