#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])


def collect_sequences(labels, seq_len, samples, out_dir, camera_index=0, warmup=30):
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # warm-up frames for exposure/white-balance
        for _ in range(max(0, warmup)):
            ok, frame = cap.read()
            if not ok:
                break
            _ = cv2.waitKey(1)

        for label in labels:
            label_dir = out_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            print(f"Label: {label} -> saving to {label_dir}")

            for sample_idx in range(samples):
                sequence = []
                print(f"  Sample {sample_idx+1}/{samples} - get ready...")
                start = time.time()
                # small countdown
                while time.time() - start < 1.0:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    cv2.putText(frame, f"{label}: starting", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
                    cv2.imshow("Collect", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                while len(sequence) < seq_len:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)
                    frame.flags.writeable = True

                    # draw hands for feedback
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)

                    cv2.putText(frame, f"{label} | frame {len(sequence)}/{seq_len} | sample {sample_idx+1}/{samples}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Collect", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                arr = np.asarray(sequence, dtype=np.float32)
                np.save(label_dir / f"{int(time.time())}_{sample_idx:04d}.npy", arr)
                print(f"    saved: shape={arr.shape}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Collect Mediapipe hand landmark sequences for gesture labels.")
    parser.add_argument('--labels', nargs='+', required=True, help='List of class labels, e.g. Click Stop Rotate No')
    parser.add_argument('--seq-len', type=int, default=30, help='Frames per sample sequence')
    parser.add_argument('--samples', type=int, default=50, help='Samples per label')
    parser.add_argument('--out', type=str, required=True, help='Output folder for sequences, e.g. data/CSRN')
    parser.add_argument('--camera', type=int, default=0, help='Camera index for cv2.VideoCapture')
    parser.add_argument('--warmup', type=int, default=30, help='Warm-up frames before collection')
    args = parser.parse_args()

    collect_sequences(args.labels, args.seq_len, args.samples, args.out, camera_index=args.camera, warmup=args.warmup)


if __name__ == '__main__':
    main()

