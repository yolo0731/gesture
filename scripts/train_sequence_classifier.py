#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def load_dataset(data_dir: Path, labels, seq_len: int):
    X, y = [], []
    for idx, label in enumerate(labels):
        label_dir = data_dir / label
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing folder: {label_dir}")
        files = sorted(label_dir.glob('*.npy'))
        if not files:
            raise FileNotFoundError(f"No .npy files under {label_dir}. Please run collect_sequences.py first.")
        for f in files:
            arr = np.load(f)
            if arr.shape[0] < seq_len:
                # pad at the end if shorter
                pad = np.zeros((seq_len - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
                arr = np.vstack([arr, pad])
            elif arr.shape[0] > seq_len:
                arr = arr[-seq_len:]
            X.append(arr)
            y.append(idx)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    return X, y


def build_model(seq_len: int, feat_dim: int, num_classes: int) -> tf.keras.Model:
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(seq_len, feat_dim)),
        BatchNormalization(),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser(description='Train a Keras sequence classifier on Mediapipe hand landmarks and export SavedModel.')
    parser.add_argument('--data', type=str, required=True, help='Dataset root, contains subfolders per label with .npy sequences')
    parser.add_argument('--labels', nargs='+', required=True, help='Class labels in order (match collection labels)')
    parser.add_argument('--seq-len', type=int, required=True, help='Sequence length expected by the model (20 or 30)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--out', type=str, required=True, help='Output SavedModel directory')
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(data_dir, args.labels, args.seq_len)
    feat_dim = X.shape[-1]

    model = build_model(args.seq_len, feat_dim, len(args.labels))

    es = EarlyStopping(monitor='val_accuracy', patience=6, mode='max', restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

    model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch,
        validation_split=args.val_split,
        callbacks=[es, rlrop],
        verbose=2
    )

    # Export as SavedModel (directory) compatible with tf.keras.models.load_model(path)
    model.save(str(out_dir))
    print(f"SavedModel exported to: {out_dir}")


if __name__ == '__main__':
    main()

