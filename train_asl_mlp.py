#!/usr/bin/env python3
"""train_asl_mlp.py — Train the ASL MLP (landmark-based, background-proof).

Data sources:
  1. MS-ASL landmarks    — training_data/samples_asl.npz (from download_msasl.py)
  2. asl_clips/ videos   — webcam recordings via Dev Panel → landmarks extracted
  3. Feedback samples    — training_data/samples_asl.npz (appended by /feedback)

Run:
    python train_asl_mlp.py
    python train_asl_mlp.py --epochs 500
    python train_asl_mlp.py --signs eat drink sit
"""

import argparse
import os
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

parser = argparse.ArgumentParser(description="Train ASL MLP (landmark-based)")
parser.add_argument("--epochs", type=int,   default=300)
parser.add_argument("--lr",     type=float, default=1e-3)
parser.add_argument("--signs",  nargs="+",  metavar="SIGN")
args = parser.parse_args()

try:
    import numpy as np
    import torch
except ImportError:
    print("❌ numpy / torch not installed.")
    sys.exit(1)

import numpy as np   # noqa: F811
import torch         # noqa: F811

from asl_dictionary import ASL_DICT
from sign_model import SignClassifier, MODEL_ASL

# ── Paths ─────────────────────────────────────────────────────────────────────
# MS-ASL landmarks extracted by download_msasl.py
MSASL_NPZ     = HERE / "training_data" / "samples_asl.npz"

# Webcam / feedback recordings
ASL_DATA_NPZ  = HERE / "training_data" / "samples_asl.npz"

# asl_clips/ folder for video-to-landmark extraction
ASL_CLIPS_DIR = HERE / "asl_clips"

# Loose video recordings named after sign
LOOSE_VIDS_DIR = Path.home() / "Downloads" / "LooseVids" / "ASL"

INPUT_DIM  = 278
MIDDLE_MCP = 9
INDEX_MCP  = 5
PINKY_MCP  = 17
FINGER_TIPS = [4, 8, 12, 16, 20]


def _normalize_hand(pts: np.ndarray) -> np.ndarray:
    h    = pts - pts[MIDDLE_MCP].copy()
    span = np.linalg.norm(h[INDEX_MCP] - h[PINKY_MCP])
    if span > 0:
        h = h / span
    dists = [np.linalg.norm(h[FINGER_TIPS[i]] - h[FINGER_TIPS[j]])
             for i in range(len(FINGER_TIPS))
             for j in range(i + 1, len(FINGER_TIPS))]
    return np.concatenate([h.flatten(), np.array(dists)])


def _frames_to_vector(frame_vectors: list) -> np.ndarray:
    if not frame_vectors:
        return np.zeros(INPUT_DIM, dtype=np.float32)
    return np.array(frame_vectors, dtype=np.float32).mean(axis=0)


# ── Source 1: MS-ASL landmarks (from download_msasl.py) ──────────────────────
def load_msasl(allowed_signs: set) -> tuple:
    if not MSASL_NPZ.exists():
        print(f"  MS-ASL      : ✗ {MSASL_NPZ} not found")
        print(f"               Run: python download_msasl.py")
        return [], []

    d      = np.load(MSASL_NPZ, allow_pickle=True)
    X_all  = list(d["X"].astype(np.float32))
    y_all  = list(d["y"].astype(str))

    if allowed_signs:
        filtered = [(x, y) for x, y in zip(X_all, y_all) if y in allowed_signs]
        X_all = [f[0] for f in filtered]
        y_all = [f[1] for f in filtered]

    counts = defaultdict(int)
    for lbl in y_all:
        counts[lbl] += 1

    total = len(y_all)
    if total:
        print(f"  MS-ASL      : ✓ {total} samples — " +
              ", ".join(f"{s}({c})" for s, c in sorted(counts.items())))
    else:
        print(f"  MS-ASL      : ✗ 0 samples match ASL_DICT signs")
        print(f"               Run download_msasl.py to populate")
    return X_all, y_all


# ── Source 2: asl_clips/ video recordings ────────────────────────────────────
def load_asl_clips(allowed_signs: set) -> tuple:
    if not ASL_CLIPS_DIR.exists():
        print(f"  asl_clips   : ✗ folder not found")
        return [], []

    try:
        import cv2
        import mp_holistic as mph
        from landmarks import extract_landmarks, normalize_landmarks
    except ImportError as e:
        print(f"  asl_clips   : ✗ import failed ({e})")
        return [], []

    X, y   = [], []
    counts = defaultdict(int)

    for mp4 in sorted(ASL_CLIPS_DIR.glob("*.mp4")):
        sign = mp4.stem.split("_")[0].lower()
        if sign not in ASL_DICT or len(sign) == 1:
            continue
        if allowed_signs and sign not in allowed_signs:
            continue

        cap       = cv2.VideoCapture(str(mp4))
        frame_vecs = []
        try:
            with mph.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as h:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    lm  = normalize_landmarks(extract_landmarks(h.process(rgb)))
                    if not np.all(lm[:126] == 0):
                        frame_vecs.append(lm)
        except Exception as e:
            print(f"  asl_clips   : ⚠ {mp4.name} — {e}")
        finally:
            cap.release()

        if frame_vecs:
            X.append(_frames_to_vector(frame_vecs))
            y.append(sign)
            counts[sign] += 1

    total = sum(counts.values())
    if total:
        print(f"  asl_clips   : ✓ {total} samples — " +
              ", ".join(f"{s}({c})" for s, c in sorted(counts.items())))
    else:
        print(f"  asl_clips   : ✗ 0 samples (record via Dev Panel)")
    return X, y


# ── Source 3: LooseVids/ASL/ folder ──────────────────────────────────────────
def load_loose_vids(allowed_signs: set) -> tuple:
    print(f"  Loose vids  : checking {LOOSE_VIDS_DIR}")
    if not LOOSE_VIDS_DIR.exists():
        print(f"  Loose vids  : ✗ not found")
        return [], []

    mp4s = list(LOOSE_VIDS_DIR.glob("*.mp4"))
    print(f"  Loose vids  : found {len(mp4s)} mp4 files")

    try:
        import cv2
        import mp_holistic as mph
        from landmarks import extract_landmarks, normalize_landmarks
    except ImportError as e:
        print(f"  Loose vids  : ✗ import failed ({e})")
        return [], []

    X, y   = [], []
    counts = defaultdict(int)

    for mp4 in sorted(mp4s):
        sign = mp4.stem.lower()
        if sign not in ASL_DICT or len(sign) == 1:
            continue
        if allowed_signs and sign not in allowed_signs:
            continue

        cap        = cv2.VideoCapture(str(mp4))
        frame_vecs = []
        try:
            with mph.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as h:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    lm  = normalize_landmarks(extract_landmarks(h.process(rgb)))
                    if not np.all(lm[:126] == 0):
                        frame_vecs.append(lm)
        except Exception as e:
            print(f"  Loose vids  : ⚠ {mp4.name} — {e}")
        finally:
            cap.release()

        if frame_vecs:
            X.append(_frames_to_vector(frame_vecs))
            y.append(sign)
            counts[sign] += 1

    total = sum(counts.values())
    if total:
        print(f"  Loose vids  : ✓ {total} samples — " +
              ", ".join(f"{s}({c})" for s, c in sorted(counts.items())))
    else:
        print(f"  Loose vids  : ✗ 0 usable samples")
    return X, y


# ── Augmentation ──────────────────────────────────────────────────────────────
def augment(X: np.ndarray, y: list, min_per_sign: int = 30) -> tuple:
    from collections import Counter
    counts = Counter(y)
    X_aug, y_aug = list(X), list(y)
    for sign, n in counts.items():
        if n >= min_per_sign:
            continue
        needed  = min_per_sign - n
        indices = [i for i, lbl in enumerate(y) if lbl == sign]
        for i in range(needed):
            src   = X[indices[i % len(indices)]]
            noise = np.random.normal(0, 0.01, src.shape).astype(np.float32)
            X_aug.append(src + noise)
            y_aug.append(sign)
    return np.array(X_aug, dtype=np.float32), y_aug


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    signs_filter = set(args.signs) if args.signs else None

    print("=" * 60)
    print("  SignFuture — ASL MLP Trainer (landmark-based)")
    print("=" * 60)
    print()
    print("  Collecting training data…")

    X_all, y_all = [], []

    # Source 1: MS-ASL landmarks
    Xm, ym = load_msasl(signs_filter or set())
    X_all += Xm
    y_all += ym

    # Source 2: asl_clips/ recordings
    Xc, yc = load_asl_clips(signs_filter or set())
    X_all += Xc
    y_all += yc

    # Source 3: LooseVids/ASL/
    Xl, yl = load_loose_vids(signs_filter or set())
    X_all += Xl
    y_all += yl

    if not X_all:
        print()
        print("  ❌ No training data found.")
        print("     → Run: python download_msasl.py")
        print("     → OR record signs via Dev Panel → Train tab (ASL mode)")
        return

    from collections import Counter
    label_counts = Counter(y_all)
    covered      = sorted(label_counts.keys())

    print()
    print(f"  ── Raw totals ──────────────────────────────────────")
    for sign in covered:
        print(f"  {sign:<16} {label_counts[sign]:>4} samples")
    print(f"  Total: {len(X_all)} samples, {len(covered)} signs")

    if len(covered) < 2:
        print("\n  ❌ Need at least 2 signs to train.")
        return

    X_np = np.array(X_all, dtype=np.float32)
    X_np, y_all = augment(X_np, y_all, min_per_sign=30)
    print(f"\n  After augmentation: {len(y_all)} samples total")

    print()
    print(f"  Training MLP — {args.epochs} epochs, lr={args.lr}")
    print(f"  Signs: {covered}")
    print()

    clf    = SignClassifier(covered)
    result = clf.train(
        X_np, y_all,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=32,
        progress_cb=lambda pct, msg: print(
            f"\r  [{'█'*(pct//5)}{'░'*(20-pct//5)}] {pct:3d}%  {msg}",
            end="", flush=True
        ),
    )

    print()
    print(f"  ✅ Accuracy : {result['accuracy']*100:.1f}%")
    print(f"  ✅ Time     : {result['time_s']:.1f}s")

    os.makedirs(os.path.dirname(MODEL_ASL) or ".", exist_ok=True)
    clf.save(MODEL_ASL)
    print(f"  ✅ Saved    → {MODEL_ASL}")
    print()
    print("  Start the server:  bash run.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
