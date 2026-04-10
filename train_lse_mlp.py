#!/usr/bin/env python3
"""train_lse_mlp.py — Train the LSE MLP (landmark-based, background-proof).

The MLP only ever sees hand/pose joint coordinates — never raw pixels.
This means background noise, lighting changes, moving cameras, etc.
have ZERO effect on the model. What gets fed in at training time is
identical in format to what MediaPipe produces live from the webcam.

Data sources (stacked — uses whatever is present, skips what isn't):
  1. LSE-FS-UVigo  (Zenodo 15797079) — preprocessed JSON keypoints
                    ~/Downloads/15797079/PROC_KPS/train+validation+test/
  2. SWL-LSE       (Zenodo 13691887) — MediaPipe .pkl skeleton sequences
                    ~/Downloads/13691887/MEDIAPIPE/
  3. samples_lse.npz — webcam recordings + feedback corrections collected via server

Output:  training_data/model_lse.pt
         (same file the server loads — drop-in replacement, no other changes needed)

Run:
    python train_lse_mlp.py
    python train_lse_mlp.py --epochs 400
    python train_lse_mlp.py --signs dolor garganta mareo
    python train_lse_mlp.py --no-swl          (skip SWL-LSE pkl files)
    python train_lse_mlp.py --no-uvigo        (skip LSE-FS-UVigo JSON)
"""

import argparse
import json
import os
import pickle
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train LSE MLP (landmark-based)")
parser.add_argument("--epochs",    type=int,   default=300)
parser.add_argument("--lr",        type=float, default=1e-3)
parser.add_argument("--signs",     nargs="+",  metavar="SIGN")
parser.add_argument("--no-swl",    action="store_true", help="Skip SWL-LSE pkl files")
parser.add_argument("--no-uvigo",  action="store_true", help="Skip LSE-FS-UVigo JSON")
args = parser.parse_args()

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    import numpy as np
    import torch
except ImportError:
    print("❌ numpy / torch not installed. Run setup.sh first.")
    sys.exit(1)

from asl_dictionary import LSE_DICT
from sign_model import SignClassifier, MODEL_LSE

# ── Paths ─────────────────────────────────────────────────────────────────────
# LSE-FS-UVigo (Zenodo 15797079) — the preprocessed keypoints zip
# Extract PROC_KPS.ZIP so you have:
#   ~/Downloads/15797079/PROC_KPS/train/
#   ~/Downloads/15797079/PROC_KPS/validation/
#   ~/Downloads/15797079/PROC_KPS/test/
UVIGO_FS_DIR = Path.home() / "Downloads" / "15797079" / "PROC_KPS" / "TRANSFORMED_KPS"

# SWL-LSE MediaPipe pkl files (Zenodo 13691887)
# Extract MEDIAPIPE.zip so you have:
#   ~/Downloads/13691887/MEDIAPIPE/*.pkl
SWL_MEDIAPIPE_DIR = Path.home() / "Downloads" / "13691887" / "MEDIAPIPE"
SWL_CSV           = Path.home() / "Downloads" / "13691887" / "ANNOTATIONS" / "videos_ref_annotations.csv"
SWL_VIDEOS_DIR    = Path.home() / "Downloads" / "13691887" / "VIDEOS_RENAMED"
SIGN4ALL_DIR      = Path.home() / "Downloads" / "sign4all"

# Webcam + feedback landmark samples stored by the server
# (training_data/samples_lse.npz — written by /collect-video and /feedback)
LSE_DATA_NPZ = HERE / "training_data" / "samples_lse.npz"

# Loose recordings ~/Downloads/LooseVids/<sign>.mp4
LOOSE_VIDS_DIR = Path.home() / "Downloads" / "LooseVids"

# ── Label mappings ────────────────────────────────────────────────────────────
# LSE-FS-UVigo JSON "label" field → LSE_DICT key
# The dataset uses Spanish health/daily-life vocabulary
UVIGO_FS_LABEL_MAP = {
    # Health signs (from esaude subset)
    "DOLOR":           "dolor",
    "GARGANTA":        "garganta",
    "MAREO":           "mareo",
    "MOCO":            "moco",
    "RESPIRAR":        "respirar",
    "TOSER":           "tos",
    "TOS":             "tos",
    "FIEBRE":          "fiebre",
    "SANGRE":          "sangre",
    "COMER":           "comer",
    "DORMIR":          "dormir",
    "BEBER":           "beber",
    "CAMINAR":         "caminar",
    "SENTAR":          "sentar",
    "SENTARSE":        "sentar",
    "ABRIR":           "abrir",
    "MIRAR":           "mirar",
    "GIRAR":           "girar",
    "INFECCION":       "infeccion",
    "INFECCIÓN":       "infeccion",
    "HINCHAZON":       "hinchazon",
    "HINCHAZÓN":       "hinchazon",
    "ZUMBIDO":         "zumbido",
    "ACUFENO":         "zumbido",
    "ACÚFENO":         "zumbido",
    # Fingerspelling — individual letters (only add if explicitly in LSE_DICT)
    # Note: single-letter labels from this dataset are NOT added here to avoid
    # contaminating sign training. Fingerspelling training needs its own dedicated pass.
    # "A", "B" etc. in the dataset are fingerspelling demos, not useful for sign recognition.

# SWL-LSE CSV label → LSE_DICT key (same as rename_dataset.py)
SWL_LABEL_MAP = {
    "DOLOR":              "dolor",
    "GARGANTA":           "garganta",
    "GARGANTA2":          "garganta",
    "MAREO":              "mareo",
    "MAREO2":             "mareo",
    "MOCO":               "moco",
    "MOCO2":              "moco",
    "RESPIRAR":           "respirar",
    "RESPIRAR2":          "respirar",
    "COMER":              "comer",
    "DORMIR":             "dormir",
    "TOSER":              "tos",
    "FIEBRE2":            "fiebre",
    "SANGRE2":            "sangre",
    "ACUFENO":            "zumbido",
    "AMIGDALAS-INFLAMAR": "hinchazon",
    "AMIGDALITIS":        "infeccion",
}


# ── Landmark constants (must match landmarks.py exactly) ─────────────────────
INPUT_DIM   = 278   # left_hand(73) + right_hand(73) + pose(132)
MIDDLE_MCP  = 9
INDEX_MCP   = 5
PINKY_MCP   = 17
FINGER_TIPS = [4, 8, 12, 16, 20]


def _normalize_hand(pts: np.ndarray) -> np.ndarray:
    """21×3 hand keypoints → 73-dim normalized vector. Matches landmarks.py."""
    h = pts - pts[MIDDLE_MCP].copy()
    span = np.linalg.norm(h[INDEX_MCP] - h[PINKY_MCP])
    if span > 0:
        h = h / span
    dists = [np.linalg.norm(h[FINGER_TIPS[i]] - h[FINGER_TIPS[j]])
             for i in range(len(FINGER_TIPS))
             for j in range(i + 1, len(FINGER_TIPS))]
    return np.concatenate([h.flatten(), np.array(dists)])   # 63 + 10 = 73


def _frames_to_vector(frame_vectors: list) -> np.ndarray:
    """List of per-frame 278-dim arrays → single mean-pooled 278-dim vector."""
    if not frame_vectors:
        return np.zeros(INPUT_DIM, dtype=np.float32)
    return np.array(frame_vectors, dtype=np.float32).mean(axis=0)


def _norm_label(s: str) -> str:
    """Strip accents, lowercase."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s.lower())
        if unicodedata.category(c) != "Mn"
    )


# ── Source 1: LSE-FS-UVigo JSON ───────────────────────────────────────────────
def load_uvigo_fs(allowed_signs: set) -> tuple[list, list]:
    """
    Read PROC_KPS JSON files. Each JSON has per-frame keypoints for ONE hand
    (the signing hand, already centred + scaled by the dataset authors).
    We place that hand into either the left or right slot (based on 'handness'),
    leave the other hand as zeros, and leave pose as zeros.
    Returns (X, y) — list of 278-dim vectors and matching sign labels.
    """
    if not UVIGO_FS_DIR.exists():
        print(f"  LSE-FS-UVigo: ✗ {UVIGO_FS_DIR} not found — skipping")
        return [], []

    X, y = [], []
    counts = defaultdict(int)
    skipped_labels = set()

    for split in ("train", "validation", "test"):
        split_dir = UVIGO_FS_DIR / split
        if not split_dir.exists():
            continue
        for jf in sorted(split_dir.glob("*.json")):
            try:
                with open(jf, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            raw_label = data.get("metadata", {}).get("label", "")
            sign = UVIGO_FS_LABEL_MAP.get(raw_label.upper())
            if sign is None:
                # Try accent-stripped lookup
                sign = UVIGO_FS_LABEL_MAP.get(_norm_label(raw_label).upper())
            if sign is None or sign not in LSE_DICT or len(sign) == 1:
                skipped_labels.add(raw_label)
                continue
            if allowed_signs and sign not in allowed_signs:
                continue

            handness = data.get("metadata", {}).get("handness", "right").lower()
            frames_data = data.get("frames", [])
            if not frames_data:
                continue

            frame_vecs = []
            for frame in frames_data:
                # PROC_KPS has only the signing hand under 'right_hand' or 'left_hand'
                hand_key = "right_hand" if handness == "right" else "left_hand"
                hand_data = frame.get(hand_key, {})
                kps = hand_data.get("keypoints", [])

                if len(kps) < 21:
                    continue

                # keypoints is a list of {x, y, z} dicts or [x, y, z] lists
                pts = []
                for kp in kps[:21]:
                    if isinstance(kp, dict):
                        pts.append([kp.get("x", 0), kp.get("y", 0), kp.get("z", 0)])
                    elif isinstance(kp, (list, tuple)) and len(kp) >= 3:
                        pts.append([kp[0], kp[1], kp[2]])
                    else:
                        pts.append([0.0, 0.0, 0.0])

                pts = np.array(pts, dtype=np.float32)   # (21, 3)
                hand_vec = _normalize_hand(pts)          # (73,)

                # Build 278-dim: lh(73) + rh(73) + pose(132)
                lh   = hand_vec if handness == "left"  else np.zeros(73, dtype=np.float32)
                rh   = hand_vec if handness == "right" else np.zeros(73, dtype=np.float32)
                pose = np.zeros(132, dtype=np.float32)
                vec  = np.concatenate([lh, rh, pose])   # (278,)
                frame_vecs.append(vec)

            if not frame_vecs:
                continue

            X.append(_frames_to_vector(frame_vecs))
            y.append(sign)
            counts[sign] += 1

    total = sum(counts.values())
    if total:
        print(f"  LSE-FS-UVigo: ✓ {total} samples across {len(counts)} signs")
        for s, c in sorted(counts.items()):
            print(f"    {s:<16} {c:>4} samples")
    else:
        print(f"  LSE-FS-UVigo: ✗ 0 usable samples found in {UVIGO_FS_DIR}")

    if skipped_labels:
        print(f"  LSE-FS-UVigo: skipped {len(skipped_labels)} unmapped labels "
              f"(e.g. {', '.join(sorted(skipped_labels)[:5])})")

    return X, y


# ── Source 2: SWL-LSE pkl files ───────────────────────────────────────────────
def _load_swl_label_map() -> dict:
    """Parse videos_ref_annotations.csv → {class_id: sign}."""
    if not SWL_CSV.exists():
        return {}
    import csv
    cid_to_sign = {}
    with open(SWL_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("LABEL", "").strip().upper()
            sign  = SWL_LABEL_MAP.get(label)
            if sign:
                try:
                    cid_to_sign[int(row["CLASS_ID"])] = sign
                except (KeyError, ValueError):
                    pass
    return cid_to_sign


def load_swl_mediapipe(allowed_signs: set) -> tuple[list, list]:
    """
    Load SWL-LSE .pkl skeleton files.
    Each pkl contains a list of frame dicts with MediaPipe landmarks.
    Format documented in SWL-LSE repo: each frame has left_hand, right_hand, pose arrays.
    """
    if not SWL_MEDIAPIPE_DIR.exists():
        print(f"  SWL-LSE pkl : ✗ {SWL_MEDIAPIPE_DIR} not found — skipping")
        return [], []

    cid_to_sign = _load_swl_label_map()
    if not cid_to_sign and SWL_CSV.exists():
        print(f"  SWL-LSE pkl : ✗ could not parse {SWL_CSV} — skipping")
        return [], []

    X, y = [], []
    counts = defaultdict(int)
    pkl_files = sorted(SWL_MEDIAPIPE_DIR.glob("*.pkl"))

    if not pkl_files:
        print(f"  SWL-LSE pkl : ✗ no .pkl files in {SWL_MEDIAPIPE_DIR}")
        return [], []

    # Probe first file to detect format
    try:
        with open(pkl_files[0], "rb") as f:
            probe = pickle.load(f)
        # Print structure hint for debugging
        if isinstance(probe, dict):
            print(f"  SWL-LSE pkl : probing format — dict keys: {list(probe.keys())[:8]}")
        elif isinstance(probe, list) and probe:
            first = probe[0]
            if isinstance(first, dict):
                print(f"  SWL-LSE pkl : probing format — list of dicts, keys: {list(first.keys())[:8]}")
            else:
                print(f"  SWL-LSE pkl : probing format — list of {type(first).__name__}")
        else:
            print(f"  SWL-LSE pkl : probing format — {type(probe).__name__}")
    except Exception as e:
        print(f"  SWL-LSE pkl : ✗ could not read first pkl: {e}")
        return [], []

    for pf in pkl_files:
        try:
            with open(pf, "rb") as f:
                data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError) as e:
            print(f"  SWL-LSE pkl : ⚠ skipping {pf.name} — {e}")
            continue

        # data is expected to be a list of sequences, or a single sequence dict
        sequences = data if isinstance(data, list) else [data]

        for seq in sequences:
            if not isinstance(seq, dict):
                continue

            # Try to get the sign label
            cid  = seq.get("class_id") or seq.get("label_id")
            sign = cid_to_sign.get(cid)
            if sign is None:
                raw = seq.get("label", "")
                sign = SWL_LABEL_MAP.get(str(raw).upper())
            if sign is None or sign not in LSE_DICT:
                continue
            if allowed_signs and sign not in allowed_signs:
                continue

            frames_raw = seq.get("frames", [])
            if not frames_raw:
                continue

            frame_vecs = []
            for frame in frames_raw:
                # Each frame expected to have flattened or shaped landmark arrays
                lh_raw   = np.array(frame.get("left_hand",  [0]*63), dtype=np.float32).flatten()[:63]
                rh_raw   = np.array(frame.get("right_hand", [0]*63), dtype=np.float32).flatten()[:63]
                pose_raw = np.array(frame.get("pose",       [0]*132), dtype=np.float32).flatten()[:132]

                # Pad if short
                lh_raw   = np.pad(lh_raw,   (0, max(0, 63  - len(lh_raw))))
                rh_raw   = np.pad(rh_raw,   (0, max(0, 63  - len(rh_raw))))
                pose_raw = np.pad(pose_raw, (0, max(0, 132 - len(pose_raw))))

                lh   = _normalize_hand(lh_raw.reshape(21, 3))   if not np.all(lh_raw == 0)   else np.zeros(73)
                rh   = _normalize_hand(rh_raw.reshape(21, 3))   if not np.all(rh_raw == 0)   else np.zeros(73)
                pose = pose_raw.reshape(33, 4).copy()
                if not np.all(pose[:, :2] == 0):
                    xy = pose[:, :2] - pose[:, :2].mean(axis=0)
                    md = np.max(np.linalg.norm(xy, axis=1))
                    if md > 0:
                        xy = xy / md
                    pose[:, :2] = xy

                frame_vecs.append(np.concatenate([lh, rh, pose.flatten()]))

            if not frame_vecs:
                continue

            X.append(_frames_to_vector(frame_vecs))
            y.append(sign)
            counts[sign] += 1

    total = sum(counts.values())
    if total:
        print(f"  SWL-LSE pkl : ✓ {total} samples across {len(counts)} signs")
        for s, c in sorted(counts.items()):
            print(f"    {s:<16} {c:>4} samples")
    else:
        print(f"  SWL-LSE pkl : ✗ 0 usable samples (check CSV + pkl format)")

    return X, y


# ── Source 3: lse_clips webcam recordings (already in training_data/lse_data.npz) ──
def load_webcam_recordings(allowed_signs: set) -> tuple[list, list]:
    if not LSE_DATA_NPZ.exists():
        print(f"  Webcam clips: ✗ no lse_data.npz yet — record via Dev Panel")
        return [], []

    d    = np.load(LSE_DATA_NPZ, allow_pickle=True)
    X_all, y_all = d["X"], d["y"]

    if allowed_signs:
        mask  = np.array([str(lbl) in allowed_signs for lbl in y_all])
        X_all = X_all[mask]
        y_all = y_all[mask]

    counts = defaultdict(int)
    for lbl in y_all:
        counts[str(lbl)] += 1

    total = len(y_all)
    if total:
        print(f"  Webcam clips: ✓ {total} samples across {len(counts)} signs")
    else:
        print(f"  Webcam clips: ✗ 0 samples match requested signs")

    return list(X_all), list(y_all.astype(str))


# ── Source 4: Loose recordings ────────────────────────────────────────────────
def load_loose_vids(allowed_signs: set) -> tuple[list, list]:
    """
    Read ~/Downloads/LooseVids/<sign>.mp4 files.
    Extracts MediaPipe landmarks from each video, mean-pools to 278-dim vector.
    """
    print(f"  Loose vids  : checking {LOOSE_VIDS_DIR}")
    if not LOOSE_VIDS_DIR.exists():
        print(f"  Loose vids  : ✗ folder not found — skipping")
        return [], []

    mp4s = list(LOOSE_VIDS_DIR.glob("*.mp4"))
    print(f"  Loose vids  : found {len(mp4s)} mp4 files")

    try:
        import cv2
        import mp_holistic as mph
        from landmarks import extract_landmarks, normalize_landmarks
    except ImportError as e:
        print(f"  Loose vids  : ✗ import failed ({e}) — skipping")
        return [], []

    X, y = [], []
    counts = defaultdict(int)

    for mp4 in sorted(mp4s):
        sign = mp4.stem.lower()
        if sign not in LSE_DICT or len(sign) == 1:  # skip bare letters
            continue
        if allowed_signs and sign not in allowed_signs:
            continue

        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            continue

        frame_vecs = []
        try:
            import mp_holistic as mph
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

        if not frame_vecs:
            continue

        X.append(_frames_to_vector(frame_vecs))
        y.append(sign)
        counts[sign] += 1

    total = sum(counts.values())
    if total:
        print(f"  Loose vids  : ✓ {total} samples — " +
              ", ".join(f"{s}({c})" for s, c in sorted(counts.items())))
    else:
        print(f"  Loose vids  : ✗ 0 usable samples in {LOOSE_VIDS_DIR}")

    return X, y


# ── Shared: extract landmarks from a video file ───────────────────────────────
def _video_to_landmark_vector(mp4_path) -> "np.ndarray | None":
    """
    Open an mp4, run MediaPipe on every frame, mean-pool to a single 278-dim vector.
    Returns None if no hand frames detected.
    """
    try:
        import cv2
        import mp_holistic as mph
        from landmarks import extract_landmarks, normalize_landmarks
    except ImportError:
        return None

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None

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
    except Exception:
        pass
    finally:
        cap.release()

    if not frame_vecs:
        return None
    return _frames_to_vector(frame_vecs)


# ── Source 5: SWL-LSE videos (VIDEOS_RENAMED/) ────────────────────────────────
def load_swl_videos(allowed_signs: set) -> tuple[list, list]:
    """Extract landmarks from SWL-LSE renamed mp4 files."""
    if not SWL_VIDEOS_DIR.exists():
        print(f"  SWL videos  : ✗ {SWL_VIDEOS_DIR} not found — skipping")
        return [], []

    X, y = [], []
    counts = defaultdict(int)

    for mp4 in sorted(SWL_VIDEOS_DIR.glob("*.mp4")):
        sign = mp4.stem.split("_")[0].lower()
        if sign not in LSE_DICT or len(sign) == 1:  # skip bare letters
            continue
        if allowed_signs and sign not in allowed_signs:
            continue
        vec = _video_to_landmark_vector(mp4)
        if vec is not None:
            X.append(vec)
            y.append(sign)
            counts[sign] += 1

    total = sum(counts.values())
    if total:
        print(f"  SWL videos  : ✓ {total} samples — " +
              ", ".join(f"{s}({c})" for s, c in sorted(counts.items())))
    else:
        print(f"  SWL videos  : ✗ 0 usable samples in {SWL_VIDEOS_DIR}")
    return X, y


# ── Source 6: Sign4all videos ─────────────────────────────────────────────────
SIGN4ALL_TO_SIGN = {
    "COMER": "comer", "BEBER": "beber", "DORMIR": "dormir",
    "CAMINAR": "caminar", "SENTARSE": "sentar", "ABRIR": "abrir", "MIRAR": "mirar",
}

def load_sign4all_videos(allowed_signs: set) -> tuple[list, list]:
    """Extract landmarks from Sign4all mp4 files."""
    if not SIGN4ALL_DIR.exists():
        print(f"  Sign4all    : ✗ {SIGN4ALL_DIR} not found — skipping")
        return [], []

    X, y = [], []
    counts = defaultdict(int)

    for folder in sorted(SIGN4ALL_DIR.iterdir()):
        if not folder.is_dir():
            continue
        sign = SIGN4ALL_TO_SIGN.get(folder.name.upper())
        if not sign:
            continue
        if allowed_signs and sign not in allowed_signs:
            continue
        for mp4 in sorted(folder.glob("*.mp4")):
            vec = _video_to_landmark_vector(mp4)
            if vec is not None:
                X.append(vec)
                y.append(sign)
                counts[sign] += 1

    total = sum(counts.values())
    if total:
        print(f"  Sign4all    : ✓ {total} samples — " +
              ", ".join(f"{s}({c})" for s, c in sorted(counts.items())))
    else:
        print(f"  Sign4all    : ✗ 0 usable samples in {SIGN4ALL_DIR}")
    return X, y


def augment(X: np.ndarray, y: list, min_per_sign: int = 30) -> tuple:
    """
    For signs below min_per_sign samples, add jittered copies until we reach it.
    Jitter = small Gaussian noise on joint coords — simulates natural variation.
    """
    from collections import Counter
    counts = Counter(y)
    X_aug, y_aug = list(X), list(y)

    for sign, n in counts.items():
        if n >= min_per_sign:
            continue
        needed  = min_per_sign - n
        indices = [i for i, lbl in enumerate(y) if lbl == sign]
        added   = 0
        while added < needed:
            src = X[indices[added % len(indices)]]
            noise = np.random.normal(0, 0.01, src.shape).astype(np.float32)
            X_aug.append(src + noise)
            y_aug.append(sign)
            added += 1

    return np.array(X_aug, dtype=np.float32), y_aug


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    signs_filter = set(args.signs) if args.signs else None

    print("=" * 60)
    print("  SignFuture — LSE MLP Trainer (landmark-based)")
    print("  Background-proof: no pixels, only joint coordinates")
    print("=" * 60)
    print()
    print("  Collecting training data…")

    X_all, y_all = [], []

    # Source 1: LSE-FS-UVigo JSON
    if not args.no_uvigo:
        Xv, yv = load_uvigo_fs(signs_filter or set())
        X_all += Xv
        y_all += yv

    # Source 2: SWL-LSE pkl — DISABLED
    # The pkl files were saved with mediapipe.framework which was removed in MediaPipe 0.10.
    # Use Source 5 (SWL videos → landmarks) instead — same data, works with current MediaPipe.
    # if not args.no_swl:
    #     Xs, ys = load_swl_mediapipe(signs_filter or set())
    #     X_all += Xs
    #     y_all += ys

    # Source 3: Webcam/feedback recordings
    Xw, yw = load_webcam_recordings(signs_filter or set())
    X_all += Xw
    y_all += yw

    # Source 4: Loose recordings
    Xl, yl = load_loose_vids(signs_filter or set())
    X_all += Xl
    y_all += yl

    # Source 5: SWL-LSE videos → landmarks
    Xsv, ysv = load_swl_videos(signs_filter or set())
    X_all += Xsv
    y_all += ysv

    # Source 6: Sign4all videos → landmarks
    Xsa, ysa = load_sign4all_videos(signs_filter or set())
    X_all += Xsa
    y_all += ysa

    if len(X_all) == 0:
        print()
        print("  ❌ No training data found from any source.")
        print("     → Download LSE-FS-UVigo: https://zenodo.org/records/15797079")
        print(f"       Extract PROC_KPS.ZIP to: {UVIGO_FS_DIR}")
        print("     → OR record signs via Dev Panel → Train tab")
        return

    from collections import Counter
    label_counts = Counter(y_all)
    covered = [s for s, c in label_counts.items() if c > 0]

    print()
    print(f"  ── Raw totals ──────────────────────────────────────")
    for sign in sorted(covered):
        print(f"  {sign:<16} {label_counts[sign]:>4} samples")
    print(f"  Total: {len(X_all)} samples, {len(covered)} signs")

    if len(covered) < 2:
        print()
        print("  ❌ Need at least 2 signs to train a classifier.")
        return

    # Augment sparse signs up to 30 samples minimum
    X_np = np.array(X_all, dtype=np.float32)
    X_np, y_all = augment(X_np, y_all, min_per_sign=30)
    print()
    print(f"  After augmentation: {len(y_all)} samples total")

    # Train
    print()
    print(f"  Training MLP — {args.epochs} epochs, lr={args.lr}")
    print(f"  Signs: {sorted(covered)}")
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

    # Save
    os.makedirs(os.path.dirname(MODEL_LSE) or ".", exist_ok=True)
    clf.save(MODEL_LSE)
    print(f"  ✅ Saved    → {MODEL_LSE}")
    print()
    print("  Start the server:  bash run.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
