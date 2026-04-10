#!/usr/bin/env python3
"""train_continuous.py — Cyclic training loop for LSE fusion model.

Runs N cycles. Each cycle:
  1. I3D  — E1 epochs on all clip data  → saves i3d_lse.pt
  2. MLP  — E2 epochs on all landmark data → saves model_lse.pt

The server picks up new model files automatically on the next request.

Run:
    python train_continuous.py                        # 10 cycles, 60 I3D, 300 MLP (defaults)
    python train_continuous.py --cycles 20
    python train_continuous.py --cycles 5 --i3d-epochs 100 --mlp-epochs 500
    python train_continuous.py --cycles 10 --no-i3d   # MLP only
    python train_continuous.py --cycles 10 --no-mlp   # I3D only
"""

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Cyclic LSE training loop")
parser.add_argument("--lang",        default="lse", choices=["asl", "lse"],
                    help="Language to train (default: lse)")
parser.add_argument("--cycles",      type=int, default=10,
                    help="Number of full I3D+MLP cycles to run (default: 10)")
parser.add_argument("--i3d-epochs",  type=int, default=60,
                    help="I3D epochs per cycle (default: 60)")
parser.add_argument("--mlp-epochs",  type=int, default=300,
                    help="MLP epochs per cycle (default: 300)")
parser.add_argument("--no-i3d",      action="store_true", help="Skip I3D training")
parser.add_argument("--no-mlp",      action="store_true", help="Skip MLP training")
parser.add_argument("--min-clips",   type=int, default=20,
                    help="Minimum clips per sign for augmentation (default: 20)")
parser.add_argument("--refresh",     action="store_true",
                    help="Force re-scan all data sources even if cache is fresh")
parser.add_argument("--swl-dir",     type=str, default=None,
                    help="Override SWL-LSE VIDEOS_RENAMED directory")
parser.add_argument("--sign4all-dir",type=str, default=None,
                    help="Override Sign4all directory")
parser.add_argument("--loose-dir",   type=str, default=None,
                    help="Override LooseVids directory")
parser.add_argument("--uvigo-dir",   type=str, default=None,
                    help="Override LSE-FS-UVigo TRANSFORMED_KPS directory")
parser.add_argument("--extra-dirs",  nargs="+", metavar="DIR",
                    help="Extra directories to scan for <sign>.mp4 files")
args = parser.parse_args()

# ── Apply CLI directory overrides ─────────────────────────────────────────────
# Defaults are set after imports so we patch them here
_DIR_OVERRIDES = {
    "swl_dir":      ("SWL_VIDEOS_DIR",  args.swl_dir),
    "sign4all_dir": ("SIGN4ALL_DIR",    args.sign4all_dir),
    "loose_dir":    ("LOOSE_VIDS_DIR",  args.loose_dir),
    "uvigo_dir":    ("UVIGO_FS_DIR",    args.uvigo_dir),
}

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("❌ PyTorch / numpy not installed. Run setup.sh first.")
    sys.exit(1)

import numpy as np                                          # noqa: F811 — PyCharm visibility
import torch                                                # noqa: F811
import torch.nn as nn                                       # noqa: F811
import torch.optim as optim                                 # noqa: F811
from torch.utils.data import DataLoader, TensorDataset     # noqa: F811

from asl_dictionary import LSE_DICT
from sign_model import (
    _I3DNet, _MLP, I3D_LSE, MODEL_LSE, load_kinetics_weights,
    DEVICE, INPUT_DIM, sequence_to_feature,
    _LANG_CLIPS, _LANG_DATA, data_path,
    load_samples, save_samples,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
SWL_VIDEOS_DIR     = Path.home() / "Downloads" / "13691887" / "VIDEOS_RENAMED"
SIGN4ALL_DIR       = Path.home() / "Downloads" / "sign4all"
LOOSE_VIDS_DIR     = Path.home() / "Downloads" / "LooseVids"
UVIGO_FS_DIR       = Path.home() / "Downloads" / "15797079" / "PROC_KPS" / "TRANSFORMED_KPS"
FEEDBACK_CLIPS_NPZ = HERE / "training_data" / "clips_lse.npz"
FEEDBACK_LM_NPZ    = HERE / "training_data" / f"samples_{args.lang}.npz"
MSASL_NPZ          = HERE / "training_data" / "samples_asl.npz"  # MS-ASL landmarks

DATA_DIR   = HERE / "training_data"
CACHE_DIR  = DATA_DIR / "clip_cache_cont"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Apply CLI overrides to path constants
if args.swl_dir:      SWL_VIDEOS_DIR = Path(args.swl_dir)
if args.sign4all_dir: SIGN4ALL_DIR   = Path(args.sign4all_dir)
if args.loose_dir:    LOOSE_VIDS_DIR = Path(args.loose_dir)
if args.uvigo_dir:    UVIGO_FS_DIR   = Path(args.uvigo_dir)
EXTRA_DIRS = [Path(d) for d in (args.extra_dirs or [])]

# ── Collected data cache ───────────────────────────────────────────────────────
# On first run, all sources are scanned and saved here.
# On subsequent runs, cache is reused unless source file counts changed.
I3D_CACHE_NPZ = DATA_DIR / "collected_clips_lse.npz"    # raw clip entries list
MLP_CACHE_NPZ = DATA_DIR / "collected_lse.npz"          # 278-dim X + y arrays
FINGERPRINT_FILE = DATA_DIR / "source_fingerprint.json" # file counts per source

MIN_CLIPS  = args.min_clips
MIN_FRAMES = 8


def _source_fingerprint() -> dict:
    """Count files in each source folder — used to detect new data."""
    def count(path, pattern="**/*"):
        if not Path(path).exists():
            return 0
        return sum(1 for _ in Path(path).glob(pattern) if _.is_file())

    return {
        "swl_videos":    count(SWL_VIDEOS_DIR, "*.mp4"),
        "sign4all":      count(SIGN4ALL_DIR,   "**/*.mp4"),
        "loose_vids":    count(LOOSE_VIDS_DIR, "*.mp4"),
        "uvigo_fs":      count(UVIGO_FS_DIR,   "**/*.json"),
        "feedback_clips": FEEDBACK_CLIPS_NPZ.stat().st_size if FEEDBACK_CLIPS_NPZ.exists() else 0,
        "feedback_lm":   FEEDBACK_LM_NPZ.stat().st_size    if FEEDBACK_LM_NPZ.exists()    else 0,
        "msasl_npz":     MSASL_NPZ.stat().st_size           if MSASL_NPZ.exists()          else 0,
        "extra_dirs":    sorted(str(d) for d in EXTRA_DIRS),
    }


def _cache_is_fresh() -> bool:
    """Return True if cache exists and source file counts haven't changed."""
    if not FINGERPRINT_FILE.exists():
        return False
    if not I3D_CACHE_NPZ.exists() or not MLP_CACHE_NPZ.exists():
        return False
    import json
    try:
        with open(FINGERPRINT_FILE) as f:
            saved = json.load(f)
        return saved == _source_fingerprint()
    except Exception:
        return False


def _save_fingerprint():
    import json
    with open(FINGERPRINT_FILE, "w") as f:
        json.dump(_source_fingerprint(), f, indent=2)
    print(f"  Fingerprint saved → {FINGERPRINT_FILE}")

# ── Reuse source mappings from train_lse.py ───────────────────────────────────
SWL_LABEL_TO_SIGN = {
    "DOLOR": "dolor", "GARGANTA": "garganta", "GARGANTA2": "garganta",
    "MAREO": "mareo", "MAREO2": "mareo", "MOCO": "moco", "MOCO2": "moco",
    "RESPIRAR": "respirar", "RESPIRAR2": "respirar", "COMER": "comer",
    "DORMIR": "dormir", "TOSER": "tos", "FIEBRE2": "fiebre",
    "SANGRE2": "sangre", "ACUFENO": "zumbido",
    "AMIGDALAS-INFLAMAR": "hinchazon", "AMIGDALITIS": "infeccion",
}
SIGN4ALL_TO_SIGN = {
    "COMER": "comer", "BEBER": "beber", "DORMIR": "dormir",
    "CAMINAR": "caminar", "SENTARSE": "sentar", "ABRIR": "abrir", "MIRAR": "mirar",
}
UVIGO_FS_LABEL_MAP = {
    "DOLOR": "dolor", "GARGANTA": "garganta", "MAREO": "mareo", "MOCO": "moco",
    "RESPIRAR": "respirar", "TOSER": "tos", "TOS": "tos", "FIEBRE": "fiebre",
    "SANGRE": "sangre", "COMER": "comer", "DORMIR": "dormir", "BEBER": "beber",
    "CAMINAR": "caminar", "SENTAR": "sentar", "SENTARSE": "sentar",
    "ABRIR": "abrir", "MIRAR": "mirar", "GIRAR": "girar",
    "INFECCION": "infeccion", "INFECCIÓN": "infeccion",
    "HINCHAZON": "hinchazon", "HINCHAZÓN": "hinchazon",
    "ZUMBIDO": "zumbido", "ACUFENO": "zumbido", "ACÚFENO": "zumbido",
}

# ── Video helpers ─────────────────────────────────────────────────────────────
try:
    import cv2 as _cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

def extract_raw_frames(path) -> list:
    if not _CV2_OK:
        return []
    cap = _cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames if len(frames) >= MIN_FRAMES else []

def frames_to_npz(frames, path):
    from sign_model import I3D_FRAMES, I3D_SIZE
    n   = len(frames)
    idx = np.linspace(0, n - 1, I3D_FRAMES).astype(int)
    sel = np.stack([_cv2.resize(
                _cv2.cvtColor(frames[i], _cv2.COLOR_BGR2RGB),
                (I3D_SIZE, I3D_SIZE))
              for i in idx], axis=0)
    np.savez_compressed(path, frames=sel)

def augment_clip(frames):
    import random
    seq    = list(frames)
    factor = random.uniform(0.72, 1.30)
    ni     = max(MIN_FRAMES, int(len(seq) * factor))
    seq    = [seq[i] for i in np.linspace(0, len(seq) - 1, ni).astype(int)]
    if random.random() < 0.5:
        delta = random.uniform(-0.20, 0.20)
        seq   = [(np.clip(f.astype(np.float32) / 255 + delta, 0, 1) * 255
                  ).astype(np.uint8) for f in seq]
    if random.random() < 0.4:
        seq = [f[:, ::-1, :].copy() for f in seq]
    return seq


# ── I3D clip dataset (disk-backed) ────────────────────────────────────────────
class DiskClipDataset(torch.utils.data.Dataset):
    def __init__(self, entries):
        self.entries = [(p, l) for p, l in entries if Path(p).exists()]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        from sign_model import I3D_FRAMES, I3D_SIZE
        path, label = self.entries[idx]
        d   = np.load(path)
        arr = d["frames"].astype(np.float32) / 255.0  # (T, H, W, 3)

        # Ensure correct shape — handle both (T,H,W,3) and unexpected formats
        if arr.ndim == 3:
            # Missing channel dim — add it
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[0] != I3D_FRAMES:
            idx_t = np.linspace(0, arr.shape[0] - 1, I3D_FRAMES).astype(int)
            arr   = arr[idx_t]

        _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr   = (arr - _MEAN) / _STD
        t     = torch.tensor(arr.transpose(3, 0, 1, 2))  # (3, T, H, W)

        # Resize spatial dims if needed (should be I3D_SIZE x I3D_SIZE)
        if t.shape[2] != I3D_SIZE or t.shape[3] != I3D_SIZE:
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0),                          # (1, 3, T, H, W)
                size=(I3D_FRAMES, I3D_SIZE, I3D_SIZE),
                mode="trilinear", align_corners=False
            ).squeeze(0)                                 # (3, T, H, W)

        return t, label


# ── Collect I3D clips ─────────────────────────────────────────────────────────
def collect_i3d_clips(signs):
    """Collect raw video clips from all sources, cache as .npz. Returns (entries, sign_count)."""
    from collections import defaultdict
    entries    = []
    sign_count = defaultdict(int)
    sign_real  = defaultdict(int)

    # Sign4all index
    sign4all_index = defaultdict(list)
    if SIGN4ALL_DIR.exists():
        for folder in sorted(SIGN4ALL_DIR.iterdir()):
            upper = folder.name.upper()
            sign  = SIGN4ALL_TO_SIGN.get(upper)
            if sign:
                sign4all_index[sign].extend(sorted(folder.glob("*.mp4")))

    for sign in signs:
        if sign not in LSE_DICT:
            continue
        real_clips = []

        # SWL-LSE
        if SWL_VIDEOS_DIR.exists():
            for mp4 in sorted(SWL_VIDEOS_DIR.glob(f"{sign}_*.mp4")):
                f = extract_raw_frames(mp4)
                if f: real_clips.append(f)

        # Sign4all
        for mp4 in sign4all_index.get(sign, []):
            f = extract_raw_frames(mp4)
            if f: real_clips.append(f)

        # LooseVids
        if LOOSE_VIDS_DIR.exists():
            loose = LOOSE_VIDS_DIR / f"{sign}.mp4"
            if loose.exists():
                f = extract_raw_frames(loose)
                if f: real_clips.append(f)

        # Feedback clips
        if FEEDBACK_CLIPS_NPZ.exists():
            d = np.load(FEEDBACK_CLIPS_NPZ, allow_pickle=True)
            for clip_frames, clip_label in zip(d["X"], d["y"]):
                if str(clip_label) == sign and isinstance(clip_frames, (list, np.ndarray)):
                    f = list(clip_frames)
                    if len(f) >= MIN_FRAMES:
                        real_clips.append(f)

        # Extra dirs — scan for <sign>.mp4 or <sign>_*.mp4
        for extra in EXTRA_DIRS:
            if not extra.exists():
                continue
            for mp4 in sorted(extra.glob(f"{sign}*.mp4")):
                f = extract_raw_frames(mp4)
                if f:
                    real_clips.append(f)

        if not real_clips:
            continue

        sign_real[sign] = len(real_clips)

        for frames in real_clips:
            out = CACHE_DIR / f"{sign}_{sign_count[sign]:04d}.npz"
            if not out.exists():
                frames_to_npz(frames, out)
            entries.append((out, sign))
            sign_count[sign] += 1

        # Augment if below MIN_CLIPS
        if sign_count[sign] < MIN_CLIPS:
            needed = MIN_CLIPS - sign_count[sign]
            cycle  = 0
            for _ in range(needed):
                base = real_clips[cycle % len(real_clips)]
                aug  = augment_clip(base)
                out  = CACHE_DIR / f"{sign}_{sign_count[sign]:04d}.npz"
                if not out.exists():
                    frames_to_npz(aug, out)
                entries.append((out, sign))
                sign_count[sign] += 1
                cycle += 1

    return entries, dict(sign_count)


# ── Collect MLP landmarks ─────────────────────────────────────────────────────
def collect_mlp_samples(signs):
    """Collect 278-dim landmark vectors from all sources. Returns (X, y)."""
    import json, unicodedata

    X_all, y_all = [], []

    def norm_label(s):
        return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                       if unicodedata.category(c) != "Mn")

    # LSE-FS-UVigo JSON
    if UVIGO_FS_DIR.exists():
        MIDDLE_MCP = 9; INDEX_MCP = 5; PINKY_MCP = 17; FINGER_TIPS = [4,8,12,16,20]

        def norm_hand(pts):
            h    = pts - pts[MIDDLE_MCP].copy()
            span = np.linalg.norm(h[INDEX_MCP] - h[PINKY_MCP])
            if span > 0: h = h / span
            dists = [np.linalg.norm(h[FINGER_TIPS[i]] - h[FINGER_TIPS[j]])
                     for i in range(5) for j in range(i+1, 5)]
            return np.concatenate([h.flatten(), np.array(dists)])

        for split in ("train", "validation", "test"):
            split_dir = UVIGO_FS_DIR / split
            if not split_dir.exists(): continue
            for jf in sorted(split_dir.glob("*.json")):
                try:
                    with open(jf, encoding="utf-8") as f:
                        data = json.load(f)
                except Exception: continue
                raw_label = data.get("metadata", {}).get("label", "")
                sign = UVIGO_FS_LABEL_MAP.get(raw_label.upper()) or \
                       UVIGO_FS_LABEL_MAP.get(norm_label(raw_label).upper())
                if not sign or sign not in LSE_DICT or sign not in signs: continue
                handness   = data.get("metadata", {}).get("handness", "right").lower()
                frame_vecs = []
                for frame in data.get("frames", []):
                    hand_key = "right_hand" if handness == "right" else "left_hand"
                    kps = frame.get(hand_key, {}).get("keypoints", [])
                    if len(kps) < 21: continue
                    pts = np.array([[k.get("x",0),k.get("y",0),k.get("z",0)]
                                    if isinstance(k, dict) else k[:3]
                                    for k in kps[:21]], dtype=np.float32)
                    hv   = norm_hand(pts)
                    lh   = hv if handness == "left"  else np.zeros(73, np.float32)
                    rh   = hv if handness == "right" else np.zeros(73, np.float32)
                    pose = np.zeros(132, np.float32)
                    frame_vecs.append(np.concatenate([lh, rh, pose]))
                if frame_vecs:
                    X_all.append(np.array(frame_vecs, np.float32).mean(axis=0))
                    y_all.append(sign)

    # Feedback / webcam samples
    if FEEDBACK_LM_NPZ.exists():
        d = np.load(FEEDBACK_LM_NPZ, allow_pickle=True)
        for x, lbl in zip(d["X"], d["y"]):
            if str(lbl) in signs:
                X_all.append(x.astype(np.float32))
                y_all.append(str(lbl))

    # MS-ASL landmarks (ASL only — only load when training ASL model)
    if args.lang == "asl" and MSASL_NPZ.exists():
        d = np.load(MSASL_NPZ, allow_pickle=True)
        count = 0
        for x, lbl in zip(d["X"], d["y"]):
            lbl_str = str(lbl).lower()
            if lbl_str in signs:
                X_all.append(x.astype(np.float32))
                y_all.append(lbl_str)
                count += 1
        if count:
            print(f"  MS-ASL: loaded {count} samples")
    if LOOSE_VIDS_DIR.exists() and _CV2_OK:
        try:
            import mp_holistic as mph
            from landmarks import extract_landmarks, normalize_landmarks
            for mp4 in sorted(LOOSE_VIDS_DIR.glob("*.mp4")):
                sign = mp4.stem.lower()
                if sign not in LSE_DICT or sign not in signs: continue
                cap = _cv2.VideoCapture(str(mp4))
                fvecs = []
                with mph.Holistic(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as h:
                    while True:
                        ret, frame = cap.read()
                        if not ret: break
                        rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
                        lm  = normalize_landmarks(extract_landmarks(h.process(rgb)))
                        if not np.all(lm[:126] == 0):
                            fvecs.append(lm)
                cap.release()
                if fvecs:
                    X_all.append(np.array(fvecs, np.float32).mean(axis=0))
                    y_all.append(sign)
        except Exception as e:
            print(f"  [mlp] LooseVids landmark extraction error: {e}")

    # Extra dirs — scan for <sign>.mp4 files
    if EXTRA_DIRS and _CV2_OK:
        try:
            import mp_holistic as mph
            from landmarks import extract_landmarks, normalize_landmarks
            for extra in EXTRA_DIRS:
                if not extra.exists():
                    continue
                for mp4 in sorted(extra.glob("*.mp4")):
                    sign = mp4.stem.split("_")[0].lower()
                    if sign not in LSE_DICT or sign not in signs:
                        continue
                    cap   = _cv2.VideoCapture(str(mp4))
                    fvecs = []
                    with mph.Holistic(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5) as h:
                        while True:
                            ret, frame = cap.read()
                            if not ret: break
                            rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
                            lm  = normalize_landmarks(extract_landmarks(h.process(rgb)))
                            if not np.all(lm[:126] == 0):
                                fvecs.append(lm)
                    cap.release()
                    if fvecs:
                        X_all.append(np.array(fvecs, np.float32).mean(axis=0))
                        y_all.append(sign)
        except Exception as e:
            print(f"  [mlp] Extra dirs landmark extraction error: {e}")

    return X_all, y_all


# ── Training loops ────────────────────────────────────────────────────────────
def train_i3d_one_cycle(entries, labels, net, opt, sch, epoch_num):
    lbl2idx = {l: i for i, l in enumerate(labels)}
    indexed = [(p, lbl2idx[l]) for p, l in entries if l in lbl2idx]
    ds = DiskClipDataset(indexed)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0,
                    pin_memory=(DEVICE.type == "cuda"))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    net.train()
    ep_loss = 0.0
    for xb, yb in dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(net(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        ep_loss += loss.item()
    sch.step()
    avg_loss = ep_loss / max(1, len(dl))

    # Quick accuracy
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in DataLoader(ds, batch_size=8):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            correct += (net(xb).argmax(1) == yb).sum().item()
            total   += len(yb)
    acc = correct / max(1, total)
    return avg_loss, acc


def train_mlp_one_cycle(X, y, labels, net, opt, sch):
    lbl2idx = {l: i for i, l in enumerate(labels)}
    valid   = [(x, lbl2idx[lbl]) for x, lbl in zip(X, y) if lbl in lbl2idx]
    if not valid:
        return 0.0, 0.0
    Xv = torch.tensor(np.array([v[0] for v in valid], np.float32)).to(DEVICE)
    yv = torch.tensor(np.array([v[1] for v in valid], np.int64)).to(DEVICE)
    dl = DataLoader(TensorDataset(Xv, yv), batch_size=32, shuffle=True)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    net.train()
    ep_loss = 0.0
    for xb, yb in dl:
        opt.zero_grad()
        loss = loss_fn(net(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        ep_loss += loss.item()
    sch.step()
    avg_loss = ep_loss / max(1, len(dl))
    net.eval()
    with torch.no_grad():
        acc = (net(Xv).argmax(1) == yv).float().mean().item()
    return avg_loss, acc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    signs = list(LSE_DICT.keys())
    lang  = args.lang

    print("=" * 60)
    print(f"  SignFuture — Continuous Training ({lang.upper()})")
    print(f"  Cycles     : {args.cycles}")
    print(f"  I3D epochs : {args.i3d_epochs}  {'(skipped)' if args.no_i3d else ''}")
    print(f"  MLP epochs : {args.mlp_epochs}  {'(skipped)' if args.no_mlp else ''}")
    print(f"  Device     : {DEVICE}")
    print("=" * 60)

    # ── Load or collect training data ─────────────────────────────────────────
    use_cache = not args.refresh and _cache_is_fresh()

    if use_cache:
        print("\n  ✅ Cache is fresh — loading from training_data/collected_*.npz")
        print("     (run with --refresh to force re-scan)")
        # Load I3D entries list from cache
        i3d_entries, i3d_sign_count = collect_i3d_clips(signs) if not args.no_i3d else ([], {})
        # Load MLP samples from cache
        if not args.no_mlp and MLP_CACHE_NPZ.exists():
            d      = np.load(MLP_CACHE_NPZ, allow_pickle=True)
            mlp_X  = list(d["X"].astype(np.float32))
            mlp_y  = list(d["y"].astype(str))
            print(f"  MLP: loaded {len(mlp_X)} samples from cache")
        else:
            mlp_X, mlp_y = [], []
    else:
        if args.refresh:
            print("\n  --refresh flag set — re-scanning all sources…")
        else:
            print("\n  No cache or sources changed — collecting data…")

        print("  Collecting I3D clip data…")
        i3d_entries, i3d_sign_count = collect_i3d_clips(signs) if not args.no_i3d else ([], {})

        print("\n  Collecting MLP landmark data…")
        mlp_X, mlp_y = collect_mlp_samples(signs) if not args.no_mlp else ([], [])

        # Save MLP samples to cache
        if mlp_X and not args.no_mlp:
            np.savez(MLP_CACHE_NPZ,
                     X=np.array(mlp_X, dtype=np.float32),
                     y=np.array(mlp_y))
            print(f"  MLP cache saved → {MLP_CACHE_NPZ}")

        _save_fingerprint()

    covered_i3d = sorted(s for s, c in i3d_sign_count.items() if c > 0)
    covered_mlp = sorted(set(mlp_y))

    if not covered_i3d and not covered_mlp:
        print("\n  ❌ No training data found from any source.")
        return

    # ── Build models ─────────────────────────────────────────────────────────
    i3d_net = i3d_opt = i3d_sch = None
    mlp_net = mlp_opt = mlp_sch = None

    if covered_i3d and not args.no_i3d:
        i3d_net = _I3DNet(len(covered_i3d)).to(DEVICE)
        load_kinetics_weights(i3d_net)
        # Load existing weights if available
        if os.path.exists(I3D_LSE):
            try:
                ck = torch.load(I3D_LSE, map_location=DEVICE, weights_only=False)
                if ck.get("labels") == covered_i3d:
                    i3d_net.load_state_dict(ck["state_dict"])
                    print("  I3D: loaded existing weights")
                else:
                    print("  I3D: label set changed — starting fresh")
            except Exception as e:
                print(f"  I3D: could not load weights ({e}) — starting fresh")
        i3d_opt = optim.AdamW(i3d_net.parameters(), lr=1e-4, weight_decay=1e-4)
        i3d_sch = optim.lr_scheduler.CosineAnnealingLR(
            i3d_opt, T_max=args.cycles * args.i3d_epochs)
        print(f"  I3D: {len(covered_i3d)} signs, {len(i3d_entries)} clips")

    if covered_mlp and not args.no_mlp:
        mlp_net = _MLP(len(covered_mlp)).to(DEVICE)
        mlp_path = os.path.join("training_data", f"model_{lang}.pt")
        if os.path.exists(mlp_path):
            try:
                ck = torch.load(mlp_path, map_location=DEVICE, weights_only=False)
                if ck.get("labels") == covered_mlp:
                    mlp_net.load_state_dict(ck["state_dict"])
                    print("  MLP: loaded existing weights")
                else:
                    print("  MLP: label set changed — starting fresh")
            except Exception as e:
                print(f"  MLP: could not load weights ({e}) — starting fresh")
        mlp_opt = optim.AdamW(mlp_net.parameters(), lr=1e-4, weight_decay=2e-4)
        mlp_sch = optim.lr_scheduler.CosineAnnealingLR(
            mlp_opt, T_max=args.cycles * args.mlp_epochs)
        print(f"  MLP: {len(covered_mlp)} signs, {len(mlp_X)} samples")

    print()

    # ── Cycle loop ────────────────────────────────────────────────────────────
    for cycle in range(1, args.cycles + 1):
        cycle_start = time.time()
        print(f"  ── Cycle {cycle}/{args.cycles} {'─'*40}")

        # Each cycle: only reload the fast feedback files (clips_lse.npz / samples_lse.npz)
        # Heavy source scanning (Sign4all, SWL videos) only happens on --refresh
        if cycle > 1:
            if not args.no_i3d and covered_i3d:
                # Re-scan feedback clips only (fast)
                new_entries, new_counts = collect_i3d_clips(signs)
                if len(new_entries) != len(i3d_entries):
                    print(f"  New I3D clips detected — updated from "
                          f"{len(i3d_entries)} to {len(new_entries)}")
                    i3d_entries    = new_entries
                    i3d_sign_count = new_counts
            if not args.no_mlp and covered_mlp:
                # Merge any new feedback landmarks into existing MLP data
                if FEEDBACK_LM_NPZ.exists():
                    d     = np.load(FEEDBACK_LM_NPZ, allow_pickle=True)
                    fb_X  = list(d["X"].astype(np.float32))
                    fb_y  = list(d["y"].astype(str))
                    # Add only samples not already in mlp_y count
                    if len(fb_X) > 0:
                        mlp_X = list(np.array(mlp_X + fb_X, np.float32))
                        mlp_y = mlp_y + fb_y

        # ── I3D: run i3d_epochs epochs ────────────────────────────────────────
        if i3d_net is not None:
            i3d_loss_hist = []
            for ep in range(1, args.i3d_epochs + 1):
                loss, acc = train_i3d_one_cycle(
                    i3d_entries, covered_i3d, i3d_net, i3d_opt, i3d_sch,
                    epoch_num=(cycle - 1) * args.i3d_epochs + ep)
                i3d_loss_hist.append(loss)
                pct = int(ep / args.i3d_epochs * 100)
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                print(f"\r  I3D [{bar}] {pct:3d}%  ep {ep}/{args.i3d_epochs}"
                      f"  loss={loss:.3f}  acc={acc*100:.1f}%",
                      end="", flush=True)

            print()  # newline after progress bar

            # Save I3D
            os.makedirs(os.path.dirname(I3D_LSE) or ".", exist_ok=True)
            torch.save({
                "labels":     covered_i3d,
                "state_dict": i3d_net.state_dict(),
                "arch":       "i3d",
            }, I3D_LSE)
            print(f"  I3D saved → {I3D_LSE}  (acc={acc*100:.1f}%)")

        # ── MLP: run mlp_epochs epochs ────────────────────────────────────────
        if mlp_net is not None:
            for ep in range(1, args.mlp_epochs + 1):
                loss, acc = train_mlp_one_cycle(
                    mlp_X, mlp_y, covered_mlp, mlp_net, mlp_opt, mlp_sch)
                pct = int(ep / args.mlp_epochs * 100)
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                print(f"\r  MLP [{bar}] {pct:3d}%  ep {ep}/{args.mlp_epochs}"
                      f"  loss={loss:.3f}  acc={acc*100:.1f}%",
                      end="", flush=True)

            print()

            # Save MLP
            mlp_path = os.path.join("training_data", f"model_{lang}.pt")
            os.makedirs(os.path.dirname(mlp_path) or ".", exist_ok=True)
            torch.save({
                "labels":     covered_mlp,
                "state_dict": mlp_net.state_dict(),
                "arch":       "mlp",
            }, mlp_path)
            print(f"  MLP saved → {mlp_path}  (acc={acc*100:.1f}%)")

        elapsed = time.time() - cycle_start
        print(f"  Cycle {cycle} done in {elapsed:.0f}s\n")

    print("=" * 60)
    print(f"  ✅ All {args.cycles} cycles complete.")
    print(f"  Restart server to reload: bash run.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
