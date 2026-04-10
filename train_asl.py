#!/usr/bin/env python3
"""train_asl.py — ASL I3D training. Disk-backed to avoid OOM.

Reads clips from ALL available sources:
  1. MS-ASL clips downloaded via download_msasl.py
  2. asl_clips/   — your own webcam recordings via Dev Panel
  3. Feedback clips — training_data/clips_asl.npz

Run:
    python train_asl.py
    python train_asl.py --epochs-i3d 120
    python train_asl.py --signs eat drink sit
    python train_asl.py --no-cache   (re-extract even if cache exists)
    python train_asl.py --no-augment (skip augmentation)
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

parser = argparse.ArgumentParser(description="Train ASL I3D model")
parser.add_argument("--i3d-only",    action="store_true")
parser.add_argument("--no-augment",  action="store_true")
parser.add_argument("--no-cache",    action="store_true")
parser.add_argument("--signs",       nargs="+", metavar="SIGN")
parser.add_argument("--epochs-i3d",  type=int, default=60)
parser.add_argument("--min-clips",   type=int, default=20)
args = parser.parse_args()

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    print("❌ PyTorch not installed. Run setup.sh first.")
    sys.exit(1)

import torch                                         # noqa: F811
import torch.nn as nn                                # noqa: F811
import torch.optim as optim                          # noqa: F811
from torch.utils.data import DataLoader, Dataset     # noqa: F811

import random
import numpy as np
import cv2
from collections import defaultdict

from asl_dictionary import ASL_DICT
from sign_model import (
    _I3DNet,
    load_kinetics_weights,
    I3D_ASL,
    DEVICE,
)

import numpy as np   # noqa: F811
import cv2           # noqa: F811

# ── Paths ─────────────────────────────────────────────────────────────────────
MSASL_CLIPS_DIR    = Path.home() / "Downloads" / "MS-ASL" / "clips"
ASL_CLIPS_DIR      = HERE / "asl_clips"
FEEDBACK_CLIPS_NPZ = HERE / "training_data" / "clips_asl.npz"
DATA_DIR           = HERE / "training_data"
CACHE_DIR          = DATA_DIR / "clip_cache_asl"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# MS-ASL label index → ASL_DICT key
# Only signs that exist in both MS-ASL and ASL_DICT
MSASL_LABEL_TO_SIGN = {
    770:  "ear",
    891:  "nose",
    967:  "mouth",
    852:  "head",
    76:   "pain",
    255:  "cough",
    843:  "dizzy",
    3:    "eat",
    56:   "drink",
    361:  "sleep",
    80:   "walk",
    18:   "sit",
}

MIN_CLIPS  = args.min_clips
MIN_FRAMES = 8


# ── Video helpers ─────────────────────────────────────────────────────────────
def extract_raw_frames(path) -> list:
    cap = cv2.VideoCapture(str(path))
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


def augment_clip(frames):
    seq    = list(frames)
    factor = random.uniform(0.72, 1.30)
    ni     = max(MIN_FRAMES, int(len(seq) * factor))
    seq    = [seq[i] for i in np.linspace(0, len(seq)-1, ni).astype(int)]
    if random.random() < 0.50:
        delta = random.uniform(-0.20, 0.20)
        seq   = [(np.clip(f.astype(np.float32)/255.0 + delta, 0, 1)*255
                  ).astype(np.uint8) for f in seq]
    if random.random() < 0.40:
        seq = [f[:, ::-1, :].copy() for f in seq]
    return seq


def frames_to_npz(frames, path):
    from sign_model import I3D_FRAMES, I3D_SIZE
    n   = len(frames)
    idx = np.linspace(0, n-1, I3D_FRAMES).astype(int)
    sel = np.stack([cv2.resize(
                cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB),
                (I3D_SIZE, I3D_SIZE))
              for i in idx], axis=0)
    np.savez_compressed(path, frames=sel)


# ── Dataset ───────────────────────────────────────────────────────────────────
class DiskClipDataset(Dataset):
    def __init__(self, entries):
        self.entries = [(p, l) for p, l in entries if Path(p).exists()]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        from sign_model import I3D_FRAMES, I3D_SIZE
        path, label = self.entries[idx]
        d   = np.load(path)
        arr = d["frames"].astype(np.float32) / 255.0
        if arr.ndim == 3:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[0] != I3D_FRAMES:
            idx_t = np.linspace(0, arr.shape[0]-1, I3D_FRAMES).astype(int)
            arr   = arr[idx_t]
        _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr   = (arr - _MEAN) / _STD
        t     = torch.tensor(arr.transpose(3, 0, 1, 2))
        if t.shape[2] != I3D_SIZE or t.shape[3] != I3D_SIZE:
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0),
                size=(I3D_FRAMES, I3D_SIZE, I3D_SIZE),
                mode="trilinear", align_corners=False
            ).squeeze(0)
        if random.random() < 0.5:
            t = torch.flip(t, dims=[3])
        t = (t + (torch.rand(1) - 0.5) * 0.2).clamp(-2.5, 2.5)
        return t, label


# ── Cache builder ─────────────────────────────────────────────────────────────
def build_cache(signs):
    entries    = []
    sign_count = defaultdict(int)
    sign_real  = defaultdict(int)

    if args.no_cache and CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(exist_ok=True)

    print("  Building dataset source indexes…")

    for sign in signs:
        if sign not in ASL_DICT:
            continue
        real_clips = []

        # ── Source 1: MS-ASL downloaded clips ─────────────────────────────────
        if MSASL_CLIPS_DIR.exists():
            sign_dir = MSASL_CLIPS_DIR / sign
            if sign_dir.exists():
                for mp4 in sorted(sign_dir.glob("*.mp4")):
                    frames = extract_raw_frames(mp4)
                    if frames:
                        real_clips.append(frames)
                        print(f"    ✓ MS-ASL {mp4.name} ({len(frames)} fr)")

        # ── Source 2: asl_clips/ webcam recordings ────────────────────────────
        if ASL_CLIPS_DIR.exists():
            stem = Path(ASL_DICT[sign]).stem
            candidates = []
            exact = ASL_CLIPS_DIR / ASL_DICT[sign]
            if exact.exists():
                candidates.append(exact)
            candidates += sorted(ASL_CLIPS_DIR.glob(f"{stem}_*.mp4"))
            for src in candidates:
                frames = extract_raw_frames(src)
                if frames:
                    real_clips.append(frames)
                    print(f"    ✓ local {src.name} ({len(frames)} fr)")

        # ── Source 3: feedback clips ───────────────────────────────────────────
        if FEEDBACK_CLIPS_NPZ.exists():
            d = np.load(FEEDBACK_CLIPS_NPZ, allow_pickle=True)
            for clip_frames, clip_label in zip(d["X"], d["y"]):
                if str(clip_label) == sign and isinstance(clip_frames, (list, np.ndarray)):
                    frames = list(clip_frames)
                    if len(frames) >= MIN_FRAMES:
                        real_clips.append(frames)
                        print(f"    ✓ feedback clip ({len(frames)} fr)")

        if not real_clips:
            print(f"    ○  {sign}: no clips found — record via Dev Panel")
            continue

        sign_real[sign] = len(real_clips)

        for frames in real_clips:
            out = CACHE_DIR / f"{sign}_{sign_count[sign]:04d}.npz"
            if not out.exists():
                frames_to_npz(frames, out)
            entries.append((out, sign))
            sign_count[sign] += 1

        if not args.no_augment and sign_count[sign] < MIN_CLIPS:
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
            print(f"    + {needed} augmented (was {sign_real[sign]} real, "
                  f"padded to {sign_count[sign]} total)")
        else:
            print(f"    → {sign_count[sign]} clips total "
                  f"({sign_real[sign]} real)")

    return entries, dict(sign_count), dict(sign_real)


# ── Training ──────────────────────────────────────────────────────────────────
def train_i3d(entries, labels, epochs):
    lbl2idx = {l: i for i, l in enumerate(labels)}
    indexed = [(path, lbl2idx[lbl]) for path, lbl in entries if lbl in lbl2idx]

    ds = DiskClipDataset(indexed)
    dl = DataLoader(ds, batch_size=4, shuffle=True,
                    num_workers=0, pin_memory=True)

    net     = _I3DNet(len(labels)).to(DEVICE)
    load_kinetics_weights(net)
    opt     = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    sch     = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"\n  Training I3D on {DEVICE} — {len(ds)} clips, "
          f"{len(labels)} signs, {epochs} epochs")
    print(f"  Labels: {labels}\n")

    net.train()
    for ep in range(1, epochs + 1):
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

        if ep % max(1, epochs // 20) == 0:
            pct = int(ep / epochs * 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct:3d}%  epoch {ep}  "
                  f"loss={ep_loss/max(1,len(dl)):.3f}",
                  end="", flush=True)

    net.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in DataLoader(ds, batch_size=8):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            correct += (net(xb).argmax(1) == yb).sum().item()
            total   += len(yb)
    acc = correct / max(1, total)
    print(f"\n  ✅ Accuracy: {acc*100:.1f}%")

    os.makedirs(os.path.dirname(I3D_ASL) or ".", exist_ok=True)
    torch.save({
        "labels":     labels,
        "state_dict": net.state_dict(),
        "arch":       "i3d",
    }, I3D_ASL)
    print(f"  ✅ Saved → {I3D_ASL}")
    return acc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    signs = list(args.signs) if args.signs else list(ASL_DICT.keys())

    print("=" * 60)
    print("  SignFuture — ASL I3D Trainer  (disk-backed, low RAM)")
    print(f"  Signs      : {len(signs)}")
    print(f"  Device     : {DEVICE}")
    vram = torch.cuda.get_device_properties(0).total_memory//1024**3 \
           if DEVICE.type == "cuda" else 0
    print(f"  VRAM       : {vram} GB" if DEVICE.type == "cuda"
          else "  VRAM       : CPU mode")
    print(f"  MS-ASL     : {'✓' if MSASL_CLIPS_DIR.exists() else '✗ not found'}")
    asl_n = len(list(ASL_CLIPS_DIR.glob("*.mp4"))) \
            if ASL_CLIPS_DIR.exists() else 0
    print(f"  asl_clips  : {asl_n} mp4 files")
    cache_n = len(list(CACHE_DIR.glob("*.npz"))) \
              if CACHE_DIR.exists() else 0
    print(f"  clip_cache : {cache_n} npz files")
    print(f"  Min clips  : {MIN_CLIPS}")
    print("=" * 60)

    print(f"\n  Building clip cache → {CACHE_DIR}")
    entries, sign_count, sign_real = build_cache(signs)

    covered = [s for s in signs if sign_count.get(s, 0) > 0]
    missing = [s for s in signs if sign_count.get(s, 0) == 0]

    print(f"\n  ── Summary ──────────────────────────────────────────")
    for s in covered:
        real = sign_real.get(s, sign_count[s])
        aug  = sign_count[s] - real
        aug_str = f"  (+{aug} aug)" if aug > 0 else ""
        print(f"  ✅  {s:<14} {real:>4} real{aug_str}  →  {sign_count[s]} total clips")
    for s in missing:
        print(f"  ○   {s:<14}   0  (record via Dev Panel or download MS-ASL)")
    print(f"\n  Total: {len(entries)} clips across {len(covered)} signs")

    if len(covered) < 2:
        print(f"\n  ❌ Need at least 2 signs with video to train.")
        return

    labels = sorted(covered)
    train_i3d(entries, labels, args.epochs_i3d)

    print(f"\n{'='*60}")
    print(f"  ✅ Done!  Start server: bash run.sh")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
