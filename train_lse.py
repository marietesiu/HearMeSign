#!/usr/bin/env python3
"""train_lse.py — LSE-only I3D training. Disk-backed to avoid OOM.

Reads clips from ALL available sources:
  1. SWL-LSE VIDEOS_RENAMED  (lab-quality reference videos, 1/sign)
  2. Sign4all                (7 756 RGB videos, 24 daily-activity signs)
  3. lse_clips/              (your own webcam recordings via Dev Panel)

  Real clips are NEVER capped — every clip found is trained on.
  Augmentation only pads signs that are BELOW MIN_CLIPS_PER_SIGN.

Instead of loading all clips into RAM, this version:
  1. Saves each clip as a small .npz on disk (training_data/clip_cache/)
  2. Trains using a DataLoader that reads one batch at a time
  3. Peak RAM usage: ~batch_size clips at a time (~200 MB max)

Run:
    python train_lse.py
    python train_lse.py --epochs-i3d 120
    python train_lse.py --signs dolor garganta mareo
    python train_lse.py --no-cache   (re-extract clips even if cache exists)
    python train_lse.py --no-augment (skip augmentation entirely)
    python train_lse.py --i3d-only
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train LSE I3D model")
parser.add_argument("--i3d-only",       action="store_true")
parser.add_argument("--mlp-only",       action="store_true")
parser.add_argument("--no-augment",     action="store_true")
parser.add_argument("--no-cache",       action="store_true",
                    help="Delete clip cache and re-extract from scratch")
parser.add_argument("--signs",          nargs="+", metavar="SIGN")
parser.add_argument("--epochs-i3d",     type=int, default=60)
parser.add_argument("--min-clips",      type=int, default=20,
                    help="Minimum clips per sign — augment up to this if below (default 20). "
                         "Signs with MORE real clips are never capped.")
args = parser.parse_args()

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    print("❌ PyTorch not installed. Run setup.sh first.")
    sys.exit(1)

import torch                                    # noqa: F811
import torch.nn as nn                           # noqa: F811
import torch.optim as optim                     # noqa: F811
from torch.utils.data import DataLoader, Dataset  # noqa: F811

import random
import numpy as np
import cv2
from collections import defaultdict

from asl_dictionary import LSE_DICT
from sign_model import (
    _I3DNet,
    load_kinetics_weights,
    I3D_LSE,
    DEVICE,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
# SWL-LSE reference videos (after running rename_dataset.py)
SWL_VIDEOS_DIR = Path.home() / "Downloads" / "13691887" / "VIDEOS_RENAMED"

# Sign4all: folder containing uppercase sign subfolders (COMER/, BEBER/ etc.)
SIGN4ALL_DIR         = Path.home() / "Downloads" / "sign4all"

# Your own webcam recordings (mp4 files)
LSE_CLIPS_DIR  = HERE / "lse_clips"
DATA_DIR       = HERE / "training_data"
CACHE_DIR      = DATA_DIR / "clip_cache"   # .npz clips live here, not in RAM
DATA_DIR.mkdir(exist_ok=True)

# Feedback-confirmed clips stored by the server (training_data/clips_lse.npz)
FEEDBACK_CLIPS_NPZ = DATA_DIR / "clips_lse.npz"

# Loose recordings named after sign (e.g. ~/Downloads/LooseVids/fiebre.mp4)
LOOSE_VIDS_DIR = Path.home() / "Downloads" / "LooseVids" 

MIN_CLIPS = args.min_clips   # augment only if BELOW this — no upper cap
MIN_FRAMES = 8

# ── Dataset source mappings ───────────────────────────────────────────────────
SWL_LABEL_TO_SIGN = {
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

SIGN4ALL_TO_SIGN = {
    "COMER":    "comer",
    "BEBER":    "beber",
    "DORMIR":   "dormir",
    "CAMINAR":  "caminar",
    "SENTARSE": "sentar",
    "ABRIR":    "abrir",
    "MIRAR":    "mirar",
}



# ── Utilities ─────────────────────────────────────────────────────────────────
def _normalise_gloss(raw: str) -> str:
    import unicodedata
    s = "".join(
        c for c in unicodedata.normalize("NFD", raw.lower())
        if unicodedata.category(c) != "Mn"
    )
    return s.replace("-", "_").replace(" ", "_")


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
    """Return one augmented variant of a clip."""
    seq = list(frames)
    factor = random.uniform(0.72, 1.30)
    ni = max(MIN_FRAMES, int(len(seq) * factor))
    seq = [seq[i] for i in np.linspace(0, len(seq)-1, ni).astype(int)]
    if random.random() < 0.50:
        delta = random.uniform(-0.20, 0.20)
        seq = [(np.clip(f.astype(np.float32)/255.0 + delta, 0, 1)*255
                ).astype(np.uint8) for f in seq]
    if random.random() < 0.40:
        seq = [f[:, ::-1, :].copy() for f in seq]
    return seq


def frames_to_small_npz(frames, path):
    """Convert frames → (16, 112, 112, 3) uint8 and save as npz. ~570 KB per clip."""
    from sign_model import I3D_FRAMES, I3D_SIZE
    n   = len(frames)
    idx = np.linspace(0, n-1, I3D_FRAMES).astype(int)
    sel = np.stack([cv2.resize(
                cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB),
                (I3D_SIZE, I3D_SIZE))
              for i in idx], axis=0)   # (16, 112, 112, 3) uint8
    np.savez_compressed(path, frames=sel)



# ── Disk-backed Dataset ───────────────────────────────────────────────────────
class DiskClipDataset(Dataset):
    """Reads pre-processed (16,112,112,3) clips from disk one at a time."""

    def __init__(self, entries):
        # entries: list of (npz_path, label_int)
        # Validate all paths exist at construction time
        self.entries = [(p, lbl) for p, lbl in entries if Path(p).exists()]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label = self.entries[idx]
        arr = np.load(path)["frames"]          # (16,112,112,3) uint8
        arr = arr.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr  = (arr - mean) / std              # (16,112,112,3)
        arr  = arr.transpose(3, 0, 1, 2)       # (3,16,112,112)
        t    = torch.tensor(arr)
        # Runtime augmentation
        if random.random() < 0.5:
            t = torch.flip(t, dims=[3])
        t = (t + (torch.rand(1) - 0.5) * 0.2).clamp(-2.5, 2.5)
        return t, label


# ── Cache builder ─────────────────────────────────────────────────────────────
def build_cache(signs):
    """
    For each sign: collect frames from ALL sources (SWL + Sign4all + lse_clips),
    save each clip as a tiny (16,112,112,3) npz file in CACHE_DIR.

    Real clips from any source are NEVER capped.
    Augmentation only pads signs below MIN_CLIPS.

    Returns list of (npz_path, sign_label).
    """
    if args.no_cache and CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print("  🗑  Cleared old cache")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Build source indexes once
    print("  Building dataset source indexes…")

    sign4all_index = defaultdict(list)
    if SIGN4ALL_DIR.exists():
        for folder in sorted(SIGN4ALL_DIR.iterdir()):
            if not folder.is_dir():
                continue
            upper = folder.name.upper()
            norm  = _normalise_gloss(folder.name)
            sign  = SIGN4ALL_TO_SIGN.get(upper) or SIGN4ALL_TO_SIGN.get(norm)
            if not sign and norm in LSE_DICT:
                sign = norm
            if sign:
                for mp4 in sorted(folder.glob("*.mp4")):
                    sign4all_index[sign].append(mp4)
        if sign4all_index:
            print(f"    Sign4all: {len(sign4all_index)} signs, "
                  f"{sum(len(v) for v in sign4all_index.values())} videos")
    else:
        print(f"    Sign4all: ✗ {SIGN4ALL_DIR} not found — skipping")

    entries    = []          # (npz_path, sign)
    sign_count = defaultdict(int)   # total clips (real + aug) per sign
    sign_real  = defaultdict(int)   # real clips per sign (for augmentation decision)

    for sign in signs:
        if sign not in LSE_DICT:
            continue

        real_clips = []   # raw frame lists for this sign

        # ── Source 1: SWL VIDEOS_RENAMED ──────────────────────────────────────
        if SWL_VIDEOS_DIR.exists():
            for mp4 in sorted(SWL_VIDEOS_DIR.glob(f"{sign}_*.mp4")):
                frames = extract_raw_frames(mp4)
                if frames:
                    real_clips.append(frames)
                    print(f"    ✓ SWL   {mp4.name} ({len(frames)} fr)")

        # ── Source 2: Sign4all ─────────────────────────────────────────────────
        for mp4 in sign4all_index.get(sign, []):
            frames = extract_raw_frames(mp4)
            if frames:
                real_clips.append(frames)
                print(f"    ✓ Sign4all {mp4.name} ({len(frames)} fr)")

        # ── Source 3: lse_clips/ own recordings ───────────────────────────────
        if LSE_CLIPS_DIR.exists():
            stem = Path(LSE_DICT[sign]).stem
            candidates = []
            exact = LSE_CLIPS_DIR / LSE_DICT[sign]
            if exact.exists():
                candidates.append(exact)
            candidates += sorted(LSE_CLIPS_DIR.glob(f"{stem}_*.mp4"))
            for src in candidates:
                frames = extract_raw_frames(src)
                if frames:
                    real_clips.append(frames)
                    print(f"    ✓ local {src.name} ({len(frames)} fr)")

        # ── Source 4: feedback-confirmed clips (training_data/clips_lse.npz) ──
        if FEEDBACK_CLIPS_NPZ.exists():
            d = np.load(FEEDBACK_CLIPS_NPZ, allow_pickle=True)
            for clip_frames, clip_label in zip(d["X"], d["y"]):
                if str(clip_label) == sign and isinstance(clip_frames, (list, np.ndarray)):
                    frames = list(clip_frames)
                    if len(frames) >= MIN_FRAMES:
                        real_clips.append(frames)
                        print(f"    ✓ feedback clip ({len(frames)} fr)")

        # ── Source 5: loose recordings ~/Downloads/LooseVids/<sign>.mp4 ───────
        if LOOSE_VIDS_DIR.exists():
            loose = LOOSE_VIDS_DIR / f"{sign}.mp4"
            if loose.exists():
                frames = extract_raw_frames(loose)
                if frames:
                    real_clips.append(frames)
                    print(f"    ✓ loose {loose.name} ({len(frames)} fr)")

        # ── Source 6: UVigo pre-extracted npz clips ────────────────────────────
        # These are placed directly in CACHE_DIR by extract_uvigo.py
        # Named: uvigo_<sign>_<file_id>_<start_ms>.npz
        for npz in sorted(CACHE_DIR.glob(f"uvigo_{sign}_*.npz")):
            entries.append((npz, sign))
            sign_count[sign] += 1
            sign_real[sign]   = sign_real.get(sign, 0) + 1
            # Don't double-count in real_clips — UVigo npz already saved

        if not real_clips:
            print(f"    ○  {sign}: no clips found — record via Dev Panel")
            continue

        sign_real[sign] = len(real_clips)

        # ── Save real clips to cache ───────────────────────────────────────────
        for frames in real_clips:
            out = CACHE_DIR / f"{sign}_{sign_count[sign]:04d}.npz"
            if not out.exists():
                frames_to_small_npz(frames, out)
            entries.append((out, sign))
            sign_count[sign] += 1

        # ── Augment only if below MIN_CLIPS ────────────────────────────────────
        if not args.no_augment and sign_count[sign] < MIN_CLIPS:
            needed    = MIN_CLIPS - sign_count[sign]
            aug_count = 0
            cycle     = 0
            while aug_count < needed:
                base = real_clips[cycle % len(real_clips)]
                aug  = augment_clip(base)
                out  = CACHE_DIR / f"{sign}_{sign_count[sign]:04d}.npz"
                if not out.exists():
                    frames_to_small_npz(aug, out)
                entries.append((out, sign))
                sign_count[sign] += 1
                aug_count        += 1
                cycle            += 1
            print(f"    + {aug_count} augmented (was {sign_real[sign]} real clips, "
                  f"padded to {sign_count[sign]} total)")
        else:
            print(f"    → {sign_count[sign]} clips total "
                  f"({sign_real[sign]} real, no augment needed)")

    return entries, dict(sign_count), dict(sign_real)


# ── Training ──────────────────────────────────────────────────────────────────
def train_i3d(entries, labels, epochs):
    """Train I3D directly from disk cache. Peak RAM = batch_size clips at a time."""
    lbl2idx = {l: i for i, l in enumerate(labels)}
    indexed = [(path, lbl2idx[lbl]) for path, lbl in entries if lbl in lbl2idx]

    ds = DiskClipDataset(indexed)
    dl = DataLoader(ds, batch_size=4, shuffle=True,
                    num_workers=0, pin_memory=True)

    net = _I3DNet(len(labels)).to(DEVICE)
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
            # Show epoch count only — no "/N" fraction
            print(f"\r  [{bar}] {pct:3d}%  epoch {ep}  "
                  f"loss={ep_loss/max(1,len(dl)):.3f}",
                  end="", flush=True)

    # Final accuracy
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in DataLoader(ds, batch_size=8):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            correct += (net(xb).argmax(1) == yb).sum().item()
            total   += len(yb)
    acc = correct / max(1, total)
    print(f"\n  ✅ Accuracy: {acc*100:.1f}%")

    # Save
    os.makedirs(os.path.dirname(I3D_LSE) or ".", exist_ok=True)
    torch.save({
        "labels":     labels,
        "state_dict": net.state_dict(),
        "arch":       "i3d",
    }, I3D_LSE)
    print(f"  ✅ Saved → {I3D_LSE}")
    return acc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    signs = list(args.signs) if args.signs else list(LSE_DICT.keys())

    print("=" * 60)
    print("  SignFuture — LSE I3D Trainer  (disk-backed, low RAM)")
    print(f"  Signs      : {len(signs)}")
    print(f"  Device     : {DEVICE}")
    vram = torch.cuda.get_device_properties(0).total_memory//1024**3 \
           if DEVICE.type == "cuda" else 0
    print(f"  VRAM       : {vram} GB" if DEVICE.type == "cuda" else "  VRAM       : CPU mode")
    print(f"  SWL video  : {'✓' if SWL_VIDEOS_DIR.exists() else '✗ not found'}")
    print(f"  Sign4all   : {'✓' if SIGN4ALL_DIR.exists() else '✗ not found'}")
    print(f"  LooseVids  : {'✓' if LOOSE_VIDS_DIR.exists() else '✗ not found'}")
    lse_n = len(list(LSE_CLIPS_DIR.glob("*.mp4"))) \
            if LSE_CLIPS_DIR.exists() else 0
    print(f"  lse_clips  : {lse_n} mp4 files")
    cache_n = len(list(CACHE_DIR.glob("*.npz"))) \
              if CACHE_DIR.exists() else 0
    print(f"  clip_cache : {cache_n} npz files")
    print(f"  Min clips  : {MIN_CLIPS} (augment only below this — no upper cap)")
    print("=" * 60)

    # ── Build / reuse disk cache ───────────────────────────────────────────────
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
        print(f"  ○   {s:<14}   0  (record via Dev Panel)")
    print(f"\n  Total: {len(entries)} clips across {len(covered)} signs")

    if len(covered) < 2:
        print(f"\n  ❌ Need at least 2 signs with video to train.")
        print(f"     Copy {len(missing)} more signs to lse_clips/ or VIDEOS_RENAMED/")
        return

    # ── Train ─────────────────────────────────────────────────────────────────
    labels = sorted(covered)
    train_i3d(entries, labels, args.epochs_i3d)

    print(f"\n{'='*60}")
    print(f"  ✅ Done!  Start server: bash run.sh")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
