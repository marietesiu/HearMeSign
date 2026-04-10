#!/usr/bin/env python3
"""extract_uvigo.py — Extract LSE sign clips from LSE-Health-UVigo dataset.

The UVigo dataset contains 272 health-domain LSE videos with ELAN annotations.
This script reads the Excel mapping file to find which video file contains
each sign and at what timestamp, cuts the clip, extracts MediaPipe landmarks,
and saves to training_data/samples_lse.npz.

It also saves raw clips to training_data/clip_cache/ for I3D training.

Dataset location:
  /home/mario/Downloads/10234465/
    Videos-LSE-Health-UVigo/  ← .mp4 files (filename = YouTube ID)
    LSE-Health-UVigo.xlsx     ← gloss annotations with timestamps

Signs found in dataset (matched to LSE_DICT):
  comer    431 clips
  sangre   385 clips
  respirar  77 clips
  dormir    44 clips
  mareo      8 clips
  Total:   945 clips

Run:
    python extract_uvigo.py                    # extract all
    python extract_uvigo.py --signs comer respirar
    python extract_uvigo.py --i3d-only         # raw clips only, skip landmarks
    python extract_uvigo.py --mlp-only         # landmarks only, skip raw clips
    python extract_uvigo.py --status           # show progress
"""

import argparse
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

parser = argparse.ArgumentParser(description="Extract UVigo LSE clips and landmarks")
parser.add_argument("--uvigo-dir",  default="/home/mario/Downloads/10234465",
                    help="Path to UVigo dataset folder")
parser.add_argument("--xlsx",       default=None,
                    help="Path to LSE-Health-UVigo.xlsx (default: uvigo-dir/LSE-Health-UVigo.xlsx)")
parser.add_argument("--signs",      nargs="+", metavar="SIGN")
parser.add_argument("--i3d-only",   action="store_true")
parser.add_argument("--mlp-only",   action="store_true")
parser.add_argument("--status",     action="store_true")
parser.add_argument("--no-cache",   action="store_true",
                    help="Re-extract even if already done")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
UVIGO_DIR   = Path(args.uvigo_dir)
VIDEOS_DIR  = UVIGO_DIR / "Videos-LSE-Health-UVigo"
XLSX_PATH   = Path(args.xlsx) if args.xlsx else UVIGO_DIR / "LSE-Health-UVigo.xlsx"
DATA_DIR    = HERE / "training_data"
CACHE_DIR   = DATA_DIR / "clip_cache"
DONE_FILE   = DATA_DIR / "uvigo_done.txt"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ── Gloss → LSE_DICT sign mapping ────────────────────────────────────────────
GLOSS_MAP = {
    "comer":              "comer",
    "dormir":             "dormir",
    "respirar":           "respirar",
    "mareo":              "mareo",
    "sangre2":            "sangre",
    "analisis-de-sangre": "sangre",
    "beber":              "beber",
    "caminar":            "caminar",
    "sentarse":           "sentar",
    "sentar":             "sentar",
    "garganta":           "garganta",
    "tos":                "tos",
    "fiebre":             "fiebre",
    "dolor":              "dolor",
    "moco":               "moco",
    "voz":                "voz",
    "cabeza":             "cabeza",
    "nariz":              "nariz",
    "oreja":              "oreja",
    "boca":               "boca",
    "cuello":             "cuello",
    "lengua":             "lengua",
    "dientes":            "dientes",
    "zumbido":            "zumbido",
    "infeccion":          "infeccion",
    "hinchazon":          "hinchazon",
    "tragar":             "tragar",
    "mirar":              "mirar",
    "girar":              "girar",
}


def norm_gloss(g: str) -> str:
    """Strip markers, parentheses, accents, lowercase."""
    g = g.strip().lstrip("*")
    g = re.sub(r"\(.*?\)", "", g).strip().lower()
    g = "".join(c for c in unicodedata.normalize("NFD", g)
                if unicodedata.category(c) != "Mn")
    return g


# ── Load Excel ────────────────────────────────────────────────────────────────
def load_annotations() -> list:
    """Returns list of (file_id, start_ms, end_ms, sign) tuples."""
    try:
        import openpyxl
    except ImportError:
        print("❌ openpyxl not installed. Run: pip install openpyxl")
        sys.exit(1)

    if not XLSX_PATH.exists():
        print(f"❌ Excel file not found: {XLSX_PATH}")
        sys.exit(1)

    wb   = openpyxl.load_workbook(str(XLSX_PATH))
    ws   = wb["GlossesContent"]
    rows = list(ws.iter_rows(values_only=True))[1:]   # skip header

    entries = []
    for file_id, start_ms, end_ms, gloss in rows:
        if not all([file_id, start_ms, end_ms, gloss]):
            continue
        g    = norm_gloss(str(gloss))
        sign = GLOSS_MAP.get(g)
        if sign:
            entries.append((str(file_id), int(start_ms), int(end_ms), sign))

    return entries


# ── Load done set ─────────────────────────────────────────────────────────────
def load_done() -> set:
    if not DONE_FILE.exists():
        return set()
    with open(DONE_FILE) as f:
        return set(line.strip() for line in f if line.strip())


def mark_done(key: str):
    with open(DONE_FILE, "a") as f:
        f.write(key + "\n")


def entry_key(file_id, start_ms, end_ms, sign):
    return f"{file_id}_{start_ms}_{end_ms}_{sign}"


# ── Video clip extraction ─────────────────────────────────────────────────────
def extract_clip_frames(mp4_path: Path, start_ms: int, end_ms: int,
                        min_frames: int = 8) -> list:
    """Extract frames between start_ms and end_ms from an mp4. Returns list of BGR frames."""
    try:
        import cv2
    except ImportError:
        return []

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return []

    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_f = int(start_ms / 1000.0 * fps)
    end_f   = int(end_ms   / 1000.0 * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frames = []
    while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_f:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    return frames if len(frames) >= min_frames else []


def save_i3d_npz(frames: list, path: Path):
    """Save clip as (16,112,112,3) npz for I3D training."""
    import numpy as np
    import cv2
    from sign_model import I3D_FRAMES, I3D_SIZE

    n   = len(frames)
    idx = np.linspace(0, n-1, I3D_FRAMES).astype(int)
    sel = np.stack([cv2.resize(
                cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB),
                (I3D_SIZE, I3D_SIZE))
              for i in idx], axis=0)
    np.savez_compressed(str(path), frames=sel)


def extract_landmarks(frames: list) -> "np.ndarray | None":
    """Run MediaPipe on frame sequence, mean-pool to 278-dim vector."""
    try:
        import numpy as np
        import cv2
        import mp_holistic as mph
        from landmarks import extract_landmarks as ext_lm, normalize_landmarks as norm_lm
    except ImportError as e:
        print(f"  ⚠ landmark deps missing: {e}")
        return None

    frame_vecs = []
    try:
        with mph.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as h:
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                lm  = norm_lm(ext_lm(h.process(rgb)))
                if not (lm[:126] == 0).all():
                    frame_vecs.append(lm)
    except Exception as e:
        print(f"  ⚠ MediaPipe error: {e}")

    if not frame_vecs:
        return None

    import numpy as np
    return np.array(frame_vecs, dtype=np.float32).mean(axis=0)


def append_landmark(vec, sign: str):
    """Append a 278-dim vector + label to samples_lse.npz."""
    import numpy as np
    npz_path = DATA_DIR / "samples_lse.npz"
    X_new    = np.array([vec], dtype=np.float32)
    y_new    = np.array([sign])
    if npz_path.exists():
        d     = np.load(str(npz_path), allow_pickle=True)
        X_all = np.concatenate([d["X"], X_new])
        y_all = np.concatenate([d["y"], y_new])
    else:
        X_all, y_all = X_new, y_new
    np.savez(str(npz_path), X=X_all, y=y_all)


# ── Status ────────────────────────────────────────────────────────────────────
def show_status(entries):
    from collections import Counter
    import numpy as np

    done   = load_done()
    counts = Counter(sign for _, _, _, sign in entries)

    print(f"\n  UVigo extraction status:")
    print(f"  Total annotations : {len(entries)}")
    print(f"  Already done      : {len(done)}")
    print(f"  Remaining         : {len(entries) - len(done)}")
    print()
    for sign, n in sorted(counts.items()):
        done_n = sum(1 for e in entries
                     if e[3] == sign and entry_key(*e) in done)
        print(f"  {sign:<16} {done_n:>4}/{n} done")

    npz = DATA_DIR / "samples_lse.npz"
    if npz.exists():
        d      = np.load(str(npz), allow_pickle=True)
        lcounts = Counter(d["y"].tolist())
        print(f"\n  samples_lse.npz : {len(d['X'])} total")
        for s, c in sorted(lcounts.items()):
            print(f"    {s:<16} {c}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  LSE-Health-UVigo Extractor")
    print("=" * 60)

    if not VIDEOS_DIR.exists():
        print(f"❌ Videos folder not found: {VIDEOS_DIR}")
        sys.exit(1)

    print(f"\n  Loading annotations from {XLSX_PATH.name}…")
    entries = load_annotations()
    print(f"  Found {len(entries)} annotated sign clips across "
          f"{len(set(e[3] for e in entries))} signs")

    # Filter by requested signs
    if args.signs:
        entries = [e for e in entries if e[3] in args.signs]
        print(f"  Filtered to {len(entries)} clips for: {args.signs}")

    if args.status:
        show_status(entries)
        return

    done = load_done() if not args.no_cache else set()

    # Count available video files
    available = set(p.stem for p in VIDEOS_DIR.glob("*.mp4"))
    print(f"  Video files on disk: {len(available)}")

    total      = len(entries)
    n_done     = 0
    n_skipped  = 0
    n_no_video = 0
    n_failed   = 0
    n_i3d      = 0
    n_mlp      = 0

    from collections import Counter
    sign_counts = Counter()

    for i, (file_id, start_ms, end_ms, sign) in enumerate(entries):
        key = entry_key(file_id, start_ms, end_ms, sign)

        if key in done:
            n_skipped += 1
            continue

        mp4 = VIDEOS_DIR / f"{file_id}.mp4"
        if not mp4.exists():
            n_no_video += 1
            continue

        # Extract frames
        frames = extract_clip_frames(mp4, start_ms, end_ms)
        if not frames:
            mark_done(key)
            n_failed += 1
            continue

        success = False

        # Save raw clip for I3D
        if not args.mlp_only:
            npz_name = f"uvigo_{sign}_{file_id}_{start_ms}.npz"
            npz_path = CACHE_DIR / npz_name
            if not npz_path.exists():
                save_i3d_npz(frames, npz_path)
            n_i3d  += 1
            success = True

        # Extract landmarks for MLP
        if not args.i3d_only:
            vec = extract_landmarks(frames)
            if vec is not None:
                append_landmark(vec, sign)
                n_mlp  += 1
                success = True

        if success:
            sign_counts[sign] += 1
            mark_done(key)
            n_done += 1

        # Progress
        processed = i + 1
        pct = int(processed / total * 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%  {processed}/{total}  "
              f"✓{n_done} ✗{n_failed} skip{n_skipped} no-vid{n_no_video}",
              end="", flush=True)

    print()
    print(f"\n  ── Results ──────────────────────────────────────────")
    for sign, count in sorted(sign_counts.items()):
        print(f"  {sign:<16} {count} clips extracted")
    print(f"\n  I3D clips saved : {n_i3d}  → training_data/clip_cache/")
    print(f"  MLP samples     : {n_mlp} → training_data/samples_lse.npz")
    print(f"  No video file   : {n_no_video} (video not in dataset)")
    print(f"  Failed          : {n_failed}")
    print()
    print("  Next steps:")
    print("    python train_lse_mlp.py   (retrain MLP with new data)")
    print("    python train_lse.py       (retrain I3D with new data)")
    print("=" * 60)


if __name__ == "__main__":
    main()
