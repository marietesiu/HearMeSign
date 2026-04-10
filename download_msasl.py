#!/usr/bin/env python3
"""download_msasl.py — Download MS-ASL dataset clips and extract MediaPipe landmarks.

Downloads all 25,513 clips (train + val + test) from YouTube via yt-dlp,
crops each clip to the bounding box of the signer, runs MediaPipe to extract
hand + pose landmarks, mean-pools to a 278-dim vector, and appends to
training_data/samples_asl.npz.

Fully resumable — already-processed clips are skipped on re-run.
Failed downloads are logged to training_data/msasl_failed.txt.

Run:
    python download_msasl.py                        # process all 25,513 clips
    python download_msasl.py --limit 100            # test run with 100 clips
    python download_msasl.py --workers 4            # parallel downloads (default 2)
    python download_msasl.py --train-only           # skip val + test
    python download_msasl.py --status               # show progress and exit
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from queue import Queue, Empty

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Download MS-ASL and extract landmarks")
parser.add_argument("--msasl-dir",   default=str(Path.home() / "Downloads" / "MS-ASL"),
                    help="Path to MS-ASL json files (default: ~/Downloads/MS-ASL)")
parser.add_argument("--limit",       type=int, default=0,
                    help="Max clips to process (0 = all)")
parser.add_argument("--workers",     type=int, default=2,
                    help="Parallel download workers (default: 2)")
parser.add_argument("--train-only",  action="store_true",
                    help="Only process train split")
parser.add_argument("--status",      action="store_true",
                    help="Show current progress and exit")
parser.add_argument("--reset-failed",action="store_true",
                    help="Retry previously failed clips")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
MSASL_DIR    = Path(args.msasl_dir)
DATA_DIR     = HERE / "training_data"
OUT_NPZ      = DATA_DIR / "samples_asl.npz"
CLIPS_DIR    = Path.home() / "Downloads" / "MS-ASL" / "clips"  # raw clips by sign
DONE_FILE    = DATA_DIR / "msasl_done.txt"
FAILED_FILE  = DATA_DIR / "msasl_failed.txt"
PROGRESS_FILE= DATA_DIR / "msasl_progress.json"
DATA_DIR.mkdir(exist_ok=True)
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

# MS-ASL label index → ASL_DICT sign name
# Only download clips for signs in ASL_DICT
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

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    print("❌ numpy not installed.")
    sys.exit(1)

import numpy as np  # noqa: F811

try:
    import cv2
    import mp_holistic as mph
    from landmarks import extract_landmarks, normalize_landmarks
    _VISION_OK = True
except ImportError as e:
    print(f"❌ Vision deps missing: {e}")
    sys.exit(1)

# ── Load dataset ──────────────────────────────────────────────────────────────
def load_samples() -> list:
    splits = ["MSASL_train.json"]
    if not args.train_only:
        splits += ["MSASL_val.json", "MSASL_test.json"]

    all_samples = []
    for fname in splits:
        fpath = MSASL_DIR / fname
        if not fpath.exists():
            print(f"⚠ {fpath} not found — skipping")
            continue
        with open(fpath) as f:
            data = json.load(f)
        all_samples.extend(data)
        print(f"  Loaded {len(data):>6} samples from {fname}")

    return all_samples


def clip_id(sample: dict) -> str:
    """Stable unique ID for a clip — hash of url + start + end."""
    key = f"{sample['url']}|{sample['start_time']:.3f}|{sample['end_time']:.3f}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def load_done() -> set:
    if not DONE_FILE.exists():
        return set()
    with open(DONE_FILE) as f:
        return set(line.strip() for line in f if line.strip())


def load_failed() -> set:
    if not FAILED_FILE.exists():
        return set()
    with open(FAILED_FILE) as f:
        return set(line.strip() for line in f if line.strip())


def mark_done(cid: str):
    with open(DONE_FILE, "a") as f:
        f.write(cid + "\n")


def mark_failed(cid: str):
    with open(FAILED_FILE, "a") as f:
        f.write(cid + "\n")


# ── Download + crop ───────────────────────────────────────────────────────────
def download_clip(sample: dict, dest: Path, timeout: int = 90) -> bool:
    """Download a YouTube clip segment via yt-dlp."""
    import subprocess
    url        = sample["url"]
    start_s    = sample["start_time"]
    end_s      = sample["end_time"]
    duration   = max(0.5, end_s - start_s)

    cmd = [
        "yt-dlp", "-q", "--no-warnings",
        "-f", "bestvideo[ext=mp4][height<=360]+bestaudio/best[height<=360]/best",
        "--merge-output-format", "mp4",
        "--download-sections", f"*{start_s:.3f}-{end_s+0.5:.3f}",
        "--force-keyframes-at-cuts",
        "-o", str(dest),
        url,
    ]
    try:
        r = subprocess.run(cmd, timeout=timeout, capture_output=True)
        return r.returncode == 0 and dest.exists() and dest.stat().st_size > 512
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def extract_clip_landmarks(video_path: Path, box: list) -> "np.ndarray | None":
    """
    Open video, crop to bounding box [y0, x0, y1, x1] (normalised),
    run MediaPipe on each frame, mean-pool to 278-dim vector.
    Returns None if no hand frames detected.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # box = [y0, x0, y1, x1] normalised
    y0 = max(0, int(box[0] * h))
    x0 = max(0, int(box[1] * w))
    y1 = min(h, int(box[2] * h))
    x1 = min(w, int(box[3] * w))

    # Fallback to full frame if box is degenerate
    if x1 - x0 < 10 or y1 - y0 < 10:
        x0, y0, x1, y1 = 0, 0, w, h

    frame_vecs = []
    try:
        with mph.Holistic(min_detection_confidence=0.4,
                          min_tracking_confidence=0.4) as hol:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cropped = frame[y0:y1, x0:x1]
                if cropped.size == 0:
                    continue
                rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                lm  = normalize_landmarks(extract_landmarks(hol.process(rgb)))
                if not np.all(lm[:126] == 0):   # at least one hand visible
                    frame_vecs.append(lm)
    except Exception:
        pass
    finally:
        cap.release()

    if not frame_vecs:
        return None
    return np.array(frame_vecs, dtype=np.float32).mean(axis=0)


# ── NPZ writer (thread-safe) ──────────────────────────────────────────────────
_npz_lock = threading.Lock()

def append_to_npz(vectors: list, labels: list):
    """Append a batch of (vector, label) pairs to samples_asl.npz."""
    if not vectors:
        return
    X_new = np.array(vectors, dtype=np.float32)
    y_new = np.array(labels)
    with _npz_lock:
        if OUT_NPZ.exists():
            d     = np.load(OUT_NPZ, allow_pickle=True)
            X_all = np.concatenate([d["X"], X_new])
            y_all = np.concatenate([d["y"], y_new])
        else:
            X_all, y_all = X_new, y_new
        np.savez(OUT_NPZ, X=X_all, y=y_all)


# ── Worker ────────────────────────────────────────────────────────────────────
def process_sample(sample: dict, tmpdir: str) -> "tuple[np.ndarray, str] | None":
    """Download, extract landmarks, return (vector, label) or None on failure.
    Also saves raw clip to CLIPS_DIR/<sign>/ for I3D training if sign is in ASL_DICT."""
    cid   = clip_id(sample)
    dest  = Path(tmpdir) / f"{cid}.mp4"
    label = sample.get("label", -1)
    sign  = MSASL_LABEL_TO_SIGN.get(label)

    try:
        if not download_clip(sample, dest):
            return None
        vec = extract_clip_landmarks(dest, sample["box"])
        if vec is None:
            return None

        # Save raw clip for I3D training if this sign is in ASL_DICT
        if sign:
            sign_dir = CLIPS_DIR / sign
            sign_dir.mkdir(exist_ok=True)
            clip_dest = sign_dir / f"{cid}.mp4"
            if not clip_dest.exists():
                import shutil
                shutil.copy2(dest, clip_dest)

        text = sign if sign else sample["text"].lower()
        return (vec, text)
    finally:
        if dest.exists():
            dest.unlink()


def worker(q: Queue, results: list, counters: dict, lock: threading.Lock):
    with tempfile.TemporaryDirectory() as tmpdir:
        while True:
            try:
                sample = q.get(timeout=5)
            except Empty:
                break

            cid    = clip_id(sample)
            result = process_sample(sample, tmpdir)

            with lock:
                counters["processed"] += 1
                if result is not None:
                    results.append(result)
                    counters["success"] += 1
                    mark_done(cid)
                else:
                    counters["failed"] += 1
                    mark_failed(cid)

                # Flush to npz every 50 successful extractions
                if len(results) >= 50:
                    vecs   = [r[0] for r in results]
                    labels = [r[1] for r in results]
                    append_to_npz(vecs, labels)
                    results.clear()

            q.task_done()


# ── Status ────────────────────────────────────────────────────────────────────
def show_status():
    done   = load_done()
    failed = load_failed()
    print(f"\n  MS-ASL download status:")
    print(f"  Completed : {len(done)}")
    print(f"  Failed    : {len(failed)}")

    if OUT_NPZ.exists():
        d      = np.load(OUT_NPZ, allow_pickle=True)
        from collections import Counter
        counts = Counter(d["y"].tolist())
        print(f"  NPZ total : {len(d['X'])} samples, {len(counts)} signs")
        print(f"  Top 10    : " + ", ".join(
            f"{s}({c})" for s, c in counts.most_common(10)))
    else:
        print(f"  NPZ total : 0 (not started)")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  MS-ASL Landmark Extractor")
    print("=" * 60)

    if args.status:
        show_status()
        return

    # Load samples
    print("\n  Loading MS-ASL json files…")
    samples = load_samples()
    if not samples:
        print("  ❌ No samples found. Check --msasl-dir path.")
        return

    # Filter already done and failed
    done   = load_done()
    failed = load_failed() if not args.reset_failed else set()
    skip   = done | failed

    pending = [s for s in samples if clip_id(s) not in skip]

    if args.limit:
        pending = pending[:args.limit]

    print(f"  Total samples : {len(samples)}")
    print(f"  Already done  : {len(done)}")
    print(f"  Failed        : {len(failed)}")
    print(f"  To process    : {len(pending)}")

    if not pending:
        print("\n  ✅ All clips already processed!")
        show_status()
        return

    print(f"  Workers       : {args.workers}")
    print(f"  Output        : {OUT_NPZ}")
    print()
    print("  Starting… (Ctrl+C to pause — progress is saved)")
    print()

    # Fill queue
    q = Queue()
    for s in pending:
        q.put(s)

    counters = {"processed": 0, "success": 0, "failed": 0}
    results  = []
    lock     = threading.Lock()
    start_t  = time.time()

    threads = []
    for _ in range(args.workers):
        t = threading.Thread(target=worker,
                             args=(q, results, counters, lock),
                             daemon=True)
        t.start()
        threads.append(t)

    # Progress display
    total = len(pending)
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(5)
            with lock:
                p   = counters["processed"]
                s   = counters["success"]
                f   = counters["failed"]
            elapsed = time.time() - start_t
            rate    = p / max(1, elapsed)
            eta_s   = (total - p) / max(0.01, rate)
            eta_m   = int(eta_s // 60)
            pct     = int(p / total * 100)
            bar     = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct:3d}%  {p}/{total}  "
                  f"✓{s} ✗{f}  "
                  f"{rate*60:.0f}/min  ETA {eta_m}min",
                  end="", flush=True)
    except KeyboardInterrupt:
        print("\n\n  ⏸ Paused — progress saved. Re-run to continue.")

    # Flush remaining results
    with lock:
        if results:
            append_to_npz([r[0] for r in results], [r[1] for r in results])
            results.clear()

    print()
    show_status()

    with lock:
        print(f"  ✅ Session done: {counters['success']} extracted, "
              f"{counters['failed']} failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
