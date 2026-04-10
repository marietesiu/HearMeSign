"""dataset_download.py — Build LSE + ASL training datasets and train models.

╔══════════════════════════════════════════════════════════════════════════════╗
║  LSE DATASET SOURCES (priority order, best → fallback)                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. SWL-LSE  (Zenodo 13691887 / github.com/mvazquezgts/SWL-LSE)             ║
║     300 health signs. Two assets used:                                       ║
║       a) MEDIAPIPE.zip → pkl files (8,000 skeleton sequences, ~27/sign)      ║
║          → feeds MLP (SignClassifier) training                               ║
║       b) VIDEOS_REF.zip → 300 reference MP4s (1 per sign, lab quality)      ║
║          → feeds I3D training (augmented to TARGET_CLIPS)                    ║
║     Download:                                                                ║
║       https://zenodo.org/records/13691887                                    ║
║     Files needed:  VIDEOS_REF.zip, MEDIAPIPE.zip,                           ║
║                    videos_ref_annotations.csv                                ║
║     After download run rename_swllse.py to rename the reference videos.     ║
║                                                                              ║
║  2. LSE-Health-UVigo  (Zenodo 10234465)                                      ║
║     273 YouTube videos of health-domain LSE. 611 unique gloss labels        ║
║     (medical terms: CORTICOIDES, HEMOGLOBINA, TVP…).                        ║
║     This script extracts clip segments using the timestamp CSV.             ║
║     Download:                                                                ║
║       https://zenodo.org/records/10234465                                    ║
║     Files needed:  LSE-Health-UVIGO-timestamps.csv  (Excel export or        ║
║                    the CSV already in your project folder)                  ║
║     yt-dlp required.  Set UVIGO_TIMESTAMPS_CSV to your local CSV path.      ║
║                                                                              ║
║  3. Sign4all  (SciDB / Nature Scientific Data, Feb 2026)                    ║
║     7,756 RGB video recordings, 24 daily-activity LSE signs (catering       ║
║     context: AGUA, CAFE, COMER, BEBER…). High density ~323/sign.            ║
║     Download:                                                                ║
║       https://www.scidb.cn/en/detail?dataSetId=12775cc0026841979bdaba60484ad067
║     Files needed: extracted video folder (MP4s in subfolders by sign name)  ║
║     Set SIGN4ALL_DIR to your extracted folder.                               ║
║                                                                              ║
║  4. lse_clips/ folder  (your own webcam recordings via Dev Panel)            ║
║     Required for signs not covered by any dataset above.                    ║
║     Single take:    lse_clips/oreja.mp4                                     ║
║     Multiple takes: lse_clips/oreja_001.mp4, oreja_002.mp4 …               ║
║     Record via Dev Panel → Train tab → collect a sign.                      ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DATASETS NOT USED (and why)                                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  zenodo.org/record/4064940  — ReproSchema (software repo), NOT a sign       ║
║                               language dataset. Wrong/misidentified link.   ║
║  github.com/arasgungore/... — Senior project using LSE_eSaude_UVIGO.        ║
║                               No new data; underlying dataset is            ║
║                               Zenodo 10234465 (already included above).    ║
║  spreadthesign.com          — Proprietary licence, no public API or bulk    ║
║                               download endpoint. Cannot be used here.       ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ASL DATASET SOURCES                                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  1. WLASL (github.com/dxli94/WLASL) — yt-dlp YouTube downloads             ║
║  2. ASL Citizen (HuggingFace) — direct MP4 downloads                        ║
║  3. asl_clips/ local recordings via Dev Panel                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

SWL-LSE SIGN MAPPING  (CSV LABEL → LSE_DICT key)
────────────────────────────────────────────────
DOLOR              → dolor          GARGANTA/GARGANTA2 → garganta
MAREO/MAREO2       → mareo          MOCO/MOCO2         → moco
RESPIRAR/RESPIRAR2 → respirar       COMER              → comer
DORMIR             → dormir         TOSER              → tos
FIEBRE2            → fiebre         SANGRE2            → sangre
ACUFENO            → zumbido        AMIGDALAS-INFLAMAR → hinchazon
AMIGDALITIS        → infeccion

Sign4all SIGN MAPPING  (folder name → LSE_DICT key)
──────────────────────────────────────────────────
COMER → comer    BEBER → beber    DORMIR → dormir
CAMINAR → caminar    SENTARSE → sentar    ABRIR → abrir    MIRAR → mirar

LSE-Health-UVigo SIGN MAPPING
──────────────────────────────────────────────────────
Auto-matched after accent normalisation. Medical/shared terms that match
your ENT signs (DOLOR, FIEBRE, INFECCION, SANGRE, RESPIRAR, COMER,
BEBER, DORMIR, CAMINAR…) are pulled automatically.

Run:
    python dataset_download.py
"""

import csv, json, os, pickle, sys, random, shutil, subprocess
import urllib.request, urllib.error
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ── path bootstrap ─────────────────────────────────────────────────────────────
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

try:
    import torch
except ImportError:
    print("\n[dataset_download] PyTorch required.\n"
          "  pip install torch torchvision\n"
          "  CUDA: pip install torch torchvision "
          "--index-url https://download.pytorch.org/whl/cu124\n")
    sys.exit(1)

from asl_dictionary import ASL_DICT, LSE_DICT
from sign_model import (
    I3DClassifier, SignClassifier,
    model_path, data_path,
    I3D_ASL, I3D_LSE,
    MODEL_ASL, MODEL_LSE,
    SAMPLES_ASL, SAMPLES_LSE,
    CLIPS_ASL, CLIPS_LSE,
    _LANG_I3D, _LANG_CLIPS,
)

# ══════════════════════════════════════════════════════════════════════════════
# ▶▶  CONFIGURE THESE PATHS  ◀◀
# ══════════════════════════════════════════════════════════════════════════════

# ── SWL-LSE (Zenodo 13691887) ─────────────────────────────────────────────────
# Where you extracted MEDIAPIPE.zip (contains .pkl skeleton files)
SWL_MEDIAPIPE_DIR = Path.home() / "Downloads" / "13691887" / "MEDIAPIPE"

# Where you extracted VIDEOS_REF.zip AFTER running rename_swllse.py
# Files should be named: garganta_001.mp4, dolor_001.mp4 etc.
SWL_VIDEOS_DIR    = Path.home() / "Downloads" / "13691887" / "VIDEOS_RENAMED"

# CSV from Zenodo download: FILENAME,CLASS_ID,LABEL
SWL_CSV           = Path.home() / "Downloads" / "13691887" / "ANNOTATIONS" / "videos_ref_annotations.csv"

# ── LSE-Health-UVigo (Zenodo 10234465) ───────────────────────────────────────
# Timestamps CSV — columns: File, Start(ms), End(ms), DT, youtube link
# This is the file you uploaded (LSE-Health-UVIGO-timestamps.csv).
# Set to None to skip this source entirely.
UVIGO_TIMESTAMPS_CSV = None

# Folder where downloaded+trimmed UVigo clips are cached (auto-created)
UVIGO_CACHE_DIR      = Path.home() / "Downloads" / "uvigo_cache"

# ── Sign4all (SciDB) ──────────────────────────────────────────────────────────
# After extracting the Sign4all download, the structure should be:
#   SIGN4ALL_DIR/COMER/video_001.mp4
#   SIGN4ALL_DIR/BEBER/video_001.mp4
# Set to None to skip this source entirely.
SIGN4ALL_DIR      = Path.home() / "Downloads" / "sign4all"

# ══════════════════════════════════════════════════════════════════════════════

# ── directories ────────────────────────────────────────────────────────────────
DATA_DIR = HERE / "training_data"
RAW_ASL  = DATA_DIR / "raw_clips" / "asl"
RAW_LSE  = DATA_DIR / "raw_clips" / "lse"
for d in (DATA_DIR, RAW_ASL, RAW_LSE):
    d.mkdir(parents=True, exist_ok=True)

# ── tuning ─────────────────────────────────────────────────────────────────────
TARGET_CLIPS = 50   # target clips per sign (real + augmented)
MIN_FRAMES   = 8    # discard clips shorter than this

# ── SWL-LSE label → your sign key ─────────────────────────────────────────────
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

# ── Sign4all folder name → your sign key ──────────────────────────────────────
SIGN4ALL_TO_SIGN = {
    "COMER":    "comer",
    "BEBER":    "beber",
    "DORMIR":   "dormir",
    "CAMINAR":  "caminar",
    "SENTARSE": "sentar",
    "ABRIR":    "abrir",
    "MIRAR":    "mirar",
}

# ── ASL remote sources ─────────────────────────────────────────────────────────
HF_BASE    = "https://huggingface.co/datasets/google/asl_citizen/resolve/main/videos"
HF_META    = "https://huggingface.co/datasets/google/asl_citizen/resolve/main/data/train-00000-of-00001.parquet"
WLASL_JSON = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _normalise_gloss(raw: str) -> str:
    """Lowercase, strip accents, replace hyphens/spaces with underscore."""
    import unicodedata
    s = "".join(
        c for c in unicodedata.normalize("NFD", raw.lower())
        if unicodedata.category(c) != "Mn"
    )
    return s.replace("-", "_").replace(" ", "_")


def _get(url: str, dest: Path, timeout: int = 25) -> bool:
    if dest.exists() and dest.stat().st_size > 1024:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SignFuture/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            dest.write_bytes(r.read())
        return dest.stat().st_size > 1024
    except Exception:
        dest.unlink(missing_ok=True)
        return False


def _ytdlp(url: str, dest: Path, timeout: int = 90) -> bool:
    if dest.exists() and dest.stat().st_size > 1024:
        return True
    try:
        r = subprocess.run(
            ["yt-dlp", "-q", "--no-warnings",
             "-f", "bestvideo[ext=mp4][height<=360]+bestaudio/best[height<=360]/best",
             "--merge-output-format", "mp4", "-o", str(dest), url],
            timeout=timeout, capture_output=True,
        )
        return r.returncode == 0 and dest.exists() and dest.stat().st_size > 1024
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _ytdlp_segment(url: str, dest: Path, start_ms: int, end_ms: int,
                    timeout: int = 120) -> bool:
    """Download a single timestamped segment from a YouTube video via yt-dlp.
    Falls back to full download + cv2 trim if --download-sections fails."""
    if dest.exists() and dest.stat().st_size > 1024:
        return True
    start_s = start_ms / 1000.0
    end_s   = end_ms   / 1000.0
    # Try native yt-dlp section download first
    try:
        r = subprocess.run(
            ["yt-dlp", "-q", "--no-warnings",
             "-f", "bestvideo[ext=mp4][height<=360]+bestaudio/best[height<=360]/best",
             "--merge-output-format", "mp4",
             "--download-sections", f"*{start_s:.3f}-{end_s:.3f}",
             "--force-keyframes-at-cuts",
             "-o", str(dest), url],
            timeout=timeout, capture_output=True,
        )
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 1024:
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    # Fallback: full download then cv2 trim
    tmp = dest.with_suffix(".full.mp4")
    if _ytdlp(url, tmp, timeout=timeout):
        ok = _trim_video(tmp, dest, start_ms, end_ms)
        tmp.unlink(missing_ok=True)
        return ok
    return False


def _trim_video(src: Path, dest: Path, start_ms: int, end_ms: int) -> bool:
    """Trim src video to [start_ms, end_ms] using cv2 frame extraction."""
    try:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            return False
        fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
        start_f = int(start_ms / 1000.0 * fps)
        end_f   = int(end_ms   / 1000.0 * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(dest),
                              cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (w, h))
        frames_written = 0
        while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_f:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frames_written += 1
        cap.release()
        out.release()
        return frames_written >= MIN_FRAMES and dest.exists()
    except Exception:
        return False


_HAS_YTDLP = None
def has_ytdlp() -> bool:
    global _HAS_YTDLP
    if _HAS_YTDLP is None:
        _HAS_YTDLP = shutil.which("yt-dlp") is not None
    return _HAS_YTDLP


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO → RAW FRAMES
# ═══════════════════════════════════════════════════════════════════════════════

def extract_raw_frames(path) -> list:
    """MP4 → list of raw BGR frames. Returns [] if < MIN_FRAMES."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames if len(frames) >= MIN_FRAMES else []


# ═══════════════════════════════════════════════════════════════════════════════
# MEDIAPIPE PKL → LANDMARK VECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def pkl_to_landmark_sequence(pkl_path) -> list:
    """Load a SWL-LSE MediaPipe pkl → list of 278-dim landmark vectors."""
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"      ⚠ pkl load failed: {pkl_path.name} — {e}")
        return []

    from landmarks import extract_landmarks, normalize_landmarks

    holistic_frames = data.get("holistic_legacy", [])
    if not holistic_frames:
        return []

    lm_seq = []
    for frame_result in holistic_frames:
        lm = normalize_landmarks(extract_landmarks(frame_result))
        if not np.all(lm[:126] == 0):   # at least one hand visible
            lm_seq.append(lm)

    return lm_seq


def _video_to_mlp(src: Path, X_mlp: list, y_mlp: list, label: str) -> int:
    """Extract MediaPipe landmarks from a video and append to MLP arrays.
    Returns 1 if successful, 0 if mediapipe not installed or no hands found."""
    try:
        import mediapipe  # noqa: F401
        import mp_holistic as mph
        from landmarks import extract_landmarks, normalize_landmarks
        cap    = cv2.VideoCapture(str(src))
        lm_seq = []
        with mph.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as h:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                lm  = normalize_landmarks(extract_landmarks(h.process(rgb)))
                if not np.all(lm[:126] == 0):
                    lm_seq.append(lm)
        cap.release()
        if lm_seq:
            from sign_model import sequence_to_feature
            X_mlp.append(sequence_to_feature(lm_seq))
            y_mlp.append(label)
            return 1
    except ImportError:
        pass   # mediapipe not installed — I3D path only
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION  (raw BGR frames)
# ═══════════════════════════════════════════════════════════════════════════════

def _frame_brightness(frame: np.ndarray) -> np.ndarray:
    delta = random.uniform(-0.20, 0.20)
    f = frame.astype(np.float32) / 255.0
    return (np.clip(f + delta, 0, 1) * 255).astype(np.uint8)

def _frame_hflip(frame: np.ndarray) -> np.ndarray:
    return frame[:, ::-1, :].copy()

def _time_warp(frames: list) -> list:
    factor = random.uniform(0.72, 1.30)
    n = max(MIN_FRAMES, int(len(frames) * factor))
    return [frames[i] for i in np.linspace(0, len(frames) - 1, n).astype(int)]

def _temporal_shift(frames: list) -> list:
    s = random.randint(0, min(5, len(frames) - MIN_FRAMES))
    result = frames[s:]
    return result if len(result) >= MIN_FRAMES else frames

def augment_clip(frames: list, n: int) -> list:
    """Return n augmented variants of a raw frame list."""
    variants = []
    for _ in range(n):
        seq = list(frames)
        if random.random() < 0.60: seq = _time_warp(seq)
        if random.random() < 0.45: seq = _temporal_shift(seq)
        if random.random() < 0.50: seq = [_frame_brightness(f) for f in seq]
        if random.random() < 0.40: seq = [_frame_hflip(f) for f in seq]
        variants.append(seq)
    return variants


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — SWL-LSE (Zenodo 13691887)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_swl_index() -> dict:
    """Parse videos_ref_annotations.csv → {sign_key: {"csv_filenames": [], "class_id": int}}"""
    index = defaultdict(lambda: {"csv_filenames": [], "class_id": None})

    if SWL_CSV.exists():
        with open(SWL_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["LABEL"].strip()
                sign  = SWL_LABEL_TO_SIGN.get(label)
                if sign:
                    index[sign]["csv_filenames"].append(row["FILENAME"].strip())
                    index[sign]["class_id"] = int(row["CLASS_ID"])
    else:
        print(f"  ⚠ SWL CSV not found at {SWL_CSV}")

    return dict(index)


def load_swl_mediapipe(sign: str, swl_index: dict) -> list:
    """Load MediaPipe pkl files for a sign → list of landmark sequences."""
    if not SWL_MEDIAPIPE_DIR.exists():
        return []

    csv_filenames = swl_index.get(sign, {}).get("csv_filenames", [])
    sequences = []

    for fname in csv_filenames:
        pkl_path = SWL_MEDIAPIPE_DIR / (fname + ".pkl")
        if not pkl_path.exists():
            pkl_path = SWL_MEDIAPIPE_DIR / (fname.replace(".mp4", "") + ".pkl")
        if not pkl_path.exists():
            continue
        seq = pkl_to_landmark_sequence(pkl_path)
        if seq:
            sequences.append(seq)

    return sequences


def load_swl_videos(sign: str) -> list:
    """Load renamed SWL reference videos → list of raw frame lists."""
    if not SWL_VIDEOS_DIR.exists():
        return []

    clips = []
    for mp4 in sorted(SWL_VIDEOS_DIR.glob(f"{sign}_*.mp4")):
        frames = extract_raw_frames(mp4)
        if frames:
            clips.append(frames)
            print(f"      ✓ SWL video  {mp4.name}  ({len(frames)} frames)")
    return clips


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — LSE-Health-UVigo (Zenodo 10234465)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_uvigo_index() -> dict:
    """
    Parse LSE-Health-UVIGO-timestamps.csv.
    CSV format: File,Start(ms),End(ms),DT,youtube link
    Returns {normalised_gloss: [(youtube_url, start_ms, end_ms), …]}
    """
    if not UVIGO_TIMESTAMPS_CSV or not UVIGO_TIMESTAMPS_CSV.exists():
        return {}

    index = defaultdict(list)
    with open(UVIGO_TIMESTAMPS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_label = row.get("DT", "").strip()
            if raw_label.startswith("DT:"):
                raw_label = raw_label[3:]
            url      = row.get("youtube link", "").strip()
            try:
                start_ms = int(row.get("Start(ms)", 0))
                end_ms   = int(row.get("End(ms)", 0))
            except (ValueError, TypeError):
                continue
            if not url or not raw_label or end_ms <= start_ms:
                continue
            norm = _normalise_gloss(raw_label)
            index[norm].append((url, start_ms, end_ms))

    print(f"  UVigo index: {len(index)} unique glosses, "
          f"{sum(len(v) for v in index.values())} total clips")
    return dict(index)


def _uvigo_gloss_to_sign(gloss_norm: str) -> str | None:
    """Map a normalised UVigo gloss to an LSE_DICT key, or None if no match."""
    if gloss_norm in LSE_DICT:
        return gloss_norm
    UVIGO_OVERRIDES = {
        "toser":    "tos",
        "fiebre":   "fiebre",
        "fiebre2":  "fiebre",
        "dolor":    "dolor",
        "infeccion":"infeccion",
        "sangre":   "sangre",
        "sangre2":  "sangre",
        "respirar": "respirar",
        "comer":    "comer",
        "beber":    "beber",
        "dormir":   "dormir",
        "caminar":  "caminar",
        "sentar":   "sentar",
        "sentarse": "sentar",
        "girar":    "girar",
        "abrir":    "abrir",
        "mirar":    "mirar",
        "mareo":    "mareo",
        "mareo2":   "mareo",
        "moco":     "moco",
        "moco2":    "moco",
    }
    return UVIGO_OVERRIDES.get(gloss_norm)


def load_uvigo_clips(sign: str, uvigo_index: dict,
                     X_i3d: list, y_i3d: list,
                     X_mlp: list, y_mlp: list) -> int:
    """
    Download and trim UVigo YouTube segments for a sign.
    Clips are cached in UVIGO_CACHE_DIR — re-runs skip already-downloaded files.
    Returns number of clips added to X_i3d.
    """
    if not has_ytdlp() or not uvigo_index:
        return 0

    UVIGO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sign_dir = UVIGO_CACHE_DIR / sign
    sign_dir.mkdir(exist_ok=True)

    matching_glosses = [g for g in uvigo_index if _uvigo_gloss_to_sign(g) == sign]
    if not matching_glosses:
        return 0

    added = 0
    for gloss in matching_glosses:
        for i, (url, start_ms, end_ms) in enumerate(uvigo_index[gloss]):
            dest = sign_dir / f"{gloss}_{i:04d}.mp4"
            if _ytdlp_segment(url, dest, start_ms, end_ms):
                frames = extract_raw_frames(dest)
                if frames:
                    X_i3d.append(frames)
                    y_i3d.append(sign)
                    added += 1
                    _video_to_mlp(dest, X_mlp, y_mlp, sign)
                    print(f"      ✓ UVigo {gloss}[{i}] ({len(frames)}fr)")
                else:
                    dest.unlink(missing_ok=True)

    return added


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 3 — Sign4all (SciDB)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_sign4all_index() -> dict:
    """
    Scan SIGN4ALL_DIR for sign subfolders that match SIGN4ALL_TO_SIGN or LSE_DICT.
    Returns {lse_dict_key: [Path, …]} for each matched sign.
    """
    if not SIGN4ALL_DIR or not SIGN4ALL_DIR.exists():
        return {}

    index = defaultdict(list)
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
                index[sign].append(mp4)

    if index:
        print(f"  Sign4all index: {len(index)} signs matched, "
              f"{sum(len(v) for v in index.values())} videos")
    return dict(index)


def load_sign4all_clips(sign: str, sign4all_index: dict,
                        X_i3d: list, y_i3d: list,
                        X_mlp: list, y_mlp: list) -> int:
    """Load Sign4all clips for a sign. Returns count added."""
    paths = sign4all_index.get(sign, [])
    added = 0
    for mp4 in paths:
        frames = extract_raw_frames(mp4)
        if frames:
            X_i3d.append(frames)
            y_i3d.append(sign)
            added += 1
            _video_to_mlp(mp4, X_mlp, y_mlp, sign)
            print(f"      ✓ Sign4all {mp4.name} ({len(frames)}fr)")
    return added


# ═══════════════════════════════════════════════════════════════════════════════
# LSE COLLECTION  (per sign — all sources stacked)
# ═══════════════════════════════════════════════════════════════════════════════

def collect_lse(label: str,
                swl_index: dict,
                uvigo_index: dict,
                sign4all_index: dict,
                X_i3d: list, y_i3d: list,
                X_mlp: list, y_mlp: list) -> tuple:
    """
    Collect training data for one LSE sign from all available sources.

    I3D path  (X_i3d / y_i3d): raw BGR frame lists → clips_lse.npz → i3d_lse.pt
    MLP path  (X_mlp / y_mlp): 278-dim feature vectors → samples_lse.npz → model_lse.pt

    Stack order:
      1. SWL-LSE MEDIAPIPE pkl  → MLP sequences (~27 per sign)
      2. SWL-LSE VIDEOS_REF mp4 → I3D raw frames (1 reference clip per sign)
      3. LSE-Health-UVigo       → I3D + MLP (timestamped YouTube segments)
      4. Sign4all               → I3D + MLP (high-density RGB videos)
      5. lse_clips/ local mp4s  → I3D + MLP (your webcam recordings)
      6. Augmentation           → pad I3D to TARGET_CLIPS if still short

    Returns (n_i3d_added, n_mlp_added).
    """
    lse_folder = HERE / "lse_clips"
    sign_dir   = RAW_LSE / label.replace(" ", "_")
    sign_dir.mkdir(exist_ok=True)

    n_i3d, n_mlp = 0, 0

    # ── 1. SWL-LSE MEDIAPIPE → MLP sequences ──────────────────────────────────
    mp_seqs = load_swl_mediapipe(label, swl_index)
    if mp_seqs:
        from sign_model import sequence_to_feature
        for seq in mp_seqs:
            X_mlp.append(sequence_to_feature(seq))
            y_mlp.append(label)
            n_mlp += 1
        print(f"      ✓ SWL MediaPipe  {n_mlp} skeleton seqs → MLP")

    # ── 2. SWL-LSE VIDEOS_REF → I3D raw frames ────────────────────────────────
    for frames in load_swl_videos(label):
        X_i3d.append(frames); y_i3d.append(label); n_i3d += 1

    # ── 3. LSE-Health-UVigo → I3D + MLP ───────────────────────────────────────
    n_uv = load_uvigo_clips(label, uvigo_index, X_i3d, y_i3d, X_mlp, y_mlp)
    if n_uv:
        n_i3d += n_uv
        print(f"      ✓ UVigo +{n_uv} clips")

    # ── 4. Sign4all → I3D + MLP ───────────────────────────────────────────────
    n_s4 = load_sign4all_clips(label, sign4all_index, X_i3d, y_i3d, X_mlp, y_mlp)
    if n_s4:
        n_i3d += n_s4
        print(f"      ✓ Sign4all +{n_s4} clips")

    # ── 5. lse_clips/ local recordings → I3D + MLP ────────────────────────────
    clip_file = LSE_DICT.get(label)
    if clip_file and lse_folder.exists():
        stem       = Path(clip_file).stem
        candidates = []
        exact      = lse_folder / clip_file
        if exact.exists():
            candidates.append(exact)
        candidates += sorted(lse_folder.glob(f"{stem}_*.mp4"))

        for src in candidates:
            frames = extract_raw_frames(src)
            if not frames:
                continue
            X_i3d.append(frames); y_i3d.append(label); n_i3d += 1
            print(f"      ✓ lse_clips/{src.name}  ({len(frames)} fr) → I3D")
            if _video_to_mlp(src, X_mlp, y_mlp, label):
                n_mlp += 1

    # ── 6. Augmentation → pad I3D to TARGET_CLIPS ─────────────────────────────
    # Collect all real clips for this label collected so far
    real_i3d_all = [X_i3d[i] for i in range(len(X_i3d)) if y_i3d[i] == label]
    needed = TARGET_CLIPS - n_i3d
    if needed > 0 and real_i3d_all:
        per, n_aug = max(1, (needed // len(real_i3d_all)) + 2), 0
        for clip in real_i3d_all:
            for aug in augment_clip(clip, per):
                if n_i3d >= TARGET_CLIPS:
                    break
                X_i3d.append(aug); y_i3d.append(label); n_i3d += 1; n_aug += 1
            if n_i3d >= TARGET_CLIPS:
                break
        if n_aug:
            print(f"      + {n_aug} augmented → I3D total: {n_i3d}")
    elif needed > 0:
        print(f"      ⚠ No video clips for I3D — record '{label}' via Dev Panel")
        print(f"         save to lse_clips/{LSE_DICT.get(label, label+'.mp4')}")

    return n_i3d, n_mlp


# ═══════════════════════════════════════════════════════════════════════════════
# ASL COLLECTION  (WLASL + ASL Citizen + augmentation)
# ═══════════════════════════════════════════════════════════════════════════════

def _wlasl_index() -> dict:
    cache = DATA_DIR / "wlasl_index.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print("    Fetching WLASL index…", end=" ", flush=True)
    try:
        with urllib.request.urlopen(WLASL_JSON, timeout=20) as r:
            data = json.load(r)
        index = {e["gloss"].lower().replace("-", " "): [i["url"] for i in e["instances"] if i.get("url")]
                 for e in data}
        cache.write_text(json.dumps(index))
        print(f"OK ({len(index)} glosses)")
        return index
    except Exception as e:
        print(f"FAILED ({e})"); return {}


def _hf_index() -> dict:
    cache = DATA_DIR / "hf_index.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print("    Fetching ASL Citizen index…", end=" ", flush=True)
    try:
        import pyarrow.parquet as pq, io
        with urllib.request.urlopen(HF_META, timeout=20) as r:
            buf = io.BytesIO(r.read())
        tbl = pq.read_table(buf).to_pydict()
        index = {}
        for gloss, fname in zip(tbl["gloss"], tbl["file_name"]):
            index.setdefault(gloss.upper(), []).append(fname)
        cache.write_text(json.dumps(index))
        print(f"OK ({len(index)} glosses)")
        return index
    except Exception as e:
        print(f"FAILED ({e}) — augmentation only"); return {}


def collect_asl(label: str, wlasl: dict, hf: dict, X: list, y: list) -> int:
    gloss     = label.upper().replace(" ", "-")
    gloss_low = label.lower()
    sign_dir  = RAW_ASL / label.replace(" ", "_")
    sign_dir.mkdir(exist_ok=True)
    real_clips = []
    added = 0

    if has_ytdlp():
        pool = wlasl.get(gloss_low, [])
        random.shuffle(pool)
        for i, url in enumerate(pool):
            if added >= TARGET_CLIPS:
                break
            dest = sign_dir / f"wlasl_{i:04d}.mp4"
            if _ytdlp(url, dest):
                frames = extract_raw_frames(dest)
                if frames:
                    real_clips.append(frames); X.append(frames); y.append(label)
                    added += 1
                    print(f"      ✓ WLASL {i:04d} ({len(frames)}fr) [{added}]")
                else:
                    dest.unlink(missing_ok=True)

    if added < TARGET_CLIPS:
        pool = hf.get(gloss, [])
        random.shuffle(pool)
        for fname in pool:
            if added >= TARGET_CLIPS:
                break
            dest = sign_dir / fname
            if _get(f"{HF_BASE}/{fname}", dest):
                frames = extract_raw_frames(dest)
                if frames:
                    real_clips.append(frames); X.append(frames); y.append(label)
                    added += 1
                    print(f"      ✓ ASLCitizen {fname[:30]} ({len(frames)}fr) [{added}]")
                else:
                    dest.unlink(missing_ok=True)

    from config import ASL_CLIPS_FOLDER
    clip_file = ASL_DICT.get(label)
    src = HERE / ASL_CLIPS_FOLDER / clip_file if clip_file else None
    if src and src.exists():
        frames = extract_raw_frames(src)
        if frames:
            real_clips.append(frames)

    needed = TARGET_CLIPS - added
    if needed > 0 and real_clips:
        per, n_aug = max(1, (needed // len(real_clips)) + 2), 0
        for clip in real_clips:
            for aug in augment_clip(clip, per):
                if added >= TARGET_CLIPS:
                    break
                X.append(aug); y.append(label); added += 1; n_aug += 1
            if added >= TARGET_CLIPS:
                break
        print(f"      + {n_aug} augmented")
    elif needed > 0:
        print(f"      ⚠ No source clips for '{label}'")
    return added


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_i3d(X: list, y: list, lang: str, out_path: str) -> dict:
    labels = sorted(set(y))
    print(f"\n  Training I3D {lang.upper()} — {len(X)} clips, {len(labels)} signs…")
    clf  = I3DClassifier(labels)
    last = [0]
    def cb(pct, msg):
        if pct != last[0]:
            print(f"\r    [{'█'*(pct//5)}{'░'*(20-pct//5)}] {pct:3d}%  {msg:<45}",
                  end="", flush=True)
            last[0] = pct
    result = clf.train(X, y, epochs=60, batch_size=4, progress_cb=cb)
    print()
    clf.save(out_path)
    return result


def train_mlp(X_raw: list, y: list, lang: str, out_path: str) -> dict:
    labels = sorted(set(y))
    print(f"\n  Training MLP {lang.upper()} — {len(X_raw)} samples, {len(labels)} signs…")
    X   = np.array(X_raw, dtype=np.float32)
    clf = SignClassifier(labels)
    last = [0]
    def cb(pct, msg):
        if pct != last[0]:
            print(f"\r    [{'█'*(pct//5)}{'░'*(20-pct//5)}] {pct:3d}%  {msg:<45}",
                  end="", flush=True)
            last[0] = pct
    result = clf.train(X, y, epochs=300, progress_cb=cb)
    print()
    clf.save(out_path)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    lse_clips_dir = HERE / "lse_clips"
    lse_clips_n   = len(list(lse_clips_dir.glob("*.mp4"))) if lse_clips_dir.exists() else 0

    uvigo_ok    = bool(UVIGO_TIMESTAMPS_CSV and UVIGO_TIMESTAMPS_CSV.exists())
    sign4all_ok = bool(SIGN4ALL_DIR and SIGN4ALL_DIR.exists())

    print("=" * 66)
    print("  SignFuture — Dataset Builder")
    print(f"  LSE signs    : {len(LSE_DICT)}")
    print(f"  ASL signs    : {len(ASL_DICT)}")
    print(f"  Target clips : {TARGET_CLIPS}/sign")
    print()
    print("  LSE sources:")
    print(f"    [1] SWL-LSE MediaPipe : {'✓ ' + str(SWL_MEDIAPIPE_DIR) if SWL_MEDIAPIPE_DIR.exists() else '✗ set SWL_MEDIAPIPE_DIR'}")
    print(f"    [1] SWL-LSE Videos    : {'✓ ' + str(SWL_VIDEOS_DIR) if SWL_VIDEOS_DIR.exists() else '✗ run rename_swllse.py first'}")
    print(f"    [2] UVigo (YouTube)   : {'✓ ' + str(UVIGO_TIMESTAMPS_CSV) if uvigo_ok else '✗ set UVIGO_TIMESTAMPS_CSV'}")
    print(f"    [3] Sign4all          : {'✓ ' + str(SIGN4ALL_DIR) if sign4all_ok else '✗ set SIGN4ALL_DIR'}")
    print(f"    [4] lse_clips/        : {'✓ ' + str(lse_clips_n) + ' files' if lse_clips_n else '⚠ empty — record via Dev Panel'}")
    print(f"    yt-dlp               : {'✓' if has_ytdlp() else '✗ pip install yt-dlp  (needed for UVigo source)'}")
    print("=" * 66)

    existing = [p for p in (CLIPS_ASL, CLIPS_LSE, SAMPLES_LSE, I3D_ASL, I3D_LSE, MODEL_LSE)
                if os.path.exists(p)]
    if existing:
        print(f"\nExisting training files: {[os.path.basename(p) for p in existing]}")
        ans = input("Overwrite? [y/N]: ").strip().lower()
        if ans != "y":
            print("Aborted."); return
        for p in existing:
            os.remove(p)

    # ── Build all dataset indexes ─────────────────────────────────────────────
    print("\n[1/5] Building dataset indexes…")

    swl_index = _build_swl_index()
    covered = sorted(s for s in LSE_DICT if s in swl_index)
    missing  = sorted(s for s in LSE_DICT if s not in swl_index)
    print(f"  SWL covers {len(covered)}/{len(LSE_DICT)} signs: {covered}")

    uvigo_index = _build_uvigo_index() if uvigo_ok else {}
    if uvigo_index:
        uvigo_matched = sorted(set(
            _uvigo_gloss_to_sign(g) for g in uvigo_index
            if _uvigo_gloss_to_sign(g)
        ))
        print(f"  UVigo matches {len(uvigo_matched)} of your signs: {uvigo_matched}")

    sign4all_index = _build_sign4all_index() if sign4all_ok else {}

    # ── ASL indexes ───────────────────────────────────────────────────────────
    print(f"\n[2/5] Fetching ASL indexes…")
    wlasl = _wlasl_index() if has_ytdlp() else {}
    hf    = _hf_index()

    # ── Collect ASL ───────────────────────────────────────────────────────────
    print(f"\n[3/5] Collecting ASL clips ({len(ASL_DICT)} signs)…")
    X_asl, y_asl = [], []
    for i, label in enumerate(ASL_DICT, 1):
        print(f"\n  [{i}/{len(ASL_DICT)}] '{label}'")
        n = collect_asl(label, wlasl, hf, X_asl, y_asl)
        print(f"  → {n} clips")

    clips_asl_arr = np.empty(len(X_asl), dtype=object)
    for i, f in enumerate(X_asl): clips_asl_arr[i] = f
    np.savez(CLIPS_ASL, X=clips_asl_arr, y=np.array(y_asl))
    print(f"\n  ✅ ASL: {len(X_asl)} clips saved")

    # ── Collect LSE ───────────────────────────────────────────────────────────
    print(f"\n[4/5] Collecting LSE data ({len(LSE_DICT)} signs)…")
    X_i3d, y_i3d = [], []
    X_mlp, y_mlp = [], []

    for i, label in enumerate(LSE_DICT, 1):
        print(f"\n  [{i}/{len(LSE_DICT)}] '{label}'")
        n_i, n_m = collect_lse(label, swl_index, uvigo_index, sign4all_index,
                                X_i3d, y_i3d, X_mlp, y_mlp)
        print(f"  → I3D: {n_i} clips   MLP: {n_m} sequences")

    clips_lse_arr = np.empty(len(X_i3d), dtype=object)
    for i, f in enumerate(X_i3d): clips_lse_arr[i] = f
    np.savez(CLIPS_LSE, X=clips_lse_arr, y=np.array(y_i3d))
    print(f"\n  ✅ LSE I3D clips : {len(X_i3d)} total")

    if X_mlp:
        np.savez(SAMPLES_LSE,
                 X=np.array(X_mlp, dtype=np.float32),
                 y=np.array(y_mlp))
        print(f"  ✅ LSE MLP samples: {len(X_mlp)} total")
        for lbl in sorted(set(y_mlp)):
            n = y_mlp.count(lbl)
            print(f"    {lbl:<20} {n:>3}  {'█'*(n//3)}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[5/5] Training models…")
    results = {}

    if X_i3d and len(set(y_i3d)) >= 2:
        results["lse_i3d"] = train_i3d(X_i3d, y_i3d, "lse", I3D_LSE)
    else:
        print("  ⚠ LSE I3D: need ≥2 signs with video → record via Dev Panel")

    if X_mlp and len(set(y_mlp)) >= 2:
        results["lse_mlp"] = train_mlp(X_mlp, y_mlp, "lse", MODEL_LSE)
    else:
        print("  ⚠ LSE MLP: need ≥2 signs with skeleton data")

    if X_asl and len(set(y_asl)) >= 2:
        results["asl_i3d"] = train_i3d(X_asl, y_asl, "asl", I3D_ASL)
    else:
        print("  ⚠ ASL I3D: need ≥2 signs with clips")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*66}")
    print("  ✅ Done!")
    for key, r in results.items():
        print(f"  {key.upper():<12} accuracy={r['accuracy']*100:.1f}%  time={r['time_s']}s")

    print(f"\n  LSE sign coverage:")
    for sign in sorted(LSE_DICT.keys()):
        i3d = "I3D✓" if sign in set(y_i3d) else "I3D✗"
        mlp = "MLP✓" if sign in set(y_mlp) else "MLP✗"
        src_tags = []
        if sign in swl_index:              src_tags.append("SWL")
        if uvigo_index and any(_uvigo_gloss_to_sign(g) == sign for g in uvigo_index):
            src_tags.append("UVigo")
        if sign4all_index and sign in sign4all_index: src_tags.append("S4A")
        src = "+".join(src_tags) if src_tags else "local"
        print(f"    {sign:<14} {i3d}  {mlp}  [{src}]")

    still_missing = [s for s in missing if s not in set(y_i3d)]
    if still_missing:
        print(f"\n  Still need recordings for {len(still_missing)} signs:")
        for s in still_missing:
            print(f"    - {s}  →  lse_clips/{LSE_DICT.get(s, s+'.mp4')}")
        print(f"\n  Record them via Dev Panel webcam → then rerun this script.")

    print(f"\n  Start server:  python web_bridge.py")
    print(f"{'='*66}")


if __name__ == "__main__":
    main()
