"""rename_dataset.py — Rename SWL-LSE VIDEOS_REF files to your LSE_DICT sign names.

Uses TWO files to build the mapping:
  videos_ref_annotations.csv : FILENAME(recXXX), CLASS_ID, LABEL — class_id → sign name
  train_labels.csv / val_labels.csv / test_labels.csv : FILENAME(timestamp), CLASS_ID

Flow:
  timestamp → class_id  (from train/val/test_labels.csv)
  class_id  → LABEL     (from videos_ref_annotations.csv)
  LABEL     → your sign (from SWL_LABEL_TO_SIGN dict)
  → copy to VIDEOS_RENAMED/signname_001.mp4

Run:
    python rename_dataset.py

Dataset: https://zenodo.org/records/13691887
Files needed from ANNOTATIONS.zip: videos_ref_annotations.csv, train_labels.csv,
                                    val_labels.csv, test_labels.csv
Files needed from VIDEOS_REF.zip:   recXXXX.mp4 reference videos
"""

import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent.resolve()   # project root

# ── Paths — all relative to project root ──────────────────────────────────────
VIDEO_DIR       = Path.home() / "Downloads" / "13691887" / "VIDEOS_REF"
# All annotation CSVs live under ANNOTATIONS/ANNOTATIONS/ after unzip
ANNOTATIONS_DIR = Path.home() / "Downloads" / "13691887" / "ANNOTATIONS"
REF_CSV   = ANNOTATIONS_DIR / "videos_ref_annotations.csv"
OUT_DIR         = Path.home() / "Downloads" / "13691887" / "VIDEOS_RENAMED"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── SWL LABEL → your LSE_DICT key ────────────────────────────────────────────
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


def main():
    # ── Verify required files exist ───────────────────────────────────────────
    if not REF_CSV.exists():
        print(f"❌ Missing: {REF_CSV}")
        print(f"   Download ANNOTATIONS.zip from https://zenodo.org/records/13691887")
        print(f"   Then: unzip ~/Downloads/ANNOTATIONS.zip -d ~/Downloads/ANNOTATIONS")
        return

    if not VIDEO_DIR.exists():
        print(f"❌ Missing video folder: {VIDEO_DIR}")
        print(f"   Download VIDEOS_REF.zip from https://zenodo.org/records/13691887")
        print(f"   Then: unzip ~/Downloads/VIDEOS_REF.zip -d ~/Downloads/VIDEOS_REF")
        return

    # ── Step 1: class_id → LABEL from videos_ref_annotations.csv ─────────────
    class_to_label   = {}   # {110: "DOLOR", 191: "MAREO", ...}
    class_to_recfile = {}   # {110: "rec0110", ...}

    with open(REF_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)   # header: FILENAME, CLASS_ID, LABEL
        for row in reader:
            cid   = int(row["CLASS_ID"])
            label = row["LABEL"].strip()
            fname = row["FILENAME"].strip()
            class_to_label[cid]   = label
            class_to_recfile[cid] = fname

    print(f"  Loaded {len(class_to_label)} class→label mappings from {REF_CSV.name}")

    # ── Step 2: find matching reference videos and copy ───────────────────────
    sign_clips = defaultdict(list)

    for class_id, rec_filename in class_to_recfile.items():
        label = class_to_label.get(class_id, "")
        sign  = SWL_LABEL_TO_SIGN.get(label)
        if not sign:
            continue

        # Try with and without .mp4 extension
        src = VIDEO_DIR / (rec_filename + ".mp4")
        if not src.exists():
            src = VIDEO_DIR / rec_filename
        if not src.exists():
            print(f"  ⚠ Not found: {rec_filename}.mp4")
            continue

        sign_clips[sign].append(src)

    # ── Step 3: copy with sequential numbering ────────────────────────────────
    total = 0
    print(f"\n  {'Sign':<14} {'Clips':>5}  Files")
    print("  " + "-" * 50)
    for sign in sorted(sign_clips):
        clips = sorted(sign_clips[sign])
        print(f"  {sign:<14} {len(clips):>5}")
        for i, src in enumerate(clips, start=1):
            dst = OUT_DIR / f"{sign}_{i:03d}.mp4"
            shutil.copy2(src, dst)
            total += 1

    print(f"\n  ✅ Copied {total} files → {OUT_DIR}")

    missing = sorted(set(
        s for s in SWL_LABEL_TO_SIGN.values() if s not in sign_clips
    ))
    if missing:
        print(f"\n  ⚠  Signs with no reference video found ({len(missing)}):")
        for s in missing:
            print(f"     - {s}")

    print(f"\n  Next steps:")
    print(f"    python dataset_download.py   # collect all datasets + train")
    print(f"    python train_lse.py          # or: train I3D directly")


if __name__ == "__main__":
    main()
