#!/usr/bin/env python3
"""rename_clips.py — Rename IMG_*.MOV clips to their sign language names.

Maps raw iPhone clips (IMG_XXXX.MOV) to the correct sign name
based on their position in the recording order.

ASL clips go to:  ~/HearMeSign/asl_clips/
LSE clips go to:  ~/HearMeSign/lse_clips/

Usage:
    # Dry run first — shows what would happen without doing anything
    python rename_clips.py --dry-run --source ~/Downloads/clips/

    # Actually copy
    python rename_clips.py --source ~/Downloads/clips/

    # Move instead of copy
    python rename_clips.py --source ~/Downloads/clips/ --move
"""

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()

parser = argparse.ArgumentParser(description="Rename IMG_*.MOV clips to sign names")
parser.add_argument("--source",   required=True, help="Folder containing IMG_*.MOV files")
parser.add_argument("--asl-dest", default=str(HERE / "asl_clips"), help="Output for ASL clips")
parser.add_argument("--lse-dest", default=str(HERE / "lse_clips"), help="Output for LSE clips")
parser.add_argument("--dry-run",  action="store_true", help="Show plan without doing it")
parser.add_argument("--move",     action="store_true", help="Move instead of copy")
args = parser.parse_args()

SOURCE   = Path(args.source)
ASL_DEST = Path(args.asl_dest)
LSE_DEST = Path(args.lse_dest)

# ── Clip → sign name mapping ──────────────────────────────────────────────────
# Position in list = recording order exactly as provided

ASL_CLIPS = [
    ("IMG_0044.mp4", "ear.mp4"),
    ("IMG_4431.mp4", "nose.mp4"),
    ("IMG_0046.mp4", "throat.mp4"),
    ("IMG_0047.mp4", "mouth.mp4"),
    ("IMG_0049.mp4", "tongue.mp4"),
    ("IMG_0050.mp4", "teeth.mp4"),
    ("IMG_0051.mp4", "head.mp4"),
    ("IMG_0053.mp4", "neck.mp4"),
    ("IMG_0054.mp4", "pain.mp4"),
    ("IMG_0055.mp4", "cough.mp4"),
    ("IMG_0057.mp4", "fever.mp4"),
    ("IMG_0059.mp4", "dizzy.mp4"),
    ("IMG_0060.mp4", "tinnitus.mp4"),
    ("IMG_0063.mp4", "blood.mp4"),
    ("IMG_0065.mp4", "infection.mp4"),
    ("IMG_0068.mp4", "swelling.mp4"),
    ("IMG_0069.mp4", "mucus.mp4"),
    ("IMG_0070.mp4", "breathe.mp4"),
    ("IMG_4394.mp4", "swallow.mp4"),
    ("IMG_4396.mp4", "look.mp4"),
    ("IMG_4397.mp4", "turn.mp4"),
    ("IMG_4398.mp4", "eat.mp4"),
    ("IMG_4400.mp4", "drink.mp4"),
    ("IMG_4402.mp4", "sleep.mp4"),
    ("IMG_4403.mp4", "walk.mp4"),
    ("IMG_4404.mp4", "sit.mp4"),
]

LSE_CLIPS = [
    ("IMG_4405.mp4", "garganta.mp4"),
    ("IMG_4406.mp4", "boca.mp4"),
    ("IMG_4407.mp4", "dientes.mp4"),
    ("IMG_4409.mp4", "cabeza.mp4"),
    ("IMG_4410.mp4", "cuello.mp4"),
    ("IMG_4411.mp4", "dolor.mp4"),
    ("IMG_4412.mp4", "tos.mp4"),
    ("IMG_4414.mp4", "fiebre.mp4"),
    ("IMG_4415.mp4", "mareo.mp4"),
    ("IMG_4416.mp4", "sangre.mp4"),
    ("IMG_4417.mp4", "infeccion.mp4"),
    ("IMG_4418.mp4", "hinchazon.mp4"),
    ("IMG_4420.mp4", "moco.mp4"),
    ("IMG_4421.mp4", "voz.mp4"),
    ("IMG_4422.mp4", "respirar.mp4"),
    ("IMG_4423.mp4", "tragar.mp4"),
    ("IMG_4424.mp4", "girar.mp4"),
    ("IMG_4425.mp4", "beber.mp4"),
    ("IMG_4427.mp4", "dormir.mp4"),
    ("IMG_4428.mp4", "caminar.mp4"),
    ("IMG_4429.mp4", "sentar.mp4"),
    ("IMG_4430.mp4", "caminar_2.mp4"),  # alternate take of caminar
]

# LSE signs that reuse ASL clips — no separate file needed.
# asl_dictionary.py already handles this by pointing to the ASL filename.
LSE_REUSES_ASL = {
    "oreja":  "asl_clips/ear.mp4",
    "oido":   "asl_clips/ear.mp4",
    "nariz":  "asl_clips/nose.mp4",
    "lengua": "asl_clips/tongue.mp4",
    "mirar":  "asl_clips/look.mp4",
    "comer":  "asl_clips/eat.mp4",
}


def find_src(name):
    """Find source file — try exact name, then lowercase extension."""
    exact = SOURCE / name
    if exact.exists():
        return exact
    lower = SOURCE / name.replace(".mp4", ".mp4")
    if lower.exists():
        return lower
    return None


def process(clips, dest, label):
    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    op = "MOVE" if args.move else "COPY"
    print(f"\n  ── {label} → {dest} ──")
    ok = skip = missing = 0

    for src_name, dst_name in clips:
        src = find_src(src_name)
        dst = dest / dst_name

        if src is None:
            print(f"  ✗ MISSING  {src_name}")
            missing += 1
            continue

        if dst.exists() and not args.dry_run:
            print(f"  ○ EXISTS   {src_name} → {dst_name}")
            skip += 1
            continue

        if args.dry_run:
            print(f"  → {src_name} → {dst_name}")
        else:
            if args.move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
            print(f"  ✓ {src_name} → {dst_name}")
        ok += 1

    print(f"  {ok} {'moved' if args.move else 'copied'}, {skip} already exist, {missing} missing")
    return ok, skip, missing


def main():
    print("=" * 60)
    print("  SignFuture — Clip Renamer")
    print(f"  Source  : {SOURCE}")
    print(f"  ASL dest: {ASL_DEST}")
    print(f"  LSE dest: {LSE_DEST}")
    print(f"  Mode    : {'DRY RUN' if args.dry_run else ('MOVE' if args.move else 'COPY')}")
    print("=" * 60)

    if not SOURCE.exists():
        print(f"\n❌ Source folder not found: {SOURCE}")
        sys.exit(1)

    mov_count = len(list(SOURCE.glob("*.MOV")) + list(SOURCE.glob("*.mov")))
    print(f"\n  Found {mov_count} .MOV files in source")

    asl_ok, _, asl_miss = process(ASL_CLIPS, ASL_DEST, "ASL")
    lse_ok, _, lse_miss = process(LSE_CLIPS, LSE_DEST, "LSE")

    print()
    print("  ── LSE signs reusing ASL clips (no copy needed) ──")
    for sign, path in LSE_REUSES_ASL.items():
        print(f"  {sign:<12} → {path}")

    print()
    if asl_miss + lse_miss:
        print(f"  ⚠️  {asl_miss + lse_miss} files not found — check source path")
    if not args.dry_run:
        print(f"  ✅ {asl_ok + lse_ok} files {'moved' if args.move else 'copied'}")
    else:
        print("  ℹ️  Dry run — add --move or remove --dry-run to apply")

    print()
    print("  Note: IMG_4430.MOV → caminar_2.mp4 (alternate take of caminar)")
    print("=" * 60)


if __name__ == "__main__":
    main()
