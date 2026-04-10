#!/usr/bin/env python3
"""train_from_feedback.py — Retrain LSE models using feedback corrections only.

This is meant to be run after collecting feedback via the /feedback endpoint.
It retrains both I3D and MLP on the confirmed corrections, then merges the
updated weights with the existing models so prior knowledge is preserved.

Run:
    python train_from_feedback.py               # default 30 I3D, 100 MLP epochs
    python train_from_feedback.py --i3d-epochs 60 --mlp-epochs 300
    python train_from_feedback.py --no-i3d      # MLP only (much faster)
    python train_from_feedback.py --status      # just show what feedback data exists
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

parser = argparse.ArgumentParser()
parser.add_argument("--i3d-epochs", type=int, default=30)
parser.add_argument("--mlp-epochs", type=int, default=100)
parser.add_argument("--no-i3d",     action="store_true")
parser.add_argument("--no-mlp",     action="store_true")
parser.add_argument("--status",     action="store_true",
                    help="Show feedback data counts and exit")
parser.add_argument("--lang",       default="lse", choices=["asl", "lse"])
args = parser.parse_args()

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("❌ PyTorch not installed.")
    sys.exit(1)

import numpy as np                                      # noqa: F811
import torch                                            # noqa: F811
import torch.nn as nn                                   # noqa: F811
import torch.optim as optim                             # noqa: F811
from torch.utils.data import DataLoader, TensorDataset # noqa: F811

from sign_model import (
    _I3DNet, _MLP, I3D_LSE, MODEL_LSE, load_kinetics_weights,
    DEVICE, INPUT_DIM, sequence_to_feature,
    _LANG_CLIPS, data_path,
)

DATA_DIR          = HERE / "training_data"
FEEDBACK_CLIPS    = DATA_DIR / f"clips_{args.lang}.npz"
FEEDBACK_SAMPLES  = DATA_DIR / f"samples_{args.lang}.npz"


def show_status():
    print(f"\n  Feedback data for {args.lang.upper()}:")
    if FEEDBACK_CLIPS.exists():
        d = np.load(FEEDBACK_CLIPS, allow_pickle=True)
        labels, counts = np.unique(d["y"], return_counts=True)
        print(f"  I3D clips  : {len(d['X'])} total")
        for lbl, cnt in zip(labels, counts):
            print(f"    {lbl:<16} {cnt:>4} clips")
    else:
        print(f"  I3D clips  : none (no feedback collected yet)")

    if FEEDBACK_SAMPLES.exists():
        d = np.load(FEEDBACK_SAMPLES, allow_pickle=True)
        labels, counts = np.unique(d["y"], return_counts=True)
        print(f"  MLP samples: {len(d['X'])} total")
        for lbl, cnt in zip(labels, counts):
            print(f"    {lbl:<16} {cnt:>4} samples")
    else:
        print(f"  MLP samples: none (no feedback collected yet)")
    print()


def train_i3d_on_feedback():
    if not FEEDBACK_CLIPS.exists():
        print("  I3D: no feedback clips yet — collect via /feedback endpoint")
        return

    d      = np.load(FEEDBACK_CLIPS, allow_pickle=True)
    clips  = list(d["X"])
    y      = list(d["y"].astype(str))
    labels = sorted(set(y))

    if len(labels) < 2:
        print(f"  I3D: need ≥2 signs in feedback (have {labels}) — skipping")
        return

    print(f"  I3D: {len(clips)} feedback clips, {len(labels)} signs: {labels}")

    # Load existing weights if labels match
    i3d_path = str(_LANG_CLIPS.get(args.lang, DATA_DIR / f"i3d_{args.lang}.pt"))
    i3d_path = str(DATA_DIR / f"i3d_{args.lang}.pt")
    net      = _I3DNet(len(labels)).to(DEVICE)
    load_kinetics_weights(net)

    if os.path.exists(i3d_path):
        try:
            ck = torch.load(i3d_path, map_location=DEVICE, weights_only=False)
            if ck.get("labels") == labels:
                net.load_state_dict(ck["state_dict"])
                print("  I3D: continuing from existing weights")
            else:
                print("  I3D: label set changed — starting fresh")
        except Exception as e:
            print(f"  I3D: could not load weights ({e}) — starting fresh")

    from sign_model import I3D_FRAMES, I3D_SIZE, frames_to_tensor

    class FeedbackClipDataset(torch.utils.data.Dataset):
        def __init__(self, clips, labels):
            lbl2idx   = {l: i for i, l in enumerate(labels)}
            self.data = [(c, lbl2idx[l]) for c, l in zip(clips, y) if l in lbl2idx]

        def __len__(self): return len(self.data)

        def __getitem__(self, idx):
            frames, label = self.data[idx]
            return frames_to_tensor(list(frames)).squeeze(0), label

    ds      = FeedbackClipDataset(clips, labels)
    dl      = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    opt     = optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
    sch     = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.i3d_epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    net.train()
    for ep in range(1, args.i3d_epochs + 1):
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
        pct = int(ep / args.i3d_epochs * 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  I3D [{bar}] {pct:3d}%  ep {ep}/{args.i3d_epochs}"
              f"  loss={ep_loss/max(1,len(dl)):.3f}", end="", flush=True)

    print()
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in DataLoader(ds, batch_size=8):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            correct += (net(xb).argmax(1) == yb).sum().item()
            total   += len(yb)
    acc = correct / max(1, total)

    torch.save({"labels": labels, "state_dict": net.state_dict(),
                "arch": "i3d"}, i3d_path)
    print(f"  I3D saved — {acc*100:.1f}% acc → {i3d_path}")


def train_mlp_on_feedback():
    if not FEEDBACK_SAMPLES.exists():
        print("  MLP: no feedback samples yet — collect via /feedback endpoint")
        return

    d      = np.load(FEEDBACK_SAMPLES, allow_pickle=True)
    X      = d["X"].astype(np.float32)
    y      = d["y"].astype(str)
    labels = sorted(set(y.tolist()))

    if len(labels) < 2:
        print(f"  MLP: need ≥2 signs in feedback (have {labels}) — skipping")
        return

    print(f"  MLP: {len(X)} feedback samples, {len(labels)} signs: {labels}")

    mlp_path = str(DATA_DIR / f"model_{args.lang}.pt")
    net      = _MLP(len(labels)).to(DEVICE)

    if os.path.exists(mlp_path):
        try:
            ck = torch.load(mlp_path, map_location=DEVICE, weights_only=False)
            if ck.get("labels") == labels:
                net.load_state_dict(ck["state_dict"])
                print("  MLP: continuing from existing weights")
            else:
                print("  MLP: label set changed — starting fresh")
        except Exception as e:
            print(f"  MLP: could not load weights ({e}) — starting fresh")

    lbl2idx = {l: i for i, l in enumerate(labels)}
    Xt = torch.tensor(X).to(DEVICE)
    yt = torch.tensor([lbl2idx[l] for l in y]).to(DEVICE)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=True)

    opt     = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=2e-4)
    sch     = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.mlp_epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    net.train()
    for ep in range(1, args.mlp_epochs + 1):
        ep_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        sch.step()
        pct = int(ep / args.mlp_epochs * 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  MLP [{bar}] {pct:3d}%  ep {ep}/{args.mlp_epochs}"
              f"  loss={ep_loss/max(1,len(dl)):.3f}", end="", flush=True)

    print()
    net.eval()
    with torch.no_grad():
        acc = (net(Xt).argmax(1) == yt).float().mean().item()

    torch.save({"labels": labels, "state_dict": net.state_dict(),
                "arch": "mlp"}, mlp_path)
    print(f"  MLP saved — {acc*100:.1f}% acc → {mlp_path}")


def main():
    print("=" * 60)
    print(f"  SignFuture — Feedback Trainer ({args.lang.upper()})")
    print("=" * 60)

    show_status()

    if args.status:
        return

    if not args.no_i3d:
        train_i3d_on_feedback()

    if not args.no_mlp:
        train_mlp_on_feedback()

    print()
    print("  Restart server to load new weights: bash run.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
