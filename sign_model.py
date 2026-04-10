"""sign_model.py — Dual-language sign language classifier.

PRIMARY MODEL: I3D (Inflated 3D ConvNet)
  Takes raw video clips as input (T × H × W × 3 frames).
  Learns spatiotemporal features directly from pixel data.
  Far better than MLP for signs with complex motion dynamics.

FALLBACK MODEL: MLP (landmark-based)
  Used automatically when I3D model file doesn't exist yet.
  Keeps the server functional during early stages before enough
  training data has been collected for I3D.

Architecture (I3D):
  Backbone : Inflated 3D ConvNet (built from scratch, trained locally)
  Input    : T=16 frames, 112×112 px, RGB normalised
  Stem     : Conv3d 7×7×7 → MaxPool → 5 Inception3D blocks
  Head     : GlobalAvgPool → Dropout(0.5) → Linear(640 → N_classes)
  Output   : Softmax over N sign classes

Two separate model files per architecture:
  training_data/i3d_asl.pt     — I3D English ASL  (primary)
  training_data/i3d_lse.pt     — I3D Spanish LSE  (primary)
  training_data/model_asl.pt   — MLP ASL fallback
  training_data/model_lse.pt   — MLP LSE fallback

The server always tries I3D first. If unavailable it falls back to MLP.
Both are trained independently and hot-swapped on language switch.

PyTorch is REQUIRED:
  pip install torch torchvision
  CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
"""

import os
import time
import threading
import numpy as np

# ── PyTorch ───────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    raise ImportError(
        "\n[sign_model] PyTorch is required.\n"
        "  pip install torch torchvision\n"
        "  CUDA build: pip install torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu121\n"
    )

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = "training_data"
I3D_ASL     = os.path.join(DATA_DIR, "i3d_asl.pt")
I3D_LSE     = os.path.join(DATA_DIR, "i3d_lse.pt")
MODEL_ASL   = os.path.join(DATA_DIR, "model_asl.pt")   # MLP fallback
MODEL_LSE   = os.path.join(DATA_DIR, "model_lse.pt")   # MLP fallback
SAMPLES_ASL = os.path.join(DATA_DIR, "samples_asl.npz")
SAMPLES_LSE = os.path.join(DATA_DIR, "samples_lse.npz")
CLIPS_ASL   = os.path.join(DATA_DIR, "clips_asl.npz")
CLIPS_LSE   = os.path.join(DATA_DIR, "clips_lse.npz")

# Legacy aliases kept so dataset_download.py imports don't break
MODEL_PATH  = MODEL_ASL
DATA_PATH   = SAMPLES_ASL

# ── I3D hyperparameters ───────────────────────────────────────────────────────
I3D_FRAMES   = 16       # frames per clip fed to I3D
I3D_SIZE     = 112      # spatial resolution (112×112 px)
I3D_CHANNELS = 3        # RGB

# ── MLP hyperparameters (fallback) ────────────────────────────────────────────
INPUT_DIM = 278

# ── Shared ────────────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.55

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[sign_model] Device: {DEVICE}"
      + (f" — {torch.cuda.get_device_name(0)}" if DEVICE.type == "cuda" else ""))


# ═══════════════════════════════════════════════════════════════════════════════
# I3D ARCHITECTURE — full Kinetics-compatible (matches piergiaj/pytorch-i3d)
# ═══════════════════════════════════════════════════════════════════════════════

# Path to Kinetics pretrained weights (rgb_imagenet.pt from piergiaj/pytorch-i3d)
# Set this to wherever you downloaded the file
KINETICS_WEIGHTS = os.path.join(
    os.path.expanduser("~"), "Downloads", "pytorch-i3d", "models", "rgb_imagenet.pt"
)


class _Unit3D(nn.Module):
    """Conv3d + BatchNorm + ReLU — matches the Unit3D in piergiaj/pytorch-i3d."""
    def __init__(self, in_ch, out_ch, kernel=(1,1,1), stride=(1,1,1),
                 padding=0, bias=False, activate=True):
        super().__init__()
        self.conv     = nn.Conv3d(in_ch, out_ch, kernel, stride=stride,
                                  padding=padding, bias=bias)
        self.bn       = nn.BatchNorm3d(out_ch, eps=0.001, momentum=0.01)
        self.activate = activate

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True) if self.activate else x


class _InceptionBlock3D(nn.Module):
    """Single Inception-3D block — matches Mixed_* blocks in piergiaj naming."""
    def __init__(self, in_ch, b0, b1r, b1, b2r, b2, bp):
        super().__init__()
        self.b0 = _Unit3D(in_ch, b0)
        self.b1 = nn.Sequential(
            _Unit3D(in_ch, b1r),
            _Unit3D(b1r, b1, kernel=(3,3,3), padding=1))
        self.b2 = nn.Sequential(
            _Unit3D(in_ch, b2r),
            _Unit3D(b2r, b2, kernel=(3,3,3), padding=1))
        self.b3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            _Unit3D(in_ch, bp))
        self.out_ch = b0 + b1 + b2 + bp

    def forward(self, x):
        return torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], dim=1)


class _I3DNet(nn.Module):
    """
    Full I3D architecture matching piergiaj/pytorch-i3d exactly.
    This allows loading Kinetics pretrained weights directly.

    Input:  (B, 3, T, H, W)
    Output: (B, n_classes) — raw logits

    All Mixed_* blocks match channel sizes from rgb_imagenet.pt.
    The logits layer is replaced with a new head for n_classes.
    """

    def __init__(self, n_classes: int, dropout: float = 0.5):
        super().__init__()

        # ── Stem (Conv3d_1a → Conv3d_2b → Conv3d_2c) ─────────────────────────
        self.Conv3d_1a_7x7 = _Unit3D(3,   64, kernel=(7,7,7), stride=(2,2,2), padding=(3,3,3))
        self.MaxPool3d_2a_3x3 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.Conv3d_2b_1x1 = _Unit3D(64,  64)
        self.Conv3d_2c_3x3 = _Unit3D(64, 192, kernel=(3,3,3), padding=1)
        self.MaxPool3d_3a_3x3 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        # ── Mixed_3b, Mixed_3c ────────────────────────────────────────────────
        self.Mixed_3b = _InceptionBlock3D(192,  64,  96, 128,  16,  32,  32)  # → 256
        self.Mixed_3c = _InceptionBlock3D(256, 128, 128, 192,  32,  96,  64)  # → 480
        self.MaxPool3d_4a_3x3 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))

        # ── Mixed_4b … Mixed_4f ───────────────────────────────────────────────
        self.Mixed_4b = _InceptionBlock3D(480, 192,  96, 208,  16,  48,  64)  # → 512
        self.Mixed_4c = _InceptionBlock3D(512, 160, 112, 224,  24,  64,  64)  # → 512
        self.Mixed_4d = _InceptionBlock3D(512, 128, 128, 256,  24,  64,  64)  # → 512
        self.Mixed_4e = _InceptionBlock3D(512, 112, 144, 288,  32,  64,  64)  # → 528
        self.Mixed_4f = _InceptionBlock3D(528, 256, 160, 320,  32, 128, 128)  # → 832
        self.MaxPool3d_5a_2x2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), padding=0)

        # ── Mixed_5b, Mixed_5c ────────────────────────────────────────────────
        self.Mixed_5b = _InceptionBlock3D(832, 256, 160, 320,  32, 128, 128)  # → 832
        self.Mixed_5c = _InceptionBlock3D(832, 384, 192, 384,  48, 128, 128)  # → 1024

        # ── Head ─────────────────────────────────────────────────────────────
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout  = nn.Dropout(dropout)
        self.logits   = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.MaxPool3d_3a_3x3(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        x = self.MaxPool3d_4a_3x3(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        x = self.MaxPool3d_5a_2x2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.logits(x)


def load_kinetics_weights(model: _I3DNet, weights_path: str = KINETICS_WEIGHTS) -> bool:
    """
    Load Kinetics pretrained weights from piergiaj/pytorch-i3d into _I3DNet.

    The piergiaj weights use a different internal naming for Unit3D layers:
      piergiaj:  Mixed_4d.b0.conv3d.weight  →  ours: Mixed_4d.b0.conv.weight
      piergiaj:  Mixed_4d.b0.bn.weight      →  ours: Mixed_4d.b0.bn.weight  (same)
      piergiaj:  Conv3d_1a_7x7.conv3d.weight → ours: Conv3d_1a_7x7.conv.weight
      piergiaj:  logits.conv3d.weight        → skip (different n_classes)

    Returns True if weights loaded successfully.
    """
    if not os.path.exists(weights_path):
        print(f"[sign_model] Kinetics weights not found at {weights_path} — training from scratch")
        return False

    try:
        pretrained = torch.load(weights_path, map_location="cpu", weights_only=False)
        model_sd   = model.state_dict()
        mapped     = {}
        skipped    = []

        for k, v in pretrained.items():
            # Skip the final logits layer — wrong number of classes
            if k.startswith("logits"):
                skipped.append(k)
                continue

            new_k = k
            # conv3d → conv (our _Unit3D uses self.conv not self.conv3d)
            new_k = new_k.replace(".conv3d.", ".conv.")
            # b1a → b1.0  (first Unit3D in branch 1 sequential)
            # b1b → b1.1  (second Unit3D in branch 1 sequential)
            # b2a → b2.0
            # b2b → b2.1
            new_k = new_k.replace(".b1a.", ".b1.0.")
            new_k = new_k.replace(".b1b.", ".b1.1.")
            new_k = new_k.replace(".b2a.", ".b2.0.")
            new_k = new_k.replace(".b2b.", ".b2.1.")
            # b3b → b3.1 (b3.0 is MaxPool, b3.1 is the Unit3D)
            new_k = new_k.replace(".b3b.", ".b3.1.")

            if new_k in model_sd and model_sd[new_k].shape == v.shape:
                mapped[new_k] = v
            else:
                skipped.append(k)

        model_sd.update(mapped)
        model.load_state_dict(model_sd)
        print(f"[sign_model] Kinetics weights loaded — "
              f"{len(mapped)}/{len(pretrained)} tensors transferred "
              f"({len(skipped)} skipped: logits + mismatches)")
        return True

    except Exception as e:
        print(f"[sign_model] Kinetics weight loading failed: {e} — training from scratch")
        return False


# ── Video preprocessing ───────────────────────────────────────────────────────

# ImageNet normalisation constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def frames_to_tensor(frames: list) -> torch.Tensor:
    """
    List of HxWx3 uint8 BGR numpy arrays (from cv2) →
    (1, 3, T, 112, 112) float32 tensor ready for I3D.

    Handles variable-length input by subsampling or padding to I3D_FRAMES.
    Converts BGR → RGB, resizes to 112×112, normalises with ImageNet stats.
    """
    import cv2 as _cv2

    n = len(frames)
    if n == 0:
        return torch.zeros(1, I3D_CHANNELS, I3D_FRAMES, I3D_SIZE, I3D_SIZE)

    # Subsample or pad to exactly I3D_FRAMES
    if n >= I3D_FRAMES:
        idx = np.linspace(0, n - 1, I3D_FRAMES).astype(int)
        sel = [frames[i] for i in idx]
    else:
        reps = (I3D_FRAMES // n) + 1
        sel  = (frames * reps)[:I3D_FRAMES]

    out = []
    for f in sel:
        # BGR → RGB
        f = _cv2.cvtColor(f, _cv2.COLOR_BGR2RGB) if f.shape[2] == 3 else f
        f = _cv2.resize(f, (I3D_SIZE, I3D_SIZE))
        f = f.astype(np.float32) / 255.0
        f = (f - _MEAN) / _STD
        out.append(f)

    arr = np.stack(out, axis=0)            # (T, H, W, 3)
    arr = arr.transpose(3, 0, 1, 2)        # (3, T, H, W)
    return torch.tensor(arr).unsqueeze(0)  # (1, 3, T, H, W)


# ── Training dataset ──────────────────────────────────────────────────────────

class _ClipDataset(Dataset):
    """Dataset of raw video clips: list of (frames_list, label_int) pairs."""

    def __init__(self, clips: list, augment: bool = False):
        self.clips   = clips
        self.augment = augment

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        frames, label = self.clips[idx]
        t = frames_to_tensor(frames).squeeze(0)  # (3, T, H, W)
        if self.augment:
            t = _augment_clip(t)
        return t, label


def _augment_clip(t: torch.Tensor) -> torch.Tensor:
    """Random horizontal flip + brightness jitter."""
    if torch.rand(1) > 0.5:
        t = torch.flip(t, dims=[3])
    t = (t + (torch.rand(1) - 0.5) * 0.2).clamp(-2.5, 2.5)
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# I3D CLASSIFIER — primary model
# ═══════════════════════════════════════════════════════════════════════════════

class I3DClassifier:
    """
    I3D-based classifier. Takes raw video frames (BGR numpy arrays from cv2).
    One instance per language. API is identical to SignClassifier (MLP).

    Training input:
      clips : list of frame lists — each item is a list of HxWx3 uint8 BGR arrays
      y     : list of label strings, same length as clips

    Inference input:
      frames: list of HxWx3 uint8 BGR arrays (extract from webcam/video with cv2)
    """

    def __init__(self, labels: list, pretrained: bool = True):
        self.labels   = sorted(set(labels))
        self.n        = len(self.labels)
        self._lbl2idx = {lbl: i for i, lbl in enumerate(self.labels)}
        self._net     = _I3DNet(self.n).to(DEVICE)
        if pretrained:
            load_kinetics_weights(self._net)

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self,
              clips: list,
              y: list,
              epochs: int = 60,
              lr: float = 1e-4,
              batch_size: int = 4,
              progress_cb=None) -> dict:
        """
        lr=1e-4 for fine-tuning pretrained Kinetics weights (lower than scratch training).
        CosineAnnealingLR + gradient clipping for stable fine-tuning.
        Label smoothing 0.1 helps generalisation with small datasets.
        """
        t0 = time.time()

        paired = [(frames, self._lbl2idx[lbl])
                  for frames, lbl in zip(clips, y)
                  if lbl in self._lbl2idx]
        if not paired:
            raise ValueError("No valid clips found — check labels match dictionary")

        ds = _ClipDataset(paired, augment=True)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=(DEVICE.type == "cuda"))

        opt     = optim.AdamW(self._net.parameters(), lr=lr, weight_decay=1e-4)
        sch     = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        self._net.train()
        for ep in range(1, epochs + 1):
            ep_loss = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(self._net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item()
            sch.step()
            if progress_cb and ep % max(1, epochs // 20) == 0:
                progress_cb(int(ep / epochs * 88),
                            f"Epoch {ep}/{epochs}  "
                            f"loss={ep_loss / max(1, len(dl)):.3f}")

        # Final accuracy on training set
        self._net.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in DataLoader(ds, batch_size=batch_size):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                correct += (self._net(xb).argmax(1) == yb).sum().item()
                total   += len(yb)
        acc = correct / max(1, total)
        if progress_cb:
            progress_cb(92, f"Train accuracy: {acc * 100:.1f}%")

        return {"accuracy": acc, "epochs": epochs,
                "time_s": round(time.time() - t0, 1)}

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, frames: list) -> tuple:
        """frames: list of HxWx3 uint8 BGR numpy arrays → (label|None, confidence)"""
        self._net.eval()
        t = frames_to_tensor(frames).to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(self._net(t), dim=1)[0].cpu().numpy()
        idx  = int(np.argmax(probs))
        conf = float(probs[idx])
        return (self.labels[idx], conf) if conf >= CONF_THRESHOLD else (None, conf)

    def predict_sequence(self, frames: list) -> tuple:
        """Same as predict — alias so web_bridge.py works without changes."""
        return self.predict(frames)

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "labels":     self.labels,
            "state_dict": self._net.state_dict(),
            "arch":       "i3d",
            "frames":     I3D_FRAMES,
            "size":       I3D_SIZE,
        }, path)
        print(f"[sign_model] I3D saved — {self.n} classes → {path}")

    @classmethod
    def load(cls, path: str) -> "I3DClassifier":
        ck  = torch.load(path, map_location=DEVICE, weights_only=False)
        obj = cls(ck["labels"])
        obj._net.load_state_dict(ck["state_dict"])
        obj._net.to(DEVICE)
        obj._net.eval()
        print(f"[sign_model] I3D loaded — {obj.n} classes: {obj.labels}")
        return obj


# ═══════════════════════════════════════════════════════════════════════════════
# MLP CLASSIFIER — fallback only
# ═══════════════════════════════════════════════════════════════════════════════

class _MLP(nn.Module):
    """
    MLP classifier for 278-dim landmark vectors.

    Uses LayerNorm instead of BatchNorm1d deliberately:
    - BatchNorm1d normalises ACROSS the batch — with augmented near-identical
      samples the batch variance approaches zero on GPU, causing NaN loss
      (confirmed PyTorch cu124 issue with extreme input values)
    - LayerNorm normalises ACROSS features within each sample — immune to
      batch composition, works correctly with batch_size=1, no NaN risk
    """
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(256, 128),       nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128,  64),       nn.LayerNorm(64),  nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def sequence_to_feature(frames: list, target_len: int = 30) -> np.ndarray:
    """Variable-length landmark frame list → single 278-dim feature (MLP only)."""
    if not frames:
        return np.zeros(INPUT_DIM, dtype=np.float32)
    arr = np.array(frames, dtype=np.float32)
    if len(arr) >= target_len:
        idx = np.linspace(0, len(arr) - 1, target_len).astype(int)
        arr = arr[idx]
    else:
        reps = (target_len // len(arr)) + 1
        arr  = np.tile(arr, (reps, 1))[:target_len]
    return arr.mean(axis=0).astype(np.float32)


class SignClassifier:
    """
    MLP landmark-based fallback. Identical public API to I3DClassifier.
    NOTE: predict_sequence() here expects landmark frame lists (278-dim vectors),
          NOT raw BGR frames. web_bridge.py feeds the right data automatically
          based on which classifier is active.
    """

    def __init__(self, labels: list):
        self.labels   = sorted(set(labels))
        self.n        = len(self.labels)
        self._lbl2idx = {lbl: i for i, lbl in enumerate(self.labels)}
        self._net     = _MLP(self.n).to(DEVICE)

    def train(self, X: np.ndarray, y: list,
              epochs: int = 300, lr: float = 1e-3,
              batch_size: int = 32, progress_cb=None) -> dict:
        from torch.utils.data import TensorDataset
        t0  = time.time()
        X   = np.array(X, dtype=np.float32)

        # Clip extreme values — landmark coords outside ±10 cause NaN in LayerNorm on GPU
        X   = np.clip(X, -10.0, 10.0)

        yi  = np.array([self._lbl2idx[lbl] for lbl in y], dtype=np.int64)
        Xt  = torch.tensor(X).to(DEVICE)
        yt  = torch.tensor(yi).to(DEVICE)
        dl  = DataLoader(TensorDataset(Xt, yt),
                         batch_size=batch_size, shuffle=True, drop_last=False)
        opt     = optim.AdamW(self._net.parameters(), lr=lr, weight_decay=2e-4)
        sch     = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
        self._net.train()
        for ep in range(1, epochs + 1):
            for xb, yb in dl:
                opt.zero_grad()
                loss = loss_fn(self._net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                opt.step()
            sch.step()
            if progress_cb and ep % 30 == 0:
                progress_cb(int(ep / epochs * 88), f"Epoch {ep}/{epochs}")
        self._net.eval()
        with torch.no_grad():
            acc = (self._net(Xt).argmax(1) == yt).float().mean().cpu().item()
        if progress_cb:
            progress_cb(92, f"Train accuracy: {acc * 100:.1f}%")
        return {"accuracy": acc, "epochs": epochs,
                "time_s": round(time.time() - t0, 1)}

    def predict(self, x: np.ndarray) -> tuple:
        self._net.eval()
        with torch.no_grad():
            probs = torch.softmax(
                self._net(torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)),
                dim=1)[0].cpu().numpy()
        idx  = int(np.argmax(probs))
        conf = float(probs[idx])
        return (self.labels[idx], conf) if conf >= CONF_THRESHOLD else (None, conf)

    def predict_sequence(self, frames: list) -> tuple:
        return self.predict(sequence_to_feature(frames))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"labels": self.labels,
                    "state_dict": self._net.state_dict(),
                    "arch": "mlp"}, path)
        print(f"[sign_model] MLP saved — {self.n} classes → {path}")

    @classmethod
    def load(cls, path: str) -> "SignClassifier":
        ck  = torch.load(path, map_location=DEVICE, weights_only=False)
        obj = cls(ck["labels"])
        obj._net.load_state_dict(ck["state_dict"])
        obj._net.to(DEVICE); obj._net.eval()
        print(f"[sign_model] MLP loaded — {obj.n} classes: {obj.labels}")
        return obj


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED LOADER — tries I3D first, falls back to MLP automatically
# ═══════════════════════════════════════════════════════════════════════════════

_LANG_I3D   = {"asl": I3D_ASL,     "lse": I3D_LSE}
_LANG_MLP   = {"asl": MODEL_ASL,   "lse": MODEL_LSE}
_LANG_DATA  = {"asl": SAMPLES_ASL, "lse": SAMPLES_LSE}
_LANG_CLIPS = {"asl": CLIPS_ASL,   "lse": CLIPS_LSE}

# I3D weight in fusion (MLP weight = 1 - I3D_WEIGHT).
# I3D captures motion better; MLP is background-proof.
# Overlap bonus amplifies any sign both models agree on.
I3D_WEIGHT    = 0.6
MLP_WEIGHT    = 0.4
OVERLAP_BONUS = 0.25   # added to combined score when both models agree on a sign
TOP_K         = 3      # how many candidates each model contributes


# ═══════════════════════════════════════════════════════════════════════════════
# FUSION CLASSIFIER — runs I3D + MLP simultaneously, merges top-3 each
# ═══════════════════════════════════════════════════════════════════════════════

class FusionClassifier:
    """
    Runs I3D and MLP in parallel. Each returns its top-3 predictions with
    confidence scores. Scores are weighted and summed per sign — any sign
    that appears in BOTH top-3 lists gets an overlap bonus, making agreement
    the strongest possible signal. The sign with the highest combined score wins.

    predict() requires BOTH raw frames AND landmark vectors simultaneously.
    web_bridge.py extracts both in one pass and passes them as a tuple.

    Falls back gracefully:
      - Only I3D available  → uses I3D alone (no fusion)
      - Only MLP available  → uses MLP alone (no fusion)
      - Both available      → full fusion
    """

    def __init__(self, i3d: I3DClassifier | None, mlp: "SignClassifier | None"):
        self.i3d    = i3d
        self.mlp    = mlp
        # Unified label set from whichever models are loaded
        labels_set  = set()
        if i3d: labels_set.update(i3d.labels)
        if mlp: labels_set.update(mlp.labels)
        self.labels = sorted(labels_set)
        self.n      = len(self.labels)

    # ── internal: get top-k (label, score) from a probability array ──────────

    @staticmethod
    def _top_k(probs: np.ndarray, labels: list, k: int) -> list:
        """Returns [(label, prob), ...] for the k highest-probability classes."""
        idx = np.argsort(probs)[::-1][:k]
        return [(labels[i], float(probs[i])) for i in idx]

    # ── internal: raw softmax probabilities from each model ──────────────────

    def _i3d_probs(self, frames: list) -> np.ndarray | None:
        if self.i3d is None or not frames:
            return None
        self.i3d._net.eval()
        t = frames_to_tensor(frames).to(DEVICE)
        with torch.no_grad():
            return F.softmax(self.i3d._net(t), dim=1)[0].cpu().numpy()

    def _mlp_probs(self, landmark_frames: list) -> np.ndarray | None:
        if self.mlp is None or not landmark_frames:
            return None
        feat = sequence_to_feature(landmark_frames)
        self.mlp._net.eval()
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.mlp._net(x), dim=1)[0].cpu().numpy()
        # If MLP produces NaN (bad weights), treat as unavailable
        if np.isnan(probs).any():
            print("[fusion] MLP produced NaN — falling back to I3D only (retrain MLP)")
            return None
        return probs

    # ── public predict ────────────────────────────────────────────────────────

    def predict(self, frames: list, landmark_frames: list) -> tuple:
        """
        Returns (label | None, confidence, debug)
        debug = {
            "i3d_top3": [(sign, pct), ...] or None,
            "mlp_top3": [(sign, pct), ...] or None,
            "overlap":  [sign, ...],
            "mode":     "fusion" | "i3d_only" | "mlp_only" | "none"
        }
        """
        i3d_probs = self._i3d_probs(frames)
        mlp_probs = self._mlp_probs(landmark_frames)

        def _fmt(top): return [(s, round(p*100, 1)) for s, p in top]

        # ── single-model fallback ─────────────────────────────────────────────
        if i3d_probs is None and mlp_probs is None:
            return None, 0.0, {"mode": "none", "i3d_top3": None, "mlp_top3": None, "overlap": []}

        if i3d_probs is None:
            top   = self._top_k(mlp_probs, self.mlp.labels, TOP_K)
            label, conf = top[0]
            debug = {"mode": "mlp_only", "i3d_top3": None, "mlp_top3": _fmt(top), "overlap": []}
            print(f"[fusion] MLP-only → {label} ({conf*100:.0f}%)")
            return (label if conf >= CONF_THRESHOLD else None), conf, debug

        if mlp_probs is None:
            top   = self._top_k(i3d_probs, self.i3d.labels, TOP_K)
            label, conf = top[0]
            debug = {"mode": "i3d_only", "i3d_top3": _fmt(top), "mlp_top3": None, "overlap": []}
            print(f"[fusion] I3D-only → {label} ({conf*100:.0f}%)")
            return (label if conf >= CONF_THRESHOLD else None), conf, debug

        # ── full fusion ───────────────────────────────────────────────────────
        i3d_top   = self._top_k(i3d_probs, self.i3d.labels, TOP_K)
        mlp_top   = self._top_k(mlp_probs, self.mlp.labels, TOP_K)
        i3d_signs = {sign for sign, _ in i3d_top}
        mlp_signs = {sign for sign, _ in mlp_top}
        overlap   = i3d_signs & mlp_signs

        scores: dict[str, float] = {}
        for sign, prob in i3d_top:
            scores[sign] = scores.get(sign, 0.0) + prob * I3D_WEIGHT
        for sign, prob in mlp_top:
            scores[sign] = scores.get(sign, 0.0) + prob * MLP_WEIGHT
        for sign in overlap:
            scores[sign] = scores.get(sign, 0.0) + OVERLAP_BONUS

        best_sign  = max(scores, key=scores.__getitem__)
        best_score = scores[best_sign]
        max_possible = I3D_WEIGHT + MLP_WEIGHT + OVERLAP_BONUS
        conf = min(best_score / max_possible, 1.0)

        debug = {
            "mode":     "fusion",
            "i3d_top3": _fmt(i3d_top),
            "mlp_top3": _fmt(mlp_top),
            "overlap":  sorted(overlap),
        }

        agreed = best_sign in overlap
        print(f"[fusion] I3D top3={[s for s,_ in i3d_top]}  "
              f"MLP top3={[s for s,_ in mlp_top]}  "
              f"→ {best_sign} ({conf*100:.0f}%){'  ✓overlap' if agreed else ''}")

        label = best_sign if conf >= CONF_THRESHOLD else None
        return label, conf, debug

    def predict_sequence(self, data) -> tuple:
        """
        Returns (label | None, confidence, debug).
        Accepts:
          - tuple (frames, landmark_frames) — fusion mode
          - list of BGR frames              — I3D only
          - list of 278-dim vectors         — MLP only
        """
        if isinstance(data, tuple) and len(data) == 2:
            return self.predict(data[0], data[1])
        # Legacy single-stream fallback
        if data and isinstance(data[0], np.ndarray) and data[0].ndim == 1:
            return self.predict([], data)   # landmark vectors → MLP only
        return self.predict(data, [])       # raw frames → I3D only


# ── Module-level singleton ────────────────────────────────────────────────────

_active_lang: str | None = None
_active_clf              = None
_active_arch: str | None = None   # "fusion", "i3d", or "mlp"
_model_lock = threading.Lock()


def _load_best(lang: str):
    """
    Load both I3D and MLP if available, wrap in FusionClassifier.
    Returns (FusionClassifier | single-model, arch_str) or (None, None).
    """
    i3d_path = _LANG_I3D.get(lang)
    mlp_path = _LANG_MLP.get(lang)

    i3d_clf = None
    mlp_clf = None

    if i3d_path and os.path.exists(i3d_path):
        try:
            i3d_clf = I3DClassifier.load(i3d_path)
        except Exception as err:
            print(f"[sign_model] I3D load failed: {err}")

    if mlp_path and os.path.exists(mlp_path):
        try:
            mlp_clf = SignClassifier.load(mlp_path)
        except Exception as err:
            print(f"[sign_model] MLP load failed: {err}")

    if i3d_clf and mlp_clf:
        clf  = FusionClassifier(i3d_clf, mlp_clf)
        arch = "fusion"
    elif i3d_clf:
        clf  = FusionClassifier(i3d_clf, None)
        arch = "i3d"
    elif mlp_clf:
        clf  = FusionClassifier(None, mlp_clf)
        arch = "mlp"
    else:
        return None, None

    print(f"[sign_model] Active model: {lang.upper()} via {arch.upper()}")
    return clf, arch


def load_model(lang: str):
    """Load + cache the best model for lang. Returns FusionClassifier or None."""
    global _active_lang, _active_clf, _active_arch
    lang = lang.lower()
    with _model_lock:
        if _active_lang == lang and _active_clf is not None:
            return _active_clf
        clf, arch = _load_best(lang)
        _active_clf  = clf
        _active_lang = lang
        _active_arch = arch
        return clf


def switch_language(lang: str):
    """Hot-swap the active model. Called by /switch-language endpoint."""
    global _active_lang, _active_clf, _active_arch
    with _model_lock:
        _active_clf  = None
        _active_lang = None
        _active_arch = None
    return load_model(lang)


def get_active():
    """Return the currently loaded classifier, or None."""
    return _active_clf


def active_arch() -> str | None:
    """Return 'fusion', 'i3d', 'mlp', or None."""
    return _active_arch


def model_ready(lang: str) -> bool:
    """True if any model (I3D or MLP) exists for this language."""
    return (os.path.exists(_LANG_I3D.get(lang, "")) or
            os.path.exists(_LANG_MLP.get(lang, "")))


def i3d_ready(lang: str) -> bool:
    """True specifically if the I3D model exists for this language."""
    return os.path.exists(_LANG_I3D.get(lang, ""))


def data_path(lang: str) -> str:
    return _LANG_DATA.get(lang.lower(), SAMPLES_ASL)


def model_path(lang: str) -> str:
    """Returns I3D path if it exists, otherwise MLP path."""
    i3d = _LANG_I3D.get(lang.lower(), I3D_ASL)
    return i3d if os.path.exists(i3d) else _LANG_MLP.get(lang.lower(), MODEL_ASL)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def save_samples(X: list, y: list, lang: str) -> int:
    """Append landmark samples to npz store (used by /collect-video)."""
    path = data_path(lang)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    X_new = np.array(X, dtype=np.float32)
    y_new = np.array(y)
    if os.path.exists(path):
        d     = np.load(path, allow_pickle=True)
        X_all = np.concatenate([d["X"], X_new])
        y_all = np.concatenate([d["y"], y_new])
    else:
        X_all, y_all = X_new, y_new
    np.savez(path, X=X_all, y=y_all)
    return len(X_all)


def load_samples(lang: str) -> tuple:
    path = data_path(lang)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No landmark data for '{lang}' at {path}.\n"
            f"Run: python dataset_download.py")
    d = np.load(path, allow_pickle=True)
    return d["X"], d["y"]


def sample_counts(lang: str) -> dict:
    path = data_path(lang)
    if not os.path.exists(path):
        return {}
    d = np.load(path, allow_pickle=True)
    labels, counts = np.unique(d["y"], return_counts=True)
    return {str(lbl): int(c) for lbl, c in zip(labels, counts)}


def delete_label_samples(label: str, lang: str) -> int:
    path = data_path(lang)
    if not os.path.exists(path):
        return 0
    d    = np.load(path, allow_pickle=True)
    mask = d["y"] != label
    np.savez(path, X=d["X"][mask], y=d["y"][mask])
    return int(mask.sum())


def clip_counts(lang: str) -> dict:
    """Number of raw video clips stored per sign label (for I3D training)."""
    path = _LANG_CLIPS.get(lang.lower())
    if not path or not os.path.exists(path):
        return {}
    d = np.load(path, allow_pickle=True)
    labels, counts = np.unique(d["y"], return_counts=True)
    return {str(lbl): int(c) for lbl, c in zip(labels, counts)}


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-TRAIN ON STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

def maybe_auto_train(lang: str) -> bool:
    """
    If no model exists but landmark data does, auto-train the MLP fallback.
    I3D is NEVER auto-trained — it needs raw video clips and takes hours.
    Trigger I3D training explicitly via dataset_download.py or /train endpoint.
    """
    mlp_p = _LANG_MLP.get(lang, "")
    i3d_p = _LANG_I3D.get(lang, "")
    dp    = data_path(lang)

    if os.path.exists(mlp_p) or os.path.exists(i3d_p):
        return False   # already have a model
    if not os.path.exists(dp):
        return False   # no training data at all

    print(f"[sign_model] No model for '{lang}', landmark data found — "
          f"auto-training MLP fallback (run dataset_download.py for I3D)…")
    try:
        X, y   = load_samples(lang)
        labels = sorted(set(y.tolist()))
        clf    = SignClassifier(labels)
        result = clf.train(X, y.tolist(), epochs=300)
        clf.save(mlp_p)
        print(f"[sign_model] MLP fallback trained: "
              f"{result['accuracy'] * 100:.1f}% in {result['time_s']}s")
        return True
    except Exception as err:
        print(f"[sign_model] Auto-train failed: {err}")
        return False