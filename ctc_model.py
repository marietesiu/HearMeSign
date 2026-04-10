"""ctc_model.py — Continuous Sign Language Recognition (CSLR) for ASL.

Architecture:
  MediaPipe landmarks (T × 278) per frame sequence
  → Linear projection (278 → 512)
  → Bidirectional LSTM × 2 layers (captures temporal flow + co-articulation)
  → Linear head (512 → vocab_size + 1 blank)
  → CTC Loss / CTC Decode

Two separate modes in the app:
  - Isolated word mode  : existing I3D + MLP fusion (unchanged)
  - Continuous mode     : this model, activated by /sign-to-text?mode=continuous

CTC blank label = 0 (index 0 reserved, all real signs start at index 1)

Training data:
  - Your existing landmark .npz files (short isolated clips — Stage 1)
  - How2Sign keypoints when available (continuous sentences — Stage 2+)

Run:
    python ctc_model.py --train          # train on available data
    python ctc_model.py --train --epochs 100
    python ctc_model.py --status         # show vocab + model info
"""

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import numpy as np
except ImportError:
    raise ImportError("PyTorch / numpy required. Run setup.sh first.")

import torch                                         # noqa: F811
import torch.nn as nn                                # noqa: F811
import torch.optim as optim                          # noqa: F811
import torch.nn.functional as F                      # noqa: F811
from torch.utils.data import DataLoader, Dataset     # noqa: F811
import numpy as np                                   # noqa: F811

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR        = HERE / "training_data"
CTC_MODEL_ASL   = str(DATA_DIR / "ctc_asl.pt")
CTC_VOCAB_FILE  = str(DATA_DIR / "ctc_vocab_asl.json")

# ── Hyperparameters ───────────────────────────────────────────────────────────
INPUT_DIM    = 278      # MediaPipe landmark vector size
HIDDEN_DIM   = 512      # BiLSTM hidden size
NUM_LAYERS   = 2        # BiLSTM depth
DROPOUT      = 0.3
BLANK_IDX    = 0        # CTC blank label index

try:
    import torch.cuda
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    DEVICE = torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# VOCABULARY
# ═══════════════════════════════════════════════════════════════════════════════

class CTCVocab:
    """
    Maps sign labels ↔ integer indices.
    Index 0 is always reserved for CTC blank.
    Real signs start at index 1.
    """
    def __init__(self, signs: list):
        self.signs   = sorted(set(signs))
        # blank=0, signs start at 1
        self.idx2sign = ["<blank>"] + self.signs
        self.sign2idx = {s: i+1 for i, s in enumerate(self.signs)}
        self.size     = len(self.idx2sign)   # includes blank

    def encode(self, sequence: list) -> list:
        """List of sign strings → list of int indices."""
        return [self.sign2idx[s] for s in sequence if s in self.sign2idx]

    def decode(self, indices: list) -> list:
        """CTC raw output → collapsed sign sequence (removes blanks + repeats)."""
        result = []
        prev   = None
        for idx in indices:
            if idx != BLANK_IDX and idx != prev:
                if idx < len(self.idx2sign):
                    result.append(self.idx2sign[idx])
            prev = idx
        return result

    def save(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump({"signs": self.signs}, f)
        print(f"[ctc_model] Vocab saved — {len(self.signs)} signs → {path}")

    @classmethod
    def load(cls, path: str) -> "CTCVocab":
        import json
        with open(path) as f:
            data = json.load(f)
        return cls(data["signs"])


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class CTCSignModel(nn.Module):
    """
    Bidirectional LSTM sequence model for continuous sign recognition.

    Input:  (T, B, 278) — sequence of landmark frames, batch second
    Output: (T, B, vocab_size) — log-softmax over vocabulary at each timestep

    The CTC decoder collapses this into a sign sequence by removing blanks
    and repeated labels.

    Designed to be frozen-backbone compatible:
    When How2Sign data is available, the projection + LSTM layers learn
    the sentence-level grammar while the landmark features remain the same
    format as the isolated word pipeline.
    """

    def __init__(self, vocab_size: int,
                 input_dim:  int = INPUT_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS,
                 dropout:    float = DROPOUT):
        super().__init__()

        # Project landmark vectors to model dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Bidirectional LSTM — captures forward and backward temporal context
        # Bidirectional doubles the output size: hidden_dim * 2
        self.lstm = nn.LSTM(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = False,    # (T, B, features)
            bidirectional = True,
            dropout = dropout if num_layers > 1 else 0.0,
        )

        # Output projection to vocabulary
        self.output_proj = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (T, B, input_dim) or (T, input_dim) for single sequence
        returns: (T, B, vocab_size) log-softmax
        """
        # Handle unbatched input
        unbatched = x.dim() == 2
        if unbatched:
            x = x.unsqueeze(1)   # (T, 1, input_dim)

        x = self.input_proj(x)         # (T, B, hidden_dim)
        x, _ = self.lstm(x)            # (T, B, hidden_dim*2)
        x = self.output_proj(x)        # (T, B, vocab_size)
        x = F.log_softmax(x, dim=-1)   # required by CTCLoss

        if unbatched:
            x = x.squeeze(1)   # (T, vocab_size)

        return x


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET — Stage 1: isolated landmark sequences
# ═══════════════════════════════════════════════════════════════════════════════

class IsolatedLandmarkDataset(Dataset):
    """
    Stage 1 training dataset — uses existing isolated word landmark data.

    Each sample is a single sign: a sequence of per-frame landmark vectors.
    The CTC target is a single-sign sequence [sign_idx].

    This lets the CTC model learn the visual appearance of each sign
    before being exposed to continuous sentences.

    Data format expected: training_data/samples_asl.npz
      X: (N, 278) — mean-pooled landmark vectors (one per clip)
      y: (N,)     — sign labels

    Since these are mean-pooled (single vector per clip), we treat each
    as a 1-frame sequence. When How2Sign per-frame data is available,
    this is replaced with actual frame sequences.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, vocab: CTCVocab):
        self.X     = X.astype(np.float32)
        self.y     = y
        self.vocab = vocab

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # (1, 278) — single frame treated as length-1 sequence
        x       = torch.tensor(self.X[idx]).unsqueeze(0)   # (1, 278)
        label   = self.vocab.encode([str(self.y[idx])])
        targets = torch.tensor(label, dtype=torch.long)
        return x, targets, torch.tensor(1), torch.tensor(len(targets))


def ctc_collate(batch):
    """Pad sequences to same length for batched CTC training."""
    xs, targets, input_lens, target_lens = zip(*batch)
    max_t   = max(x.shape[0] for x in xs)
    padded  = torch.zeros(max_t, len(xs), xs[0].shape[-1])
    for i, x in enumerate(xs):
        padded[:x.shape[0], i, :] = x
    targets     = torch.cat(targets)
    input_lens  = torch.stack(input_lens)
    target_lens = torch.stack(target_lens)
    return padded, targets, input_lens, target_lens


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER WRAPPER — same API as SignClassifier / I3DClassifier
# ═══════════════════════════════════════════════════════════════════════════════

class CTCClassifier:
    """
    CTC-based continuous sign recogniser.
    Public API matches SignClassifier so web_bridge.py can use it.

    predict_sequence(landmark_frames) → (text | None, confidence)
    where text is a space-joined sequence of recognised signs.
    """

    def __init__(self, signs: list):
        self.vocab  = CTCVocab(signs)
        self.labels = self.vocab.signs
        self._net   = CTCSignModel(self.vocab.size).to(DEVICE)

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 100, lr: float = 1e-3,
              progress_cb=None) -> dict:
        t0      = time.time()
        X       = np.clip(X.astype(np.float32), -10.0, 10.0)
        ds      = IsolatedLandmarkDataset(X, y, self.vocab)
        dl      = DataLoader(ds, batch_size=32, shuffle=True,
                             collate_fn=ctc_collate, drop_last=False)
        opt     = optim.AdamW(self._net.parameters(), lr=lr, weight_decay=1e-4)
        sch     = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        ctc     = nn.CTCLoss(blank=BLANK_IDX, reduction="mean",
                             zero_infinity=True)

        self._net.train()
        for ep in range(1, epochs + 1):
            ep_loss = 0.0
            for x_seq, targets, in_lens, tgt_lens in dl:
                x_seq   = x_seq.to(DEVICE)
                targets = targets.to(DEVICE)
                opt.zero_grad()
                logits  = self._net(x_seq)           # (T, B, vocab)
                loss    = ctc(logits, targets, in_lens, tgt_lens)
                if not torch.isnan(loss):
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                    opt.step()
                    ep_loss += loss.item()
            sch.step()
            if progress_cb and ep % max(1, epochs // 20) == 0:
                progress_cb(int(ep / epochs * 88), f"Epoch {ep}/{epochs}")

        if progress_cb:
            progress_cb(92, "Evaluating…")

        return {"epochs": epochs, "time_s": round(time.time() - t0, 1)}

    def predict_sequence(self, landmark_frames: list) -> tuple:
        """
        landmark_frames: list of 278-dim numpy arrays (one per frame)
        Returns: (space-joined sign string | None, confidence)
        """
        if not landmark_frames:
            return None, 0.0

        self._net.eval()
        arr = np.clip(np.array(landmark_frames, dtype=np.float32), -10.0, 10.0)
        x   = torch.tensor(arr).to(DEVICE)   # (T, 278)

        with torch.no_grad():
            logits = self._net(x)             # (T, vocab_size)
            probs  = torch.exp(logits)        # convert log_softmax → probs

        # Greedy CTC decode
        best_path = probs.argmax(dim=-1).cpu().tolist()   # (T,)
        signs     = self.vocab.decode(best_path)

        if not signs:
            return None, 0.0

        # Confidence = mean of max prob at each non-blank timestep
        non_blank_probs = [
            float(probs[t].max()) for t in range(len(best_path))
            if best_path[t] != BLANK_IDX
        ]
        conf = float(np.mean(non_blank_probs)) if non_blank_probs else 0.0

        text = " ".join(signs)
        return (text, conf) if conf >= 0.3 else (None, conf)

    def save(self, model_path: str, vocab_path: str = CTC_VOCAB_FILE):
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        torch.save({
            "signs":      self.vocab.signs,
            "state_dict": self._net.state_dict(),
            "arch":       "ctc_bilstm",
        }, model_path)
        self.vocab.save(vocab_path)
        print(f"[ctc_model] Saved → {model_path}")

    @classmethod
    def load(cls, model_path: str) -> "CTCClassifier":
        ck  = torch.load(model_path, map_location=DEVICE, weights_only=False)
        obj = cls(ck["signs"])
        obj._net.load_state_dict(ck["state_dict"])
        obj._net.to(DEVICE)
        obj._net.eval()
        print(f"[ctc_model] Loaded — {len(ck['signs'])} signs: {ck['signs']}")
        return obj


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE TRAINING SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════

def _train_standalone():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",   action="store_true")
    parser.add_argument("--status",  action="store_true")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--lr",      type=float, default=1e-3)
    args = parser.parse_args()

    samples_path = DATA_DIR / "samples_asl.npz"

    if args.status:
        print(f"\n  CTC model status:")
        print(f"  Model    : {'✓' if os.path.exists(CTC_MODEL_ASL) else '✗ not trained'}")
        print(f"  Vocab    : {'✓' if os.path.exists(CTC_VOCAB_FILE) else '✗ not found'}")
        print(f"  ASL data : {'✓' if samples_path.exists() else '✗ run download_msasl.py'}")
        if samples_path.exists():
            d = np.load(samples_path, allow_pickle=True)
            from collections import Counter
            counts = Counter(d["y"].tolist())
            print(f"  Samples  : {len(d['X'])} across {len(counts)} signs")
        return

    if not args.train:
        print("Usage: python ctc_model.py --train [--epochs N] [--lr F]")
        print("       python ctc_model.py --status")
        return

    if not samples_path.exists():
        print(f"❌ No ASL landmark data at {samples_path}")
        print("   Run: python download_msasl.py")
        return

    d      = np.load(samples_path, allow_pickle=True)
    X, y   = d["X"], d["y"]
    signs  = sorted(set(y.tolist()))

    print("=" * 60)
    print("  SignFuture — CTC Continuous ASL Trainer (Stage 1)")
    print(f"  Samples  : {len(X)}")
    print(f"  Signs    : {len(signs)}")
    print(f"  Device   : {DEVICE}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Note     : Stage 1 uses isolated clips as length-1 sequences.")
    print(f"             Add How2Sign keypoints for continuous sentence training.")
    print("=" * 60)

    clf = CTCClassifier(signs)

    def cb(pct, msg):
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%  {msg}", end="", flush=True)

    result = clf.train(X, y, epochs=args.epochs, lr=args.lr, progress_cb=cb)
    print()
    print(f"  ✅ Done in {result['time_s']}s")
    clf.save(CTC_MODEL_ASL)
    print(f"\n  Next step: integrate with web_bridge.py using ?mode=continuous")
    print("=" * 60)


if __name__ == "__main__":
    _train_standalone()
