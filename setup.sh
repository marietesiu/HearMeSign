#!/bin/bash
# SignFuture — Full Setup Script
# Arch Linux / Manjaro, RTX 3050 Laptop GPU
#
# Run from your project folder:
#   cd ~/HearMeSign && bash setup.sh
#
# This script:
#   1. Installs NVIDIA drivers + CUDA (if not already present)
#   2. Installs system dependencies
#   3. Sets up pyenv + Python 3.11.9
#   4. Creates the .venv virtual environment
#   5. Installs ALL Python dependencies (including PyTorch CUDA)
#   6. Downloads MediaPipe model files (hand + pose)
#   7. Creates required project folders
#   8. Creates required project folders
#   9. Configures mDNS (avahi) for phone connectivity
#  10. Installs systemd service (Flask auto-starts on boot)

set -e

echo "════════════════════════════════════════════════════════"
echo "  SignFuture — Arch Linux Setup"
echo "  Machine: RTX 3050 Laptop (CUDA 12.1)"
echo "════════════════════════════════════════════════════════"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
echo "  Project: $PROJECT_DIR"
echo ""

# ── Sudo detection ────────────────────────────────────────────────────────────
if [ "$EUID" -eq 0 ]; then
  SUDO=""
elif command -v sudo &>/dev/null; then
  SUDO="sudo"
else
  echo "❌ Need root or sudo. Run: su -c 'bash setup.sh'"
  exit 1
fi

# ── 1. NVIDIA + CUDA (skip if already installed) ──────────────────────────────
echo "[1/8] Checking NVIDIA drivers..."
if ! command -v nvidia-smi &>/dev/null; then
  echo "  Installing nvidia-open + CUDA..."
  $SUDO pacman -S --needed --noconfirm nvidia-open nvidia-utils cuda cudnn
else
  echo "  ✓ nvidia-smi found: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
fi

# ── 2. System packages ────────────────────────────────────────────────────────
echo ""
echo "[2/8] Installing system packages..."
$SUDO pacman -S --needed --noconfirm \
  python python-pip git ffmpeg \
  libgl glib2 x264 mesa \
  base-devel openssl zlib xz tk libffi \
  pyenv \
  avahi nss-mdns

# ── 3. pyenv setup ────────────────────────────────────────────────────────────
echo ""
echo "[3/8] Configuring pyenv..."
if ! grep -q "pyenv init" ~/.bashrc; then
  cat >> ~/.bashrc << 'EOF'

# pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
EOF
  echo "  Added pyenv to ~/.bashrc"
fi
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# ── 4. Python 3.11.9 ──────────────────────────────────────────────────────────
echo ""
echo "[4/8] Python 3.11.9 via pyenv..."
pyenv install 3.11.9 -s
PYTHON="$HOME/.pyenv/versions/3.11.9/bin/python"
echo "  ✓ $($PYTHON --version)"

# ── 5. Virtual environment ────────────────────────────────────────────────────
echo ""
echo "[5/8] Creating virtual environment (.venv)..."
if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "  Deactivating existing venv: $VIRTUAL_ENV"
  deactivate 2>/dev/null || true
fi
$PYTHON -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet
echo "  ✓ .venv created with $(python --version)"

# ── 6. Python packages ────────────────────────────────────────────────────────
echo ""
echo "[6/8] Installing Python packages..."

pip install --quiet \
  flask flask-cors \
  mediapipe opencv-python \
  numpy scipy scikit-learn pyarrow huggingface_hub \
  gtts \
  yt-dlp requests

# PyTorch — cu124 works on BOTH RTX 3050 (laptop) and RTX 3080 (desktop)
echo ""
echo "  Installing PyTorch CUDA 12.4..."
pip install --quiet \
  torch torchvision \
  --index-url https://download.pytorch.org/whl/cu124

echo "  ✓ All Python packages installed"

# ── 7. MediaPipe model files ──────────────────────────────────────────────────
echo ""
echo "[7/8] Downloading MediaPipe model files..."
mkdir -p models

HAND_MODEL="models/hand_landmarker.task"
POSE_MODEL="models/pose_landmarker_lite.task"

if [ ! -f "$HAND_MODEL" ]; then
  echo "  Downloading hand_landmarker.task (~8 MB)..."
  curl -sL "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" \
    -o "$HAND_MODEL"
  echo "  ✓ hand_landmarker.task"
else
  echo "  ✓ hand_landmarker.task already present"
fi

if [ ! -f "$POSE_MODEL" ]; then
  echo "  Downloading pose_landmarker_lite.task (~5 MB)..."
  curl -sL "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" \
    -o "$POSE_MODEL"
  echo "  ✓ pose_landmarker_lite.task"
else
  echo "  ✓ pose_landmarker_lite.task already present"
fi

# ── 8. Project folders ────────────────────────────────────────────────────────
echo ""
echo "[8/8] Creating project folders..."
mkdir -p training_data
mkdir -p asl_clips
mkdir -p lse_clips
mkdir -p training_data/raw_clips/asl
mkdir -p training_data/raw_clips/lse
mkdir -p ~/Downloads/13691887
mkdir -p ~/Downloads/sign4all
echo "  training_data/  asl_clips/  lse_clips/  ~/Downloads/13691887/  ~/Downloads/sign4all/"

# ── 9. mDNS (avahi) — phone reaches laptop via hear-me-sign.local ─────────────
echo ""
echo "[9/10] Configuring mDNS (avahi)..."

NSSWITCH=/etc/nsswitch.conf
if ! grep -q "mdns_minimal" "$NSSWITCH"; then
  $SUDO sed -i 's/^hosts:.*/hosts: mDNS_minimal [NOTFOUND=return] resolve [!UNAVAIL=return] files myhostname dns/' "$NSSWITCH"
  echo "  nsswitch.conf updated"
else
  echo "  nsswitch.conf already configured"
fi

$SUDO systemctl enable --now avahi-daemon
echo "  avahi-daemon enabled and running"
echo "  Phone can reach server at: http://$(hostname).local:8000"

# ── 10. Systemd service — Flask starts automatically on boot ──────────────────
echo ""
echo "[10/10] Installing SignFuture systemd service..."

SERVICE_FILE=/etc/systemd/system/signfuture.service
CURRENT_USER=$(whoami)

$SUDO tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=SignFuture Flask Backend
After=network.target avahi-daemon.service
Wants=avahi-daemon.service

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/.venv/bin/python $PROJECT_DIR/web_bridge.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

$SUDO systemctl daemon-reload
$SUDO systemctl enable signfuture.service
echo "  signfuture.service installed and enabled"
echo "  Flask will start automatically on every boot"
echo ""
echo "  To start now without rebooting:"
echo "    sudo systemctl start signfuture.service"
echo "  To check status:"
echo "    sudo systemctl status signfuture.service"

# ── Final verification ────────────────────────────────────────────────────────
echo ""
echo "================================================"
echo "  Verification"
echo "================================================"

PASS=0; FAIL=0

check() {
  local label="$1"; local cmd="$2"
  if eval "$cmd" &>/dev/null; then
    echo "  OK  $label"; PASS=$((PASS+1))
  else
    echo "  FAIL  $label"; FAIL=$((FAIL+1))
  fi
}

check "Python 3.11"           "$PROJECT_DIR/.venv/bin/python --version | grep -q '3.11'"
check "PyTorch"               "$PROJECT_DIR/.venv/bin/python -c 'import torch'"
check "Flask"                 "$PROJECT_DIR/.venv/bin/python -c 'import flask'"
check "OpenCV"                "$PROJECT_DIR/.venv/bin/python -c 'import cv2'"
check "NumPy"                 "$PROJECT_DIR/.venv/bin/python -c 'import numpy'"
check "gTTS"                  "$PROJECT_DIR/.venv/bin/python -c 'from gtts import gTTS'"
check "avahi-daemon running"  "systemctl is-active --quiet avahi-daemon"
check "signfuture.service"    "systemctl is-enabled --quiet signfuture"
check "asl_clips/ folder"     "[ -d '$PROJECT_DIR/asl_clips' ]"
check "lse_clips/ folder"     "[ -d '$PROJECT_DIR/lse_clips' ]"
check "training_data/ folder" "[ -d '$PROJECT_DIR/training_data' ]"

echo ""
if [ "$FAIL" -eq 0 ]; then
  echo "  All $PASS checks passed — SignFuture is ready."
  echo ""
  echo "  Start the server:  sudo systemctl start signfuture"
  echo "  Phone URL:         http://$(hostname).local:8000"
else
  echo "  $PASS passed, $FAIL failed — check output above."
fi
echo "================================================"
