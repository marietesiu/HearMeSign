"""gen_wg_qr.py — Generate a WireGuard QR code for the phone peer.

Usage:
    python3 gen_wg_qr.py

What it does:
  1. Auto-detects your current public IP (works from any network/location)
  2. Builds the phone WireGuard config with that IP as the endpoint
  3. Saves a QR code to /tmp/wg-phone.png
  4. Prints the wg set command to register the phone as a peer

Run this any time your public IP changes (e.g. you move networks).
Then re-scan the QR on your phone to update the endpoint.
"""

import subprocess
import sys

try:
    import qrcode
except ImportError:
    print("Missing qrcode — run: pip install 'qrcode[pil]' --break-system-packages")
    sys.exit(1)

# ── Keys (laptop = server, phone = client) ────────────────────────────────────
LAPTOP_PRIVATE_KEY = "sIxB0EXIMHg4ptQILNSnNUWSofUINmRIck2VWD1eD1M="
LAPTOP_PUBLIC_KEY  = "fYOOUIGGo6vvCqHgvt2tEAXzC/fA+bIkFPtylPBYHGg="
PHONE_PRIVATE_KEY  = "MCxHyYsMp+3oTj5fCLp9ZJjyJtQvxVMgnuk2XcCU8Hw="
PHONE_PUBLIC_KEY   = "4zIadvcS/w0/+6jBxwf5DStaX4PMwTThDw8jotmH3F8="

WG_PORT      = 51820
LAPTOP_WG_IP = "10.0.0.1"
PHONE_WG_IP  = "10.0.0.2"
QR_OUT       = "/tmp/wg-phone.png"

# ── Auto-detect current public IP ─────────────────────────────────────────────
print("Detecting public IP...")
try:
    public_ip = subprocess.check_output(
        "curl -s --max-time 5 ifconfig.me",
        shell=True
    ).decode().strip()
    if not public_ip:
        raise ValueError("Empty response")
except Exception as e:
    print(f"Could not detect public IP: {e}")
    public_ip = input("Enter your public IP manually: ").strip()

print(f"  Public IP : {public_ip}")
print(f"  Endpoint  : {public_ip}:{WG_PORT}")

# ── Build phone config ────────────────────────────────────────────────────────
config = f"""\
[Interface]
PrivateKey = {PHONE_PRIVATE_KEY}
Address = {PHONE_WG_IP}/24
DNS = 1.1.1.1

[Peer]
PublicKey = {LAPTOP_PUBLIC_KEY}
Endpoint = {public_ip}:{WG_PORT}
AllowedIPs = {LAPTOP_WG_IP}/32
PersistentKeepalive = 25
"""

# ── Generate QR ───────────────────────────────────────────────────────────────
qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L)
qr.add_data(config)
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white")
img.save(QR_OUT)

print(f"\nQR code saved → {QR_OUT}")
print("Open it and scan with: WireGuard app → + → Scan from QR code\n")

# ── Print peer registration command ──────────────────────────────────────────
print("Then register the phone peer on the tunnel (if not already done):")
print(f"  sudo wg set wg0 peer {PHONE_PUBLIC_KEY} allowed-ips {PHONE_WG_IP}/32")
print( "  sudo wg-quick save wg0\n")
print("To test after scanning:")
print(f"  Phone browser → http://{LAPTOP_WG_IP}:8000/health")
print(f"  Laptop        → sudo wg show")
