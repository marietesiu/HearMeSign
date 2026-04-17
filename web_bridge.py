"""web_bridge.py — SignFuture Flask backend. Dual-language, no ffmpeg, HTTP POST.

Primary model : I3D  (training_data/i3d_asl.pt  /  i3d_lse.pt)
Fallback model: MLP  (training_data/model_asl.pt / model_lse.pt)
The server loads I3D if the file exists, otherwise falls back to MLP silently.

Endpoints:
  GET  /                      → index.html
  GET  /clips/<filename>      → MP4 clip files
  GET  /health                → JSON status (both models)
  GET  /test-clip             → first available clip (connectivity test)
  POST /switch-language       → {"lang":"asl"|"lse"}  hot-swap active model
  POST /text-to-sign          → {"text":"..."}         → {"clips":[...]}
  POST /audio-to-sign         → {"text":"..."}         → {"clips":[...]}
  POST /sign-to-text          → <video bytes>          → {"text":"...", "session_id":"..."}
  POST /feedback              → {session_id, correct, label?} → correction + retrain at N=10
  POST /sign-to-audio         → <video bytes>          → MP3
  POST /collect-video         → <video bytes> + header → {"stored":N}
  POST /train                 → {}                     → starts background train
  GET  /train-status          → {"state","pct","msg","result"}
  GET  /samples               → {"label": count, ...}
  POST /delete-label          → {"label":"..."}        → {"remaining":N}
  GET  /test-clip             → first MP4 (connectivity check)

Access via WireGuard tunnel: phone connects to http://10.0.0.1:8000
Run gen_wg_qr.py on the laptop to regenerate the phone QR if your IP changes.
"""

import io
import os
import time
import threading
import uuid
from collections import deque

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

from asl_dictionary import ASL_DICT, LSE_DICT
from config import USE_SPANISH, ASL_CLIPS_FOLDER, PUBLIC_URL, CLOUDFARE_TUNNEL
from matcher import match_phrases
from text_processing import clean_text
import sign_model as sm

# ── Optional deps ──────────────────────────────────────────────────────────
try:
    from gtts import gTTS
    _TTS_OK = True
except ImportError:
    gTTS = None
    _TTS_OK = False
    print("[bridge] gTTS not installed — pip install gtts")

try:
    import collections, tempfile, cv2
    import mp_holistic as mph
    from landmarks import extract_landmarks, normalize_landmarks
    _VISION_OK = True
except ImportError as _e:
    collections = None
    tempfile = None
    cv2 = None
    mph = None
    extract_landmarks = None
    normalize_landmarks = None
    _VISION_OK = False
    print(f"[bridge] Vision deps missing ({_e})")

# ── Flask ──────────────────────────────────────────────────────────────────
app  = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload
HERE = os.path.dirname(os.path.abspath(__file__))

@app.errorhandler(404)
def not_found(e):
    # Return minimal JSON for unknown routes (suppresses verbose Flask HTML 404 pages)
    return jsonify({"error": "not found"}), 404

# ── Feedback / session store ───────────────────────────────────────────────
# Keeps the last 50 prediction sessions in memory so /feedback can reference them.
# Each entry: {video_bytes, raw_frames, lang, prediction, debug}
FEEDBACK_RETRAIN_AFTER = 10          # retrain after this many confirmed corrections
_session_store: dict   = {}          # session_id → session dict
_session_order: deque  = deque(maxlen=50)   # evict oldest when full
_feedback_lock         = threading.Lock()
_pending_corrections: list = []      # confirmed (video_bytes, label, lang) waiting for retrain


# ── Active language (mirrors config.py USE_SPANISH on startup) ─────────────
_lang_lock   = threading.Lock()
_active_lang = "lse" if USE_SPANISH else "asl"   # "asl" or "lse"

# ── Training state ─────────────────────────────────────────────────────────
_train_state   = {"state": "idle", "pct": 0, "msg": "", "result": None}
_train_lock    = threading.Lock()
_prepare_state = {"state": "idle", "pct": 0, "msg": ""}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _lang() -> str:
    with _lang_lock:
        return _active_lang


def _clf():
    """Return the active classifier (loaded on demand)."""
    return sm.load_model(_lang())


def _dict():
    return LSE_DICT if _lang() == "lse" else ASL_DICT


def _resolve_clips(text: str) -> list[str]:
    is_spanish = (_lang() == "lse")
    words = clean_text(text, use_spanish=is_spanish)
    return match_phrases(words, _dict(), use_spanish=is_spanish)


def _tts(text: str) -> bytes:
    # gTTS >= 2.5.0 broke write_to_fp(BytesIO) — it now expects a real file path.
    # Use a named temp file and read it back instead.
    import tempfile
    lang_code = "es" if _lang() == "lse" else "en"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        gTTS(text=text, lang=lang_code).save(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    except Exception as e:
        print(f"[bridge] TTS error: {e}")
        raise RuntimeError(f"Text-to-speech failed: {e}") from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _clips_dir(lang: str | None = None) -> str:
    """Return the clip folder for the given language (defaults to active lang)."""
    l = lang or _lang()
    if l == "lse":
        return os.path.join(HERE, "lse_clips")
    return os.path.join(HERE, ASL_CLIPS_FOLDER)


def _decode_video(video_bytes: bytes,
                  timeout_s: float = 30.0) -> list:
    """
    Core decoder: video bytes → list of raw BGR frames (HxWx3 uint8 numpy arrays).

    Tries .webm, .mp4, .avi suffixes — cv2 codec support varies by platform.
    Hard 30s timeout prevents corrupt video from hanging forever.
    Returns ALL frames (not filtered). Caller decides what to do with them.
    """
    result_box: list = []
    error_box:  list = []

    def _worker():
        frames   = []
        last_err = None
        for suffix in (".webm", ".mp4", ".avi"):
            tmp = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                    f.write(video_bytes)
                    tmp = f.name
                cap = cv2.VideoCapture(tmp)
                if not cap.isOpened():
                    cap.release()
                    continue
                local_frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    local_frames.append(frame)   # raw BGR, HxWx3 uint8
                cap.release()
                frames = local_frames
                print(f"[bridge] Decoded suffix={suffix}, "
                      f"frames={len(frames)}, "
                      f"size={len(video_bytes) // 1024}KB")
                break
            except Exception as e:
                last_err = e
            finally:
                if tmp and os.path.exists(tmp):
                    try: os.remove(tmp)
                    except: pass
        if last_err and not frames:
            error_box.append(last_err)
        result_box.append(frames)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        print(f"[bridge] _decode_video TIMED OUT after {timeout_s}s")
        return []
    if error_box:
        print(f"[bridge] _decode_video error: {error_box[0]}")
    return result_box[0] if result_box else []


def _extract_raw_frames(video_bytes: bytes) -> list:
    """
    I3D path: video bytes → list of raw BGR frames.
    No MediaPipe involved — I3D reads pixels directly.
    Returns empty list if no frames decoded.
    """
    return _decode_video(video_bytes)


def _extract_landmarks(video_bytes: bytes) -> list:
    """
    MLP fallback path: video bytes → list of 278-dim landmark vectors.
    Only called when active model is MLP (no i3d_<lang>.pt exists yet).
    Filters to frames where at least one hand is visible.
    """
    if not _VISION_OK:
        return []
    raw_frames = _decode_video(video_bytes)
    if not raw_frames:
        return []
    lm_frames = []
    with mph.Holistic(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as h:
        for frame in raw_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lm  = normalize_landmarks(extract_landmarks(h.process(rgb)))
            if not np.all(lm[:126] == 0):   # at least one hand detected
                lm_frames.append(lm)
    print(f"[bridge] Landmark extraction: {len(lm_frames)}/{len(raw_frames)} hand frames")
    return lm_frames


def _extract_both(video_bytes: bytes) -> tuple:
    """
    Fusion path: decode video once, extract BOTH raw frames AND landmarks
    in a single pass. Returns (raw_frames, landmark_frames).

    raw_frames      → fed to I3D (pixel-based)
    landmark_frames → fed to MLP (coordinate-based, background-proof)

    Either list may be empty if extraction fails — FusionClassifier handles
    graceful degradation to single-model mode automatically.
    """
    raw_frames = _extract_raw_frames(video_bytes)
    if not raw_frames:
        return [], []

    lm_frames = []
    if extract_landmarks is not None and normalize_landmarks is not None:
        try:
            with mph.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as h:
                for frame in raw_frames:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    lm  = normalize_landmarks(extract_landmarks(h.process(rgb)))
                    if not np.all(lm == 0):
                        lm_frames.append(lm)
        except Exception as e:
            print(f"[bridge] Landmark extraction error: {e}")

    print(f"[bridge] Fusion: {len(raw_frames)} raw frames, "
          f"{len(lm_frames)} landmark frames")
    return raw_frames, lm_frames


def _not_trained_error():
    lang = _lang()
    return jsonify({
        "error": (
            f"The {lang.upper()} model hasn't been trained yet. "
            f"Run  python dataset_download.py  to build the dataset and train, "
            f"or use the Dev Panel → 🧠 Train tab to record and train manually."
        ),
        "not_trained": True,
        "lang": lang,
    }), 503


# ═══════════════════════════════════════════════════════════════════════════════
# STATIC
# ═══════════════════════════════════════════════════════════════════════════════


@app.route("/")
def index():
    f = os.path.join(HERE, "index.html")
    return send_from_directory(HERE, "index.html") if os.path.exists(f) \
        else ("<h2>SignFuture backend running. "
              "<a href='/health'>/health</a></h2>", 200)


@app.route("/clips/<path:filename>")
def serve_clip(filename):
    # Try active language folder first, fall back to asl_clips
    for folder in (_clips_dir(), os.path.join(HERE, ASL_CLIPS_FOLDER)):
        fpath = os.path.join(folder, filename)
        if os.path.exists(fpath):
            resp = send_from_directory(folder, filename)
            # Cache clips aggressively — they never change between requests.
            # The browser will serve from disk on repeat plays instead of
            # hitting the network again, making subsequent playback instant.
            resp.headers["Cache-Control"] = "public, max-age=86400, immutable"
            return resp
    return send_from_directory(_clips_dir(), filename)  # will 404 cleanly


@app.route("/test-clip")
def test_clip():
    d = _clips_dir()
    if not os.path.isdir(d):
        return jsonify({"error": f"Clips folder not found: {d}"}), 404
    mp4s = [f for f in os.listdir(d) if f.endswith(".mp4")]
    if not mp4s:
        return jsonify({"error": f"No clip files found in {d}"}), 404
    return send_from_directory(d, mp4s[0])


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH + LANGUAGE SWITCH
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/health")
def health():
    lang = _lang()
    _clf_inst = _clf()
    _clips_path = _clips_dir()
    return jsonify({
        "status":        "ok",
        "active_lang":   lang,
        "active_arch":   sm.active_arch(),
        "asl_ready":     sm.model_ready("asl"),
        "lse_ready":     sm.model_ready("lse"),
        "asl_i3d":       sm.i3d_ready("asl"),
        "lse_i3d":       sm.i3d_ready("lse"),
        "model_labels":  _clf_inst.labels if _clf_inst else [],
        "tts":           _TTS_OK,
        "vision":        _VISION_OK,
        "clips_count":   len([f for f in os.listdir(_clips_path) if f.endswith(".mp4")])
                         if os.path.isdir(_clips_path) else 0,
        "samples_asl":   sm.sample_counts("asl"),
        "samples_lse":   sm.sample_counts("lse"),
        "samples":       sm.sample_counts(lang),
        "public_url":    PUBLIC_URL,
    })


@app.route("/switch-language", methods=["POST"])
def switch_language():
    """
    Hot-swap the active model without restarting.
    Body: {"lang": "asl"} or {"lang": "lse"}
    """
    global _active_lang
    data = request.get_json(force=True, silent=True) or {}
    lang = data.get("lang", "").lower()
    if lang not in ("asl", "lse"):
        return jsonify({"error": "lang must be 'asl' or 'lse'"}), 400

    with _lang_lock:
        _active_lang = lang

    clf = sm.switch_language(lang)
    return jsonify({
        "active_lang":  lang,
        "model_ready":  clf is not None,
        "model_labels": clf.labels if clf else [],
    })


@app.route("/ping")
def ping():
    """Raw connectivity check — returns timestamp and server info."""
    import time as _time
    return jsonify({
        "pong": True,
        "ts":   _time.time(),
        "lang": _lang(),
        "pid":  os.getpid(),
    })


@app.route("/echo", methods=["POST"])
def echo():
    """
    Echoes back the raw bytes received so the client can verify
    the request body arrives intact end-to-end.
    Returns: {received_bytes, sha256, content_type}
    """
    import hashlib
    body = request.get_data()
    return jsonify({
        "received_bytes": len(body),
        "sha256":         hashlib.sha256(body).hexdigest(),
        "content_type":   request.content_type,
        "ok":             True,
    })


@app.route("/probe", methods=["POST"])
def probe():
    """
    Video probe endpoint — receives video bytes, attempts to decode with cv2,
    reports exactly what happened without running MediaPipe.
    Lets the test page verify video decoding works independently of model state.
    Returns: {received_bytes, writable, decodable, frame_count, width, height, fps, error}
    """
    video = request.get_data()
    result = {
        "received_bytes": len(video),
        "writable":       False,
        "decodable":      False,
        "frame_count":    0,
        "width":          0,
        "height":         0,
        "fps":            0,
        "error":          None,
    }
    if not video:
        result["error"] = "No bytes received"
        return jsonify(result), 400

    import tempfile, time as _time
    # Try .webm first, then .mp4 — cv2 codec support varies by platform
    for suffix in (".webm", ".mp4", ".avi"):
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(video)
                tmp = f.name
            result["writable"] = True
            cap = cv2.VideoCapture(tmp)
            if cap.isOpened():
                result["decodable"] = True
                result["width"]     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                result["height"]    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                result["fps"]       = round(cap.get(cv2.CAP_PROP_FPS), 1)
                count = 0
                while count < 300:   # read up to 300 frames
                    ret, _ = cap.read()
                    if not ret: break
                    count += 1
                result["frame_count"] = count
                cap.release()
                result["suffix_used"] = suffix
                break  # success
        except Exception as e:
            result["error"] = str(e)
        finally:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)

    if not result["decodable"]:
        result["error"] = (result["error"] or
            "cv2 could not open video — codec may be missing on this server. "
            "Install: pip install opencv-python (not headless) and ensure libx264 is present.")

    status = 200 if result["decodable"] else 422
    return jsonify(result), status


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSLATION
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/text-to-sign", methods=["POST"])
def text_to_sign():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or data.get("content") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    clips = _resolve_clips(text)
    if not clips:
        return jsonify({
            "error": f'No sign clips found for "{text}". '
                     f'Try: {", ".join(list(_dict())[:4])}.'
        }), 404
    urls = [f"/clips/{os.path.basename(c)}" for c in clips]
    print(f"[/text-to-sign] ({_lang()}) '{text}' → {urls}")
    return jsonify({"clips": urls, "transcript": text, "lang": _lang()})


@app.route("/audio-to-sign", methods=["POST"])
def audio_to_sign():
    # Browser runs SpeechRecognition and sends text — no server-side audio needed
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or data.get("transcript") or "").strip()
    if not text:
        return jsonify({
            "error": "No transcript received. "
                     "Chrome speech recognition required."
        }), 400
    clips = _resolve_clips(text)
    if not clips:
        return jsonify({
            "error": f'No sign clips for "{text}". '
                     f'Try: {", ".join(list(_dict())[:4])}.'
        }), 404
    urls = [f"/clips/{os.path.basename(c)}" for c in clips]
    return jsonify({"clips": urls, "transcript": text, "lang": _lang()})


# ── CTC continuous model (lazy-loaded) ────────────────────────────────────────
_ctc_clf  = None
_ctc_lock = threading.Lock()

def _ctc_model():
    """Lazy-load CTC model. Returns None if not trained yet."""
    global _ctc_clf
    with _ctc_lock:
        if _ctc_clf is not None:
            return _ctc_clf
        ctc_path = os.path.join(sm.DATA_DIR, "ctc_asl.pt")
        if not os.path.exists(ctc_path):
            return None
        try:
            from ctc_model import CTCClassifier
            _ctc_clf = CTCClassifier.load(ctc_path)
            print(f"[bridge] CTC model loaded — continuous ASL mode available")
        except Exception as e:
            print(f"[bridge] CTC model load failed: {e}")
        return _ctc_clf


@app.route("/sign-to-text", methods=["POST"])
def sign_to_text():
    if not _VISION_OK:
        return jsonify({"error": "Vision not available (mediapipe/cv2 missing)"}), 503
    video = request.get_data()
    if not video:
        return jsonify({"error": "No video received"}), 400

    # ── Continuous mode (CTC) — ?mode=continuous or X-Mode: continuous ────────
    mode = (request.args.get("mode") or
            request.headers.get("X-Mode") or "isolated").lower()

    if mode == "continuous" and _lang() == "asl":
        ctc = _ctc_model()
        if ctc is None:
            return jsonify({
                "error": "CTC continuous model not trained yet.",
                "hint":  "Run: python ctc_model.py --train"
            }), 503
        _, lm_frames = _extract_both(video)
        if not lm_frames:
            return jsonify({"error": "No hand detected."}), 422
        try:
            text, conf = ctc.predict_sequence(lm_frames)
        except Exception as e:
            print(f"[/sign-to-text] CTC error: {e}")
            return jsonify({"error": f"CTC model failed: {e}"}), 500
        print(f"[/sign-to-text] CTC -> {text} ({conf*100:.0f}%)")
        if text is None:
            return jsonify({
                "error":      f"No signs recognised ({conf*100:.0f}%).",
                "confidence": round(conf, 3),
                "mode":       "continuous",
            }), 422
        return jsonify({
            "type":       "continuous",
            "text":       text,
            "confidence": round(conf, 3),
            "lang":       _lang(),
            "mode":       "continuous",
        })

    # ── Isolated word mode (I3D + MLP fusion) — default ───────────────────────
    clf = _clf()
    if clf is None:
        return _not_trained_error()
    arch = sm.active_arch()
    print(f"[/sign-to-text] ({_lang()}/{arch}) {len(video)/1024:.1f} KB")
    raw_frames, lm_frames = _extract_both(video)
    if not raw_frames and not lm_frames:
        return jsonify({
            "error": "No frames extracted. Make sure your hand is clearly visible."
        }), 422
    try:
        label, conf, debug = clf.predict_sequence((raw_frames, lm_frames))
        print(f"[/sign-to-text] -> {label} ({conf*100:.0f}%)")
    except Exception as e:
        print(f"[/sign-to-text] model error: {e}")
        return jsonify({"error": f"Model inference failed: {e}"}), 500

    sid = str(uuid.uuid4())[:8]
    with _feedback_lock:
        if len(_session_order) == _session_order.maxlen:
            old_sid = _session_order[0]
            _session_store.pop(old_sid, None)
        _session_store[sid] = {
            "video_bytes": video,
            "raw_frames":  raw_frames,
            "lang":        _lang(),
            "prediction":  label,
            "debug":       debug,
        }
        _session_order.append(sid)

    if label is None:
        return jsonify({
            "error":      f"Sign not recognised ({conf*100:.0f}% confidence). "
                          "Hold the sign clearly for 2-3 seconds.",
            "confidence": round(conf, 3),
            "session_id": sid,
            "debug":      debug,
        }), 422
    return jsonify({"type": "sign", "text": label, "confidence": round(conf, 3),
                    "lang": _lang(), "session_id": sid, "debug": debug})


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Confirm or correct a prediction.
    """
    data = request.get_json(silent=True) or {}
    sid  = (data.get("session_id") or "").strip()
    if not sid:
        return jsonify({"error": "Missing session_id"}), 400

    with _feedback_lock:
        session = _session_store.get(sid)
    if session is None:
        return jsonify({"error": f"Session '{sid}' not found or expired"}), 404

    correct = bool(data.get("correct", True))
    lang    = session["lang"]

    if correct:
        label = session["prediction"]
        if not label:
            return jsonify({"error": "Original prediction was None — "
                                     "provide correct label with correct=false"}), 400
    else:
        label = (data.get("label") or "").strip().lower()
        if not label:
            return jsonify({"error": "correct=false requires a 'label' field"}), 400

    # Store raw frames for I3D and landmarks for MLP.
    # Repetitions: correct = 1x (reinforce), wrong = 10x (heavy correction).
    raw_frames  = session["raw_frames"]
    video_bytes = session["video_bytes"]
    reps        = 1 if correct else 10

    stored_i3d = stored_mlp = 0

    if raw_frames:
        clips_path = sm._LANG_CLIPS.get(lang,
                         os.path.join(sm.DATA_DIR, f"clips_{lang}.npz"))
        os.makedirs(os.path.dirname(clips_path) or ".", exist_ok=True)
        # Stack reps copies of the clip
        new_clips = np.empty(reps, dtype=object)
        for i in range(reps):
            new_clips[i] = raw_frames
        new_y = np.array([label] * reps)
        if os.path.exists(clips_path):
            d     = np.load(clips_path, allow_pickle=True)
            all_X = np.concatenate([d["X"], new_clips])
            all_y = np.concatenate([d["y"], new_y])
        else:
            all_X, all_y = new_clips, new_y
        np.savez(clips_path, X=all_X, y=all_y)
        stored_i3d = reps

    if _VISION_OK and extract_landmarks and normalize_landmarks:
        lm_frames = _extract_landmarks(video_bytes)
        if lm_frames:
            feat  = sm.sequence_to_feature(lm_frames)
            feats = [feat] * reps
            stored_mlp = sm.save_samples(feats, [label] * reps, lang)

    print(f"[/feedback] sid={sid} correct={correct} label={label} "
          f"reps={reps} i3d={stored_i3d} mlp={stored_mlp}")

    # Always trigger immediate background retrain after feedback
    retrain_started = False
    with _feedback_lock:
        _pending_corrections.append((label, lang))
        pending_count = len(_pending_corrections)

    # Retrain immediately — every feedback call triggers it
    with _feedback_lock:
        _pending_corrections.clear()
    print(f"[/feedback] triggering immediate retrain for {lang.upper()} "
          f"(correct={correct}, reps={reps})")
    t = threading.Thread(target=_run_training, args=(lang, lambda p, m:
        print(f"[retrain] {p}% {m}")), daemon=True)
    t.start()
    retrain_started = True

    response = {
        "stored":           True,
        "label":            label,
        "lang":             lang,
        "reps":             reps,
        "retrain_started":  retrain_started,
    }

    return jsonify(response)


@app.route("/sign-to-audio", methods=["POST"])
def sign_to_audio():
    if not _TTS_OK:
        return jsonify({"error": "TTS unavailable — pip install gtts"}), 503
    if not _VISION_OK:
        return jsonify({"error": "Vision not available"}), 503
    clf = _clf()
    if clf is None:
        return _not_trained_error()
    video = request.get_data()
    if not video:
        return jsonify({"error": "No video received"}), 400
    arch = sm.active_arch()
    print(f"[/sign-to-audio] ({_lang()}/{arch}) {len(video)/1024:.1f} KB")
    raw_frames, lm_frames = _extract_both(video)
    if not raw_frames and not lm_frames:
        return jsonify({"error": "No hand detected."}), 422
    label, conf, debug = clf.predict_sequence((raw_frames, lm_frames))
    if label is None:
        return jsonify({
            "error":      f"Sign not recognised ({conf*100:.0f}%). "
                          "Hold the sign clearly for 2–3 seconds.",
            "confidence": round(conf, 3),
            "debug":      debug,
        }), 422
    print(f"[/sign-to-audio] '{label}' → TTS")
    try:
        mp3 = _tts(label)
    except Exception as e:
        return jsonify({"error": f"TTS failed: {e}"}), 500
    return send_file(io.BytesIO(mp3), mimetype="audio/mpeg",
                     download_name="speech.mp3")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/collect-video", methods=["POST"])
def collect_video():
    """
    Receive raw video bytes + X-Sign-Label header, store training sample.

    I3D  path: stores raw BGR frames into clips_<lang>.npz (object array of frame lists)
    MLP  path: extracts landmarks, pools to 278-dim, stores in samples_<lang>.npz
    """
    if not _VISION_OK:
        return jsonify({"error": "Vision deps not available"}), 503
    label = (request.headers.get("X-Sign-Label") or "").strip().lower()
    lang  = (request.headers.get("X-Sign-Lang")  or "").strip().lower()
    if lang not in ("asl", "lse"):
        lang = _lang()
    if not label:
        return jsonify({"error": "Missing X-Sign-Label header"}), 400
    video = request.get_data()
    if not video:
        return jsonify({"error": "No video received"}), 400

    arch = sm.active_arch() or "fusion"   # default to fusion

    # Always extract both — fusion stores both; single-model modes ignore what they don't need
    raw_frames, lm_frames = _extract_both(video)

    if not raw_frames and not lm_frames:
        return jsonify({"error": "Could not decode video. Try again."}), 422

    stored_i3d = 0
    stored_mlp = 0

    # ── Store raw frames for I3D ──────────────────────────────────────────────
    if raw_frames and arch in ("i3d", "fusion"):
        clips_path = sm._LANG_CLIPS.get(lang,
                         os.path.join(sm.DATA_DIR, f"clips_{lang}.npz"))
        os.makedirs(os.path.dirname(clips_path) or ".", exist_ok=True)
        new_clip    = np.empty(1, dtype=object)
        new_clip[0] = raw_frames
        new_y       = np.array([label])
        if os.path.exists(clips_path):
            d      = np.load(clips_path, allow_pickle=True)
            all_X  = np.concatenate([d["X"], new_clip])
            all_y  = np.concatenate([d["y"], new_y])
        else:
            all_X, all_y = new_clip, new_y
        np.savez(clips_path, X=all_X, y=all_y)
        stored_i3d = 1
        print(f"[/collect-video] I3D ({lang}) '{label}': +1 clip ({len(raw_frames)} frames)")

    # ── Store landmarks for MLP ───────────────────────────────────────────────
    if lm_frames and arch in ("mlp", "fusion"):
        feats      = [sm.sequence_to_feature(lm_frames)]
        stored_mlp = sm.save_samples(feats, [label], lang)
        print(f"[/collect-video] MLP ({lang}) '{label}': +1 landmark sample")

    counts = sm.sample_counts(lang)
    return jsonify({
        "stored": max(stored_i3d, stored_mlp),
        "total":  counts.get(label, 0),
        "counts": counts,
        "lang":   lang,
        "received_bytes":   len(video),
        "frames_extracted": len(raw_frames),
        "arch": arch,
    })


@app.route("/learn-signs")
def learn_signs():
    """
    Return [{label, clipUrl, ai_scoreable}] for every dictionary word that has
    a resolvable clip. Mirrors the fallback logic in serve_clip(): checks the
    language's own clips folder first, then falls back to asl_clips so that
    LSE signs which share ASL clip files still appear in the learn grid.
    Works even when no model is trained — users can still watch clips.
    """
    lang = request.args.get('lang', '').lower()
    if lang not in ('asl', 'lse'):
        lang = _lang()

    dictionary      = LSE_DICT if lang == "lse" else ASL_DICT
    lang_clips_dir  = _clips_dir(lang)
    asl_clips_dir   = os.path.join(HERE, ASL_CLIPS_FOLDER)

    clf = sm.load_model(lang)
    model_label_set = set(l.lower() for l in (clf.labels if clf else []))

    signs = []
    for label, clip_file in dictionary.items():
        # Check language folder first, then asl_clips — same order as serve_clip()
        found = any(
            os.path.exists(os.path.join(folder, clip_file))
            for folder in (lang_clips_dir, asl_clips_dir)
        )
        if found:
            signs.append({
                "label":        label,
                "clipUrl":      f"/clips/{clip_file}",
                "ai_scoreable": label in model_label_set,
            })

    signs.sort(key=lambda s: (not s["ai_scoreable"], s["label"]))
    return jsonify({"signs": signs, "lang": lang})


@app.route("/samples")
def get_samples():
    # Accept optional ?lang=asl|lse so the Dev Panel can query both independently
    lang = request.args.get('lang', '').lower()
    if lang not in ('asl', 'lse'):
        lang = _lang()
    return jsonify(sm.sample_counts(lang))


@app.route("/delete-label", methods=["POST"])
def del_label():
    data  = request.get_json(force=True, silent=True) or {}
    label = (data.get("label") or "").strip().lower()
    lang  = (data.get("lang") or "").lower()
    if lang not in ("asl", "lse"):
        lang = _lang()
    if not label:
        return jsonify({"error": "No label"}), 400
    remaining = sm.delete_label_samples(label, lang)
    return jsonify({"deleted": label, "lang": lang, "remaining_total": remaining})


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING  (fine-tune / retrain on collected webcam data)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_training(lang: str, progress_cb):
    """
    Background training thread. Trains BOTH I3D and MLP when data is available
    for each, so the fusion classifier always has both models up to date.
    Called by /train endpoint and auto-triggered after N feedback corrections.
    """
    try:
        clips_path   = sm._LANG_CLIPS.get(lang,
                           os.path.join(sm.DATA_DIR, f"clips_{lang}.npz"))
        samples_path = sm.data_path(lang)

        has_clips   = os.path.exists(clips_path)
        has_samples = os.path.exists(samples_path)

        if not has_clips and not has_samples:
            progress_cb(0, "No training data found. Collect clips via Dev Panel.",
                        done=True, error=True)
            return

        results = {}

        # ── Train I3D if raw clips exist ──────────────────────────────────────
        if has_clips:
            d      = np.load(clips_path, allow_pickle=True)
            clips  = list(d["X"])
            y      = list(d["y"])
            labels = sorted(set(y))
            if len(labels) >= 2:
                progress_cb(3, f"I3D: training {lang.upper()} — "
                               f"{len(clips)} clips, {len(labels)} signs…")
                clf_i3d = sm.I3DClassifier(labels)
                res_i3d = clf_i3d.train(
                    clips, y, epochs=60,
                    progress_cb=lambda p, m: progress_cb(3 + int(p * 0.45), m),
                )
                out_path = sm._LANG_I3D.get(lang,
                               os.path.join(sm.DATA_DIR, f"i3d_{lang}.pt"))
                clf_i3d.save(out_path)
                results["i3d"] = res_i3d
                progress_cb(48, f"I3D done — {res_i3d['accuracy']*100:.1f}% accuracy")
            else:
                progress_cb(48, f"I3D skipped — need ≥2 signs (have {len(labels)})")

        # ── Train MLP if landmark samples exist ───────────────────────────────
        if has_samples:
            X, y   = sm.load_samples(lang)
            labels = sorted(set(y.tolist()))
            if len(labels) >= 2:
                progress_cb(50, f"MLP: training {lang.upper()} — "
                                f"{len(X)} samples, {len(labels)} signs…")
                clf_mlp = sm.SignClassifier(labels)
                res_mlp = clf_mlp.train(
                    X, y.tolist(), epochs=300,
                    progress_cb=lambda p, m: progress_cb(50 + int(p * 0.45), m),
                )
                clf_mlp.save(sm.model_path(lang))
                results["mlp"] = res_mlp
                progress_cb(95, f"MLP done — {res_mlp['accuracy']*100:.1f}% accuracy")
            else:
                progress_cb(95, f"MLP skipped — need ≥2 signs (have {len(labels)})")

        if not results:
            progress_cb(0, "Not enough signs to train (need ≥2 per model).",
                        done=True, error=True)
            return

        # Reload the fusion model with both updated files
        sm.switch_language(lang)

        summary_parts = []
        if "i3d" in results:
            summary_parts.append(f"I3D {results['i3d']['accuracy']*100:.1f}%")
        if "mlp" in results:
            summary_parts.append(f"MLP {results['mlp']['accuracy']*100:.1f}%")
        summary = " | ".join(summary_parts)

        progress_cb(100, f"✅ {lang.upper()} retrained — {summary}",
                    done=True, result=results)

    except FileNotFoundError as e:
        progress_cb(0, str(e), done=True, error=True)
    except Exception as e:
        progress_cb(0, f"Training failed: {e}", done=True, error=True)


@app.route("/train", methods=["POST"])
def train():
    with _train_lock:
        if _train_state["state"] == "running":
            return jsonify({"error": "Training already in progress"}), 409
        _train_state.update({"state": "running", "pct": 0,
                             "msg": "Starting…", "result": None})
    data = request.get_json(force=True, silent=True) or {}
    lang = (data.get("lang") or "").lower()
    if lang not in ("asl", "lse"):
        lang = _lang()

    def cb(pct, msg, done=False, error=False, result=None):
        with _train_lock:
            _train_state.update({"pct": pct, "msg": msg})
            if done:
                _train_state["state"]  = "error" if error else "done"
                _train_state["result"] = result

    threading.Thread(target=_run_training, args=(lang, cb), daemon=True).start()
    return jsonify({"started": True, "lang": lang})


@app.route("/train-status")
def train_status():
    with _train_lock:
        return jsonify(dict(_train_state))



@app.route("/prepare", methods=["POST"])
def prepare():
    """
    Triggers dataset_download.py as a background subprocess.
    The frontend's Dev Panel 'Download Dataset & Train' button calls this.
    Progress is polled via /prepare-status.
    """
    global _prepare_state
    with _train_lock:
        if _prepare_state.get("state") == "running":
            return jsonify({"error": "Already running"}), 409
        _prepare_state = {"state": "running", "pct": 0, "msg": "Starting…"}

    def _run():
        global _prepare_state
        import subprocess, sys
        script = os.path.join(HERE, "dataset_download.py")
        if not os.path.exists(script):
            with _train_lock:
                _prepare_state = {"state": "error", "pct": 0,
                                  "msg": "dataset_download.py not found"}
            return
        try:
            proc = subprocess.Popen(
                [sys.executable, script],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=HERE,
            )
            pct = 1
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    pct = min(pct + 1, 95)
                    with _train_lock:
                        _prepare_state["msg"] = line
                        _prepare_state["pct"] = pct
                    print(f"[/prepare] {line}")
            proc.wait()
            ok = proc.returncode == 0
            with _train_lock:
                _prepare_state = {
                    "state": "done" if ok else "error",
                    "pct":   100 if ok else pct,
                    "msg":   "✅ Done! Restart server to load new model." if ok
                             else f"❌ Exited with code {proc.returncode}",
                }
        except Exception as exc:
            with _train_lock:
                _prepare_state = {"state": "error", "pct": 0,
                                  "msg": f"❌ {exc}"}

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"started": True})


@app.route("/prepare-status")
def prepare_status():
    """Poll endpoint for /prepare progress."""
    with _train_lock:
        return jsonify(dict(_prepare_state))



@app.errorhandler(413)
def too_large(_e):
    return jsonify({
        "error": "Upload too large. Max 50 MB. "
                 "Shorten your recording or check your WireGuard / network config.",
        "max_bytes": 50 * 1024 * 1024,
    }), 413


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Auto-train MLP fallback if data exists but no model at all
    sm.maybe_auto_train("asl")
    sm.maybe_auto_train("lse")

    # Pre-load the active language model
    try:
        clf = sm.load_model(_active_lang)
        if clf:
            print(f"[bridge] Model loaded for {_active_lang}")
    except Exception as e:
        print(f"[bridge] Model load failed at startup: {e}")
        clf = None

    cd = _clips_dir()
    clips_n = len([f for f in os.listdir(cd) if f.endswith(".mp4")]) \
              if os.path.isdir(cd) else 0
    clips_folder_name = "lse_clips" if _active_lang == "lse" else ASL_CLIPS_FOLDER

    def _model_status(lang):
        if sm.i3d_ready(lang):
            return f"✓ I3D ({sm._LANG_I3D[lang]})"
        if sm.model_ready(lang):
            return f"✓ MLP fallback ({sm._LANG_MLP[lang]})"
        return "✗ not trained — run dataset_download.py"

    print("=" * 60)
    print("  SignFuture Web Bridge")
    print(f"  Active lang : {_active_lang.upper()}")
    print(f"  Active arch : {sm.active_arch() or 'none'}")
    print(f"  ASL model   : {_model_status('asl')}")
    print(f"  LSE model   : {_model_status('lse')}")
    print(f"  TTS         : {'✓ gTTS' if _TTS_OK else '✗ pip install gtts'}")
    print(f"  Vision      : {'✓ mediapipe+cv2' if _VISION_OK else '✗ optional'}")
    print(f"  Clips       : {clips_n} files in {clips_folder_name}/")

    # ── SSL — required for camera/mic on non-localhost (Safari + mobile Chrome) ──
    _cert = os.path.join(HERE, "cert.pem")
    _key  = os.path.join(HERE, "key.pem")
    if os.path.exists(_cert) and os.path.exists(_key):
        proto = "https"
    else:
        proto = "http"
        print("[bridge] WARNING: cert.pem / key.pem not found — serving HTTP.")
        print("[bridge]          Camera and mic will NOT work on Safari / mobile.")
        print("[bridge]          Generate them with:")
        print("[bridge]          openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj '/CN=hear-me-sign.local'")

    print(f"  Local       : {proto}://localhost:8000")
    print(f"  Tunnel      : (start cloudflared and update config.py — not needed to run)")
    print("=" * 60)

    # Suppress noisy SSLError / BrokenPipe tracebacks from Werkzeug.
    # These happen when a mobile client drops a video range request mid-stream
    # (e.g. navigating away while a clip is buffering). They are harmless —
    # the client simply disconnected — but without suppression they spam the log.
    import logging, traceback as _tb

    class _QuietWSGIHandler(logging.Handler):
        """Drop log records that are just client-disconnect noise."""
        def emit(self, record):
            msg = record.getMessage() + (record.exc_text or "")
            if any(k in msg for k in ("SSLError", "BrokenPipe", "unknown error", "ConnectionResetError")):
                return
            # Re-emit to stderr so real errors still show
            import sys
            print(record.getMessage(), file=sys.stderr)

    _wlog = logging.getLogger("werkzeug")
    _wlog.handlers = [_QuietWSGIHandler()]
    _wlog.propagate = False

    ssl_ctx = (_cert, _key) if proto == "https" else None
    import sys

    tunnel_url = next((a for a in sys.argv[1:] if a.startswith("http")), "")
    if tunnel_url:
        import config as _cfg

        _cfg.CLOUDFARE_TUNNEL = tunnel_url
        _cfg.PUBLIC_URL = tunnel_url
        print(f"[bridge] Tunnel URL set to: {tunnel_url}")

    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True,
            use_reloader=False)
