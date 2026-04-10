"""mp_holistic.py — MediaPipe 0.10+ compatible holistic wrapper.

MediaPipe 0.10 removed solutions.holistic. This module reimplements the same
interface using the new Hand Landmarker + Pose Landmarker Tasks API, returning
a results object with the same attributes that landmarks.py expects:
  .left_hand_landmarks
  .right_hand_landmarks
  .pose_landmarks

Each landmark has .x, .y, .z attributes (and .visibility for pose).
"""

try:
    import mediapipe as mp
    import numpy as np
    import cv2
    _VISION_OK = True
except ImportError:
    _VISION_OK = False


# ── Shim landmark objects (mimic mediapipe NormalizedLandmark) ────────────────

class _Landmark:
    __slots__ = ("x", "y", "z", "visibility")
    def __init__(self, x=0.0, y=0.0, z=0.0, visibility=1.0):
        self.x = x; self.y = y; self.z = z; self.visibility = visibility

class _LandmarkList:
    def __init__(self, landmarks):
        self.landmark = landmarks

class _Results:
    def __init__(self):
        self.left_hand_landmarks  = None
        self.right_hand_landmarks = None
        self.pose_landmarks       = None


# ── Parse Tasks API results into shim objects ─────────────────────────────────

def _hand_landmarks_from_result(hand_result):
    left = right = None
    if hand_result is None:
        return None, None
    for i, handedness_list in enumerate(hand_result.handedness):
        if i >= len(hand_result.hand_landmarks):
            break
        side = handedness_list[0].category_name.lower()
        ll   = _LandmarkList([_Landmark(lm.x, lm.y, lm.z)
                               for lm in hand_result.hand_landmarks[i]])
        if side == "left":
            left = ll
        else:
            right = ll
    return left, right

def _pose_landmarks_from_result(pose_result):
    if pose_result is None or not pose_result.pose_landmarks:
        return None
    return _LandmarkList([
        _Landmark(lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0))
        for lm in pose_result.pose_landmarks[0]
    ])


# ── Model file helper ─────────────────────────────────────────────────────────

def _find_or_download(filename, url, search_dirs):
    import os, urllib.request
    for d in search_dirs:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    os.makedirs(search_dirs[0], exist_ok=True)
    dest = os.path.join(search_dirs[0], filename)
    print(f"[mp_holistic] Downloading {filename}…")
    urllib.request.urlretrieve(url, dest)
    print(f"[mp_holistic] Saved → {dest}")
    return dest


# ── Holistic class ────────────────────────────────────────────────────────────

class Holistic:
    """
    MediaPipe 0.10+ compatible holistic landmark extractor.
    Same interface as the old mediapipe.solutions.holistic.Holistic.

        with Holistic(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as h:
            results = h.process(rgb_frame)
    """

    def __init__(self, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, model_complexity=1):
        if not _VISION_OK:
            raise ImportError("mediapipe / cv2 / numpy not installed")
        self._det_conf    = min_detection_confidence
        self._track_conf  = min_tracking_confidence
        self._hand        = None
        self._pose        = None

    def __enter__(self):
        self._init(); return self

    def __exit__(self, *_):
        self.close()

    def open(self):
        self._init(); return self

    def close(self):
        for attr in ("_hand", "_pose"):
            obj = getattr(self, attr, None)
            if obj:
                obj.close()
                setattr(self, attr, None)

    def _init(self):
        import os
        here        = os.path.dirname(os.path.abspath(__file__))
        search_dirs = [os.path.join(here, "models"),
                       os.path.expanduser("~/.cache/mediapipe"), "/tmp"]

        hand_model = _find_or_download(
            "hand_landmarker.task",
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task",
            search_dirs)

        pose_model = _find_or_download(
            "pose_landmarker_lite.task",
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
            search_dirs)

        Base    = mp.tasks.BaseOptions
        Mode    = mp.tasks.vision.RunningMode
        HLM     = mp.tasks.vision.HandLandmarker
        HLMOpts = mp.tasks.vision.HandLandmarkerOptions
        PLM     = mp.tasks.vision.PoseLandmarker
        PLMOpts = mp.tasks.vision.PoseLandmarkerOptions

        self._hand = HLM.create_from_options(HLMOpts(
            base_options=Base(model_asset_path=hand_model),
            running_mode=Mode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=self._det_conf,
            min_hand_presence_confidence=self._det_conf,
            min_tracking_confidence=self._track_conf,
        ))
        self._pose = PLM.create_from_options(PLMOpts(
            base_options=Base(model_asset_path=pose_model),
            running_mode=Mode.IMAGE,
            min_pose_detection_confidence=self._det_conf,
            min_tracking_confidence=self._track_conf,
        ))

    def process(self, rgb_frame) -> _Results:
        if self._hand is None:
            self._init()
        img         = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        hand_result = self._hand.detect(img)
        pose_result = self._pose.detect(img)
        r = _Results()
        r.left_hand_landmarks, r.right_hand_landmarks = \
            _hand_landmarks_from_result(hand_result)
        r.pose_landmarks = _pose_landmarks_from_result(pose_result)
        return r
