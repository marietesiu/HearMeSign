"""landmarks.py — MediaPipe landmark extraction and normalization."""

import numpy as np

WRIST       = 0
THUMB_TIP   = 4
INDEX_MCP   = 5
MIDDLE_MCP  = 9
RING_MCP    = 13
PINKY_MCP   = 17
MIDDLE_TIP  = 12
FINGER_TIPS = [THUMB_TIP, 8, MIDDLE_TIP, 16, 20]


def extract_landmarks(results) -> np.ndarray:
    """Extracts and flattens hand + pose landmarks. Returns zeros for undetected parts."""
    lh = (np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
          if results.left_hand_landmarks else np.zeros(63))
    rh = (np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
          if results.right_hand_landmarks else np.zeros(63))
    pose = (np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(132))
    return np.concatenate([lh, rh, pose])


def _normalize_hand(hand: np.ndarray) -> np.ndarray:
    """Normalizes hand anchored on middle knuckle, robust to loose/long sleeves. Returns (73,)."""
    hand = hand - hand[MIDDLE_MCP].copy()
    span = np.linalg.norm(hand[INDEX_MCP] - hand[PINKY_MCP])
    if span > 0:
        hand = hand / span
    dists = [np.linalg.norm(hand[FINGER_TIPS[i]] - hand[FINGER_TIPS[j]])
             for i in range(len(FINGER_TIPS)) for j in range(i + 1, len(FINGER_TIPS))]
    return np.concatenate([hand.flatten(), np.array(dists)])


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalizes raw landmarks (258,) → invariant to sleeves, position, scale. Returns (278,)."""
    lh_raw = landmarks[0:63].reshape(21, 3)
    rh_raw = landmarks[63:126].reshape(21, 3)
    lh     = _normalize_hand(lh_raw) if not np.all(lh_raw == 0) else np.zeros(73)
    rh     = _normalize_hand(rh_raw) if not np.all(rh_raw == 0) else np.zeros(73)

    pose = landmarks[126:258].reshape(33, 4).copy()
    if not np.all(pose[:, :2] == 0):
        xy = pose[:, :2] - pose[:, :2].mean(axis=0)
        md = np.max(np.linalg.norm(xy, axis=1))
        if md > 0:
            xy = xy / md
        pose[:, :2] = xy

    return np.concatenate([lh, rh, pose.flatten()])
