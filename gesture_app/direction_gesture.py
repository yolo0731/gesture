import cv2
import numpy as np
from collections import deque, Counter

try:
    import mediapipe as mp
except ImportError as e:
    raise RuntimeError("mediapipe is required for direction_gesture. Please install it.") from e


def _compute_direction(label_points):
    """
    Compute direction from index finger PIP(6) -> TIP(8) vector.

    label_points: list of (id, x, y) for 21 landmarks in pixel coords.
    Returns one of: 'up', 'down', 'left', 'right', or None if not confident.
    """
    id_pip = 6
    id_tip = 8

    pip = label_points[id_pip]
    tip = label_points[id_tip]
    _, x0, y0 = pip
    _, x1, y1 = tip

    dx = x1 - x0
    dy = y1 - y0
    length = (dx * dx + dy * dy) ** 0.5

    if length < 25:
        return None

    axis_bias = 1.2
    if abs(dx) > abs(dy) * axis_bias:
        return 'left' if dx > 0 else 'right'
    elif abs(dy) > abs(dx) * axis_bias:
        return 'down' if dy > 0 else 'up'
    else:
        return None


def _draw_overlay(img, label):
    h, w = img.shape[:2]
    cv2.rectangle(img, (10, 10), (220, 80), (0, 0, 0), -1)
    cv2.putText(img, label, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)


def direction_gesture(ui=None):
    """
    Webcam-based index finger direction detector (up/down/left/right).

    Press 'q' to quit.
    """
    preferred_indexes = [6, 0, 1]
    cap = None
    for idx in preferred_indexes:
        temp = cv2.VideoCapture(idx)
        if temp is not None and temp.read()[0]:
            cap = temp
            break
        if temp is not None:
            temp.release()
    if cap is None:
        raise RuntimeError("Unable to access any webcam (tried indices 6, 0, 1)")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    history = deque(maxlen=7)
    last_announced = None

    try:
        while True:
            ok, img = cap.read()
            if not ok:
                continue

            h, w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            results = hands.process(img_rgb)

            current_label = None

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                lm_list = []
                for lid, lm in enumerate(hand.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    lm_list.append([lid, x, y])

                mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

                current_label = _compute_direction(lm_list)

            history.append(current_label)
            labels = [x for x in history if x is not None]
            if labels:
                most_common, count = Counter(labels).most_common(1)[0]
                if count >= 3:
                    if most_common != last_announced:
                        last_announced = most_common
                        if ui is not None and hasattr(ui, 'textBrowser') and hasattr(ui.textBrowser, 'append'):
                            ui.textBrowser.append(most_common)
                    _draw_overlay(img, most_common)

            cv2.imshow("Direction", img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

