# -*- coding: utf-8 -*-
"""
简单的手部关键点可视化 Demo（摄像头）。
"""
import mediapipe as mp
import cv2
import numpy as np


def run():
    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=2, model_complexity=0,
                          min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    while True:
        success, img = cap.read()
        if not success:
            continue
        image_height, image_width, _ = np.shape(img)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

        cv2.imshow("hands", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()

