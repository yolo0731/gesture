# -*- coding: utf-8 -*-
"""
ASL 数字手势识别（静态）。
"""

import mediapipe as mp
import cv2
import numpy as np
from tracking import definition
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pyttsx3


def number_gesture(ui):
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

            list_lms = []
            for i in range(21):
                position_x = hand.landmark[i].x * image_width
                position_y = hand.landmark[i].y * image_height
                list_lms.append([int(position_x), int(position_y)])

            list_lms = np.array(list_lms, dtype=np.int32)
            hull_index = [0,1,2,3,6,10,14,19,18,17,0]
            hull = cv2.convexHull(list_lms[hull_index])
            cv2.polylines(img, [hull], True, (0, 255, 0), 2)

            fingertips_list = [4,8,12,16,20]
            up_fingers = []
            a = list_lms[8]
            print(a)
            with open('test.txt', 'a') as f:
                f.write(str(a) + '\r\n')
            for i in fingertips_list:
                i_ordinate = (int(list_lms[i][0]), int(list_lms[i][1]))
                dist = cv2.pointPolygonTest(hull, i_ordinate, True)
                if dist < 0:
                    up_fingers.append(i)

            str_guester = definition.get_str_guester(up_fingers)
            cv2.putText(img, ' %s' % (str_guester), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,0), 5, cv2.LINE_AA)
            ui.textBrowser.append(str_guester)

        cv2.imshow("hands", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
