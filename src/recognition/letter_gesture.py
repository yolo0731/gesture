import cv2
import mediapipe as mp
import time
from .letter_interpre import interpret
import pyttsx3

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from collections import deque


def add_space(text):
    return text + " "


def delete(text):
    return text[:-1]


def clear(text):
    return ""


def letter_gesture(ui):
    THUMB = object()
    INDEX = object()
    MIDDLE = object()
    RING = object()
    PINKY = object()

    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1)
    mpDraw = mp.solutions.drawing_utils

    prev_time = time.time()
    curr_time = time.time()

    letters = ""

    while True:
        _, img = cap.read()
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                lm_list = []
                for id, lm in enumerate(handLms.landmark):
                    cx, cy, cz = int(lm.x*w), int(lm.y*h), lm.z*c
                    lm_list.append([id, cx, cy, cz])

                if str(interpret(lm_list)) != 0:
                    cv2.putText(img, str(interpret(lm_list)), (w-300, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (52, 195, 235), 3)
                    ui.textBrowser.append(interpret(lm_list))
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                curr_time = time.time()
                diff_time = curr_time - prev_time
                if diff_time < 1:
                    display_time = 3
                elif diff_time < 2:
                    display_time = 2
                elif diff_time <= 3:
                    display_time = 1
        else:
            prev_time = time.time()

        curr_time = time.time()
        if curr_time - prev_time > 3:
            try:
                letters += interpret(lm_list)
            except TypeError:
                pass
            cv2.putText(img, "captured", (w//2 - 200, h//2), cv2.FONT_HERSHEY_DUPLEX, 3, (235, 107, 52), 3)
            prev_time = time.time()

        cv2.imshow('OpenCV Feed', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    engine = pyttsx3.init()
    print('准备开始语音播报...')
    engine.say(letters)
    engine.runAndWait()
    engine.stop()


def JZRec(ui):
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    actions = np.array(['Jj', 'Zz', 'None'])
    colors = [(245,117,16), (117,245,16), (16,117,245), (50,50,50)]

    def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        return output_frame

    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_styled_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

    def extract_keypoints(results):
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([lh, rh])

    sequence = []
    sentence = []
    threshold = 0.8
    predictions = []
    smoothed = deque(maxlen=10)
    from utils import paths
    base = paths.models_dir()
    model_dir = base / "action.h5_3"
    model_h5 = base / "action.h5_3.h5"
    load_path = None
    if model_dir.exists():
        load_path = model_dir
    elif model_h5.exists():
        load_path = model_h5
    if load_path is None:
        print(f"未找到J/Z轨迹模型: 期望 {model_dir}/ (SavedModel) 或 {model_h5} (.h5 文件)。")
        print("请参考README采集并训练，或放置完整模型后重试。")
        return
    print(f"加载J/Z轨迹模型: {load_path}")
    model = load_model(str(load_path), compile=True)

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        last_line = ""
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                smoothed.append(res)
                avg_res = np.mean(smoothed, axis=0)
                pred_idx = int(np.argmax(avg_res))
                predictions.append(pred_idx)
                if len(predictions) >= 10:
                    window = predictions[-10:]
                    vals, counts = np.unique(window, return_counts=True)
                    most_common = int(vals[np.argmax(counts)])
                else:
                    most_common = pred_idx
                if most_common == pred_idx:
                    if avg_res[pred_idx] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])
                if len(sentence) > 5:
                    sentence = sentence[-5:]
                image = prob_viz(avg_res, actions, image, colors)
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            header = ' '.join(sentence)
            cv2.putText(image, header, (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if ui is not None and header and header != last_line:
                ui.textBrowser.append(header)
                last_line = header
            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
