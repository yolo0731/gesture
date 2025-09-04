# -*- coding: utf-8 -*-
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque


def CSRN(ui):
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    actions = np.array(['Click', 'Stop', 'Rotate', 'No'])
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
    predictions = []
    smoothed = deque(maxlen=10)
    threshold = 0.8
    from utils import paths
    base = paths.models_dir()
    model_dir = base / "CSRN_Model_2"
    model_h5 = base / "CSRN_Model_2.h5"
    try:
        load_path = None
        if model_dir.exists():
            load_path = model_dir
        elif model_h5.exists():
            load_path = model_h5
        if load_path is None:
            print(f"未找到动态手势模型: 期望 {model_dir}/ (SavedModel) 或 {model_h5} (.h5 文件)。")
            print("请按 README 运行训练脚本或放置完整模型后重试。")
            return
        print(f"加载动态手势模型: {load_path}")
        model = load_model(str(load_path), compile=True)
    except Exception as e:
        print(f"加载动态手势模型失败: {e}")
        return

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
            sequence = sequence[-20:]
            if len(sequence) == 20:
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
                if most_common == pred_idx and avg_res[pred_idx] > threshold:
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
            if ui is not None and header != last_line and header.strip():
                ui.textBrowser.append(header)
                last_line = header
            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
