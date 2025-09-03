# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:22:18 2022

@author: ZH5002
"""

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from collections import deque

def CSRN(ui):
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities
    # actions = np.array(['Jj', 'Zz','None'])
    actions = np.array(['Click','Stop','Rotate','No'])
    colors = [(245,117,16), (117,245,16),(16,117,245),(50,50,50)]
    def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
        return output_frame

    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results


    def draw_styled_landmarks(image, results):
        # Draw face connections
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
        #                           mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
        #                           mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        #                           ) 
        # # Draw pose connections
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        #                           mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
        #                           mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        #                           ) 
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 ) 
        # Draw right hand connections  
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 ) 
    def extract_keypoints(results):
        # pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([lh, rh])



    # 1. New detection variables
    sequence = []
    sentence = []
    predictions = []
    smoothed = deque(maxlen=10)
    threshold = 0.8
    from . import paths
    model = load_model(str(paths.models_dir() / "CSRN_Model_2"), compile=True)

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        last_line = ""
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-20:]
            
            if len(sequence) == 20:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                smoothed.append(res)
                avg_res = np.mean(smoothed, axis=0)
                pred_idx = int(np.argmax(avg_res))
                predictions.append(pred_idx)
                
                
            #3. Viz logic
                if len(predictions) >= 10:
                    # 使用众数稳定决策，替代 np.unique 的不稳定行为
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

                # Viz probabilities（使用平滑后的分数）
                image = prob_viz(avg_res, actions, image, colors)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            header = ' '.join(sentence)
            cv2.putText(image, header, (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if ui is not None and header != last_line and header.strip():
                ui.textBrowser.append(header)
                last_line = header
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
