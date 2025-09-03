import cv2
import mediapipe as mp
import time
from .Finger import Finger
from .letter_interpre import interpret
import pyttsx3

import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,load_model
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

    # Finger objects
    THUMB = Finger()
    INDEX = Finger()
    MIDDLE = Finger()
    RING = Finger()
    PINKY = Finger()

    # captures video from webcam
    # NOTE: input value can vary between -1, 0, 1, 2 (differs per device, 0 or 1 is common)
    # WARNING: VideoCapture does not work if another application is using camera (ie. video calling)
    cap = cv2.VideoCapture(0)

    # from pre-trained Mediapipe to draw hand landmarks and connections
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1)
    mpDraw = mp.solutions.drawing_utils

    # used to calculate FPS
    # pTime = 0  # previous time
    # cTime = 0  # current time

    # used to record letter every 3 seconds hand is in frame
    prev_time = time.time()
    curr_time = time.time()

    letters = ""

    while True:
        # reads image from webcam
        _, img = cap.read()
        h, w, c = img.shape                 # get height, width, depth

        # converts default image value to RGB value
        # NOTE: when printing back to the screen, use default value (img) NOT imgRGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False  # improves performance
        # use Mediapipe to process converted RGB value
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:

            for handLms in results.multi_hand_landmarks:
                # creates list of all landmarks for easier indexing
                # list will have 21 values -> lm_list[0] will be first landmark
                lm_list = []

                # id corresponds to landmark #
                #   -> 21 landmarks in total (4 on non-thumb fingers, rest on thumb and palm)
                # lm corresponds to landmark value
                #   -> each lm has x coordinate and y coordinate
                #   -> default values are in ratio (value between 0 and 1)
                #   -> to convert to pixel value, multiple by width and height of screen
                for id, lm in enumerate(handLms.landmark):
                    # convert to x, y pixel values
                    cx, cy, cz = int(lm.x*w), int(lm.y*h), lm.z*c

                    lm_list.append([id, cx, cy, cz])

                # writes text to screen
                if str(interpret(lm_list)) != 0:
                    cv2.putText(img, str(interpret(lm_list)), (w-300, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (52, 195, 235), 3)
                    ui.textBrowser.append(interpret(lm_list)) 
                    
                
                

                # draw hand landmarks and connections
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                # countdown timer
                curr_time = time.time()
                diff_time = curr_time - prev_time
                if diff_time < 1:
                    display_time = 3
                elif diff_time < 2:
                    display_time = 2
                elif diff_time <= 3:
                    display_time = 1
        else:
            # reset timer when hand not in frame
            prev_time = time.time()
        
        # capture letter of hand every three seconds
        curr_time = time.time()
        if curr_time - prev_time > 3:
            try:
                letters += interpret(lm_list)
            except TypeError:
                pass
            cv2.putText(img, "captured", (w//2 - 200,h//2), cv2.FONT_HERSHEY_DUPLEX, 3, (235, 107, 52), 3)
            prev_time = time.time()
        
        #cv2.putText(img, letters, (10, h - 50), cv2.FONT_HERSHEY_DUPLEX, 3, (235, 143, 52), 3)

        
        # print FPS on screen (not console)
        # cTime = time.time()
        # fps = 1/(cTime-pTime)
        # pTime = cTime
        # cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)

        # print current image captured from webcam
        cv2.imshow("Image", img)
        key = cv2.waitKey(10)
        
        # press Q to quit or "stop" button
        if key & 0xFF == ord('q'):  # 按q退出
            break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()
    

    engine = pyttsx3.init() #初始化
    print('准备开始语音播报...')
    engine.say(letters)  
    engine.runAndWait()  
    engine.stop()
    
    
def JZRec(ui):
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities
    # actions = np.array(['Jj', 'Zz','None'])
    actions = np.array(['Jj', 'Zz','None'])
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
    threshold = 0.8
    predictions = []
    smoothed = deque(maxlen=10)
    from . import paths
    model = load_model(str(paths.models_dir() / "action.h5_3"), compile=True) 

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
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                smoothed.append(res)
                avg_res = np.mean(smoothed, axis=0)
                pred_idx = int(np.argmax(avg_res))
                predictions.append(pred_idx)
                
                
            #3. Viz logic
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

                # Viz probabilities
                image = prob_viz(avg_res, actions, image, colors)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            header = ' '.join(sentence)
            cv2.putText(image, header, (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if ui is not None and header and header != last_line:
                ui.textBrowser.append(header)
                last_line = header
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        
        
# letter_gesture()

# JZRec()
