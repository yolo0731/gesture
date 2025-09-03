# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 08:24:59 2021

@author: admin
"""

import mediapipe as mp
import cv2
import numpy as np
from . import definition
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pyttsx3

def number_gesture(ui):
    cap = cv2.VideoCapture(0)# 打开摄像头
    # 定义手 检测对象
    mpHands = mp.solutions.hands
    #手部检测器 (跟踪加监测（静态图像改成1），检测置信，跟踪置信手掌)
    #model_complexity 目标跟踪加检测，根据当前目标点的位置，预测下一时刻可能出现的点的位置
    hands = mpHands.Hands(max_num_hands=2,model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    while True:
        success, img = cap.read()# 读一帧图像
        if not success:
            continue
        image_height, image_width, _ = np.shape(img)
        # BGR转换为RGB,不然蓝色调会变多
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 得到检测结果
        results = hands.process(imgRGB)
        if results.multi_hand_landmarks:#如果有一帧图像被检测出来
            for hand in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img,hand,mpHands.HAND_CONNECTIONS)#关键点用线描出来
                
                
                
                
                
            # 采集所有关键点的坐标
            list_lms = []    
            for i in range(21):
                position_x = hand.landmark[i].x*image_width#此处已经归一化
                position_y = hand.landmark[i].y*image_height
                list_lms.append([int(position_x),int(position_y)])
            #position_x = hand.landmark[8].x*image_width#此处已经归一化
            #position_y = hand.landmark[8].y*image_height
            #list_lms.append([int(position_x),int(position_y)])
            
            
            # Hull Construction 
            list_lms = np.array(list_lms,dtype=np.int32)
            hull_index = [0,1,2,3,6,10,14,19,18,17,0]
            hull = cv2.convexHull(list_lms[hull_index])#寻找 Hull
            # 绘制Hull 多边形 闭合
            cv2.polylines(img,[hull], True, (0, 255, 0), 2)
            
            
            
            # 查找外部的点数
            #n_fig = -1
            fingertips_list = [4,8,12,16,20] 
            up_fingers = []
            collector = []
            #食指指尖坐标采集
            a=list_lms[8]
            print(a)
            
            f = open('test.txt', 'a')

            f.write(str(a))
            f.write('\r\n')
            #f.close()
            
            #使用最短距离函数判断内外关系
            for i in fingertips_list:
                i_ordinate = (int(list_lms[i][0]),int(list_lms[i][1]))#每一个点的坐标
                dist= cv2.pointPolygonTest(hull,i_ordinate,True)#True表示求距离 找每一个指尖到凸包的最短距离
                if dist <0: #在Hull外
                    up_fingers.append(i)
            
            # print(up_fingers)
            # print(list_lms)
            # print(np.shape(list_lms))
            #根据在外面的指尖的个数判断数字
            str_guester = definition.get_str_guester(up_fingers)
            # engine = pyttsx3.init() #初始化
            # print('准备开始语音播报...')
            # engine.say(str_guester)
              
            # engine.runAndWait()  
            # engine.stop()
            # collector.append(str_guester)
            
            cv2.putText(img,' %s'%(str_guester),(90,90),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,0),5,cv2.LINE_AA)
            
            ui.textBrowser.append(str_guester) 
             
            # for i in fingertips_list:
            #     pos_x = hand.landmark[i].x*image_width
            #     pos_y = hand.landmark[i].y*image_height
            #     # 画点
            #     cv2.circle(img, (int(pos_x),int(pos_y)), 3, (0,255,255),-1)
                    
       
        cv2.imshow("hands",img)

        #key =  cv2.waitKey(1) & 0xFF   

        # 按键 "q" 退出
        #if key ==  ord('q'):
        
        if cv2.waitKey(10) & 0xFF == ord('q'):  # 按q退出
            break
    cap.release() 
    cv2.destroyAllWindows()

       
# engine = pyttsx3.init() #初始化
# print('准备开始语音播报...')
# engine.say(collector)
  
# engine.runAndWait()  
# engine.stop()
# return str_guester    


    
    
    
    
    
    
    
    
    
    
    
    
