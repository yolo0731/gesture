# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 09:38:56 2021

@author: admin
"""
import mediapipe as mp
import cv2
import numpy as np

def get_angle(v1,v2): #空间角度公式
    angle = np.dot(v1,v2)/(np.sqrt(np.sum(v1*v1))*np.sqrt(np.sum(v2*v2)))
    angle = np.arccos(angle)/3.14*180
    
    return angle 
    
    
def get_str_guester(up_fingers):
    
    if len(up_fingers)==1 and up_fingers[0]==8:
        
        #v1 = list_lms[6]-list_lms[7]
        #v2 = list_lms[8]-list_lms[7]
        
        #angle = get_angle(v1,v2)
       
        #if angle<170:
            #number = "9"
        #else:'''
        number = "1"
    
    # elif len(up_fingers)==1 and up_fingers[0]==4:
    #     number = "Good"
    
    # elif len(up_fingers)==1 and up_fingers[0]==20:
    #     number = "Bad"
        
    # elif len(up_fingers)==1 and up_fingers[0]==12:
    #     number = "FXXX"
   
    elif len(up_fingers)==2 and up_fingers[0]==8 and up_fingers[1]==12:
        number = "2"
        
    elif len(up_fingers)==3 and up_fingers[0]==8 and up_fingers[1]==12 and up_fingers[2]==16:
        number = "6"
        
    elif len(up_fingers)==3 and up_fingers[0]==8 and up_fingers[1]==16 and up_fingers[2]==20:
        number = "8"
    
    elif len(up_fingers)==3 and up_fingers[0]==4 and up_fingers[1]==8 and up_fingers[2]==12:
        number = "3"
        
    elif len(up_fingers)==3 and up_fingers[0]==8 and up_fingers[1]==12 and up_fingers[2]==20:
        number = "7"
    
    #elif len(up_fingers)==3 and up_fingers[0]==4 and up_fingers[1]==8 and up_fingers[2]==12:
  
        #dis_8_12 = list_lms[8,:] - list_lms[12,:]
        #dis_8_12 = np.sqrt(np.dot(dis_8_12,dis_8_12))#勾股定理
        
        #dis_4_12 = list_lms[4,:] - list_lms[12,:]
        #dis_4_12 = np.sqrt(np.dot(dis_4_12,dis_4_12))
        
        #if dis_4_12/(dis_8_12+1) <3:
            #number = "7"
        
        # elif dis_4_12/(dis_8_12+1) >5:
        #     number = "Gun"
        #else:
            #number = " "
            
    # elif len(up_fingers)==3 and up_fingers[0]==4 and up_fingers[1]==8 and up_fingers[2]==20:
    #     number = "ROCK"
    
    
    elif len(up_fingers)==4 and up_fingers[0]==8 and up_fingers[1]==12 and up_fingers[2]==16 and up_fingers[3]==20:
        number = "4"
    
    elif len(up_fingers)==5:
        number = "5"
        
    elif len(up_fingers)==0:
        number = "0"
    elif len(up_fingers)==3 and up_fingers[0]==12 and up_fingers[1]==16 and up_fingers[2]==20:
        number = "9"
    else:
        number = " "
        
    return number