# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 10:27:10 2022

@author: admin
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from .written_number import written_number

import torch
from .cnntrain_mnist import ConvNet
import glob
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import torchvision
from skimage import io,transform
import pyttsx3
def get_one_center(img):
    #img = cv2.imread(image, 0)

    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(img, 127, 255, 0)

    # calculate moments of binary image
    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    cX = (M["m10"] / M["m00"])
    cY = (M["m01"] / M["m00"])
    return cX,cY



def custom_blur_demo(image):
  kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
  image = cv2.filter2D(image, -1, kernel=kernel)
  
def blur_demo(image):
  image = cv2.blur(image, (15, 1))
  
  
def number_rec(ui):
    written_number(ui)
    return  # 调用written_number后直接返回，不执行后续代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from . import paths
    model = torch.load(str(paths.models_dir() / 'MNIST1.pth')) # 加载模型
    #model = model.to(device)
    model.eval()    #把模型转为test模式



    img = cv2.imread('./test.jpg', 0)  #以灰度图的方式读取要预测的图片
    # plt.imshow(img)
    # plt.show()
    # dst = cv2.fastNlMeansDenoising(img,None,10,10,7,21)
    #img = cv2.blur(img, (3, 3))
    #img = cv2.medianBlur(img, 3)
    # plt.imshow(img)
    # plt.show()
    img =img[0:400,400:800] 
    #img = cv2.bilateralFilter(img, 0, 5, 100, 15)
    #img = cv2.GaussianBlur(img, (3, 3), 0)
    #gray = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
    #medina = cv2.medianBlur(gray,5)#中值去噪声
    # plt.imshow(img)
    # plt.show()
    
   
    
    img = cv2.resize(img, (20, 20),Image.ANTIALIAS)
    custom_blur_demo(img)
    blur_demo(img)
    # # Find indices where we have mass
    # mass_x, mass_y = np.where(img >= 255)
    # # mass_x and mass_y are the list of x indices and y indices of mass pixels

    # cent_x = np.average(mass_x)
    # cent_y = np.average(mass_y)
    # print(cent_x,cent_y)
    #cv2.circle(img, (int(cent_x), int(cent_y)), 1, (255, 0, 0))
    cx,cy=get_one_center(img)
    print(cx,cy)
    M = np.float32([[1,0,(10-cx)],[0,1,10-cy]])
    res = cv2.warpAffine(img, M, (20, 20))
    
    img = cv2.copyMakeBorder(res, 4, 4, 4, 4, cv2.BORDER_CONSTANT,value=[0,0,0])
    # cx,cy=get_one_center(img)
    # print(cx,cy)
    # plt.imshow(img)
    # plt.show()
    # cv2.imwrite("D:/Automation/Final Project/usable/fashion_mnist/saved/test_cut.jpg", img)
    
    
    # #img = cv2.medianBlur(img, 3)
   
    # # plt.imshow(img)
    # # plt.show()
   

    # # height,width=img.shape
    # # dst=np.zeros((height,width),np.uint8)
    # # for i in range(height):
    # #     for j in range(width):
    # #         dst[i,j]=255-img[i,j]

    # # img = dst
    # #plt.imshow(img)
    # #plt.show()
    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])

    img = transform(img).unsqueeze(0)#可以理解为将此数据视为一个整体而给予的一个索引，方便网络对数据的批处理。
    output = model(img)
    #print(output)
    label = torch.argmax(output)
    #print(label)#0-9=0-9
    index=label.item()
    number=["0","1","2","3","4","5","6","7","8","9"]
    print("The predicted number is ",number[index])
    result= number[index]
    engine = pyttsx3.init() #初始化
    ui.textBrowser.append(result) 
    print('准备开始语音播报...')
    engine.say(result)
    
    engine.runAndWait()
    engine.stop()

    
    
    # # img=np.array(img).astype(np.float32)
    # # img=np.expand_dims(img,0)
    # # img=np.expand_dims(img,0)#扩展后，为[1，1，28，28]
    # # img=torch.from_numpy(img)
    # # img = img.to(device)
    # # output=model(Variable(img))
    # # prob = F.softmax(output, dim=1)
    # # prob = Variable(prob)
    # # prob = prob.cpu().numpy()  #用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
    # # print(prob)  #prob是10个分类的概率
    # # pred = np.argmax(prob) #选出概率最大的一个
    # # print(pred.item())


