import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np
from .duan_validation_2 import ConvNet2
#from ResNet_tf import resnet18
from PIL import Image
from torchvision import datasets, transforms
import pyttsx3
from .written_letter import written_letter
def get_one_center(img):
    #img = cv2.imread(imae, 0)

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
def letter_rec(ui):
    written_letter(ui)
    return  # 调用written_letter后直接返回，不执行后续代码
    from . import paths
    model= ConvNet2()
    model.load_state_dict(torch.load(str(paths.models_dir() / 'EMNIST2_5.18.pth'), map_location='cpu'))
    model.eval()
    img = cv2.imread('./test_letter.jpg', 0)  #以灰度图的方式读取要预测的图片
    # plt.imshow(img)
    # plt.show()
    # dst = cv2.fastNlMeansDenoising(img,None,10,10,7,21)
    #img = cv2.blur(img, (3, 3))
    #img = cv2.medianBlur(img, 3)
    # plt.imshow(img)
    # plt.show()
    img =img[0:400,400:800]
    #img = cv2.resize(img, (28, 28))
    #img = cv2.bilateralFilter(img, 0, 5, 100, 15)
    #img = cv2.GaussianBlur(img, (3, 3), 0)
    #gray = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
    #medina = cv2.medianBlur(gray,5)#中值去噪声
    # plt.imshow(img)
    # plt.show()
    #img = cv2.resize(img,(28,28))
    # transform=transforms.Compose([
    #                     transforms.ToTensor(),
                        
    #                     transforms.Normalize((0.1723,), (0.3309,))
    #                    ])
    
    
    
    # img = transform(img).unsqueeze(0)
   
    
   
    
    img = cv2.resize(img, (20, 20),Image.ANTIALIAS)
    # custom_blur_demo(img)
    # blur_demo(img)
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
    print(cx,cy)
    
    
   
   
     
     
   



       

    transform=transforms.Compose([
                        
                        transforms.ToTensor(),
                        #transforms.CenterCrop((28,28)),
                        transforms.Normalize((0.1723,), (0.3309,))
                        ])

    plt.imshow(img)
    plt.show()
    img = transform(img).unsqueeze(0)#可以理解为将此数据视为一个整体而给予的一个索引，方便网络对数据的批处理。
    
    output = model(img)
    #print(output)
    label = torch.argmax(output)
    #print(label)#1-26==Aa-Zz
    index=label.item()
    #print(index)
    letter=["A/a","B/b","C/c","D/d","E/e","F/f","G/g","H/h","I/i","J/j","K/k","L/l","M/m","N/n","O/o","P/p"
            ,"Q/q","R/r","S/s","T/t","U/u","V/v","W/w","X/x","Y/y","Z/z"]
    print("The predicted letter is ",letter[index-1])
    result = letter[index-1]
    ui.textBrowser.append(result) 
    engine = pyttsx3.init() #初始化
    print('准备开始语音播报...')
    engine.say(result)
    
    engine.runAndWait()
    engine.stop()
    
    
    

    
    

# #index和字母转换
#     if index == 1 :
#        index ="A/a"
       
#     if index == 2:
#         index="B/b"
           
#     if index == 4:
#         index="D/d" 
              
#     if index == 5:
#         index="E/e" 
                 
#     if index == 6:
#         index="F/f" 
                    
#     if index == 7:
#         index="G/g" 
                       
#     if index == 8:
#         index="H/h" 
                          
#     if index == 9:
#         index="I/i" 
                             
#     if index == 10:
#         index="J/j" 
                                
#     if index == 11:
#         index="K/k" 
                                   
#     if index == 3:
#         index="C/c" 
                                      
#     if index == 12:
#         index="L/l" 
                                         
#     if index == 13:
#         index="M/m"
                                            
#     if index == 14:
#         index="N/n"
                                               
#     if index == 15:
#         index="O/o"
                                                  
#     if index == 16:
#         index="P/p"
                                                     
#     if index == 17:
#         index="Q/q"
                                                        
#     if index == 18:
#         index="R/r"
                                                           
#     if index == 19:
#         index="S/s"
                                                              
#     if index == 20:
#         index="T/t"
                                                                 
#     if index == 21:
#         index="U/u" 
                                                                    
#     if index == 22:
#         index="V/v"
                                                                       
#     if index == 23:
#         index="W/w"
                                                                          
#     if index == 24:
#         index="X/x"
                                                                             
#     if index == 25:
#         index="Y/y" 
                                                                                
#     if index == 26:
#         index="Z/z" 
    
#     print(index)   
              
              
    
 
    
