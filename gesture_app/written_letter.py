# 保存canvas+停止绘画？？？
import numpy as np
import os
import cv2
from . import HandTrackingModule_letter as htm
import torch
from .duan_validation_2 import ConvNet2
from torchvision import transforms
from PIL import Image

def custom_blur_demo(image):
    """锐化"""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) 
    return cv2.filter2D(image, -1, kernel=kernel)

def blur_demo(image):
    """模糊"""
    return cv2.blur(image, (15, 1))

def get_one_center(img):
    """计算图像质心"""
    # 确保图像是灰度图像
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    M = cv2.moments(thresh)
    if M["m00"] != 0:
        cX = (M["m10"] / M["m00"])
        cY = (M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return cX, cY

def recognize_letter_from_image(image_path):
    """使用letter_rec.py的识别逻辑识别图片中的字母"""
    try:
        # 加载模型
        from . import paths
        model = ConvNet2()
        model.load_state_dict(torch.load(str(paths.models_dir() / 'EMNIST2_5.18.pth'), map_location='cpu'))
        model.eval()
        
        # 读取图片 (使用letter_rec.py中的逻辑)
        img = cv2.imread(image_path, 0)  # 灰度图
        if img is None:
            return None
            
        # 提取绘画区域 (和letter_rec.py一样)
        img = img[0:400, 400:800] 
        
        # 锐化和模糊 (letter_rec.py的处理)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        img = cv2.filter2D(img, -1, kernel=kernel)
        img = cv2.blur(img, (15, 1))
        
        # 调整大小到20x20 (使用INTER_LANCZOS4避免PIL错误)
        img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_LANCZOS4)
        
        # 计算质心并居中 (letter_rec.py的逻辑)
        ret, thresh = cv2.threshold(img, 127, 255, 0)
        M = cv2.moments(thresh)
        if M["m00"] != 0:
            cx = (M["m10"] / M["m00"])
            cy = (M["m01"] / M["m00"])
            M = np.float32([[1,0,(10-cx)],[0,1,10-cy]])
            img = cv2.warpAffine(img, M, (20, 20))
        
        # 添加边框到28x28
        img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=[0,0,0])
        
        # 转换和预测 (letter_rec.py的逻辑)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        img_tensor = transform(img).unsqueeze(0)
        output = model(img_tensor)
        label = torch.argmax(output)
        index = label.item()
        
        # 字母转换 (和letter_rec.py一样)
        letter=["A/a","B/b","C/c","D/d","E/e","F/f","G/g","H/h","I/i","J/j","K/k","L/l","M/m","N/n","O/o","P/p"
                ,"Q/q","R/r","S/s","T/t","U/u","V/v","W/w","X/x","Y/y","Z/z"]
        
        if index >= 1 and index <= 26:
            return letter[index-1]
        else:
            return None
        
    except Exception as e:
        print(f"字母识别错误: {e}")
        return None

def written_letter(ui=None):
    brushThickness = 50
    eraserThickness = 15

    #folderPath = "Paint"
    #myList = os.listdir(folderPath)
    #print(myList)
    #overlayList = []

    #for imPath in myList:
       # image = cv2.imread(f'{folderPath}/{imPath}')
       # overlayList.append(image)
    #print(len(overlayList))

    #header = overlayList[0]
    drawColor = (255, 255, 255)
    videoSourceIndex = 0
    # Use V4L2 backend for Linux instead of DSHOW (Windows-specific)
    cap = cv2.VideoCapture(videoSourceIndex, cv2.CAP_V4L2)
    #cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头索引 {videoSourceIndex}")
        return
    
    cap.set(3, 1280)#长
    cap.set(4, 720)#宽

    detector = htm.handDetector()
    xp, yp = 0, 0
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
    
    # 识别相关变量
    prediction_text = ""
    frame_count = 0
    last_prediction = None
    
    # 初始化输出文件
    try:
        with open('test_letter.txt', 'w') as f:
            f.write("手写字母识别结果:\n")
        print("✅ test_letter.txt文件初始化成功")
    except Exception as e:
        print(f"❌ test_letter.txt文件初始化失败: {e}")




    while cap.isOpened():
        #import image
        success, img = cap.read()
        
        # Check if frame was read successfully
        if not success:
            print("错误: 无法读取摄像头帧")
            break

        # find hand landmarks
        img = cv2.flip(img, 1)#水平翻转，便于书写
        img = detector.findHands(img)#找一帧手
        lmList = detector.findPosition(img, draw=False)#找手的坐标
        a1,b1,c1,d1=400,0,400,400
        cv2.rectangle(img,(a1,b1),(a1+c1,b1+d1),(0,255,0),0)
        if len(lmList) != 0:
            #print(lmList)

            # tip of index and middle finger
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            # check which fingures are up
            fingers = detector.fingersUp()
            # print(fingers)

            # if selection mode - two fingers are up
            #if fingers[1] and fingers[2]:
             #   xp, yp = 0, 0
              #  print("Selection Mode")
               # if y1 < 125:
                    # checking for click
                #    if 250<x1<450:
                 #       header = overlayList[0]
                  #      drawColor = (255, 0, 255)
                   # elif 550<x1<750:
                    #    header = overlayList[1]
                     #   drawColor = (255, 0, 100)
                    #elif 800<x1<950:
                     #   header = overlayList[2]
                      #  drawColor = (0, 255, 0)
                    #elif 1050<x1<1200:
                     #   header = overlayList[3]
                      #  drawColor = (0, 0, 0)

               # cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
               
            # if drawing mode - index finger is up
            if  fingers[1] and fingers[2] == False:
                
                cv2.circle(img, (x1,y1), 15, drawColor, cv2.FILLED)#在每一点画圆圈
                
                print("Drawing Mode")

                if xp == 0 and yp == 0:#实现不从原点跟随
                    xp, yp = x1, y1

                # if drawColor == (255, 0, 255):
                #     cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                #     cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                # else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)#当前点和之前点之间连线
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            #cv2.rectangle(img,(xp,yp),(xp+500,yp+500),(255,255,255),3)
            xp, yp = x1, y1#不断更新实现两点不断连线
                    

                
        
        # imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)#灰度化
        # _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)#二值化
        # imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)#黑白化
        # img = cv2.bitwise_and(img, imgInv)

        # 实时保存test_letter.jpg并识别（每10帧执行一次）
        frame_count += 1
        if frame_count % 10 == 0:
            # 保存Canvas到test_letter.jpg
            cv2.imwrite("./test_letter.jpg", imgCanvas)
            
            # 检查画布是否有内容
            roi_debug = imgCanvas[0:400, 400:800]
            roi_sum = np.sum(roi_debug)
            
            if roi_sum > 0:  # 画布有内容时才进行识别
                # 使用letter_rec的识别逻辑识别test_letter.jpg
                predicted_letter = recognize_letter_from_image("./test_letter.jpg")
                
                if predicted_letter is not None:
                    # 实时输出到控制台（像数字识别那样刷屏显示）
                    print(f"[{predicted_letter}]")
                    
                    prediction_text = f"{predicted_letter}"
                    
                    # **像letter_gesture.py那样，每次识别成功都更新UI**
                    if ui is not None:
                        ui.textBrowser.append(str(predicted_letter))
                    
                    # 保存到test_letter.txt文件（像数字识别那样）
                    try:
                        with open('test_letter.txt', 'a') as f:
                            f.write(f"[{predicted_letter}]\n")
                    except Exception as e:
                        print(f"文件写入错误: {e}")
                    
                    # 如果识别结果发生变化，输出更明显的提示
                    if last_prediction != predicted_letter:
                        print(f"🔥 识别到新字母: {predicted_letter}")
                        last_prediction = predicted_letter
                else:
                    prediction_text = "识别失败"
                    print("⚠️  画布有内容但识别失败")
            else:
                prediction_text = ""
                if frame_count % 100 == 0:  # 每100帧提示一次
                    print("ℹ️  画布区域为空")

        img = cv2.bitwise_or(img, imgCanvas)#通过与黑色背景图的或运算，实现白色轨迹在img上的保留
        
        # 在图像上显示识别结果（像letter_gesture.py那样的大字体黄色显示）
        if prediction_text and prediction_text != "识别失败":
            # 大字体显示字母（像letter_gesture.py那样）
            cv2.putText(img, ' %s' % prediction_text, (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 5, cv2.LINE_AA)
        elif prediction_text == "识别失败":
            # 识别失败的提示
            cv2.putText(img, "Fail", (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        cv2.imshow("Canvas", imgCanvas)
        cv2.imshow("Image", img)
        cv2.imwrite("./test_letter.jpg", imgCanvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按q退出
            break
        elif key == ord('c') or key == ord('C'):  # 清空画布
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)
            prediction_text = ""
            last_prediction = None
            print("🧹 画布已清空，准备识别新字母")
            
            # 清空test_letter.jpg
            cv2.imwrite("./test_letter.jpg", imgCanvas)
            
            # 在test_letter.txt中记录清空操作
            try:
                with open('test_letter.txt', 'a') as f:
                    f.write("--- 画布已清空 ---\n")
            except:
                pass
    
    cap.release()
    cv2.destroyAllWindows()

