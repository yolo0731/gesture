# 保存canvas+停止绘画？？？
import numpy as np
import os
import cv2
from tracking import HandTrackingModule as htm
import torch
from ml.cnntrain_mnist import ConvNet
from torchvision import transforms
import sys
from PIL import Image

_DIGIT_MODEL = None

def _get_digit_model():
    global _DIGIT_MODEL
    if _DIGIT_MODEL is None:
        from utils import paths
        # 兼容历史保存方式：模型可能以 __main__.ConvNet 名称被pickle
        try:
            # 将当前包内的 ConvNet 映射到 __main__.ConvNet，便于 torch.load 定位
            import ml.cnntrain_mnist as cm
            setattr(sys.modules.get('__main__'), 'ConvNet', cm.ConvNet)
        except Exception:
            pass
        _DIGIT_MODEL = torch.load(str(paths.models_dir() / 'MNIST1.pth'), map_location='cpu')
        _DIGIT_MODEL.eval()
    return _DIGIT_MODEL

def recognize_digit_from_image(image_path):
    """使用number_rec.py的识别逻辑识别图片中的数字"""
    try:
        # 仅首次加载模型，后续复用缓存，避免每次识别都重复加载
        model = _get_digit_model()
        
        # 读取图片 (使用number_rec.py中的逻辑)
        img = cv2.imread(image_path, 0)  # 灰度图
        if img is None:
            return None
            
        # 提取绘画区域 (和number_rec.py一样)
        img = img[0:400, 400:800] 
        
        # 锐化和模糊 (number_rec.py的处理)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        img = cv2.filter2D(img, -1, kernel=kernel)
        img = cv2.blur(img, (15, 1))
        
        # 调整大小到20x20 (修复ANTIALIAS错误)
        img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_LANCZOS4)
        
        # 计算质心并居中 (number_rec.py的逻辑)
        ret, thresh = cv2.threshold(img, 127, 255, 0)
        M = cv2.moments(thresh)
        if M["m00"] != 0:
            cx = (M["m10"] / M["m00"])
            cy = (M["m01"] / M["m00"])
            M = np.float32([[1,0,(10-cx)],[0,1,10-cy]])
            img = cv2.warpAffine(img, M, (20, 20))
        
        # 添加边框到28x28
        img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=[0,0,0])
        
        # 转换和预测 (number_rec.py的逻辑)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        img_tensor = transform(img).unsqueeze(0)
        output = model(img_tensor)
        label = torch.argmax(output)
        index = label.item()
        
        return index
        
    except Exception as e:
        print(f"识别错误: {e}")
        return None

def written_number(ui=None):
    brushThickness = 50
    eraserThickness = 15

    drawColor = (255, 255, 255)
    videoSourceIndex = 0
    # Use V4L2 backend for Linux instead of DSHOW (Windows-specific)
    cap = cv2.VideoCapture(videoSourceIndex, cv2.CAP_V4L2)
    if not cap.isOpened():
        # 回退到默认后端（跨平台）
        cap = cv2.VideoCapture(videoSourceIndex)
    
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
        with open('test.txt', 'w') as f:
            f.write("手写数字识别结果:\n")
        print("✅ test.txt文件初始化成功")
    except Exception as e:
        print(f"❌ test.txt文件初始化失败: {e}")

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("错误: 无法读取摄像头帧")
            break

        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        a1,b1,c1,d1=400,0,400,400
        cv2.rectangle(img,(a1,b1),(a1+c1,b1+d1),(0,255,0),0)
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            fingers = detector.fingersUp()

            if fingers[1] and fingers[2] == False:
                cv2.circle(img, (x1,y1), 15, drawColor, cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

        frame_count += 1
        if frame_count % 10 == 0:
            cv2.imwrite("./test.jpg", imgCanvas)
            roi_debug = imgCanvas[0:400, 400:800]
            roi_sum = np.sum(roi_debug)
            if roi_sum > 0:
                predicted_digit = recognize_digit_from_image("./test.jpg")
                if predicted_digit is not None:
                    print(f"[{predicted_digit}]")
                    prediction_text = f"{predicted_digit}"
                    if ui is not None:
                        ui.textBrowser.append(str(predicted_digit))
                    try:
                        with open('test.txt', 'a') as f:
                            f.write(f"[{predicted_digit}]\n")
                    except Exception as e:
                        print(f"文件写入错误: {e}")
                    if last_prediction != predicted_digit:
                        print(f"🔥 识别到新数字: {predicted_digit}")
                        last_prediction = predicted_digit
                else:
                    prediction_text = "识别失败"
                    print("⚠️  画布有内容但识别失败")
            else:
                prediction_text = ""
                if frame_count % 100 == 0:
                    print("ℹ️  画布区域为空")

        img = cv2.bitwise_or(img, imgCanvas)
        if prediction_text and prediction_text != "识别失败":
            cv2.putText(img, ' %s' % prediction_text, (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 5, cv2.LINE_AA)
        elif prediction_text == "识别失败":
            cv2.putText(img, "识别失败", (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.imshow("Canvas", imgCanvas)
        cv2.imshow("Image", img)
        cv2.imwrite("./test.jpg", imgCanvas)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') or key == ord('C'):
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)
            prediction_text = ""
            last_prediction = None
            print("🧹 画布已清空，准备识别新数字")
            cv2.imwrite("./test.jpg", imgCanvas)
            try:
                with open('test.txt', 'a') as f:
                    f.write("--- 画布已清空 ---\n")
            except:
                pass
    cap.release()
    cv2.destroyAllWindows()
