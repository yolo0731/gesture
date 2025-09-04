# 保存canvas+停止绘画？？？
import numpy as np
import os
import cv2
from tracking import HandTrackingModule_letter as htm
import torch
from ml.duan_validation_2 import ConvNet2
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
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    M = cv2.moments(thresh)
    if M["m00"] != 0:
        cX = (M["m10"] / M["m00"]) ; cY = (M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return cX, cY

def recognize_letter_from_image(image_path):
    try:
        from utils import paths
        model_path = paths.models_dir() / 'EMNIST2_5.18.pth'
        if not model_path.exists():
            print(f"未找到字母模型权重: {model_path}")
            return None
        model = ConvNet2()
        model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        model.eval()
        img = cv2.imread(image_path, 0)
        if img is None:
            return None
        img = img[0:400, 400:800]
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        img = cv2.filter2D(img, -1, kernel=kernel)
        img = cv2.blur(img, (15, 1))
        img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_LANCZOS4)
        ret, thresh = cv2.threshold(img, 127, 255, 0)
        M = cv2.moments(thresh)
        if M["m00"] != 0:
            cx = (M["m10"] / M["m00"]) ; cy = (M["m01"] / M["m00"])
            M = np.float32([[1,0,(10-cx)],[0,1,10-cy]])
            img = cv2.warpAffine(img, M, (20, 20))
        img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=[0,0,0])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1723,), (0.3309,))
        ])
        img_tensor = transform(img).unsqueeze(0)
        output = model(img_tensor)
        label = torch.argmax(output)
        index = label.item()
        letter=["A/a","B/b","C/c","D/d","E/e","F/f","G/g","H/h","I/i","J/j","K/k","L/l","M/m","N/n","O/o","P/p"
                ,"Q/q","R/r","S/s","T/t","U/u","V/v","W/w","X/x","Y/y","Z/z"]
        if 1 <= index <= 26:
            return letter[index-1]
        return None
    except Exception as e:
        print(f"字母识别错误: {e}")
        return None

def written_letter(ui=None):
    brushThickness = 50
    eraserThickness = 15
    drawColor = (255, 255, 255)
    videoSourceIndex = 0
    cap = cv2.VideoCapture(videoSourceIndex, cv2.CAP_V4L2)
    if not cap.isOpened():
        err = f"错误: 无法打开摄像头索引 {videoSourceIndex}。可运行 scripts/detect_and_update_camera_index.py 修复。"
        print(err)
        if ui is not None:
            try:
                ui.textBrowser.append(err)
            except Exception:
                pass
        return
    cap.set(3, 1280)
    cap.set(4, 720)
    detector = htm.handDetector()
    xp, yp = 0, 0
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
    prediction_text = ""
    frame_count = 0
    last_prediction = None
    try:
        with open('test_letter.txt', 'w') as f:
            f.write("手写字母识别结果:\n")
        print("✅ test_letter.txt文件初始化成功")
    except Exception as e:
        print(f"❌ test_letter.txt文件初始化失败: {e}")

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
            cv2.imwrite("./test_letter.jpg", imgCanvas)
            roi_debug = imgCanvas[0:400, 400:800]
            roi_sum = np.sum(roi_debug)
            if roi_sum > 0:
                predicted_letter = recognize_letter_from_image("./test_letter.jpg")
                if predicted_letter is not None:
                    print(f"[{predicted_letter}]")
                    prediction_text = f"{predicted_letter}"
                    if ui is not None:
                        ui.textBrowser.append(str(predicted_letter))
                    try:
                        with open('test_letter.txt', 'a') as f:
                            f.write(f"[{predicted_letter}]\n")
                    except Exception as e:
                        print(f"文件写入错误: {e}")
                    if last_prediction != predicted_letter:
                        print(f"🔥 识别到新字母: {predicted_letter}")
                        last_prediction = predicted_letter
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
            cv2.putText(img, "Fail", (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.imshow("Canvas", imgCanvas)
        cv2.imshow("Image", img)
        cv2.imwrite("./test_letter.jpg", imgCanvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') or key == ord('C'):
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)
            prediction_text = ""
            last_prediction = None
            print("🧹 画布已清空，准备识别新字母")
            cv2.imwrite("./test_letter.jpg", imgCanvas)
            try:
                with open('test_letter.txt', 'a') as f:
                    f.write("--- 画布已清空 ---\n")
            except:
                pass
    cap.release()
    cv2.destroyAllWindows()
