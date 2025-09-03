import cv2

def find_first_available_camera(max_index=10):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                return i
    return None

# 找到第一个可用的摄像头索引
index = find_first_available_camera()
if index is None:
    print("❌ 没有找到可用的摄像头，请检查设备。")
else:
    print(f"✅ 找到摄像头，索引号是 {index}")

    # 打开摄像头
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"❌ 摄像头 {index} 打不开。")
    else:
        print(f"✅ 摄像头 {index} 已打开，按 Q 退出窗口。")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取视频帧。")
                break

            cv2.imshow(f"Camera {index}", frame)

            # 按下 q 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
