import cv2
import numpy as np
import mediapipe as mp

# 初始化 MediaPipe Pose（可考慮根據需要調整其他參數）
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def extract_skeleton_points(image):
    if not isinstance(image, np.ndarray):
        try:
            image = np.array(image)
        except Exception as e:
            raise ValueError("無法轉換影像為 numpy.ndarray") from e

    image = np.array(image, copy=True)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    
    if image is None:
        raise ValueError("讀取到的影像為 None")
    if image.dtype != np.uint8:
        raise ValueError(f"錯誤的影像型態: {image.dtype}, 應為 uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("影像形狀不符合要求，必須為 3-channel 影像")
    
    # --- 新增: 將影像轉換為正方形，提供 IMAGE_DIMENSIONS  ---
    h, w, _ = image.shape
    if h != w:
        size = max(h, w)
        square_image = np.zeros((size, size, 3), dtype=np.uint8)
        square_image[:h, :w] = image
        image = square_image
    
    rgb_image = image[..., ::-1].copy()  # 直接用 numpy 反轉 BGR -> RGB
    
    try:
        results = pose.process(rgb_image)
    except Exception as e:
        raise Exception(f"骨架解析失敗：{e}")
    
    if results.pose_landmarks:
        return [(lm.x, lm.y, lm.visibility) for lm in results.pose_landmarks.landmark]
    return []

def normalize_skeleton_data(skeleton_data, scaler, time_steps=120):
    if len(skeleton_data) >= time_steps:
        slice_data = skeleton_data[-time_steps:]
    else:
        padding = [[(0, 0, 0)] * 33] * (time_steps - len(skeleton_data))
        slice_data = padding + skeleton_data
    accel = []
    for i in range(1, len(slice_data)):
        frame_accel = [((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
                       for p1, p2 in zip(slice_data[i - 1], slice_data[i])]
        accel.append(frame_accel)
    if len(accel) < time_steps:
        accel = [[0] * 33] * (time_steps - len(accel)) + accel
    combo = []
    for i in range(time_steps):
        skeleton_frame = np.array(slice_data[i]).flatten()
        accel_frame = np.array(accel[i])
        combo.append(np.hstack((skeleton_frame, accel_frame)))
    combo = np.array(combo).reshape(1, time_steps, 132)
    flat = combo.reshape(1, -1)
    scaled = scaler.transform(flat)
    return scaled.reshape(1, time_steps, 132)