import cv2
import numpy as np
import mediapipe as mp
from scipy.interpolate import interp1d

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

def kalman_filter_1d(data, process_variance=1e-5, measurement_variance=1e-2):
    """
    對一維資料進行 Kalman 濾波。
    """
    n = len(data)
    x = data[0]
    P = 1.0
    Q = process_variance
    R = measurement_variance
    result = [x]
    for i in range(1, n):
        # 預測
        x = x
        P = P + Q
        # 更新
        K = P / (P + R)
        x = x + K * (data[i] - x)
        P = (1 - K) * P
        result.append(x)
    return result

def interpolate_skeleton_data(skeleton_data, target_length):
    """
    使用 Kalman 濾波器平滑骨架資料，並補足到 target_length。
    """
    skeleton_np = np.array(skeleton_data)  # shape: (T, 33, 3)
    T, J, D = skeleton_np.shape  # T: 幀數, J: 關節數, D: 維度(x, y, v)

    # 對每個關節的每個維度做 Kalman 濾波
    filtered = np.zeros_like(skeleton_np)
    for j in range(J):
        for d in range(D):
            filtered[:, j, d] = kalman_filter_1d(skeleton_np[:, j, d])

    # 若不足 target_length，則重複最後一幀補齊
    if T < target_length:
        pad = np.tile(filtered[-1:], (target_length - T, 1, 1))
        filtered = np.concatenate([filtered, pad], axis=0)
    elif T > target_length:
        filtered = filtered[-target_length:]

    return filtered.tolist()