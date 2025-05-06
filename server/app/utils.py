import cv2
import numpy as np
import mediapipe as mp

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 提取骨架點
def extract_skeleton_points(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        return [(lm.x, lm.y, lm.visibility) for lm in results.pose_landmarks.landmark]
    return []

# 正規化骨架資料
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