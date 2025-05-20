import os
import cv2
import mediapipe as mp
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from keras import models, layers
from sklearn.preprocessing import MinMaxScaler
import pickle
from tqdm import tqdm  # 新增 tqdm 進度條功能
import random
import matplotlib.pyplot as plt

# === 配置區 ===
NORMAL_FOLDER = 'medias/train_video/normal'
FALL_FOLDER = 'medias/train_video/fall'
OUTPUT_NORMAL_FOLDER = 'outputs/skeletons/N01/normal'
OUTPUT_FALL_FOLDER = 'outputs/skeletons/N01/fall'
MODEL_SAVE_PATH = 'outputs/models/cnn_lstm_fall_detection_model_N01_1.h5'
SCALER_SAVE_PATH = 'outputs/models/scaler_N01_1.pkl'
TEST_VIDEO_PATH = 'medias/test/IMG_8049.mp4'

REGENERATE_SKELETON = False  # 是否重新生成骨架數據
TIME_STEPS = 60  # 推論所需的幀數
INFERENCE_FREQUENCY = 15  # 每隔多少幀進行一次推論

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def extract_skeleton_points(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image = np.ascontiguousarray(rgb_image)  # 確保記憶體排列連續
    results = pose.process(rgb_image)
    if results.pose_landmarks:
        return [(lm.x, lm.y, lm.visibility) for lm in results.pose_landmarks.landmark]
    return []

def process_and_save_skeleton_data(video_path, output_file, progress_bar=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file: {video_path}")
        return

    skeleton_data = []
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            break

        skeleton_points = extract_skeleton_points(frame)
        if skeleton_points:
            skeleton_data.append(skeleton_points)
        else:
            skeleton_data.append([(0, 0, 0)] * 33)
        if progress_bar:
            progress_bar.update(1)
    cap.release()

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(skeleton_data, f)

def count_frames_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        success, _ = cap.read()
        if not success:
            break
        count += 1
    cap.release()
    return count

def process_videos_from_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    video_files = [video_file for video_file in os.listdir(folder_path) if video_file.lower().endswith(('.mp4', '.avi', '.mov'))]

    # 統計所有影片實際處理的總幀數
    total_frames = 0
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        total_frames += count_frames_in_video(video_path)

    # 建立進度條
    with tqdm(total=total_frames, desc=f"Processing frames in {folder_path}") as pbar:
        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            output_file = os.path.join(output_folder, f'{os.path.splitext(video_file)[0]}.json')
            process_and_save_skeleton_data(video_path, output_file, progress_bar=pbar)

def calculate_acceleration(skeleton_data):
    acceleration = []
    for i in range(1, len(skeleton_data)):
        frame_accel = [
            ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
            for p1, p2 in zip(skeleton_data[i - 1], skeleton_data[i])
        ]
        acceleration.append(frame_accel)
    return acceleration

def normalize_skeleton_data(skeleton_data, scaler, time_steps=60):
    skeleton_data = skeleton_data[-time_steps:] if len(skeleton_data) >= time_steps else skeleton_data
    if len(skeleton_data) < time_steps:
        padding = [[(0, 0, 0)] * 33] * (time_steps - len(skeleton_data))
        skeleton_data = padding + skeleton_data

    accel_features = calculate_acceleration(skeleton_data)
    if len(accel_features) < time_steps:
        accel_features = [[0] * 33] * (time_steps - len(accel_features)) + accel_features

    combined_data = [np.hstack((np.ravel(frame), accel)) for frame, accel in zip(skeleton_data, accel_features)]
    flattened_data = np.array(combined_data).reshape(1, -1)
    normalized_data = scaler.transform(flattened_data)
    return normalized_data.reshape(time_steps, 132)

def load_data(normal_folder, fall_folder, time_steps=60):
    X, y = [], []

    def pad_or_truncate(skeleton_data, time_steps):
        if len(skeleton_data) >= time_steps:
            return skeleton_data[:time_steps]
        else:
            padding = [[(0, 0, 0)] * 33] * (time_steps - len(skeleton_data))
            return skeleton_data + padding

    # 取得 fall 資料夾內所有 json 檔案列表
    fall_files = [file for file in os.listdir(fall_folder) if file.lower().endswith('.json')]
    # 取得 normal 資料夾內所有 json 檔案列表
    normal_files = [file for file in os.listdir(normal_folder) if file.lower().endswith('.json')]

    # 處理 normal 檔案 (label = 0)
    for file in normal_files:
        with open(os.path.join(normal_folder, file), 'r') as f:
            skeleton_data = json.load(f)
            skeleton_data = pad_or_truncate(skeleton_data, time_steps)
            accel_features = calculate_acceleration(skeleton_data)
            if len(accel_features) < time_steps:
                accel_features = [[0] * 33] * (time_steps - len(accel_features)) + accel_features
            combined_data = [np.hstack((np.ravel(frame), accel)) for frame, accel in zip(skeleton_data, accel_features)]
            X.append(combined_data)
            y.append(0)

    # 處理 fall 檔案 (label = 1)
    for file in fall_files:
        with open(os.path.join(fall_folder, file), 'r') as f:
            skeleton_data = json.load(f)
            skeleton_data = pad_or_truncate(skeleton_data, time_steps)
            accel_features = calculate_acceleration(skeleton_data)
            if len(accel_features) < time_steps:
                accel_features = [[0] * 33] * (time_steps - len(accel_features)) + accel_features
            combined_data = [np.hstack((np.ravel(frame), accel)) for frame, accel in zip(skeleton_data, accel_features)]
            X.append(combined_data)
            y.append(1)

    return np.array(X), np.array(y)


    model = models.Sequential()
    model.add(layers.LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(layers.Dropout(0.5))
    model.add(layers.LSTM(64))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_lstm_model(input_shape):
    model = models.Sequential()
    
    # CNN layers for spatial feature extraction
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.5))
    
    # LSTM layers for temporal feature extraction
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.LSTM(64))
    model.add(layers.Dropout(0.5))
    
    # Fully connected layer for classification
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


    X, y = load_data(normal_folder, fall_folder, time_steps)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_train_norm = scaler.fit_transform(X_train_flat)
    X_test_norm = scaler.transform(X_test_flat)

    X_train_reshaped = X_train_norm.reshape(X_train.shape[0], time_steps, -1)
    X_test_reshaped = X_test_norm.reshape(X_test.shape[0], time_steps, -1)

    input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
    model = build_lstm_model(input_shape)

    model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test))

    model.save(model_save_path)
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scaler, f)

    return model

def train_cnn_lstm_model(normal_folder, fall_folder, model_save_path, scaler_save_path, time_steps=60):
    X, y = load_data(normal_folder, fall_folder, time_steps)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_train_norm = scaler.fit_transform(X_train_flat)
    X_test_norm = scaler.transform(X_test_flat)

    X_train_reshaped = X_train_norm.reshape(X_train.shape[0], time_steps, -1)
    X_test_reshaped = X_test_norm.reshape(X_test.shape[0], time_steps, -1)

    input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
    model = build_cnn_lstm_model(input_shape)

    model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test))

    model.save(model_save_path)
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scaler, f)

    return model

def predict_fall(skeleton_sequence, model, scaler):
    skeleton_sequence = np.array(skeleton_sequence)
    skeleton_sequence = skeleton_sequence.reshape(1, skeleton_sequence.shape[0], -1)
    prediction = model.predict(skeleton_sequence)
    return prediction[0][0]

def display_video_with_skeleton_and_fall_detection(video_path, model, scaler, time_steps=60, inference_frequency=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file: {video_path}")
        return
    
    skeleton_data = []
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        skeleton_points = extract_skeleton_points(frame)
        skeleton_data.append(skeleton_points)

        h, w, _ = frame.shape
        if skeleton_points:
            points = [(int(point[0] * w), int(point[1] * h)) for point in skeleton_points]
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)

        if frame_count % inference_frequency == 0 and len(skeleton_data) >= time_steps:
            recent_skeleton_data = skeleton_data[-time_steps:]
            normalized_skeleton_data = normalize_skeleton_data(recent_skeleton_data, scaler, time_steps)
            fall_probability = predict_fall(normalized_skeleton_data, model, scaler)

            # 根據 fall_probability 判斷顯示的文字
            if fall_probability > 0.6:
                fall_probability_text = f"Fall Detected: {fall_probability:.2f}"
                cv2.putText(frame, fall_probability_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, 'Warning: Fall Detected!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            elif fall_probability < 0.4:
                fall_probability_text = f"Normal: {fall_probability:.2f}"
                cv2.putText(frame, fall_probability_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                fall_probability_text = f"Uncertain / Unknown: {fall_probability:.2f}"
                cv2.putText(frame, fall_probability_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow('Fall Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

def visualize_score_distribution(y_true, y_scores):
    """
    視覺化二元分類模型的分數分佈圖。
    
    :param y_true: list or array, 真實標籤 (0: Normal, 1: Fall)
    :param y_scores: list or array, 模型預測分數 (sigmoid 輸出)
    """
    # 分離分數到兩個列表
    fall_scores = [score for score, label in zip(y_scores, y_true) if label == 1]
    normal_scores = [score for score, label in zip(y_scores, y_true) if label == 0]

    # 測試用繪圖 (檢查繪圖是否正常)
    plt.figure()
    plt.plot(np.random.normal(size=100), label="Test Plot")
    plt.legend()
    plt.show()

    # 繪製分數分佈圖
    plt.figure(figsize=(10, 6))
    plt.hist(normal_scores, bins=20, alpha=0.7, label='Normal (label=0)', color='green')
    plt.hist(fall_scores, bins=20, alpha=0.7, label='Fall (label=1)', color='red')

    # 添加閾值線
    plt.axvline(x=0.4, color='blue', linestyle='--', label='Threshold: 0.4')
    plt.axvline(x=0.6, color='orange', linestyle='--', label='Threshold: 0.6')

    # 添加圖例、標籤和標題
    plt.legend(loc='upper center')
    plt.xlabel('Prediction Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution of Fall and Normal Classes')

    # 顯示圖表
    plt.show()

if __name__ == '__main__':
    if REGENERATE_SKELETON:
        import shutil
        import tkinter as tk
        from tkinter import messagebox

        # 顯示確認對話視窗
        root = tk.Tk()
        root.withdraw()  # 隱藏主視窗
        answer = messagebox.askyesno("覆蓋確認", "確定要覆蓋現有的骨架資料嗎？此動作會清空輸出骨架檔案的位置。")
        root.destroy()

        if answer:
            # 若確認覆蓋，先清空輸出骨架的資料夾
            for folder in [OUTPUT_NORMAL_FOLDER, OUTPUT_FALL_FOLDER]:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                os.makedirs(folder)
            # 重新生成骨架資料
            process_videos_from_folder(NORMAL_FOLDER, OUTPUT_NORMAL_FOLDER)
            process_videos_from_folder(FALL_FOLDER, OUTPUT_FALL_FOLDER)
        else:
            print("已取消覆蓋，將沿用現有骨架資料。")
    
    print("開始訓練模型...")
    # model = train_lstm_model(OUTPUT_NORMAL_FOLDER, OUTPUT_FALL_FOLDER, MODEL_SAVE_PATH, SCALER_SAVE_PATH, TIME_STEPS)
    model = train_cnn_lstm_model(OUTPUT_NORMAL_FOLDER, OUTPUT_FALL_FOLDER, MODEL_SAVE_PATH, SCALER_SAVE_PATH, TIME_STEPS)
    print("訓練完成，模型與 scaler 已儲存。")

    model = models.load_model(MODEL_SAVE_PATH)
    with open(SCALER_SAVE_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # 測試用隨機數據
    np.random.seed(42)
    y_true_test = np.random.choice([0, 1], size=100, p=[0.7, 0.3])  # 70% Normal, 30% Fall
    y_scores_test = np.random.uniform(0, 1, size=100)  # 隨機分數在 [0, 1] 範圍內

    # 呼叫 visualize_score_distribution 函數
    visualize_score_distribution(y_true_test, y_scores_test)

    display_video_with_skeleton_and_fall_detection(TEST_VIDEO_PATH, model, scaler, TIME_STEPS, INFERENCE_FREQUENCY)