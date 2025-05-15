from flask import Blueprint, request, jsonify
from ..fall_model import load_fall_model
from ..utils import extract_skeleton_points, normalize_skeleton_data
import numpy as np
import cv2

fall_detection_bp = Blueprint("fall_detection_bp", __name__)

# 載入模型與 Scaler
model, scaler = load_fall_model()

# 用於存儲每個使用者的緩衝區
user_video_buffers = {}   # 使用者原始影像緩衝區
user_skeleton_buffers = {}  # 使用者骨架緩衝區
user_update_counts = {}   # 使用者小緩衝區更新計數

# 可調整參數
LARGE_BUFFER_SIZE = 180  # 原始影像緩衝區大小
SMALL_BUFFER_SIZE = 120  # 骨架緩衝區大小（必須大於 PREDICTION_INTERVAL）
PREDICTION_INTERVAL = 30  # 每次推論所需的幀數

@fall_detection_bp.route("/detect_fall_frame", methods=["POST"])
def detect_fall_frame():
    global user_video_buffers, user_skeleton_buffers, user_update_counts

    # 獲取使用者 ID
    user_id = request.form.get("id")
    if not user_id:
        return jsonify({"error": "缺少使用者 ID"}), 400

    # 一次接收多張圖片，必須有 PREDICTION_INTERVAL 張圖片
    frames_files = request.files.getlist("frames")
    if len(frames_files) != PREDICTION_INTERVAL:
        return jsonify({"error": f"必須上傳 {PREDICTION_INTERVAL} 張圖片，目前接收到 {len(frames_files)} 張"}), 400

    # 如果 ID 不存在於字典中，初始化新的緩衝區和更新計數
    if user_id not in user_video_buffers:
        user_video_buffers[user_id] = []
        user_skeleton_buffers[user_id] = []
        user_update_counts[user_id] = 0

    try:
        # 逐一處理每一張圖片
        for file in frames_files:
            # 將影像轉換為 OpenCV 格式
            file_bytes = np.frombuffer(file.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # 儲存原始影像到大緩衝區
            user_video_buffers[user_id].append(frame)
            if len(user_video_buffers[user_id]) > LARGE_BUFFER_SIZE:
                user_video_buffers[user_id] = user_video_buffers[user_id][-LARGE_BUFFER_SIZE:]

            # 提取骨架點並儲存到小緩衝區
            skeleton_points = extract_skeleton_points(frame)
            if skeleton_points:
                user_skeleton_buffers[user_id].append(skeleton_points)
                user_update_counts[user_id] += 1
                print(f"[DEBUG] User {user_id}: Skeleton buffer updated. Total frames: {len(user_skeleton_buffers[user_id])}")

            if len(user_skeleton_buffers[user_id]) > SMALL_BUFFER_SIZE:
                user_skeleton_buffers[user_id] = user_skeleton_buffers[user_id][-SMALL_BUFFER_SIZE:]

        # 推論邏輯優先使用小緩衝區並檢查更新計數
        result = "Insufficient data"
        if len(user_skeleton_buffers[user_id]) >= SMALL_BUFFER_SIZE:
            if user_update_counts[user_id] >= PREDICTION_INTERVAL:
                norm_data = normalize_skeleton_data(user_skeleton_buffers[user_id], scaler, time_steps=SMALL_BUFFER_SIZE)
                pred = model.predict(norm_data)[0][0]
                result = "Fall" if pred >= 0.5 else "Non-Fall"
                user_update_counts[user_id] = 0
                print(f"[DEBUG] User {user_id}: Prediction made. Result: {result}")

        print(f"[DEBUG] User {user_id}: Frame count: {len(user_video_buffers[user_id])}, Update count: {user_update_counts[user_id]}")
        return jsonify({"id": user_id, "result": result}), 200

    except Exception as e:
        print(f"[ERROR] User {user_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500