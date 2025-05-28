import math
import os
import tempfile
import datetime
import threading
import cv2
import numpy as np
from flask import Blueprint, request, jsonify
from ..fall_model import load_fall_model
from ..utils import extract_skeleton_points, normalize_skeleton_data
from ..db import get_connection

fall_detection_bp = Blueprint("fall_detection_bp", __name__)

TEMP_VIDEO_DIR = "tmp"
VIDEOS_DIR = "fall_videos"

model, scaler = load_fall_model()

# 使用 temp 檔案儲存影片段
user_temp_files = {}            # 用於儲存跌倒前的影片檔 (pre-fall)
user_post_temp_files = {}       # 用於跌倒後上傳的影片檔 (post-fall)
user_skeleton_buffers = {}
user_post_fall_buffers = {}
user_update_counts = {}
user_can_save = {}

SMALL_BUFFER_SIZE = 60          # 用於骨架推論最低骨架資料筆數
BUFFER_UPDATE_SIZE = 30
SERVER_MAX_FPS = 30             # 伺服器最大處理 FPS

PRE_FALL_SECONDS = 2            # 前置影片秒數上限 (秒)
POST_FALL_SECONDS = 2           # 跌倒後影片秒數上限 (秒)

def save_video(user_id, frames):
    if not frames:
        return None
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    video_filename = f"{user_id}_{date_str}_001.mp4"
    video_path = os.path.join(VIDEOS_DIR, video_filename)
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, SERVER_MAX_FPS, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    return video_filename

def save_to_db(user_id, location, pose_before_fall, video_filename, result):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = """
                INSERT INTO fall_events (user_id, detected_time, location, pose_before_fall, video_filename)
                VALUES (%s, %s, %s, %s)
                """
        values = (user_id, datetime.datetime.now(), location, pose_before_fall, video_filename, result)
        cursor.execute(query, values)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"[ERROR] DB 儲存失敗: {e}")

def get_frames_in_file(file_path):
    cap = cv2.VideoCapture(file_path)
    frame_count = 0
    if cap.isOpened():
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def get_total_frames_in_files(file_list):
    total = 0
    for file_path in file_list:
        total += get_frames_in_file(file_path)
    return total

def update_temp_files(user_id, max_frames):
    """
    當前置暫存區累計幀數超過 max_frames 時，
    由最舊檔案開始刪除，直到累計幀數低於或等於上限。
    """
    while get_total_frames_in_files(user_temp_files.get(user_id, [])) > max_frames and user_temp_files[user_id]:
        old = user_temp_files[user_id].pop(0)
        if os.path.exists(old):
            os.remove(old)

def update_post_files(user_id, max_frames):
    """
    當後置暫存區累計幀數超過 max_frames 時，
    由最舊檔案開始刪除，直到累計幀數低於或等於上限。
    """
    while get_total_frames_in_files(user_post_temp_files.get(user_id, [])) > max_frames and user_post_temp_files[user_id]:
        old = user_post_temp_files[user_id].pop(0)
        if os.path.exists(old):
            os.remove(old)

@fall_detection_bp.route("/detect_fall_video", methods=["POST"])
def detect_fall_video():
    global user_temp_files, user_post_temp_files, user_skeleton_buffers, user_update_counts, user_can_save, user_post_fall_buffers

    user_id = request.form.get("id")
    if not user_id:
        return jsonify({"error": "缺少使用者 ID"}), 400

    video_file = request.files.get("video")
    if not video_file:
        return jsonify({"error": "未接收到影片檔案"}), 400

    if not video_file.filename.lower().endswith(".mp4"):
        return jsonify({"error": "檔案格式不符，請上傳 MP4 影片"}), 400

    # 初始化使用者暫存區與布林變數
    if user_id not in user_temp_files:
        user_temp_files[user_id] = []
        user_skeleton_buffers[user_id] = []
        user_update_counts[user_id] = 0
        user_can_save[user_id] = True  # 預設為 True，允許儲存
        user_post_fall_buffers[user_id] = []  # 初始化跌倒後緩衝區

    prediction_result = "Insufficient data"

    try:
        # 將上傳的影片存成 temporary 檔
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=TEMP_VIDEO_DIR) as tmp:
            video_path = tmp.name
            tmp.write(video_file.read())

        # 取得本次上傳影片的 FPS 和總幀數
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": "影片讀取失敗"}), 400

        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        current_file_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] User {user_id}: Current video FPS = {video_fps}, frames = {current_file_frames}")

        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            skeleton = extract_skeleton_points(frame)
            if skeleton:
                user_skeleton_buffers[user_id].append(skeleton)
                user_update_counts[user_id] += 1

            # 如果已偵測為跌倒，將幀存入跌倒後緩衝區
            if not user_can_save[user_id]:
                user_post_fall_buffers[user_id].append(frame)

            frame_index += 1
        cap.release()

        # 列印累計的骨架幀數
        total_frames = len(user_skeleton_buffers[user_id])
        print(f"[INFO] User {user_id}: Total accumulated skeleton frames = {total_frames}")

        # 如果累計骨架資料達到要求，進行推論
        if total_frames >= SMALL_BUFFER_SIZE:
            norm_data = normalize_skeleton_data(user_skeleton_buffers[user_id], scaler, time_steps=SMALL_BUFFER_SIZE)
            all_pred = model.predict(norm_data)
            print(f"[DEBUG] User {user_id}: Prediction result = {all_pred}")
            pred = all_pred[0][0]
            if pred >= 0.5:
                prediction_result = "fall"
                print(f"[INFO] User {user_id}: 跌倒事件偵測到。")
                user_can_save[user_id] = False  # 偵測到跌倒後禁止再次儲存
            else:
                prediction_result = "Non_fall"
                print(f"[INFO] User {user_id}: 未偵測到跌倒事件。")
                user_can_save[user_id] = True  # 回歸非跌倒時設為 True，允許儲存

            # 清除最早的影片檔案
            if user_temp_files[user_id]:
                old_file = user_temp_files[user_id].pop(0)
                if os.path.exists(old_file):
                    os.remove(old_file)

            # 清除已處理的骨架資料
            user_skeleton_buffers[user_id] = user_skeleton_buffers[user_id][BUFFER_UPDATE_SIZE:]

        # 將本次影片加入暫存區
        user_temp_files[user_id].append(video_path)
        response = jsonify({"id": user_id, "result": prediction_result})
        response.status_code = 200

        return response

    except Exception as e:
        print(f"[ERROR] User {user_id}: {e}")
        return jsonify({"error": str(e)}), 500