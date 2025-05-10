import math
import os
import tempfile
import datetime
import cv2
import numpy as np
from flask import Blueprint, request, jsonify
from ..fall_model import load_fall_model
from ..utils import extract_skeleton_points, normalize_skeleton_data
from ..db import get_connection

fall_detection_bp = Blueprint("fall_detection_bp", __name__)

TEMP_VIDEO_DIR = "tmp"
VIDEOS_DIR = os.path.join(os.getcwd(), "videos")

model, scaler = load_fall_model()

# 使用 temp 檔案儲存影片段
user_temp_files = {}       # 用於儲存跌倒前的影片檔 (pre-fall)
user_post_temp_files = {}  # 用於跌倒後上傳的影片檔 (post-fall)
user_skeleton_buffers = {}
user_update_counts = {}
user_buffer_filled = {}    # 紀錄每個使用者是否累計到足夠的幀數

SMALL_BUFFER_SIZE = 60       # 用於骨架推論最低骨架資料筆數
PREDICTION_INTERVAL = 30
BUFFER_UPDATE_SIZE = 30
SERVER_MAX_FPS = 30    # 伺服器最大處理 FPS

PRE_FALL_SECONDS = 2   # 前置影片秒數上限 (秒)
POST_FALL_SECONDS = 1  # 跌倒後影片秒數上限 (秒)
MIN_REQUIRED_FRAMES = PRE_FALL_SECONDS * SERVER_MAX_FPS  # 前置至少需要的幀數

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

def save_to_db(user_id, video_filename, result):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = """
                INSERT INTO fall_events (user_id, detected_time, video_filename, result)
                VALUES (%s, %s, %s, %s)
                """
        values = (user_id, datetime.datetime.now(), video_filename, result)
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
    global user_temp_files, user_post_temp_files, user_skeleton_buffers, user_update_counts, user_buffer_filled

    user_id = request.form.get("id")
    if not user_id:
        return jsonify({"error": "缺少使用者 ID"}), 400

    video_file = request.files.get("video")
    if not video_file:
        return jsonify({"error": "未接收到影片檔案"}), 400

    if not video_file.filename.lower().endswith(".mp4"):
        return jsonify({"error": "檔案格式不符，請上傳 MP4 影片"}), 400

    # 初始化使用者暫存區、骨架資料與狀態旗標
    if user_id not in user_temp_files:
        user_temp_files[user_id] = []
        user_post_temp_files[user_id] = []
        user_skeleton_buffers[user_id] = []
        user_update_counts[user_id] = 0
        user_buffer_filled[user_id] = False

    fall_detected = False
    prediction_result = "Insufficient data"

    try:
        # 將上傳的影片存成 temporary 檔（亂數名稱由 NamedTemporaryFile 產生）
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=TEMP_VIDEO_DIR) as tmp:
            video_path = tmp.name
            tmp.write(video_file.read())

        # 取得本次上傳影片的總幀數
        current_file_frames = get_frames_in_file(video_path)

        # 利用存檔的影片進行骨架抽取與跌倒判斷
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": "影片讀取失敗"}), 400

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        drop_rate = math.ceil(video_fps / SERVER_MAX_FPS) if video_fps > SERVER_MAX_FPS else 1
        
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 每秒抽取一個 frame 更新骨架資料（僅在尚未偵測跌倒時）
            if frame_index % max(1, int(video_fps)) == 0 and not fall_detected:
                skeleton = extract_skeleton_points(frame)
                if skeleton:
                    user_skeleton_buffers[user_id].append(skeleton)
                    user_update_counts[user_id] += 1

                if (len(user_skeleton_buffers[user_id]) >= SMALL_BUFFER_SIZE and 
                    user_update_counts[user_id] >= PREDICTION_INTERVAL):
                    norm_data = normalize_skeleton_data(user_skeleton_buffers[user_id], scaler, time_steps=SMALL_BUFFER_SIZE)
                    pred = model.predict(norm_data)[0][0]
                    if pred >= 0.5:
                        prediction_result = "fall"
                        fall_detected = True
                        print(f"[INFO] User {user_id}: 跌倒事件偵測到。")
                    else:
                        prediction_result = "Non_fall"
                    user_update_counts[user_id] = 0
                    if len(user_skeleton_buffers[user_id]) >= BUFFER_UPDATE_SIZE:
                        user_skeleton_buffers[user_id] = user_skeleton_buffers[user_id][BUFFER_UPDATE_SIZE:]
            frame_index += 1
        cap.release()

        # 累計前置暫存區影片的幀數，加上本次影片幀數
        prev_frames = get_total_frames_in_files(user_temp_files.get(user_id, []))
        accumulated_frames = prev_frames + current_file_frames
        print(f"[DEBUG] User {user_id}: Accumulated frames (pre-temp + current) = {accumulated_frames}")

        # 檢查是否已經累計到足夠的幀數並更新旗標
        if not user_buffer_filled[user_id]:
            if accumulated_frames >= MIN_REQUIRED_FRAMES:
                user_buffer_filled[user_id] = True
                print(f"[INFO] User {user_id}: Buffer has reached sufficient frames: {accumulated_frames}")
            else:
                prediction_result = "Insufficient data"
        else:
            # 一旦達到足夠幀數，以後回傳只會是 fall 或 Non_fall，如果預測結果仍然是不足，則訂為 Non_fall
            if prediction_result == "Insufficient data":
                prediction_result = "Non_fall"

        if prediction_result == "fall":
            # 若跌倒，將本次影片加入 post-fall 暫存區
            user_post_temp_files[user_id].append(video_path)
            max_post_frames = POST_FALL_SECONDS * SERVER_MAX_FPS
            update_post_files(user_id, max_post_frames)
            # 組合前置與後置影片檔
            combined_clip_paths = user_temp_files[user_id] + user_post_temp_files[user_id]
            combined_frames = []
            for clip in combined_clip_paths:
                cap_clip = cv2.VideoCapture(clip)
                while True:
                    ret, frame = cap_clip.read()
                    if not ret:
                        break
                    combined_frames.append(frame)
                cap_clip.release()
            video_filename = save_video(user_id, combined_frames)
            if video_filename:
                save_to_db(user_id, video_filename, "fall")
                return jsonify({"id": user_id, "result": "fall", "video": video_filename}), 200
            else:
                return jsonify({"error": "跌倒影片儲存失敗"}), 500

        else:
            # 若推論結果為 Non_fall 或 Insufficient data，將本次影片加入前置暫存區
            user_temp_files[user_id].append(video_path)
            update_temp_files(user_id, MIN_REQUIRED_FRAMES)
            return jsonify({"id": user_id, "result": prediction_result}), 200

    except Exception as e:
        print(f"[ERROR] User {user_id}: {e}")
        return jsonify({"error": str(e)}), 500