import math
import os
import shutil
import tempfile
import datetime
import threading
import cv2
import numpy as np
from flask import Blueprint, request, jsonify, send_file
from ..fall_model import load_fall_model
from ..utils import extract_skeleton_points, normalize_skeleton_data, interpolate_skeleton_data
from ..db import get_connection
from ..service.fall_video_service import list_fall_video_data_from_reange, get_video_filename_with_id

fall_bp = Blueprint("fall_bp", __name__)

TEMP_VIDEO_DIR = "sources/tmp"
VIDEOS_DIR = "sources/fall_videos"

# 清空 tmp 資料夾
def clear_temp_directory():
    if os.path.exists(TEMP_VIDEO_DIR):
        shutil.rmtree(TEMP_VIDEO_DIR)  # 刪除整個資料夾
    os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)  # 重新建立空的資料夾

# 在應用程式啟動時清空暫存區
clear_temp_directory()

model, scaler = load_fall_model()

# 使用 temp 檔案儲存影片段
user_temp_videos_path = {}            # 用於儲存跌倒前的影片檔 (pre-fall)
user_skeleton_buffers = {}
user_can_save = {}
user_processing_files = {}      # 用於儲存跌倒事件期間的影片檔案
user_is_falling = {}            # 用於標記使用者是否正在處理跌倒事件
user_falling_end = {}           # 用於標記使用者是否已經結束跌倒事件

SMALL_BUFFER_SIZE = 60          # 用於骨架推論最低骨架資料筆數
BUFFER_UPDATE_SIZE = 30
SERVER_MAX_FPS = 60             # 伺服器最大處理 FPS

FALL_THRESHOLD_UPPER = 0.7      # 預測值大於此為跌倒
FALL_THRESHOLD_LOWER = 0.5      # 預測值小於此為非跌倒

PRE_FALL_COUNTS = 4             # 前置影片筆數
POST_FALL_COUNTS = 2            # 跌倒後影片筆數

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

def save_to_db(user_id, location, pose_before_fall, video_filename):
    """
    儲存跌倒事件資訊到資料庫。

    :param user_id: 使用者 ID
    :param location: 跌倒事件發生的位置
    :param pose_before_fall: 跌倒前的姿勢
    :param video_filename: 儲存的影片檔案名稱
    :param result: 跌倒事件的結果（fall 或 non_fall）
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = """
                INSERT INTO fall_events (user_id, detected_time, location, pose_before_fall, video_filename)
                VALUES (%s, %s, %s, %s, %s)
                """
        values = (user_id, datetime.datetime.now(), location, pose_before_fall, video_filename)
        cursor.execute(query, values)
        conn.commit()

        # 獲取自動生成的 record_id
        record_id = cursor.lastrowid
        print(f"[INFO] DB 儲存成功:\n"
              f"record_id={record_id}\n"
              f"user_id={user_id}\n"
              f"location={location}\n"
              f"pose_before_fall={pose_before_fall}\n"
              f"video_filename={video_filename}")
        
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
    while get_total_frames_in_files(user_temp_videos_path.get(user_id, [])) > max_frames and user_temp_videos_path[user_id]:
        old = user_temp_videos_path[user_id].pop(0)
        if os.path.exists(old):
            os.remove(old)

def merge_videos(video_files, output_file):
    """
    合併指定的影片檔案清單，並儲存為一個影片檔。

    :param video_files: 影片檔案清單（完整路徑）
    :param output_file: 合併後影片的輸出路徑
    """
    if not video_files:
        print("❌ 未找到任何可合併的影片檔案。")
        return

    # 讀第一支影片取得影片參數
    cap0 = cv2.VideoCapture(video_files[0])
    if not cap0.isOpened():
        print(f"❌ 無法讀取影片：{video_files[0]}")
        return
    width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap0.get(cv2.CAP_PROP_FPS)
    cap0.release()

    # 確保輸出資料夾存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 初始化 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # 依序寫入每支影片
    for vf in video_files:
        print(f"▶️ 正在合併：{vf}")
        cap = cv2.VideoCapture(vf)
        if not cap.isOpened():
            print(f"⚠️ 無法讀取影片，已跳過：{vf}")
            continue
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()
    print(f"✅ 影片合併完成，已儲存至：{output_file}")


@fall_bp.route("/fall_video", methods=["POST"])
def detect_fall_video():
    # 全域變數
    global user_temp_videos_path, user_processing_files, user_skeleton_buffers, user_can_save, user_is_falling, user_falling_end 

    #確認使用者 ID 與影片檔案是否存在
    user_id = request.form.get("id")
    if not user_id:
        return jsonify({"error": "缺少使用者 ID"}), 400

    video_file = request.files.get("video")
    if not video_file:
        return jsonify({"error": "未接收到影片檔案"}), 400

    if not video_file.filename.lower().endswith(".mp4"):
        return jsonify({"error": "檔案格式不符，請上傳 MP4 影片"}), 400

    # 初始化使用者暫存區與布林變數
    if user_id not in user_temp_videos_path:
        user_temp_videos_path[user_id] = []
        user_processing_files[user_id] = []
        user_skeleton_buffers[user_id] = []
        user_can_save[user_id] = True     # 預設為 True，允許儲存
        user_is_falling[user_id] = False  # 預設為 False，表示未處理跌倒事件
        user_falling_end[user_id] = True  

    #預設結果為資料不足
    prediction_result = "Insufficient data"

    try:
        # 將上傳的影片存成 temporary 檔
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=TEMP_VIDEO_DIR) as tmp:
            video_path = tmp.name
            tmp.write(video_file.read())

        # 取得本次上傳影片資訊
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": "影片讀取失敗"}), 400

        # 取得影片的 FPS 與幀數
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        current_file_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] User {user_id}: Current video FPS = {video_fps}, frames = {current_file_frames}")

        frame_index = 0
        # 讀取影片並儲存骨架點
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            skeleton = extract_skeleton_points(frame)
            if skeleton:
                user_skeleton_buffers[user_id].append(skeleton)

            frame_index += 1
        cap.release()
        
        # 列印累計的骨架幀數
        total_frames = len(user_skeleton_buffers[user_id])
        print(f"[INFO] User {user_id}: Total accumulated skeleton frames = {total_frames}")

        if total_frames > SMALL_BUFFER_SIZE:
            # 先做線性插值補幀
            # skeleton_for_pred = interpolate_skeleton_data(
            #     user_skeleton_buffers[user_id], SMALL_BUFFER_SIZE
            # )
            # norm_data = normalize_skeleton_data(
            #     skeleton_for_pred, scaler, time_steps=SMALL_BUFFER_SIZE
            # )
            norm_data = normalize_skeleton_data(
                user_skeleton_buffers[user_id], scaler, time_steps=SMALL_BUFFER_SIZE
            )
            all_pred = model.predict(norm_data)
            print(f"[DEBUG] User {user_id}: Prediction result = {all_pred}")
            pred = all_pred[0][0]

            # 判斷結果
            if pred > FALL_THRESHOLD_UPPER:
                prediction_result = "fall"
                print(f"[INFO] User {user_id}: 跌倒事件偵測到。")

                if not user_is_falling[user_id]:
                    user_processing_files[user_id] = user_temp_videos_path[user_id].copy()
                    user_is_falling[user_id] = True
                    user_falling_end[user_id] = False

                user_processing_files[user_id].append(video_path)

                if len(user_processing_files[user_id]) >= (PRE_FALL_COUNTS + POST_FALL_COUNTS) and not user_falling_end[user_id]:
                    merged_video_path = os.path.join(VIDEOS_DIR, f"{user_id}_merged_fall_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.mp4")
                    merge_videos(user_processing_files[user_id], merged_video_path)
                    print(f"[INFO] User {user_id}: 合併影片已儲存至 {merged_video_path}")
                    # 清空兩個暫存區
                    user_processing_files[user_id] = []
                    user_temp_videos_path[user_id] = []

                    save_to_db(
                        user_id=user_id,
                        location="客廳",
                        pose_before_fall="走路中",
                        video_filename=os.path.basename(merged_video_path),
                        result="fall"
                    )
                    user_is_falling[user_id] = False
                    user_falling_end[user_id] = True

            elif pred < FALL_THRESHOLD_LOWER:
                prediction_result = "Non-fall"
                # 非跌倒
                if user_is_falling.get(user_id, False) and user_processing_files.get(user_id):
                    # 僅在等待合併時才強制合併
                    merged_video_path = os.path.join(VIDEOS_DIR, f"{user_id}_merged_fall_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.mp4")
                    merge_videos(user_processing_files[user_id], merged_video_path)
                    print(f"[INFO] User {user_id}: 合併影片已儲存至 {merged_video_path}")
                    save_to_db(
                        user_id=user_id,
                        location="客廳",
                        pose_before_fall="走路中",
                        video_filename=os.path.basename(merged_video_path),
                        result="fall"
                    )
                    user_processing_files[user_id] = []
                    user_temp_videos_path[user_id] = []
                    user_is_falling[user_id] = False
                    user_falling_end[user_id] = True

            else:
                prediction_result = "unknown"
                print(f"[INFO] User {user_id}: 預測值介於上下限，狀態未知，影片不合併。")
                # 不做影片合併，只繼續累積

            # 清除舊的骨架資料
            user_skeleton_buffers[user_id] = user_skeleton_buffers[user_id][BUFFER_UPDATE_SIZE:]

        # 將本次影片加入暫存區
        user_temp_videos_path[user_id].append(video_path)

        # 確保暫存檔案數量不超過上限
        update_temp_files(user_id, max_frames=PRE_FALL_COUNTS * SERVER_MAX_FPS)

        response = jsonify({"id": user_id, "result": prediction_result})
        response.status_code = 200

        return response

    except Exception as e:
        print(f"[ERROR] User {user_id}: {e}")
        return jsonify({"error": str(e)}), 500
    
@fall_bp.route("/fall_videos_data", methods=["GET"])
def get_merged_fall_videos():
    """
    查詢合併後的跌倒影片資料（只回傳資料庫資料，不傳送影片檔案）
    參數:
        user_id: int (必填)
        start_date: yyyy-mm-dd (可選)
        end_date: yyyy-mm-dd (可選)
        limit: int (可選，最多5，預設5)
    回傳:
        JSON 格式的影片資料
    """
    user_id = request.args.get("user_id", type=int)
    start_str = request.args.get("start_date")
    end_str = request.args.get("end_date")
    limit = request.args.get("limit", default=5, type=int)
    if limit > 5:
        limit = 5
    if limit < 1:
        limit = 1

    if not user_id:
        return jsonify({"error": "缺少 user_id 參數"}), 400

    # 處理時間區間
    try:
        if end_str:
            end = datetime.datetime.strptime(end_str, "%Y-%m-%d") + datetime.timedelta(days=1)
        else:
            end = datetime.datetime.now() + datetime.timedelta(days=1)
        if start_str:
            start = datetime.datetime.strptime(start_str, "%Y-%m-%d")
        else:
            start = None
    except Exception:
        return jsonify({"error": "日期格式錯誤，請用 yyyy-mm-dd"}), 400
       
    data = list_fall_video_data_from_reange(user_id, start, end, limit)
    # 統一格式
    for row in data:
        if isinstance(row["detected_time"], datetime.datetime):
            row["detected_time"] = row["detected_time"].strftime("%Y-%m-%d %H:%M:%S")
    return jsonify(data)

@fall_bp.route("/fall_video_file", methods=["GET"])
def get_merged_fall_video_file():
    record_id = request.args.get("record_id", type=int)
    if not record_id:
        return jsonify({"error": "缺少 record_id 參數"}), 400

    video_filename = get_video_filename_with_id(record_id)
    if not video_filename:
        return jsonify({"error": "找不到影片"}), 404

    video_path = os.path.join(VIDEOS_DIR, video_filename)
    if not os.path.exists(video_path):
        return jsonify({"error": "影片檔案不存在於伺服器"}), 404

    return send_file(
        os.path.abspath(video_path),
        as_attachment=True,
        mimetype="video/mp4"
    )