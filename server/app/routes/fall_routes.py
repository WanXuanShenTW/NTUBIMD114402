from flask import Blueprint, request, jsonify
from ..db import get_connection
import os
from werkzeug.utils import secure_filename
from datetime import datetime

fall_bp = Blueprint("fall_bp", __name__)
FALL_VIDEO_FOLDER = os.path.join("static", "fall_videos")
os.makedirs(FALL_VIDEO_FOLDER, exist_ok=True)

@fall_bp.route('/add_fall_incident', methods=['POST'])
def add_fall_incident():
    user_id = request.form.get('user_id')
    detected_time_str = request.form.get('detected_time')
    location = request.form.get('location')
    pose_before_fall = request.form.get('pose_before_fall')
    video_filename = request.files.get('video_filename')

    if not all([user_id, detected_time_str, location, pose_before_fall, video_filename]):
        return jsonify({'error': '缺少必要欄位或影片'}), 400

    try:
        detected_time = datetime.strptime(detected_time_str, "%Y-%m-%d %H:%M:%S")
        timestamp = detected_time.strftime('%Y%m%d_%H%M%S')
        filename = f"user{user_id}_{timestamp}.mp4"
        save_path = os.path.join(FALL_VIDEO_FOLDER, secure_filename(filename))
        video_filename.save(save_path)

        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""INSERT INTO fall_incident_report (user_id, detected_time, location, pose_before_fall, video_filename)
                       VALUES (%s, %s, %s, %s, %s)""", (user_id, detected_time, location, pose_before_fall, filename))
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'message': '跌倒事件已新增成功', 'video_filename': filename}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
