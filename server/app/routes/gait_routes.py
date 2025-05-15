from flask import Blueprint, request, jsonify
from ..db import get_connection
import os
from werkzeug.utils import secure_filename
from datetime import datetime

gait_bp = Blueprint("gait_bp", __name__)
GAIT_VIDEO_FOLDER = os.path.join("static", "gait_instability_videos")
os.makedirs(GAIT_VIDEO_FOLDER, exist_ok=True)

@gait_bp.route('/add_gait_instability', methods=['POST'])
def add_gait_instability():
    user_id = request.form.get('user_id')
    detected_time_str = request.form.get('detected_time')
    video = request.files.get('video')

    if not all([user_id, detected_time_str, video]):
        return jsonify({'error': '缺少必要欄位或影片'}), 400

    try:
        detected_time = datetime.strptime(detected_time_str, "%Y-%m-%d %H:%M:%S")
        timestamp = detected_time.strftime('%Y%m%d_%H%M%S')
        filename = f"user{user_id}_{timestamp}.mp4"
        save_path = os.path.join(GAIT_VIDEO_FOLDER, secure_filename(filename))
        video.save(save_path)

        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""INSERT INTO gait_instability_records (user_id, detected_time, video_filename)
                       VALUES (%s, %s, %s)""", (user_id, detected_time, filename))
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'message': '步態不穩事件已新增成功', 'video_filename': filename}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
