from flask import Blueprint, request, jsonify
from ..db import get_connection
from datetime import datetime, timedelta

video_bp = Blueprint("video_bp", __name__)

@video_bp.route("/video-events")
def get_video_events():
    user_id = request.args.get("user_id", type=int)
    start_str = request.args.get("start_date")
    end_str = request.args.get("end_date")

    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d") + timedelta(days=1)

    conn = get_connection()
    cursor = conn.cursor()
    query = """SELECT 'fall' AS event_type, user_id, detected_time AS start_time, video_filename
               FROM fall_events
               WHERE user_id = %s AND video_filename IS NOT NULL AND detected_time BETWEEN %s AND %s
               UNION ALL
               SELECT 'leave_bed' AS event_type, user_id, leave_time AS start_time, video_filename
               FROM leave_bed_events
               WHERE user_id = %s AND is_abnormal = 1 AND video_filename IS NOT NULL AND leave_time BETWEEN %s AND %s
               ORDER BY start_time DESC"""
    values = (user_id, start, end, user_id, start, end)
    cursor.execute(query, values)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    results = [{"event_type": r[0], "user_id": r[1], "start_time": r[2].strftime("%Y-%m-%d %H:%M:%S"), "video_filename": r[3]} for r in rows]
    return jsonify(results)
