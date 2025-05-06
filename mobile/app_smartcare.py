
from flask import Flask, request, jsonify,send_from_directory, abort
from flask_mysqldb import MySQL
import os
from datetime import datetime, timedelta
app = Flask(__name__)

# MySQL 設定（你已經有設定這段）
app.config['MYSQL_HOST'] = '140.131.114.242'
app.config['MYSQL_USER'] = 'smartcare_db'
app.config['MYSQL_PASSWORD'] = 'SmartCare114@2'
app.config['MYSQL_DB'] = '114-402'

mysql = MySQL(app)

#搜尋影片功能
@app.route("/video-events")
def get_video_events():
    user_id = request.args.get("user_id", type=int)
    start_str = request.args.get("start_date")
    end_str = request.args.get("end_date")
    
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d") + timedelta(days=1)  # 包含整天

    cursor = mysql.connection.cursor()

    query = """
        SELECT 'fall' AS event_type, user_id, detected_time AS start_time, video_filename
        FROM fall_events
        WHERE user_id = %s
          AND video_filename IS NOT NULL
          AND detected_time BETWEEN %s AND %s

        UNION ALL

        SELECT 'leave_bed' AS event_type, user_id, leave_time AS start_time, video_filename
        FROM leave_bed_events
        WHERE user_id = %s
          AND is_abnormal = 1
          AND video_filename IS NOT NULL
          AND leave_time BETWEEN %s AND %s

        ORDER BY start_time DESC
    """

    values = (user_id, start, end, user_id, start, end)
    cursor.execute(query, values)
    rows = cursor.fetchall()

    results = []
    for row in rows:
        results.append({
            "event_type": row[0],
            "user_id": row[1],
            "start_time": row[2].strftime("%Y-%m-%d %H:%M:%S"),
            "video_filename": row[3]
        })

    return jsonify(results)



#取出影片功能(根據查詢結果，要輸入事件類型、影片檔名)


# 設定影片根資料夾
VIDEO_BASE_DIR = os.path.join(app.root_path, 'static')

@app.route("/videos/<event_type>/<video_filename>")
def get_video_by_type(event_type, video_filename):
    # 根據事件類型，決定資料夾
    if event_type == "fall":
        folder = os.path.join(VIDEO_BASE_DIR, "fall_videos")
    elif event_type == "leave_bed":
        folder = os.path.join(VIDEO_BASE_DIR, "leave_bed_videos")
    else:
        return abort(404, description="Unknown event type.")

    try:
        return send_from_directory(folder, video_filename)
    except FileNotFoundError:
        return abort(404, description="Video file not found.")
    


# 加入蒐藏功能
@app.route('/add_favorite', methods=['POST'])
def add_favorite():
    data = request.get_json()

    user_id = data.get('user_id')
    video_filename = data.get('video_filename')
    video_type = data.get('video_type')

    # 基本驗證
    if not user_id or not video_filename or not video_type:
        return jsonify({"error": "缺少必要欄位"}), 400

    try:
        cur = mysql.connection.cursor()
        cur.execute('''
            INSERT INTO video_favorites (user_id, video_filename, video_type)
            VALUES (%s, %s, %s)
        ''', (user_id, video_filename, video_type))
        mysql.connection.commit()
        cur.close()

        return jsonify({
            "message": "影片已加入收藏",
            "video_filename": video_filename,
            "video_type": video_type
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

#瀏覽典藏影片
@app.route('/favorites/<int:user_id>', methods=['GET'])
def get_favorites(user_id):
    cur = mysql.connection.cursor()

    cur.execute('''
        SELECT video_filename, video_type, added_at
        FROM video_favorites
        WHERE user_id = %s
        ORDER BY added_at DESC
    ''', (user_id,))

    rows = cur.fetchall()
    cur.close()

    # 將查詢結果轉為 JSON 格式
    result = []
    for row in rows:
        result.append({
            "video_filename": row[0],
            "video_type": row[1],
            "added_at": row[2].strftime("%Y-%m-%d %H:%M:%S")
        })

    return jsonify({
        "user_id": user_id,
        "favorites": result
    })

#取消收藏
@app.route('/remove_favorite', methods=['POST'])
def remove_favorite():
    data = request.get_json()

    user_id = data.get('user_id')
    video_filename = data.get('video_filename')
    video_type = data.get('video_type')

    # 檢查是否缺少欄位
    if not user_id or not video_filename or not video_type:
        return jsonify({"error": "缺少必要欄位"}), 400

    try:
        cur = mysql.connection.cursor()

        cur.execute('''
            DELETE FROM video_favorites 
            WHERE user_id = %s AND video_filename = %s AND video_type = %s
        ''', (user_id, video_filename, video_type))

        mysql.connection.commit()
        cur.close()

        return jsonify({
            "message": "影片已取消收藏",
            "video_filename": video_filename,
            "video_type": video_type
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 根據影片檔名取得事件日期
@app.route('/video-date/<video_filename>', methods=['GET'])
def get_video_date(video_filename):
    cur = mysql.connection.cursor()

    # 嘗試從 fall_events 查找
    cur.execute("""
        SELECT detected_time FROM fall_events
        WHERE video_filename = %s
        LIMIT 1
    """, (video_filename,))
    row = cur.fetchone()

    # 如果找不到，就去 leave_bed_events 查
    if not row:
        cur.execute("""
            SELECT leave_time FROM leave_bed_events
            WHERE video_filename = %s
            LIMIT 1
        """, (video_filename,))
        row = cur.fetchone()

    cur.close()

    # 若仍找不到
    if not row:
        return jsonify({'error': '找不到對應的影片日期'}), 404

    # 成功回傳
    return jsonify({
        'video_filename': video_filename,
        'date': row[0].strftime('%Y/%m/%d')
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)