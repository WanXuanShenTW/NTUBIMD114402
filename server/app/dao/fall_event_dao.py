import datetime
from typing import Any, Dict, List, Optional

def insert_fall_event(
    conn,
    user_id: int,
    location: str,
    pose_before_fall: str,
    video_filename: str
) -> Optional[int]:
    """
    將跌倒事件紀錄插入資料庫。

    :param conn: 資料庫連線物件，由 service 層傳入
    :param user_id: 使用者 ID
    :param location: 跌倒發生位置（如 "廚房", "浴室"）
    :param pose_before_fall: 跌倒前的動作描述（如 "站立", "行走"）
    :param video_filename: 儲存的影片檔名（如 "fall_20250530_001.mp4"）
    :return: 成功回傳 record_id，失敗回傳 None
    """
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO fall_events (user_id, detected_time, location, pose_before_fall, video_filename)
            VALUES (%s, %s, %s, %s, %s)
        """
        values = (
            user_id,
            datetime.datetime.now(),
            location,
            pose_before_fall,
            video_filename
        )
        cursor.execute(query, values)
        conn.commit()
        record_id = cursor.lastrowid
        print(f"[INFO] 新增跌倒事件成功: record_id={record_id}")
        return record_id
    except Exception as e:
        print(f"[ERROR] 新增跌倒事件失敗: {e}")
        return None

def fetch_fall_events_by_user_and_time_range(
    conn,
    user_id: int,
    start: Optional[datetime.datetime] = None,
    end: Optional[datetime.datetime] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    try:
        cursor = conn.cursor()
        query = """
            SELECT record_id, user_id, detected_time, location, pose_before_fall, video_filename
            FROM `114-402`.fall_events
            WHERE user_id = %s AND video_filename IS NOT NULL
        """
        values = [user_id]
        if start and end:
            query += " AND detected_time BETWEEN %s AND %s"
            values.extend([start, end])
        elif start:
            query += " AND detected_time >= %s"
            values.append(start)
        elif end:
            query += " AND detected_time <= %s"
            values.append(end)
        query += " ORDER BY detected_time DESC LIMIT %s"
        values.append(limit)
        cursor.execute(query, tuple(values))
        rows = cursor.fetchall()
        result = []
        for row in rows:
            result.append({
                "record_id": row[0],
                "user_id": row[1],
                "detected_time": row[2],
                "location": row[3],
                "pose_before_fall": row[4],
                "video_filename": row[5]
            })
        return result
    except Exception as e:
        print(f"[ERROR] 查詢跌倒事件失敗: {e}")
        return []

def get_fall_event_video_filename_by_id(conn, record_id: int) -> Optional[str]:
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT video_filename FROM `114-402`.fall_events WHERE record_id = %s AND video_filename IS NOT NULL",
            (record_id,)
        )
        row = cursor.fetchone()
        if row and row[0]:
            return row[0]
        return None
    except Exception as e:
        print(f"[ERROR] 查詢影片檔名失敗: {e}")
        return None